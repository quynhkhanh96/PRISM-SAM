"""
Adaptation sites for PRISM-SAM and the R1/R5 controls (TMLR Review 1).

One entry point: `apply_adaptation(model, adapt_cfg)`. The config is a dict
(normally the `adapt` block of an experiment's config.json):

    site:   'tok' | 'twt' | 'nontwt' | 'dec' | 'enc'
    param:  'lora' | 'full' | None          (None only for site='tok')
    tok:    bool  — mask tokens trainable (True in every main variant)
    lora:   {rank, alpha, dropout, scope, proj, last_k}   (param='lora' only)

Variant definitions (agreed Nov 19, 2026):

  tok          mask tokens only.
  twt-lora     LoRA on TWT attention projections; scope in {full, min, final},
               proj in {qkvo, qk, vo}. PRISM-SAM = scope=full, proj=qkvo, r8a8.
  twt-full     every TWT parameter (transformer.*), full rank.
  nontwt-full  output_hypernetworks_mlps + output_upscaling, full rank.
               IoU head and iou_token are EXCLUDED: the IoU head receives no
               gradient in our recipe (score loss is never fed iou_preds) and
               cannot affect any metric (PR and CE both select best-of-4 by
               IoU vs GT).
  nontwt-lora  LoRA on all nn.Linear of output_hypernetworks_mlps.
               Output-upscaling transposed convs stay FROZEN (LoRA applied to
               dense weight matrices only; they are trained in nontwt-full).
               Main setting: r18a18 (param-matched to TWT-LoRA, alpha/r = 1).
  enc-lora     LoRA (independent q, k, v, o adapters) on the last `last_k`
               ViT blocks of SAM1's image encoder. Main setting: last_k=2, r6a6.
  dec-full     the whole mask_decoder, full rank (pre-revision Concern-3 run).
               Kept exactly as before, IoU head included (it stays at its
               pretrained value; see `param_report`).

`tok` must be True for 'dec' (it is part of the decoder) and is ignored for
'tok'. Every unknown value raises — a typo must never silently train the
wrong variant.
"""

from collections import OrderedDict

from torch import nn

from .lora import LoRALinear, LoRAFusedQKV

SITES = ("tok", "twt", "nontwt", "dec", "enc")
PARAMS = ("lora", "full")
TWT_SCOPES = ("full", "min", "final")
PROJS = ("qkvo", "qk", "vo")
_PROJ_NAMES = {
    "qkvo": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "qk": ["q_proj", "k_proj"],
    "vo": ["v_proj", "out_proj"],
}


# ---------------------------------------------------------------- helpers

def get_mask_decoder(model):
    """SAM1: model.mask_decoder; SAM2: model.sam_mask_decoder."""
    if hasattr(model, "mask_decoder"):
        return model.mask_decoder
    if hasattr(model, "sam_mask_decoder"):
        return model.sam_mask_decoder
    raise AttributeError("model has neither 'mask_decoder' (SAM1) nor 'sam_mask_decoder' (SAM2)")


def _set_trainable(module, flag=True):
    for p in module.parameters():
        p.requires_grad_(flag)


def _lora_cfg(cfg):
    lc = cfg.get("lora") or {}
    for k in ("rank", "alpha"):
        if k not in lc:
            raise ValueError(f"param='lora' requires lora.{k}")
    return lc["rank"], lc["alpha"], lc.get("dropout", 0.05)


# ------------------------------------------------------------- TWT (LoRA)

def loraize_attention(attn, proj, rank, alpha, dropout):
    """TWT Attention module (q_proj, k_proj, v_proj, out_proj)."""
    if proj not in _PROJ_NAMES:
        raise ValueError(f"unknown proj {proj!r}; expected one of {PROJS}")
    for name in _PROJ_NAMES[proj]:
        base = getattr(attn, name)  # AttributeError if missing — intended
        if isinstance(base, LoRALinear):
            raise RuntimeError(f"{name} is already LoRA-wrapped")
        setattr(attn, name, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))


def lora_twt(decoder, scope, proj, rank, alpha, dropout):
    """Same sites as pre-revision lora.py::inject_lora_into_mask_decoder."""
    if scope not in TWT_SCOPES:
        raise ValueError(f"unknown scope {scope!r}; expected one of {TWT_SCOPES}")
    twt = decoder.transformer
    if scope in ("min", "full"):
        for block in twt.layers:
            loraize_attention(block.cross_attn_token_to_image, proj, rank, alpha, dropout)
            if scope == "full":
                loraize_attention(block.self_attn, proj, rank, alpha, dropout)
                loraize_attention(block.cross_attn_image_to_token, proj, rank, alpha, dropout)
    loraize_attention(twt.final_attn_token_to_image, proj, rank, alpha, dropout)


# --------------------------------------------------------- non-TWT (LoRA)

def lora_hypernetworks(decoder, rank, alpha, dropout):
    """
    LoRA on every nn.Linear of output_hypernetworks_mlps (4 MLPs x 3 layers).
    Replaces layers by index in MLP.layers — the pre-revision
    `lora_hyper_mlp` flag used setattr(mlp, 'Linear', ...) and therefore never
    replaced anything (flag was always False, so no reported result is
    affected).
    """
    n = 0
    for mlp in decoder.output_hypernetworks_mlps:
        for i, layer in enumerate(mlp.layers):
            if not isinstance(layer, nn.Linear):
                raise TypeError(f"unexpected layer type in hypernetwork MLP: {type(layer)}")
            mlp.layers[i] = LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout)
            n += 1
    return n


# ------------------------------------------------------- image encoder (LoRA)

def lora_image_encoder(model, last_k, proj, rank, alpha, dropout):
    if not hasattr(model, "image_encoder") or not hasattr(model.image_encoder, "blocks"):
        raise NotImplementedError("enc-lora is implemented for SAM1's ViT image encoder only")
    if proj != "qkvo":
        raise ValueError("enc-lora supports proj='qkvo' only")
    blocks = model.image_encoder.blocks
    if not (1 <= last_k <= len(blocks)):
        raise ValueError(f"last_k must be in [1, {len(blocks)}], got {last_k}")
    for blk in blocks[len(blocks) - last_k:]:
        attn = blk.attn
        attn.qkv = LoRAFusedQKV(attn.qkv, rank=rank, alpha=alpha, dropout=dropout)
        attn.proj = LoRALinear(attn.proj, rank=rank, alpha=alpha, dropout=dropout)


# ---------------------------------------------------------------- entry point

def validate_adapt_cfg(cfg):
    site, param = cfg.get("site"), cfg.get("param")
    if site not in SITES:
        raise ValueError(f"unknown site {site!r}; expected one of {SITES}")
    if site == "tok":
        if param not in (None, "none"):
            raise ValueError("site='tok' takes no param")
    elif param not in PARAMS:
        raise ValueError(f"unknown param {param!r}; expected one of {PARAMS}")
    if site == "enc" and param != "lora":
        raise ValueError("site='enc' is defined for param='lora' only")
    if site == "dec" and param == "lora":
        raise ValueError("site='dec' is defined for param='full' only (twt/nontwt cover LoRA)")
    if site == "dec" and not cfg.get("tok", True):
        raise ValueError("site='dec' trains the whole decoder, mask tokens included: tok must be True")
    if "tok" not in cfg and site != "tok":
        raise ValueError("adapt.tok must be set explicitly")


def apply_adaptation(model, cfg):
    """Freeze everything, then open exactly the parameters of the variant."""
    validate_adapt_cfg(cfg)
    site, param = cfg["site"], cfg.get("param")
    dec = get_mask_decoder(model)

    _set_trainable(model, False)

    if site == "twt":
        if param == "lora":
            r, a, d = _lora_cfg(cfg)
            lc = cfg["lora"]
            lora_twt(dec, lc.get("scope", "full"), lc.get("proj", "qkvo"), r, a, d)
        else:
            _set_trainable(dec.transformer, True)

    elif site == "nontwt":
        if param == "lora":
            r, a, d = _lora_cfg(cfg)
            lora_hypernetworks(dec, r, a, d)
        else:
            _set_trainable(dec.output_hypernetworks_mlps, True)
            _set_trainable(dec.output_upscaling, True)

    elif site == "enc":
        r, a, d = _lora_cfg(cfg)
        lc = cfg["lora"]
        lora_image_encoder(model, lc["last_k"], lc.get("proj", "qkvo"), r, a, d)

    elif site == "dec":
        _set_trainable(dec, True)

    # mask tokens (site='tok' always; others per flag)
    if site == "tok" or cfg.get("tok", False):
        _set_trainable(dec.mask_tokens, True)

    return model


def trainable_parameters(model):
    """What the optimizer must receive — ALL trainable params, not only the decoder's."""
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------- reporting

def _group_of(name):
    if name.startswith(("mask_decoder.", "sam_mask_decoder.")):
        sub = name.split(".", 1)[1]
        for prefix, g in (
            ("transformer.", "twt"),
            ("mask_tokens.", "mask_tokens"),
            ("iou_token.", "iou_token"),
            ("output_hypernetworks_mlps.", "hypernet_mlps"),
            ("output_upscaling.", "upscaling"),
            ("iou_prediction_head.", "iou_head"),
        ):
            if sub.startswith(prefix):
                return g
        return "decoder_other"
    if name.startswith("image_encoder."):
        return "image_encoder"
    if name.startswith("prompt_encoder."):
        return "prompt_encoder"
    return "other"


# Parameters that are trainable in some variants but never receive gradient
# under our recipe (see module docstring). Reported separately.
NO_GRAD_GROUPS = ("iou_head",)


def param_report(model):
    groups = OrderedDict()
    for name, p in model.named_parameters():
        if p.requires_grad:
            g = _group_of(name)
            groups[g] = groups.get(g, 0) + p.numel()
    total = sum(groups.values())
    effective = total - sum(groups.get(g, 0) for g in NO_GRAD_GROUPS)
    return {"by_group": dict(groups), "trainable": total, "trainable_effective": effective}
