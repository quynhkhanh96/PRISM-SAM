"""
Model construction and checkpoint I/O.

  build_model(exp_cfg, sam_checkpoint)   -> pretrained SAM + adaptation
  save_trainable(model, path)            -> only parameters with requires_grad
  load_finetuned(model, path)            -> new (trainable-only) OR legacy
                                            (full state_dict) checkpoints

Every driver (train, PR eval, CE eval, attention/causal analysis) builds its
model through `build_model` from the experiment's config.json, so analysis
code is variant-agnostic.

Legacy (pre-revision) checkpoints are full `model.state_dict()` dumps. They
load strictly into a model whose adaptation matches how they were trained
(LoRA submodule names are unchanged), which is how Batch 0 re-dumps
per-instance results for existing runs. Pair them with a config produced by
`legacy_adapt_cfg`.
"""

import torch

from . import adapt as _adapt

SAM1_TYPES = ("vit_b", "vit_l", "vit_h")


def build_sam1(model_type="vit_l", checkpoint=None):
    from segment_anything.build_sam import sam_model_registry
    if model_type not in SAM1_TYPES:
        raise ValueError(f"unknown SAM1 model_type {model_type!r}")
    return sam_model_registry[model_type](checkpoint=checkpoint)


def build_model(exp_cfg, sam_checkpoint=None):
    """
    exp_cfg: dict with at least
        backbone: 'sam1' (SAM2 builder added with the SAM2 port)
        model_type: 'vit_l'
        adapt: {...}  (see adapt.py)
    """
    backbone = exp_cfg.get("backbone", "sam1")
    if backbone == "sam1":
        model = build_sam1(exp_cfg.get("model_type", "vit_l"), sam_checkpoint)
    else:
        raise NotImplementedError(f"backbone {backbone!r} not wired yet")
    if exp_cfg.get("adapt") is not None:
        _adapt.apply_adaptation(model, exp_cfg["adapt"])
    return model


# ------------------------------------------------------------------ ckpt I/O

def trainable_state_dict(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


def save_trainable(model, path, extra=None):
    torch.save({"format": "prismsam-trainable-v1",
                "state_dict": trainable_state_dict(model),
                "extra": extra or {}}, path)


def load_finetuned(model, path, map_location="cpu"):
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict) and obj.get("format") == "prismsam-trainable-v1":
        sd = obj["state_dict"]
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        got = set(sd)
        if got != expected:
            raise RuntimeError(
                "checkpoint/variant mismatch\n"
                f"  missing   : {sorted(expected - got)[:10]}\n"
                f"  unexpected: {sorted(got - expected)[:10]}")
        res = model.load_state_dict(sd, strict=False)
        if res.unexpected_keys:
            raise RuntimeError(f"unexpected keys: {res.unexpected_keys[:10]}")
        return "trainable"
    # legacy: full state_dict -> strict load (fails loudly on variant mismatch)
    model.load_state_dict(obj, strict=True)
    return "legacy"


def legacy_adapt_cfg(finetune_mode="lora", scope="full", proj="qkvo", rank=8, alpha=8):
    """Adapt-config equivalent of a pre-revision run (mask tokens were always trainable)."""
    if finetune_mode == "full_decoder":
        return {"site": "dec", "param": "full", "tok": True}
    return {"site": "twt", "param": "lora", "tok": True,
            "lora": {"rank": rank, "alpha": alpha, "dropout": 0.05,
                     "scope": scope, "proj": proj}}
