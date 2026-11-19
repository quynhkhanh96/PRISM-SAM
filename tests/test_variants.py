"""
Pre-flight checks for the R1/R5 variants. Run BEFORE launching any training:

    python tests/test_variants.py            # CPU is enough (~ a few minutes)
    python tests/test_variants.py --device cuda

1. counts      trainable #params per variant == EXPECTED_TRAINABLE, and the
               trainable groups are exactly the intended ones.
2. identity    LoRA variants at init (B = 0) reproduce vanilla SAM outputs
               bit-for-bit in eval mode.
3. gradients   one forward/backward with the training loss: every trainable
               tensor receives a non-zero gradient (except the IoU head, which
               the recipe never trains), no frozen tensor receives one, and the
               optimizer list covers every trainable tensor.
4. ckpt        new trainable-only checkpoints round-trip; a checkpoint from
               one variant refuses to load into another; a legacy full
               state_dict of the pre-revision PRISM-SAM loads strictly into
               the new 'twt-lora' variant.

Random-init SAM is used (no checkpoint needed): these are structural tests.
ViT-L forward at 1024 px needs > 8 GB RAM on CPU (global-attention maps);
on a small machine run  --model_type vit_b --skip counts  for tests 2-4.
"""

import argparse
import copy
import gc
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from prismsam import adapt  # noqa: E402
from prismsam.losses import SAMMultimaskLoss, mask_iou_from_logits  # noqa: E402
from prismsam.lora import LoRALinear, LoRAFusedQKV  # noqa: E402
from prismsam.models import (build_sam1, load_finetuned, save_trainable,  # noqa: E402
                             legacy_adapt_cfg)
from prismsam.variants import VARIANTS, EXPECTED_TRAINABLE, variant_name  # noqa: E402

EXPECTED_GROUPS = {
    "tok": {"mask_tokens"},
    "twt-lora": {"twt", "mask_tokens"},
    "twt-full": {"twt", "mask_tokens"},
    "nontwt-full": {"hypernet_mlps", "upscaling", "mask_tokens"},
    "nontwt-full-tok0": {"hypernet_mlps", "upscaling"},
    "nontwt-lora": {"hypernet_mlps", "mask_tokens"},
    "enc-lora": {"image_encoder", "mask_tokens"},
    "dec-full": {"twt", "mask_tokens", "iou_token", "hypernet_mlps", "upscaling", "iou_head"},
}


def make_batch(device, K=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    image = torch.rand(3, 1024, 1024, generator=g) * 255
    pts = torch.randint(100, 900, (K, 1, 2), generator=g).float()
    batched_input = [{
        "image": image.to(device), "original_size": (1024, 1024),
        "point_coords": pts.to(device), "point_labels": torch.ones(K, 1, device=device),
    }]
    gt = (torch.rand(K, 256, 256, generator=g) > 0.5).float().to(device)
    return batched_input, gt


def perturb_lora_B(model, std=1e-2):
    """B is zero-initialised, so dL/dA == 0 at step 0. Test-only perturbation."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.B.weight.normal_(0, std)
            elif isinstance(m, LoRAFusedQKV):
                for b in m.B:
                    b.weight.normal_(0, std)


def test_counts(base):
    print("\n[1] trainable parameter counts")
    for name, cfg in VARIANTS.items():
        m = adapt.apply_adaptation(copy.deepcopy(base), cfg)
        rep = adapt.param_report(m)
        groups = set(rep["by_group"])
        ok = rep["trainable"] == EXPECTED_TRAINABLE[name] and groups == EXPECTED_GROUPS[name]
        print(f"  {'OK ' if ok else 'FAIL'} {name:18s} {rep['trainable']:>10,}  "
              f"(effective {rep['trainable_effective']:,})  {variant_name(cfg)}")
        print(f"       groups: {rep['by_group']}")
        assert ok, f"{name}: got {rep}, expected {EXPECTED_TRAINABLE[name]} / {EXPECTED_GROUPS[name]}"
        del m
        gc.collect()


def test_identity_at_init(base, device):
    print("\n[2] LoRA at init == vanilla SAM (eval mode)")
    batched_input, _ = make_batch(device)
    ref_model = copy.deepcopy(base).to(device).eval()
    with torch.no_grad():
        ref = ref_model(batched_input, multimask_output=True)[0]["low_res_logits"]
    del ref_model
    for name in ("twt-lora", "nontwt-lora", "enc-lora"):
        m = adapt.apply_adaptation(copy.deepcopy(base), VARIANTS[name]).to(device).eval()
        with torch.no_grad():
            out = m(batched_input, multimask_output=True)[0]["low_res_logits"]
        diff = (out - ref).abs().max().item()
        print(f"  {'OK ' if diff == 0 else 'FAIL'} {name:12s} max|diff| = {diff:.3e}")
        assert diff == 0, name
        del m
        gc.collect()


def test_gradients(base, device):
    print("\n[3] gradient flow (one step of the training loss)")
    crit = SAMMultimaskLoss(focal_weight=1.0, dice_weight=1.0, score_weight=1.0)
    batched_input, gt = make_batch(device)
    for name, cfg in VARIANTS.items():
        m = adapt.apply_adaptation(copy.deepcopy(base), cfg)
        perturb_lora_B(m)
        m.to(device).train()
        opt_params = adapt.trainable_parameters(m)
        n_req = sum(1 for p in m.parameters() if p.requires_grad)
        assert len(opt_params) == n_req

        out = m(batched_input, multimask_output=True)
        preds = torch.vstack([o["low_res_logits"] for o in out])
        loss = crit(preds, gt)["loss"]  # same call as train.py: no iou_preds
        loss.backward()
        # The loss back-propagates only the best-of-4 mask per prompt, so the
        # hypernetwork MLP of a mask index that wins for no prompt in this
        # batch legitimately gets no gradient at this step.
        with torch.no_grad():
            ious = torch.stack([mask_iou_from_logits(preds[:, i], gt)
                                for i in range(preds.shape[1])], 1)
        winners = set(ious.argmax(1).tolist())
        idle = {f"output_hypernetworks_mlps.{j}." for j in range(preds.shape[1]) if j not in winners}

        bad_dead, no_grad_ok, leaked = [], [], []
        for n, p in m.named_parameters():
            g = adapt._group_of(n)
            if p.requires_grad:
                has = p.grad is not None and p.grad.abs().sum().item() > 0
                if g in adapt.NO_GRAD_GROUPS:
                    no_grad_ok.append(n) if not has else None
                elif not has and not any(t in n for t in idle):
                    bad_dead.append(n)
            elif p.grad is not None:
                leaked.append(n)
        ok = not bad_dead and not leaked
        extra = f"  (IoU head without grad, expected: {len(no_grad_ok)} tensors)" if no_grad_ok else ""
        extra += f"  [winning mask idx {sorted(winners)}]" if idle and cfg['site'] in ('nontwt', 'dec') else ""
        print(f"  {'OK ' if ok else 'FAIL'} {name:18s} loss={loss.item():.4f}{extra}")
        assert not bad_dead, f"{name}: trainable but no gradient: {bad_dead[:8]}"
        assert not leaked, f"{name}: frozen but got gradient: {leaked[:8]}"
        del m, out, preds, loss, opt_params
        gc.collect()


def test_checkpoints(base):
    print("\n[4] checkpoint I/O")
    # new format round-trip
    m1 = adapt.apply_adaptation(copy.deepcopy(base), VARIANTS["nontwt-lora"])
    perturb_lora_B(m1)
    buf = io.BytesIO()
    save_trainable(m1, buf)
    buf.seek(0)
    m2 = adapt.apply_adaptation(copy.deepcopy(base), VARIANTS["nontwt-lora"])
    assert load_finetuned(m2, buf) == "trainable"
    for (n, a), (_, b) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(a, b), n
    print("  OK  trainable-only round trip (nontwt-lora)")
    del m1, m2
    gc.collect()

    # cross-variant load must fail
    buf.seek(0)
    m3 = adapt.apply_adaptation(copy.deepcopy(base), VARIANTS["twt-lora"])
    try:
        load_finetuned(m3, buf)
        raise AssertionError("loaded nontwt-lora ckpt into twt-lora without error")
    except RuntimeError:
        print("  OK  cross-variant load refused")
    del m3
    gc.collect()

    # legacy: pre-revision PRISM-SAM == twt-lora; full state_dict, strict
    legacy = adapt.apply_adaptation(copy.deepcopy(base), legacy_adapt_cfg())
    perturb_lora_B(legacy)
    buf = io.BytesIO()
    torch.save(legacy.state_dict(), buf)
    del legacy
    gc.collect()
    buf.seek(0)
    m4 = adapt.apply_adaptation(copy.deepcopy(base), VARIANTS["twt-lora"])
    assert load_finetuned(m4, buf) == "legacy"
    print("  OK  legacy full state_dict -> twt-lora (strict)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model_type", default="vit_l",
                    help="counts are only defined for vit_l; on a small-RAM CPU box run "
                         "identity/grads with vit_b (same block structure)")
    ap.add_argument("--skip", nargs="*", default=[], choices=["counts", "identity", "grads", "ckpt"])
    args = ap.parse_args()
    torch.manual_seed(0)
    base = build_sam1(args.model_type, checkpoint=None)
    if "counts" not in args.skip and args.model_type != "vit_l":
        print("[1] skipped: EXPECTED_TRAINABLE is for vit_l")
    elif "counts" not in args.skip:
        test_counts(base)
    if "identity" not in args.skip:
        test_identity_at_init(base, args.device)
    if "grads" not in args.skip:
        test_gradients(base, args.device)
    if "ckpt" not in args.skip:
        test_checkpoints(base)
    print("\nALL PASSED")
