"""
Train one variant for one seed (OCHuman, SAM1).

    python scripts/train.py --config configs/och_twt-lora.yaml --seed 0

Layout written (results_root/results/<exp_name>/):
    config.json          resolved config (fixed settings included); written once,
                         later seeds must resolve to the identical config
    s{seed}/train.log    args, #params by group, per-epoch losses
    s{seed}/train_meta.json   few-shot file path + md5, git hash, param report
    s{seed}/ckpt/final.pth    trainable parameters only (prismsam-trainable-v1)

Recipe = pre-revision train.py, unchanged:
  - global seed 42 set before model construction (LoRA init draws from it in
    the same module order as before), shuffle=True DataLoader, verbatim
    augmentation and collate_fn;
  - AdamW(wd=0), grad-clip 1.0, AMP, multimask_output=True, best-of-4 loss
    called WITHOUT iou_preds (so the IoU head never trains);
  - cosine LR stepped once per EPOCH with T_max = #epochs;
  - no validation, no checkpoint selection: final state is saved.
The schedule is specified in total optimisation steps; epochs = total_steps /
steps_per_epoch must be an integer (N=100, batch 2, 5000 steps -> 100 epochs).

Optimizer receives adapt.trainable_parameters(model): every trainable tensor,
image-encoder adapters included.
"""

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from collections import defaultdict  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from prismsam import adapt, paths  # noqa: E402
from prismsam.augment import SegmentationAugmentor, collate_fn  # noqa: E402
from prismsam.datasets import och  # noqa: E402
from prismsam.losses import SAMMultimaskLoss  # noqa: E402
from prismsam.models import build_model, save_trainable  # noqa: E402
from prismsam.variants import VARIANTS, variant_name  # noqa: E402

TRAIN_DEFAULTS = {
    "lr": 1e-4, "total_steps": 5000, "batch_instances": 2, "K": 3,
    "weight_decay": 0.0, "max_grad_norm": 1.0, "amp": True,
    "focal_weight": 1.0, "dice_weight": 1.0, "score_weight": 1.0,
    "global_seed": 42,
}


# ------------------------------------------------------------------ helpers

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fmt_lr(lr):
    m, e = f"{lr:.0e}".split("e")
    return f"{m}e{int(e)}"            # 1e-4, 3e-5


def fmt_tau(tau):
    return f"{int(round(tau * 100)):03d}"   # 0.35 -> 035


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_hash():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def resolve_config(raw):
    """yaml -> fully explicit config (what config.json stores)."""
    cfg = {"data": raw.get("data", "och"), "backbone": raw.get("backbone", "sam1"),
           "model_type": raw.get("model_type", "vit_l")}
    if cfg["data"] != "och":
        raise NotImplementedError("only data='och' is wired in this script")
    if "variant" in raw:
        if "adapt" in raw:
            raise ValueError("give either 'variant' or 'adapt', not both")
        cfg["variant"] = raw["variant"]
        cfg["adapt"] = json.loads(json.dumps(VARIANTS[raw["variant"]]))
    else:
        cfg["adapt"] = raw["adapt"]
    adapt.validate_adapt_cfg(cfg["adapt"])

    t = dict(TRAIN_DEFAULTS)
    unknown = set(raw.get("train", {})) - set(t)
    if unknown:
        raise ValueError(f"unknown train keys: {unknown}")
    t.update(raw.get("train", {}))
    t["lr"] = float(t["lr"])
    cfg["train"] = t

    b = raw["benchmark"]
    cfg["benchmark"] = {"tau": float(b["tau"]), "N": int(b["N"]), "fewshot": b["fewshot"]}

    cfg["exp_name"] = raw.get("exp_name") or "__".join([
        cfg["data"], variant_name(cfg["adapt"]), f"lr{fmt_lr(t['lr'])}",
        f"tau{fmt_tau(cfg['benchmark']['tau'])}-N{cfg['benchmark']['N']}",
        f"steps{t['total_steps']}"])
    return cfg


def fewshot_path(cfg, seed):
    keys = dict(paths._load())
    return cfg["benchmark"]["fewshot"].format(seed=seed, **keys)


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True, help="few-shot (train) seed: 0..4, pilot 10/11")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true", help="allow re-training an existing seed")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = resolve_config(yaml.safe_load(f))
    t = cfg["train"]

    exp_dir = os.path.join(paths.get("results_root"), "results", cfg["exp_name"])
    sdir = os.path.join(exp_dir, f"s{args.seed}")
    ckpt_path = os.path.join(sdir, "ckpt", "final.pth")
    if os.path.exists(ckpt_path) and not args.overwrite:
        raise FileExistsError(f"{ckpt_path} exists (use --overwrite)")
    os.makedirs(os.path.join(sdir, "ckpt"), exist_ok=True)

    cfg_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            old = json.load(f)
        if old != cfg:
            raise RuntimeError(f"{cfg_path} differs from the resolved config of this run")
    else:
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

    log_path = os.path.join(sdir, "train.log")
    logf = open(log_path, "a")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    log("=== CONFIG ===\n" + json.dumps(cfg, indent=2) + f"\nseed={args.seed}\n==============")

    # --- model (seed BEFORE construction: LoRA init consumes the RNG) ---
    set_seed(t["global_seed"])
    model = build_model(cfg, sam_checkpoint=paths.sam1_checkpoint(cfg["model_type"]))
    rep = adapt.param_report(model)
    log(f"Trainable: {rep['trainable']:,} (effective {rep['trainable_effective']:,}) "
        f"by group: {rep['by_group']}")
    model.to(args.device)

    # --- data ---
    fs_path = fewshot_path(cfg, args.seed)
    ds = och.OCHumanPromptDataset(
        ochuman_root=paths.get("ochuman_root"), fewshot_path=fs_path,
        num_shots=cfg["benchmark"]["N"], num_prompts=t["K"],
        augment=SegmentationAugmentor(target_size=1024), device=args.device)
    loader = DataLoader(ds, batch_size=t["batch_instances"], collate_fn=collate_fn, shuffle=True)
    steps_per_epoch = len(loader)
    epochs = t["total_steps"] / steps_per_epoch
    if not float(epochs).is_integer():
        raise ValueError(f"total_steps={t['total_steps']} not a multiple of steps_per_epoch="
                         f"{steps_per_epoch} (N={cfg['benchmark']['N']}, batch={t['batch_instances']})")
    epochs = int(epochs)
    log(f"few-shot: {fs_path} (md5 {md5(fs_path)}), steps/epoch={steps_per_epoch}, epochs={epochs}")

    with open(os.path.join(sdir, "train_meta.json"), "w") as f:
        json.dump({"seed": args.seed, "fewshot": os.path.abspath(fs_path), "fewshot_md5": md5(fs_path),
                   "git": git_hash(), "param_report": rep, "steps_per_epoch": steps_per_epoch,
                   "epochs": epochs}, f, indent=2)

    # --- optimisation (identical to pre-revision train_lora_fixed) ---
    criterion = SAMMultimaskLoss(focal_weight=t["focal_weight"], dice_weight=t["dice_weight"],
                                 score_weight=t["score_weight"])
    params = adapt.trainable_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = t["amp"] and args.device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    steps_done = 0
    for epoch in range(1, epochs + 1):
        model.train()
        n, run = 0, defaultdict(float)
        for batched_input, targets in tqdm(loader, total=steps_per_epoch, leave=False):
            if not len(batched_input):
                continue
            bs = len(batched_input)
            n += bs
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(batched_input, multimask_output=True)
                preds = torch.vstack([o["low_res_logits"] for o in out])
                loss_dict = criterion(preds, targets)
                loss = loss_dict["loss"]
            scaler.scale(loss).backward()
            if t["max_grad_norm"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, t["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            steps_done += 1
            run["loss_total"] += loss.item() * bs
            for k, v in loss_dict.items():
                if k != "loss_total":
                    run[k] += v.item() * bs
        scheduler.step()
        avg = {k: v / max(1, n) for k, v in run.items()}
        if not math.isfinite(avg.get("loss_total", float("inf"))):
            log(f"[WARNING] non-finite loss at epoch {epoch}")
        log(f"Epoch {epoch:03d}/{epochs} | lr={scheduler.get_last_lr()[0]:.2e} | "
            + " | ".join(f"{k}={v:.5f}" for k, v in avg.items()))

    save_trainable(model, ckpt_path, extra={"exp_name": cfg["exp_name"], "seed": args.seed,
                                            "steps_done": steps_done})
    log(f"Done: {steps_done} optimiser steps ({epochs} epochs). Saved {ckpt_path}")
    logf.close()


if __name__ == "__main__":
    main()