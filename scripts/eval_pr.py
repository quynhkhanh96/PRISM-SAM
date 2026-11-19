"""
Prompt-robustness (PR) evaluation on OCHuman, with per-instance output.

Three ways to specify the model (exactly one):

  New experiment (config.json + trainable-only ckpt):
    python scripts/eval_pr.py --exp_dir <results>/<exp_name> --train_seed 0 --test_seed 0
    -> writes <exp_dir>/s0/eval/pr__och-test_s0.{instances,prompts}.parquet

  Legacy checkpoint (pre-revision full state_dict, Batch 0 re-dumps):
    python scripts/eval_pr.py --legacy_ckpt <old_run>/final.pth \
        --legacy_mode lora --scope full --proj qkvo --rank 8 --alpha 8 \
        --method och__twt-lora-scfull-qkvo-r8a8__tok1__... --train_seed 0 \
        --test_file <legacy_och_dir>/benchmark_data/ochuman_test_seed0.json --test_seed 0 \
        --out_dir <results>/<canonical_name>/s0/eval

  Zero-shot baseline (folder with config.json whose adapt is null, e.g. zs__sam1):
    python scripts/eval_pr.py --exp_dir <results>/zs__sam1 --test_seed 0
  or ad hoc:  --zs sam1 --test_seed 0 --method zs__sam1 --out_dir ...

--test_file defaults to <results_root>/benchmarks/och/test/test_s{test_seed}.json

Metric constants are those of the pre-revision test.py.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from prismsam import paths  # noqa: E402
from prismsam.config import test_path  # noqa: E402
from prismsam.datasets import och  # noqa: E402
from prismsam.metrics import pr_instance_values  # noqa: E402
from prismsam.models import build_model, load_finetuned, legacy_adapt_cfg  # noqa: E402
from prismsam.records import PRWriter  # noqa: E402
from prismsam.utils import read_image  # noqa: E402
from segment_anything.predictor import SamPredictor  # noqa: E402

TAU_REMOTE = 0.5
MIN_COMPONENT_AREA = 150
OCCLUSION_MIN_COMPONENTS = 2


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


def get_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--exp_dir")
    src.add_argument("--legacy_ckpt")
    src.add_argument("--zs", choices=["sam1"])
    p.add_argument("--train_seed", type=int, default=-1)
    p.add_argument("--test_file", default=None,
                   help="default: <results_root>/benchmarks/och/test/test_s{test_seed}.json")
    p.add_argument("--test_seed", type=int, required=True)
    p.add_argument("--bench", default=None, help="default: och-test_s{test_seed}")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--method", default=None)
    p.add_argument("--model_type", default="vit_l")
    p.add_argument("--device", default="cuda")
    # legacy-only
    p.add_argument("--legacy_mode", choices=["lora", "full_decoder"], default="lora")
    p.add_argument("--scope", default="full")
    p.add_argument("--proj", default="qkvo")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="debug: first N instances only")
    return p.parse_args()


def resolve(args):
    """-> (exp_cfg, ckpt_path or None, method, out_dir); sets args.train_seed=-1 for zero-shot."""
    if args.exp_dir:
        with open(os.path.join(args.exp_dir, "config.json")) as f:
            cfg = json.load(f)
        method = args.method or cfg.get("exp_name", os.path.basename(os.path.normpath(args.exp_dir)))
        if cfg.get("adapt") is None:                      # zero-shot baseline folder
            args.train_seed = -1
            sdir = os.path.join(args.exp_dir, f"s{args.test_seed}")
            return cfg, None, method, args.out_dir or os.path.join(sdir, "eval")
        if args.train_seed < 0:
            raise ValueError("--train_seed is required with --exp_dir")
        sdir = os.path.join(args.exp_dir, f"s{args.train_seed}")
        return (cfg, os.path.join(sdir, "ckpt", "final.pth"), method,
                args.out_dir or os.path.join(sdir, "eval"))
    if args.out_dir is None or args.method is None:
        raise ValueError("--out_dir and --method are required for --legacy_ckpt / --zs")
    if args.legacy_ckpt:
        if args.train_seed < 0:
            raise ValueError("--train_seed is required with --legacy_ckpt")
        cfg = {"backbone": "sam1", "model_type": args.model_type,
               "adapt": legacy_adapt_cfg(args.legacy_mode, args.scope, args.proj,
                                         args.rank, args.alpha)}
        return cfg, args.legacy_ckpt, args.method, args.out_dir
    args.train_seed = -1
    cfg = {"backbone": "sam1", "model_type": args.model_type, "adapt": None}
    return cfg, None, args.method, args.out_dir


def main():
    args = get_args()
    cfg, ckpt, method, out_dir = resolve(args)
    if args.test_file is None:
        args.test_file = test_path(och.DATASET, args.test_seed)
    bench = args.bench or f"och-test_s{args.test_seed}"

    model = build_model(cfg, sam_checkpoint=paths.sam1_checkpoint(cfg.get("model_type", "vit_l")))
    ckpt_kind = load_finetuned(model, ckpt) if ckpt else "zero-shot"
    model.to(args.device).eval()
    predictor = SamPredictor(model)
    print(f"[eval_pr] method={method} ckpt={ckpt} ({ckpt_kind}) bench={bench}")

    root = paths.get("ochuman_root")
    index = och.OchumanIndex.from_root(root)
    records = och.load_records(args.test_file)
    if args.limit:
        records = records[:args.limit]

    writer = PRWriter(method, args.train_seed, args.test_seed, och.DATASET, och.CATEGORY)
    with torch.no_grad():
        for rec in tqdm(records):
            predictor.set_image(read_image(och.image_path(root, rec)))
            gt = och.gt_mask(rec)
            preds_pp = []
            for point in rec["points"]:
                masks, _, _ = predictor.predict(point_coords=np.array([point]),
                                                point_labels=np.array([1]))
                preds_pp.append(masks)
            inst, prompts = pr_instance_values(
                gt, preds_pp, tau_remote=TAU_REMOTE, min_component_area=MIN_COMPONENT_AREA,
                occlusion_min_components=OCCLUSION_MIN_COMPONENTS)
            writer.add(index.instance_id(rec), inst, prompts)

    meta = {"config": cfg, "ckpt": ckpt, "ckpt_kind": ckpt_kind,
            "test_file": os.path.abspath(args.test_file), "test_file_md5": md5(args.test_file),
            "git": git_hash(), "tau_remote": TAU_REMOTE,
            "min_component_area": MIN_COMPONENT_AREA,
            "occlusion_min_components": OCCLUSION_MIN_COMPONENTS, "limit": args.limit}
    stem, agg = writer.save(out_dir, bench, meta)
    print(f"[eval_pr] saved {stem}.*")
    for k, v in agg.items():
        print(f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()