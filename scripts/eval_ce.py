"""
Click-efficiency (CE / NoC) evaluation on OCHuman, any SAM1 variant.

    python scripts/eval_ce.py --exp_dir <results>/<exp_name> --train_seed 0 --test_seed 0
    python scripts/eval_ce.py --exp_dir <results>/zs__sam1 --test_seed 0
    -> <exp_dir>/s{seed}/eval/ce__och-test_s{seed}.jsonl  (+ .summary.json)

Protocol = pre-revision Concern #4, unchanged:
  RITM/SimpleClick click simulation (prismsam/clicker.py, verbatim), k_max=5,
  IoU target 0.90 and RPR=1 target, multimask_output=True with best-vs-GT
  candidate selection, no mask input between rounds, loop stops once both
  targets are reached. The test file's pre-sampled `points` are NOT used:
  clicks are generated from the GT, so the protocol is deterministic and the
  test seed only selects the file (instances are identical across seeds).

One JSON line per instance (R7 CE spec; per-click rows are the lists):
  method, train_seed, test_seed, dataset, category, instance_id,
  legacy_instance_id ('{image_id}_{k:03d}', the old drivers' id, for
  regression only), num_gt_components, n_clicks_used, iou_per_click,
  rpr_per_click, noc_at_iou_target, noc_at_rpr1
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from prismsam import paths  # noqa: E402
from prismsam.ce_metrics import ce_summary  # noqa: E402
from prismsam.clicker import run_click_loop, click_to_xy_label  # noqa: E402
from prismsam.config import test_path  # noqa: E402
from prismsam.datasets import och  # noqa: E402
from prismsam.metrics import remote_part_recall, decompose_components  # noqa: E402
from prismsam.models import build_model, load_finetuned  # noqa: E402
from prismsam.utils import read_image  # noqa: E402
from segment_anything.predictor import SamPredictor  # noqa: E402

TAU_REMOTE = 0.5
MIN_COMPONENT_AREA = 150


def make_pred_fn(predictor):
    def pred_fn(clicker):
        xy = [click_to_xy_label(c) for c in clicker.get_clicks()]
        coords = np.array([[x, y] for (x, y, _) in xy])
        labels = np.array([1 if pos else 0 for (_, _, pos) in xy])
        masks, _, _ = predictor.predict(point_coords=coords, point_labels=labels,
                                        multimask_output=True)
        return list(masks)
    return pred_fn


def compute_rpr(gt_mask, candidates):
    return remote_part_recall(gt_mask, [list(candidates)], tau=TAU_REMOTE,
                              min_component_area=MIN_COMPONENT_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--train_seed", type=int, default=-1)
    ap.add_argument("--test_seed", type=int, required=True)
    ap.add_argument("--test_file", default=None)
    ap.add_argument("--k_max", type=int, default=5)
    ap.add_argument("--iou_target", type=float, default=0.90)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(os.path.join(args.exp_dir, "config.json")) as f:
        cfg = json.load(f)
    method = cfg.get("exp_name", os.path.basename(os.path.normpath(args.exp_dir)))
    if cfg.get("adapt") is None:
        args.train_seed, ckpt = -1, None
        sdir = os.path.join(args.exp_dir, f"s{args.test_seed}")
    else:
        if args.train_seed < 0:
            raise ValueError("--train_seed is required for a fine-tuned experiment")
        sdir = os.path.join(args.exp_dir, f"s{args.train_seed}")
        ckpt = os.path.join(sdir, "ckpt", "final.pth")
    out_dir = os.path.join(sdir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"ce__och-test_s{args.test_seed}")
    test_file = args.test_file or test_path(och.DATASET, args.test_seed)

    model = build_model(cfg, sam_checkpoint=paths.sam1_checkpoint(cfg.get("model_type", "vit_l")))
    ckpt_kind = load_finetuned(model, ckpt) if ckpt else "zero-shot"
    model.to(args.device).eval()
    predictor = SamPredictor(model)
    pred_fn = make_pred_fn(predictor)
    print(f"[eval_ce] method={method} ckpt={ckpt} ({ckpt_kind}) test={test_file}")

    root = paths.get("ochuman_root")
    index = och.OchumanIndex.from_root(root)
    records = och.load_records(test_file)
    if args.limit:
        records = records[:args.limit]

    keys = {"method": method, "train_seed": args.train_seed, "test_seed": args.test_seed,
            "dataset": och.DATASET, "category": och.CATEGORY}
    rows, per_image, seen = [], {}, set()
    with torch.no_grad(), open(stem + ".jsonl", "w") as fout:
        for rec in tqdm(records):
            k = per_image.get(rec["image_id"], 0)
            per_image[rec["image_id"]] = k + 1
            iid = index.instance_id(rec)
            if iid in seen:
                raise ValueError(f"duplicate instance_id {iid}")
            seen.add(iid)

            predictor.set_image(read_image(och.image_path(root, rec)))
            gt = och.gt_mask(rec).astype(bool)
            n_comp = len(decompose_components(gt, min_component_area=MIN_COMPONENT_AREA))
            res = run_click_loop(predict_fn=pred_fn, gt_mask=gt, k_max=args.k_max,
                                 iou_target=args.iou_target, compute_rpr_fn=compute_rpr)
            row = {**keys, "instance_id": iid, "legacy_instance_id": f"{rec['image_id']}_{k:03d}",
                   "num_gt_components": n_comp,
                   "n_clicks_used": res["n_clicks_used"],
                   "iou_per_click": [float(v) for v in res["iou_per_click"]],
                   "rpr_per_click": [float(v) for v in res["rpr_per_click"]],
                   "noc_at_iou_target": res["noc_at_iou_target"],
                   "noc_at_rpr1": res["noc_at_rpr1"]}
            fout.write(json.dumps(row) + "\n")
            rows.append(row)

    summary = {"keys": keys, "summary": ce_summary(rows, k_max=args.k_max, iou_target=args.iou_target),
               "meta": {"config": cfg, "ckpt": ckpt, "ckpt_kind": ckpt_kind,
                        "test_file": os.path.abspath(test_file), "k_max": args.k_max,
                        "iou_target": args.iou_target, "tau_remote": TAU_REMOTE,
                        "min_component_area": MIN_COMPONENT_AREA, "limit": args.limit,
                        "device_name": (torch.cuda.get_device_name()
                                        if args.device.startswith("cuda") and torch.cuda.is_available()
                                        else args.device),
                        "torch": torch.__version__, "cuda": torch.version.cuda}}
    with open(stem + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    s = summary["summary"]
    for lab in ("full", "occluded"):
        c1, noc = s[lab]["click1"], s[lab]["noc"]
        print(f"[{lab}] n={c1['n']}  click1 IoU>=0.9: {c1['pct_iou_hit_on_click1']:.2f}%  "
              f"RPR=1: {c1['pct_rpr_hit_on_click1']:.2f}%  | NoC@90 reached-mean "
              f"{noc['noc_at_iou_target']['mean']}  censored {noc['noc_at_iou_target']['pct_censored']:.1f}%")
    print(f"[eval_ce] saved {stem}.jsonl")


if __name__ == "__main__":
    main()