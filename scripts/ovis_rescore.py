"""
R3 OVIS diagnostic, part A: re-score the EXISTING OVIS hard-case pools (and
hence every 100-shot training set, which are subsets of them) with SAM1 at the
STORED click point, recording per-component IoUs, so that any mining criterion
(tau, area) can be evaluated without re-mining.

    python scripts/ovis_rescore.py --ovis_root /data/khanhdtq/ovis_data \
        --pool_dir   <EDA-OVIS>/ovis_hardcases_by_category_sam1_seed0_trainvalid \
        --fewshot_dir <EDA-OVIS>            # holds ovis_fewshot_{cat}_train_seed{s}.json
    python scripts/ovis_rescore.py ... --categories Dog,Horse,Cat,Turtle,Vehical   # 5 only
    python scripts/ovis_rescore.py ... --summary_only

What it answers
  * fraction of each pool / each 100-shot set that is still hard under OCHuman's
    criterion (0.35, 0), under (0.35, 150) and (0.25, 150);
  * a LOWER BOUND on the pool size a tau=0.35 mining would have produced (a real
    re-mining could also pick frames that were blocked by min_frame_gap /
    max_per_track after a tau=0.5 hit);
  * how far the missed components are from being recovered (IoU bands).
Sanity: tau 0.5 / area 150 recomputed from the record must equal the stored rpr
(mismatches counted; GPU numerical noise can flip borderline components).

SAM call identical to the pre-revision SAM1PointPredictor: SamPredictor.predict
with one positive point, default multimask (4 masks), masks cast to uint8,
best-of-4 against GT inside remote_part_recall.

Output: <out_dir>/rescore.jsonl (one line per pool case, resumable),
        <out_dir>/ovis_rescore.md, ovis_rescore_pool.csv, ovis_rescore_fewshot.csv
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from prismsam import mining  # noqa: E402

CHOSEN = ["Dog", "Horse", "Cat", "Turtle", "Vehical"]
ORIG = (0.5, 150)                                        # pre-revision OVIS criterion
CRITERIA = [(0.5, 150), (0.35, 150), (0.25, 150), (0.35, 0), (0.5, 0)]
SEEDS = [0, 1, 2, 3, 4]


def case_key(c):
    return f"{c['video_id']}:{c['ann_id']}:{c['frame_idx']}:{c['point_xy'][0]}:{c['point_xy'][1]}"


def load_pools(pool_dir, categories):
    pools = {}
    for f in sorted(glob.glob(os.path.join(pool_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(f))[0]
        if categories and name not in categories:
            continue
        with open(f) as fh:
            pools[name] = json.load(fh)
    return pools


def rescore(args, pools, out_path):
    from prismsam.datasets import ovis
    from prismsam.models import build_sam1
    from prismsam.utils import read_image
    import torch
    from segment_anything.predictor import SamPredictor

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(line)["key"] for line in f}
    jobs = {}
    for cat, cases in pools.items():
        for c in cases:
            k = case_key(c)
            if k not in done:
                jobs.setdefault(c["image_path"], []).append((cat, c, k))
    print(f"{sum(len(v) for v in jobs.values())} cases to score on {len(jobs)} frames "
          f"({len(done)} already done)")
    if not jobs:
        return
    index = ovis.OvisIndex(ovis.load_merged(args.ovis_root))
    if args.ckpt:
        ckpt = args.ckpt
    else:
        from prismsam import paths
        ckpt = paths.sam1_checkpoint(args.model_type)
    sam = build_sam1(args.model_type, ckpt)
    sam.to(device=args.device)
    sam.eval()   # SAM1 has no dropout / batch-norm: identical to the pre-revision call
    predictor = SamPredictor(sam)
    t0 = time.time()
    with open(out_path, "a") as out:
        for n, (img, items) in enumerate(sorted(jobs.items())):
            predictor.set_image(read_image(os.path.join(args.ovis_root, img)))
            for cat, c, k in items:
                if index.image_rel_path(c["ann_id"], c["frame_idx"]) != img:
                    raise RuntimeError(f"{k}: image path mismatch with annotations")
                gt = index.gt_mask(c["ann_id"], c["frame_idx"])
                with torch.no_grad():
                    masks = predictor.predict(point_coords=np.array([c["point_xy"]]),
                                              point_labels=np.array([1]))[0]
                masks = [m.astype(np.uint8) for m in masks]
                rec = mining.scan_record(gt, masks)
                rec.pop("failure_mode")          # OCHuman-specific field, not used here
                out.write(json.dumps({"key": k, "category": cat, "stored_rpr": c["rpr"],
                                      "occlusion_label": c["occlusion_label"], **rec}) + "\n")
            if n % 200 == 0:
                out.flush()
                print(f"  {n}/{len(jobs)} frames, {time.time() - t0:.0f}s")


def summarize(args, pools, out_path, out_dir):
    recs = {}
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            recs[r["key"]] = r
    missing = [case_key(c) for cs in pools.values() for c in cs if case_key(c) not in recs]
    if missing:
        raise RuntimeError(f"{len(missing)} pool cases not scored yet (e.g. {missing[:3]})")

    pool_rows, sanity = [], {}
    for cat, cases in pools.items():
        rs = [recs[case_key(c)] for c in cases]
        bad = sum(abs(mining.rpr_from_record(r, *ORIG) - r["stored_rpr"]) > 1e-9 for r in rs)
        sanity[cat] = bad
        row = {"category": cat, "pool_tau050a150": len(rs), "sanity_mismatch": bad}
        for crit in CRITERIA[1:]:
            k = f"hard_tau{int(crit[0]*100):03d}a{crit[1]}"
            row[k] = sum(mining.is_hard(r, *crit) for r in rs)
        row["lb_pool_tau035a150_ge100"] = row["hard_tau035a150"] >= 100
        pool_rows.append(row)

    fs_rows = []
    for cat in [c for c in CHOSEN if c in pools]:
        for s in SEEDS:
            p = os.path.join(args.fewshot_dir, f"ovis_fewshot_{cat}_train_seed{s}.json")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                rs = [recs[case_key(c)] for c in json.load(f)]
            row = {"category": cat, "seed": s, "n": len(rs)}
            for crit in CRITERIA:
                row[f"frac_hard_tau{int(crit[0]*100):03d}a{crit[1]}"] = float(np.mean(
                    [mining.is_hard(r, *crit) for r in rs]))
            fs_rows.append(row)

    # IoU of missed components (orig criterion: area >= 150, IoU < 0.5), chosen categories
    bands = [(0.0, 0.05), (0.05, 0.25), (0.25, 0.35), (0.35, 0.5)]
    band_rows = []
    for cat in [c for c in CHOSEN if c in pools] + ["ALL"]:
        cats = list(pools) if cat == "ALL" else [cat]
        v = np.array([c_["iou_p150"] for k_ in cats for c in pools[k_]
                      for c_ in recs[case_key(c)]["comps"]
                      if c_["area"] >= 150 and c_["iou_p150"] < 0.5])
        band_rows.append((cat, len(v), [100 * np.mean((v >= a) & (v < b)) if len(v) else np.nan
                                        for a, b in bands]))

    for name, rows in [("ovis_rescore_pool.csv", pool_rows), ("ovis_rescore_fewshot.csv", fs_rows)]:
        if rows:
            with open(os.path.join(out_dir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    L = ["# R3 — OVIS re-scoring of the existing hard-case pools (stored click points)", "",
         "Pool = all cases mined pre-revision with (τ=0.5, area 150). Re-scored with SAM1 at the "
         "stored point. `hard_tauXXXaYYY` = number still hard under that criterion. Under a lower τ "
         "a real re-mining could also visit frames blocked by min_frame_gap / max_per_track after a "
         "τ=0.5 hit, so the τ=0.35 counts are LOWER BOUNDS of a τ=0.35 pool.", "",
         f"Sanity (stored rpr reproduced at τ=0.5/150): {sum(sanity.values())} mismatching cases "
         f"of {sum(len(c) for c in pools.values())}.", "",
         "## Pools", "", "| category | pool (0.5,150) | (0.35,150) | (0.25,150) | (0.35,0) OCH crit. | "
         "(0.5,0) | ≥100 at (0.35,150) (lower bound) |", "|---|---|---|---|---|---|---|"]
    for r in sorted(pool_rows, key=lambda r: -r["pool_tau050a150"]):
        b = "**" if r["category"] in CHOSEN else ""
        L.append(f"| {b}{r['category']}{b} | {r['pool_tau050a150']} | {r['hard_tau035a150']} | "
                 f"{r['hard_tau025a150']} | {r['hard_tau035a0']} | {r['hard_tau050a0']} | "
                 f"{'yes' if r['lb_pool_tau035a150_ge100'] else 'no'} |")
    if fs_rows:
        L += ["", "## 100-shot training sets (chosen categories): fraction still hard, mean over seeds (min–max)",
              "", "| category | " + " | ".join(f"({t}, {a})" for t, a in CRITERIA) + " |",
              "|---|" + "---|" * len(CRITERIA)]
        for cat in CHOSEN:
            sub = [r for r in fs_rows if r["category"] == cat]
            if not sub:
                continue
            cells = []
            for t, a in CRITERIA:
                v = np.array([r[f"frac_hard_tau{int(t*100):03d}a{a}"] for r in sub])
                cells.append(f"{v.mean():.3f} ({v.min():.2f}–{v.max():.2f})")
            L.append(f"| {cat} | " + " | ".join(cells) + " |")
    L += ["", "## Missed components (area ≥150, IoU < 0.5 at the stored click): % per IoU band", "",
          "| category | n missed comps | " + " | ".join(f"[{a}, {b})" for a, b in bands) + " |",
          "|---|---|" + "---|" * len(bands)]
    for cat, n, pct in band_rows:
        L.append(f"| {cat} | {n} | " + " | ".join(f"{p:.1f}" for p in pct) + " |")
    md = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "ovis_rescore.md"), "w") as f:
        f.write(md)
    print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ovis_root", required=True)
    ap.add_argument("--pool_dir", required=True)
    ap.add_argument("--fewshot_dir", required=True)
    ap.add_argument("--categories", default=None, help="comma list; default: every pool file")
    ap.add_argument("--out_dir", default=None, help="default <results_root>/analysis/R3/ovis")
    ap.add_argument("--model_type", default="vit_l")
    ap.add_argument("--ckpt", default=None, help="default: paths.sam1_checkpoint(model_type)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--summary_only", action="store_true")
    args = ap.parse_args()
    if args.out_dir is None:
        from prismsam import paths
        args.out_dir = os.path.join(paths.get("results_root"), "analysis", "R3", "ovis")
    os.makedirs(args.out_dir, exist_ok=True)
    cats = set(args.categories.split(",")) if args.categories else None
    pools = load_pools(args.pool_dir, cats)
    print("pools:", {k: len(v) for k, v in pools.items()})
    out_path = os.path.join(args.out_dir, "rescore.jsonl")
    if not args.summary_only:
        rescore(args, pools, out_path)
    summarize(args, pools, out_path, args.out_dir)


if __name__ == "__main__":
    main()