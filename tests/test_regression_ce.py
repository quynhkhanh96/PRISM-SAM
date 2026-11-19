"""
Regression check for click efficiency: a new ce__*.jsonl against the
pre-revision noc_*_seed{s}.jsonl of the same model and seed.

    python tests/test_regression_ce.py --new <exp>/s0/eval/ce__och-test_s0.jsonl \
        --legacy <click-efficiency dir>/noc_prismsam_seed0.jsonl

Rows are paired by the old drivers' synthetic id ('{image_id}_{k:03d}', stored
as legacy_instance_id). Click 1 is a deterministic function of the GT, so
click-1 IoU must match to float noise. Later clicks depend on the previous
mask; cross-GPU noise (A40 vs A100) can move a click and make a trajectory
diverge, so aggregates are compared with tolerances expressed in instance
flips (1 instance = 0.026 pp on the full set, 0.043 pp on the occluded subset).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prismsam.ce_metrics import ce_summary  # noqa: E402

TOL_CLICK1_IOU = 1e-4       # per instance
TOL_PP_CLICK1 = 0.2         # percentage points, click-1 success rates
TOL_PP_CENSORED = 0.5       # percentage points, %censored
TOL_NOC_MEAN = 0.02         # clicks, mean NoC over reached instances


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def flat(s):
    out = {}
    for lab in ("full", "occluded"):
        c1, noc = s[lab]["click1"], s[lab]["noc"]
        out[f"{lab}/n"] = c1["n"]
        out[f"{lab}/click1 IoU>=.9 (%)"] = c1["pct_iou_hit_on_click1"]
        out[f"{lab}/click1 RPR=1 (%)"] = c1["pct_rpr_hit_on_click1"]
        for k, short in (("noc_at_iou_target", "NoC@90"), ("noc_at_rpr1", "NoC-RPR1")):
            out[f"{lab}/{short} mean"] = noc[k]["mean"]
            out[f"{lab}/{short} censored (%)"] = noc[k]["pct_censored"]
    return out


def tol_for(k):
    if k.endswith("/n"):
        return 0
    if "click1" in k:
        return TOL_PP_CLICK1
    if "censored" in k:
        return TOL_PP_CENSORED
    return TOL_NOC_MEAN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--legacy", required=True)
    args = ap.parse_args()

    new, old = load(args.new), load(args.legacy)
    old_by_id = {r["instance_id"]: r for r in old}
    ok = True
    if len(new) != len(old) or set(old_by_id) != {r["legacy_instance_id"] for r in new}:
        print(f"FAIL instance sets differ (new {len(new)}, legacy {len(old)})")
        sys.exit(1)

    n_comp_bad, c1_bad, diverged = 0, 0, 0
    for r in new:
        o = old_by_id[r["legacy_instance_id"]]
        n_comp_bad += r["num_gt_components"] != o["num_gt_components"]
        c1_bad += abs(r["iou_per_click"][0] - o["iou_per_click"][0]) > TOL_CLICK1_IOU
        same = (len(r["iou_per_click"]) == len(o["iou_per_click"]) and
                all(abs(a - b) <= TOL_CLICK1_IOU for a, b in zip(r["iou_per_click"], o["iou_per_click"])))
        diverged += not same
    n = len(new)
    print(f"instances: {n} | num_gt_components mismatch: {n_comp_bad} | "
          f"click-1 IoU mismatch (>{TOL_CLICK1_IOU:g}): {c1_bad} | "
          f"trajectories differing anywhere: {diverged} ({100 * diverged / n:.2f}%)")
    ok &= n_comp_bad == 0

    a, b = flat(ce_summary(old)), flat(ce_summary(new))
    print(f"{'metric':34s} {'legacy':>10s} {'new':>10s} {'|diff|':>9s}")
    for k in a:
        va, vb = a[k], b[k]
        if va is None or vb is None:
            good = va is None and vb is None
            d = float("nan")
        else:
            d = abs(va - vb)
            good = d <= tol_for(k)
        ok &= good
        fa = "None" if va is None else f"{va:10.4f}"
        fb = "None" if vb is None else f"{vb:10.4f}"
        print(f"{k:34s} {fa:>10s} {fb:>10s} {d:9.4f} {'OK' if good else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()