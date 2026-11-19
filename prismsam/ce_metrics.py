"""
Click-efficiency (CE) summaries. Section 1 is copied VERBATIM from the
pre-revision aggregate_noc_results.py (summarize) and noc_click1_analysis.py
(single_click_success_rate, post_click1_trend). Section 2 wraps them into the
per-file summary written next to each ce__*.jsonl.
"""

import numpy as np

# ============================ Section 1 (verbatim) ============================

def summarize(records, k_max=5, occluded_only=False):
    if occluded_only:
        records = [r for r in records if r["num_gt_components"] >= 2]

    n = len(records)
    if n == 0:
        return None

    noc_iou = [r["noc_at_iou_target"] for r in records]
    noc_rpr = [r["noc_at_rpr1"] for r in records]

    n_censored_iou = sum(1 for v in noc_iou if v is None)
    n_censored_rpr = sum(1 for v in noc_rpr if v is None)

    # Convention for censored instances (didn't reach target within k_max):
    # report mean/median over REACHED instances only, and separately report
    # % censored. Do NOT silently impute k_max+1 into the mean — that
    # understates how bad censored cases are AND conflates "just missed" with
    # "never gets there", which reviewers will (rightly) flag if not careful.
    reached_iou = [v for v in noc_iou if v is not None]
    reached_rpr = [v for v in noc_rpr if v is not None]

    def stats(vals):
        if not vals:
            return {"mean": None, "median": None}
        return {"mean": float(np.mean(vals)), "median": float(np.median(vals))}

    return {
        "n_instances": n,
        "k_max": k_max,
        "noc_at_iou_target": {
            **stats(reached_iou),
            "pct_censored": 100.0 * n_censored_iou / n,
            "n_censored": n_censored_iou,
        },
        "noc_at_rpr1": {
            **stats(reached_rpr),
            "pct_censored": 100.0 * n_censored_rpr / n,
            "n_censored": n_censored_rpr,
        },
    }


def single_click_success_rate(records, iou_target=0.90, rpr_target=1.0, occluded_only=False):
    if occluded_only:
        records = [r for r in records if r["num_gt_components"] >= 2]
    n = len(records)
    if n == 0:
        return None

    n_iou_hit = sum(1 for r in records
                     if len(r["iou_per_click"]) >= 1 and r["iou_per_click"][0] >= iou_target)
    n_rpr_hit = sum(1 for r in records
                     if r.get("rpr_per_click") and len(r["rpr_per_click"]) >= 1
                     and r["rpr_per_click"][0] >= rpr_target)

    return {
        "n": n,
        "pct_iou_hit_on_click1": 100.0 * n_iou_hit / n,
        "pct_rpr_hit_on_click1": 100.0 * n_rpr_hit / n,
    }


def post_click1_trend(records, iou_target=0.90, occluded_only=False, delta_eps=0.01):
    """Restricted to instances CENSORED at iou_target (never reached it
    within k_max) -- asks what happened to IoU after click 1 for exactly
    the cases the model 'struggled' on."""
    if occluded_only:
        records = [r for r in records if r["num_gt_components"] >= 2]

    deltas = []
    for r in records:
        vals = r["iou_per_click"]
        if len(vals) < 2:
            continue  # only 1 click recorded, nothing to compare
        if max(vals) >= iou_target:
            continue  # not censored -- excluded, this check is about censored cases only
        deltas.append(vals[-1] - vals[0])

    if not deltas:
        return {"n_censored_with_2plus_clicks": 0}

    n_improved = sum(1 for d in deltas if d > delta_eps)
    n_worsened = sum(1 for d in deltas if d < -delta_eps)
    n_plateau = len(deltas) - n_improved - n_worsened

    return {
        "n_censored_with_2plus_clicks": len(deltas),
        "mean_delta_click1_to_last": float(np.mean(deltas)),
        "pct_improved_gt_1pt": 100.0 * n_improved / len(deltas),
        "pct_plateaued": 100.0 * n_plateau / len(deltas),
        "pct_worsened_gt_1pt": 100.0 * n_worsened / len(deltas),
    }


# ============================ Section 2 (new) =================================

def ce_summary(records, k_max=5, iou_target=0.90, rpr_target=1.0):
    out = {}
    for label, occ in (("full", False), ("occluded", True)):
        out[label] = {
            "noc": summarize(records, k_max=k_max, occluded_only=occ),
            "click1": single_click_success_rate(records, iou_target=iou_target,
                                                rpr_target=rpr_target, occluded_only=occ),
            "post_click1_trend": post_click1_trend(records, iou_target=iou_target,
                                                   occluded_only=occ),
        }
    return out