"""
Hard-case mining for OCHuman (TMLR R3; reusable for R9).

Section 1 is copied VERBATIM from the pre-revision Concern1/utils.py (the code
that mined the reported OCHuman few-shot sets; md5 of the source file
0bca22007e1841a23175c5433165c189, block md5 03ccdddb35ae72051ed7cb0fb8b54a11
after CRLF->LF). Do not edit it.

Section 2 (new) turns one SAM call into a criterion-free record: for every GT
connected component (8-connectivity, no area filter) its area and its best IoU
with the predicted components of the best-of-4 mask, once with all predicted
components (min_component_area=0) and once with predicted components >= 150 px.
From that record `is_hard(rec, tau, area)` reproduces

    remote_part_recall(gt, [preds], tau=tau, min_component_area=area) < 1.0

exactly (same components, same IoU values, same comparison), so every mining
criterion (tau, area) can be derived from one pass without re-running SAM.

Pre-revision OCHuman mining = tau 0.35, area 0 (min_component_area left at its
default 0 in create_benchmark.py), hard AND failure_mode == 'C' (>= 2 GT
components of >= 150 px), early stop at N.
"""

import cv2
import numpy as np

from . import metrics as M

# =========================================================================
# Section 1 -- VERBATIM (pre-revision Concern1/utils.py)
# =========================================================================

# --- Failure mode utils ---
PAIRED_OBJECT_CLASSES = {
    "skis",
}

def sample_point_largest_component_safe(
    gt_mask: np.ndarray,
    dist_percentile: float = 0.6,
    min_pixels: int = 30,
):
    """
    Sample a point from the largest GT component, avoiding boundary-adjacent pixels
    using distance transform.
    """
    gt_mask = gt_mask.astype(np.uint8)

    # 1) Decompose components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gt_mask, connectivity=8)

    if num_labels <= 1:
        # degenerate case
        ys, xs = np.where(gt_mask > 0)
        idx = np.random.randint(len(xs))
        return int(xs[idx]), int(ys[idx])

    # 2) Select largest component (skip background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)
    comp_mask = (labels == largest_label).astype(np.uint8)

    # 3) Distance transform inside largest component
    dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)

    if dist.max() == 0:
        ys, xs = np.where(comp_mask > 0)
        idx = np.random.randint(len(xs))
        return int(xs[idx]), int(ys[idx])

    # 4) Keep only pixels far enough from boundary
    thresh = np.percentile(dist[dist > 0], dist_percentile * 100)
    ys, xs = np.where(dist >= thresh)

    if len(xs) < min_pixels:
        # fallback: any pixel in largest component
        ys, xs = np.where(comp_mask > 0)

    idx = np.random.randint(len(xs))
    return int(xs[idx]), int(ys[idx])


def decompose_components(mask: np.ndarray, min_area: int = 0):
    """
    Decompose a binary mask into connected components.
    Returns a list of component masks (H, W).
    """
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    comps = []
    for lbl in range(1, num_labels):  # skip background = 0
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_area:
            comp = (labels == lbl).astype(np.uint8)
            comps.append(comp)

    return comps


def classify_failure_mode(
    gt_mask: np.ndarray,
    category_name: str,
    min_component_area: int = 32,
    debug: bool = False
):
    """
    Classify failure mode for a hard case (remote_part_recall < 1.0).

    Returns:
        "A" : single-component GT (thin / extension failure)
        "B" : paired-object semantic (2 components + paired class)
        "C" : true multi-component / occlusion
    """
    # Decompose GT components
    gt_comps = decompose_components(gt_mask, min_area=min_component_area)
    num_comps = len(gt_comps)
    if debug:
        print('#comp =', num_comps)

    # (A) Single-component GT
    if num_comps == 1:
        return "A"

    # (B) Paired-object semantics
    if num_comps == 2 and category_name.lower() in PAIRED_OBJECT_CLASSES:
        return "B"

    # (C) Everything else
    return "C"


# =========================================================================
# Section 2 -- criterion-free scan records (new)
# =========================================================================

PRED_AREAS = (0, 150)   # predicted-component filters stored per GT component
FAILURE_MODE_MIN_AREA = 150   # = MIN_COMPONENT_AREA in pre-revision create_benchmark.py


def _max_iou(gt_comp, pred_comps):
    # remote_part_recall returns 0.0 when there is no predicted component:
    # storing 0.0 makes `iou >= tau` False for every tau > 0, i.e. the same verdict.
    if not pred_comps:
        return 0.0
    return float(max(M.iou(gt_comp, p) for p in pred_comps))


def scan_record(gt_mask, pred_masks):
    """pred_masks: the 4 candidate masks SAM returns for ONE point prompt."""
    best_pred, best_iou, (_, best_idx) = M._best_pred_mask_over_all_prompts(gt_mask, [pred_masks])
    gt_comps = M.decompose_components(gt_mask, 0)
    pred_comps = {a: M.decompose_components(best_pred, a) for a in PRED_AREAS}
    comps = []
    for g in gt_comps:
        comps.append({"area": int(g.sum()),
                      **{f"iou_p{a}": _max_iou(g, pred_comps[a]) for a in PRED_AREAS}})
    return {
        "best_iou": float(best_iou), "best_mask_idx": int(best_idx),
        "n_pred_comps": {str(a): len(pred_comps[a]) for a in PRED_AREAS},
        "comps": comps,
        "failure_mode": classify_failure_mode(gt_mask=gt_mask, category_name="person",
                                              min_component_area=FAILURE_MODE_MIN_AREA),
    }


def rpr_from_record(rec, tau, area):
    """= remote_part_recall(gt, [preds], tau, min_component_area=area)."""
    if area not in PRED_AREAS:
        raise ValueError(f"area={area} not stored (have {PRED_AREAS})")
    comps = [c for c in rec["comps"] if c["area"] >= area]
    if not comps:
        return 1.0
    if rec["n_pred_comps"][str(area)] == 0:      # same order of checks as remote_part_recall
        return 0.0
    hits = sum(1 for c in comps if c[f"iou_p{area}"] >= tau)
    return float(hits) / float(len(comps))


def is_hard(rec, tau, area):
    return rpr_from_record(rec, tau, area) < 1.0


def is_selected(rec, tau, area):
    """Pre-revision selection rule: hard under (tau, area) AND failure mode C."""
    return rec["failure_mode"] == "C" and is_hard(rec, tau, area)


def missed_components(rec, tau, area):
    """GT components (under the area filter) not recovered at tau."""
    return [c for c in rec["comps"] if c["area"] >= area and c[f"iou_p{area}"] < tau]
