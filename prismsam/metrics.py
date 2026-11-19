"""
Benchmark metrics.

Section 1 is copied VERBATIM from the pre-revision benchmark_metrics.py
(identical md5 across all six old code folders), so every number is computed
by the same code as the reported results. Do not edit it.

Section 2 (new) returns raw per-prompt / per-instance values for the
per-instance result files (R7 spec), built only from Section 1 functions.

Note on OccludedInstanceF1: it is NOT a dataset-level F1 from TP/FP/FN. It is
the per-instance Dice of the best mask over K x 4, averaged over instances with
>= 2 GT components of area >= min_component_area. Its per-instance value equals
`dice` restricted to `is_occluded`, so it is bootstrapped directly.
"""

# ============================ Section 1 (verbatim) ============================
from typing import List, Tuple, Dict, Optional
import numpy as np

# ====== basic utilities ======

def _to_bool(mask: np.ndarray) -> np.ndarray:
    # accept {0,1} uint8 or bool; returns bool
    return (mask > 0).astype(bool)

def iou(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = _to_bool(gt); pred = _to_bool(pred)
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return float(inter) / float(union + 1e-8)

def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = _to_bool(gt); pred = _to_bool(pred)
    inter = np.logical_and(gt, pred).sum()
    s = gt.sum() + pred.sum()
    return (2.0 * float(inter)) / float(s + 1e-8)

def best_of_n_iou(gt: np.ndarray, masks: List[np.ndarray]) -> Tuple[float, int]:
    scores = [iou(gt, m) for m in masks]
    j = int(np.argmax(scores))
    return scores[j], j

def best_of_n_dice(gt: np.ndarray, masks: List[np.ndarray]) -> Tuple[float, int]:
    scores = [dice(gt, m) for m in masks]
    j = int(np.argmax(scores))
    return scores[j], j

# ====== connected components ======

import cv2
def connected_components(mask: np.ndarray, connectivity: int = 8) -> Tuple[np.ndarray, int]:
    # OpenCV returns (num_labels, labels), where labels in {0..num_labels-1}
    num, labels = cv2.connectedComponents(_to_bool(mask).astype(np.uint8), connectivity=connectivity)
    return labels, num

def decompose_components(mask: np.ndarray, min_component_area: int = 0) -> List[np.ndarray]:
    """
    Decompose a binary mask into connected components, filtering out components
    smaller than min_component_area pixels.

    Args:
        mask: binary mask (H, W), uint8 or bool
        min_component_area: minimum number of pixels for a component to be kept.
            Use 0 (default) to keep all components.
            Use a positive value (e.g. 150) to filter annotation artifacts.

    Returns:
        List of bool component masks (H, W), one per valid component.
        Background (label 0) is always excluded.
    """
    labels, num = connected_components(mask)
    comps = []
    for lab in range(1, num):  # skip background label 0
        comp = (labels == lab)
        if comp.sum() >= min_component_area:
            comps.append(comp)
    return comps

# ====== prompt robustness ======

def mean_iou_per_prompt(
    gt: np.ndarray,
    preds_per_prompt: List[List[np.ndarray]],  # length K; each is list of 4 masks
) -> Tuple[float, float]:
    """
    For each prompt k, take best-of-4 IoU with GT; return (mean, variance) over K prompts.
    """
    best_ious = []
    for masks4 in preds_per_prompt:
        best, _ = best_of_n_iou(gt, masks4)
        best_ious.append(best)
    best_ious = np.array(best_ious, dtype=np.float32)
    return float(best_ious.mean()), float(best_ious.var())

# ====== occlusion-focused metrics ======

def _best_pred_mask_over_all_prompts(
    gt: np.ndarray,
    preds_per_prompt: List[List[np.ndarray]],
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Collapse K×4 predictions to the single predicted mask with highest IoU to GT.
    Returns (best_mask, best_iou, (prompt_idx, mask_idx_within_prompt)).
    """
    best_score, best_pm, best_mm = -1.0, -1, -1
    for pk, masks4 in enumerate(preds_per_prompt):
        s, j = best_of_n_iou(gt, masks4)
        if s > best_score:
            best_score, best_pm, best_mm = s, pk, j
    return preds_per_prompt[best_pm][best_mm], best_score, (best_pm, best_mm)

def remote_part_recall(
    gt: np.ndarray,
    preds_per_prompt: List[List[np.ndarray]],
    tau: float = 0.35,
    min_component_area: int = 0,
) -> float:
    """
    Component-level recall: fraction of GT connected components that are 'hit'
    by some predicted component with IoU >= tau. Uses the single best predicted
    mask over all prompts (best IoU to GT), then decomposes both GT and prediction
    into components.

    min_component_area is applied to both GT and predicted components, ensuring
    that annotation artifacts and spurious predicted blobs do not affect the metric.
    """
    gt_comps = decompose_components(gt, min_component_area)
    if len(gt_comps) == 0:
        return 1.0  # degenerate: no valid GT parts → treat as trivially satisfied

    best_pred, _, _ = _best_pred_mask_over_all_prompts(gt, preds_per_prompt)
    pred_comps = decompose_components(best_pred, min_component_area)
    if len(pred_comps) == 0:
        return 0.0

    hits = sum(
        1 for Gi in gt_comps
        if max(iou(Gi, Pj) for Pj in pred_comps) >= tau
    )
    return float(hits) / float(len(gt_comps))

def fragment_iou(
    gt: np.ndarray,
    preds_per_prompt: List[List[np.ndarray]],
    min_component_area: int = 0,
) -> float:
    """
    Average, over GT components, of the best IoU with any predicted component.
    Uses the single best predicted mask (over K×4) for component matching.

    min_component_area is applied to both GT and predicted components.
    """
    gt_comps = decompose_components(gt, min_component_area)
    if len(gt_comps) == 0:
        return 1.0

    best_pred, _, _ = _best_pred_mask_over_all_prompts(gt, preds_per_prompt)
    pred_comps = decompose_components(best_pred, min_component_area)
    if len(pred_comps) == 0:
        return 0.0

    vals = [max(iou(Gi, Pj) for Pj in pred_comps) for Gi in gt_comps]
    return float(np.mean(vals))

def occluded_instance_f1(
    gt: np.ndarray,
    preds_per_prompt: List[List[np.ndarray]],
    min_components: int = 2,
    min_component_area: int = 0,
) -> Optional[float]:
    """
    Dice/F1 computed only for occluded instances (GT with >= min_components valid components).
    Uses best mask over all prompts (K×4).
    Returns None if this instance does not qualify as occluded under the given thresholds.

    min_component_area is applied when counting GT components to determine occlusion status,
    preventing annotation artifacts from inflating OccludedInstanceCount.
    """
    if len(decompose_components(gt, min_component_area)) < min_components:
        return None
    best_pred, _, _ = _best_pred_mask_over_all_prompts(gt, preds_per_prompt)
    return dice(gt, best_pred)

# ====== core per-instance metrics ======

def instance_core_metrics(
    gt: np.ndarray,
    preds_per_prompt: List[List[np.ndarray]],
) -> Dict[str, float]:
    """
    Core: mIoU & Dice using best mask over all prompts (K×4).
    """
    best_pred, best_iou, _ = _best_pred_mask_over_all_prompts(gt, preds_per_prompt)
    return {
        "IoU": best_iou,
        "Dice": dice(gt, best_pred),
    }


# ============================ Section 2 (new) =================================

def remote_part_counts(gt, preds_per_prompt, tau, min_component_area):
    """(n_gt_components, n_hit) behind remote_part_recall; same matching rule."""
    gt_comps = decompose_components(gt, min_component_area)
    if len(gt_comps) == 0:
        return 0, 0
    best_pred, _, _ = _best_pred_mask_over_all_prompts(gt, preds_per_prompt)
    pred_comps = decompose_components(best_pred, min_component_area)
    if len(pred_comps) == 0:
        return len(gt_comps), 0
    hits = sum(1 for Gi in gt_comps if max(iou(Gi, Pj) for Pj in pred_comps) >= tau)
    return len(gt_comps), hits


def pr_instance_values(gt, preds_per_prompt, tau_remote=0.5, min_component_area=150,
                       occlusion_min_components=2):
    """
    Returns (instance_row, prompt_rows) for one test instance.
    preds_per_prompt: K prompts x 4 candidate masks.
    """
    prompt_rows = []
    best_ious = []
    for k, masks4 in enumerate(preds_per_prompt):
        s, j = best_of_n_iou(gt, masks4)
        best_ious.append(s)
        prompt_rows.append({"prompt_id": k, "iou": float(s),
                            "dice": float(dice(gt, masks4[j])), "best_mask_idx": int(j)})

    core = instance_core_metrics(gt, preds_per_prompt)
    m, v = mean_iou_per_prompt(gt, preds_per_prompt)          # float32, as reported
    rpr = remote_part_recall(gt, preds_per_prompt, tau=tau_remote,
                             min_component_area=min_component_area)
    n_comp, n_hit = remote_part_counts(gt, preds_per_prompt, tau_remote, min_component_area)
    if n_comp > 0:
        assert abs(rpr - n_hit / n_comp) < 1e-12, (rpr, n_hit, n_comp)
    frag = fragment_iou(gt, preds_per_prompt, min_component_area=min_component_area)
    occ = occluded_instance_f1(gt, preds_per_prompt, min_components=occlusion_min_components,
                               min_component_area=min_component_area)
    _, _, (bp, bm) = _best_pred_mask_over_all_prompts(gt, preds_per_prompt)

    inst = {
        "iou": float(core["IoU"]), "dice": float(core["Dice"]),
        "mean_iou_pp": float(m), "var_iou_pp": float(v),
        "rpr": float(rpr), "n_gt_components": int(n_comp), "n_comp_hit": int(n_hit),
        "fragment_iou": float(frag),
        "is_occluded": occ is not None,
        "occ_dice": float(occ) if occ is not None else float("nan"),
        "best_prompt_id": int(bp), "best_mask_idx": int(bm),
    }
    return inst, prompt_rows