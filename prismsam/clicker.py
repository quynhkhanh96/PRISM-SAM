# prismsam/clicker.py — VERBATIM copy of the pre-revision shared_clicker.py
# (Concern #4). Torch-free; the same file is used by the SimpleClick env.
# Do not edit: every CE number (old and new) must come from this exact code.
"""
shared_clicker.py — copy this exact file into BOTH the `simpleclick` conda
env's working dir AND the PRISM-SAM env's working dir. Only depends on
numpy, cv2, copy -- no torch, no conflict between envs.

Section 1 (Clicker, Click): verbatim from isegm/inference/clicker.py.
Section 2: generic click-loop, model-agnostic. The predict_fn contract is:

    predict_fn(clicker) -> List[np.ndarray]   # candidate masks this round

For SimpleClick this list has length 1 (architecture has no ambiguity
output). For PRISM-SAM/SAM this list has length N (multimask_output=True).
Either way, run_click_loop selects the best candidate BY IoU VS GT (the
"annotator can tell a good mask from a bad one by eye" simplification --
this is the same criterion benchmark_metrics.remote_part_recall uses
internally, per its docstring: "Uses the single best predicted mask ...
best IoU to GT"), and that selected mask is what's fed forward into the
NEXT click's FN/FP computation -- consistent with what's being scored.
"""

import numpy as np
from copy import deepcopy
import cv2


# ============================================================
# Section 1 -- verbatim from isegm/inference/clicker.py
# ============================================================

class Clicker(object):
    def __init__(self, gt_mask=None, init_clicks=None, ignore_label=-1, click_indx_offset=0):
        self.click_indx_offset = click_indx_offset
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset_clicks()

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)

    def make_next_click(self, pred_mask):
        assert self.gt_mask is not None
        click = self._get_next_click(pred_mask)
        self.add_click(click)

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]

    def _get_next_click(self, pred_mask, padding=True):
        fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
        fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)

        return Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0]))

    def add_click(self, click):
        coords = click.coords
        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1
        self.clicks_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords
        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True

    def reset_clicks(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=bool)
        self.num_pos_clicks = 0
        self.num_neg_clicks = 0
        self.clicks_list = []

    def get_state(self):
        return deepcopy(self.clicks_list)

    def set_state(self, state):
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self):
        return len(self.clicks_list)


class Click:
    def __init__(self, is_positive, coords, indx=None):
        self.is_positive = is_positive
        self.coords = coords  # (y, x)
        self.indx = indx

    @property
    def coords_and_indx(self):
        return (*self.coords, self.indx)

    def copy(self, **kwargs):
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy


# ============================================================
# Section 2 -- generic driver loop
# ============================================================

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return 1.0 if union == 0 else float(inter) / float(union)


def select_best_candidate(candidates, gt_mask):
    """Best-vs-GT selection by IoU -- same criterion benchmark_metrics.
    remote_part_recall uses internally, so the mask fed forward here and
    the mask scored by remote_part_recall() are guaranteed to be the same
    one (both pick argmax IoU vs GT)."""
    ious = [compute_iou(c, gt_mask) for c in candidates]
    best_idx = int(np.argmax(ious))
    return candidates[best_idx], ious[best_idx]


def run_click_loop_with_artifacts(predict_fn, gt_mask: np.ndarray, k_max: int = 5,
                                    iou_target: float = 0.90, compute_rpr_fn=None):
    """
    Same protocol as run_click_loop, but additionally records every click
    (coords + sign) and every selected mask -- for qualitative visualization
    of a SMALL, hand-picked set of instances only. Do NOT use this for the
    full 3819x5 run (masks are expensive to store); use run_click_loop for
    that and this variant only for the qualitative-figure subset.

    Returns everything run_click_loop returns, PLUS:
        "clicks": list of (x, y, is_positive) tuples, one per round
        "masks": list of bool HxW arrays, one per round (the selected best
                 candidate that round -- same one used to generate the next click)
    """
    clicker = Clicker(gt_mask=gt_mask.astype(np.uint8))
    pred_mask = np.zeros_like(gt_mask, dtype=bool)

    iou_per_click, rpr_per_click = [], [] if compute_rpr_fn is not None else None
    clicks_record, masks_record = [], []
    noc_at_iou_target, noc_at_rpr1 = None, None

    for step in range(k_max):
        clicker.make_next_click(pred_mask)
        last_click = clicker.get_clicks()[-1]
        clicks_record.append(click_to_xy_label(last_click))

        candidates = predict_fn(clicker)
        best_mask, best_iou = select_best_candidate(candidates, gt_mask)
        pred_mask = best_mask.astype(bool)
        masks_record.append(pred_mask.copy())
        iou_per_click.append(best_iou)

        rpr = None
        if compute_rpr_fn is not None:
            rpr = compute_rpr_fn(gt_mask, candidates)
            rpr_per_click.append(rpr)

        n = step + 1
        if noc_at_iou_target is None and best_iou >= iou_target:
            noc_at_iou_target = n
        if compute_rpr_fn is not None and noc_at_rpr1 is None and rpr >= 1.0:
            noc_at_rpr1 = n
        if (noc_at_iou_target is not None) and (compute_rpr_fn is None or noc_at_rpr1 is not None):
            break

    return {
        "n_clicks_used": len(iou_per_click),
        "iou_per_click": iou_per_click,
        "rpr_per_click": rpr_per_click,
        "noc_at_iou_target": noc_at_iou_target,
        "noc_at_rpr1": noc_at_rpr1,
        "clicks": clicks_record,
        "masks": masks_record,
    }


def click_to_xy_label(click):
    """Convert a Click (coords=(y,x)) to (x, y, is_positive) for SAM-style APIs."""
    y, x = click.coords
    return x, y, click.is_positive


def run_click_loop(predict_fn, gt_mask: np.ndarray, k_max: int = 5,
                    iou_target: float = 0.90, compute_rpr_fn=None):
    """
    predict_fn: callable(clicker: Clicker) -> List[np.ndarray]
        Candidate masks for the CURRENT accumulated click state (len=1 for
        SimpleClick, len=N for PRISM-SAM/SAM multimask_output=True).
    compute_rpr_fn: callable(gt_mask, candidates_this_round) -> float in [0,1].
        Expected to internally do its own best-vs-GT selection (matches
        benchmark_metrics.remote_part_recall(gt, [candidates], tau=0.5,
        min_component_area=150) exactly -- pass a functools.partial with
        tau/min_component_area already bound). If None, RPR is skipped.

    Returns dict with n_clicks_used, iou_per_click, rpr_per_click,
    noc_at_iou_target (None if censored at k_max), noc_at_rpr1 (None if censored).
    """
    clicker = Clicker(gt_mask=gt_mask.astype(np.uint8))
    pred_mask = np.zeros_like(gt_mask, dtype=bool)  # feeds click round 1's FN/FP

    iou_per_click = []
    rpr_per_click = [] if compute_rpr_fn is not None else None
    noc_at_iou_target = None
    noc_at_rpr1 = None

    for step in range(k_max):
        clicker.make_next_click(pred_mask)
        candidates = predict_fn(clicker)  # List[np.ndarray], this round's N masks

        best_mask, best_iou = select_best_candidate(candidates, gt_mask)
        pred_mask = best_mask.astype(bool)  # forwarded to next round's click generation
        iou_per_click.append(best_iou)

        rpr = None
        if compute_rpr_fn is not None:
            rpr = compute_rpr_fn(gt_mask, candidates)
            rpr_per_click.append(rpr)

        n = step + 1
        if noc_at_iou_target is None and best_iou >= iou_target:
            noc_at_iou_target = n
        if compute_rpr_fn is not None and noc_at_rpr1 is None and rpr >= 1.0:
            noc_at_rpr1 = n

        done_iou = noc_at_iou_target is not None
        done_rpr = (compute_rpr_fn is None) or (noc_at_rpr1 is not None)
        if done_iou and done_rpr:
            break

    return {
        "n_clicks_used": len(iou_per_click),
        "iou_per_click": iou_per_click,
        "rpr_per_click": rpr_per_click,
        "noc_at_iou_target": noc_at_iou_target,
        "noc_at_rpr1": noc_at_rpr1,
    }