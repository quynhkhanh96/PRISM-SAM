"""
Remote-component attention enrichment (Section 5) and raw attention readouts (R2).

Section 1 is copied VERBATIM from the pre-revision
Rerun-ablations-and-analysis/attention_alignment_analysis.py
(md5 8de93bf8ddcdb20acb192a9bc2dafb01), constants and "Core metric" +
"GT component count" blocks, so A_enrich_remote is computed by the same code as
the reported Section 5 / S4.2 numbers. Do not edit it.

Section 2 (new) reads every token->image attention of the TWT for the point
token, per head, and records the raw attention mass on GT regions, so that
enrichment for any layer / head aggregation can be recomputed later from the
saved files without rerunning inference. Section 2 reuses the Section 1
component logic; `region_masks_64` must reproduce `get_remote_component_masks`
exactly for the remote union (checked at runtime by eval_attn.py).
"""

# ============================ Section 1 (verbatim) ============================
import cv2
import numpy as np

tau_remote         = 0.5
min_component_area = 150
POINT_TOKEN_IDX    = 5    # 0=iou, 1–4=mask tokens, 5=point token
ATTN_GRID          = 64   # SAM image embedding spatial size: 64×64 = 4096 tokens
N_TOKENS           = ATTN_GRID * ATTN_GRID   # 4096

# ── Core metric ───────────────────────────────────────────────────────────────

def get_point_attn_map(attn_module, eps=1e-9):
    """
    Extract the point token's head-averaged attention over image tokens.
    Returns np.ndarray of shape [64, 64], or None.
    """
    if not hasattr(attn_module, 'last_attn') or attn_module.last_attn is None:
        return None
    attn = attn_module.last_attn  # [B, H, Q, K]
    if attn.size(-2) <= POINT_TOKEN_IDX:
        return None
    # Average over heads, drop batch → [K]
    point_attn = attn[0].mean(dim=0)[POINT_TOKEN_IDX]   # [4096]
    point_attn = point_attn / (point_attn.sum() + eps)
    return point_attn.cpu().numpy().reshape(ATTN_GRID, ATTN_GRID)  # [64, 64]


SAM_INPUT_SIZE = 1024   # SAM resizes longest side to 1024 before padding


def sam_preprocess_mask(mask, img_h, img_w, feat_size=ATTN_GRID):
    """
    Replicate SAM's image preprocessing pipeline on a binary mask so that
    the downsampled mask aligns with the 64×64 attention token grid.

    Pipeline:
      1. Resize longest side to 1024, preserve aspect ratio
      2. Pad shorter side to 1024×1024 (right and bottom padding)
      3. Downsample 1024×1024 → feat_size×feat_size (64×64)

    Returns binary np.ndarray [feat_size, feat_size].
    """
    target = SAM_INPUT_SIZE

    if img_h >= img_w:
        scale = target / float(img_h)
        new_h, new_w = target, int(round(img_w * scale))
    else:
        scale = target / float(img_w)
        new_w, new_h = target, int(round(img_h * scale))

    mask_resized = cv2.resize(mask.astype(np.uint8), (new_w, new_h),
                              interpolation=cv2.INTER_NEAREST)

    mask_padded = np.zeros((target, target), dtype=np.uint8)
    mask_padded[:new_h, :new_w] = mask_resized

    mask_ds = cv2.resize(mask_padded, (feat_size, feat_size),
                         interpolation=cv2.INTER_NEAREST)
    return mask_ds.astype(bool)


def transform_point_to_sam_space(px, py, img_h, img_w):
    """
    Apply the same scale (no padding needed for point lookup — padding is
    zero-filled so a point outside new_h×new_w is background by definition).

    Returns (px_new, py_new) in the resized (pre-pad) coordinate space,
    clamped to valid range.
    """
    target = SAM_INPUT_SIZE

    if img_h >= img_w:
        scale = target / float(img_h)
    else:
        scale = target / float(img_w)

    px_new = int(round(px * scale))
    py_new = int(round(py * scale))

    new_h = int(round(img_h * scale)) if img_h < img_w else target
    new_w = int(round(img_w * scale)) if img_w < img_h else target

    px_new = max(0, min(px_new, new_w - 1))
    py_new = max(0, min(py_new, new_h - 1))
    return px_new, py_new


def get_remote_component_masks(gt_mask, prompt_point, img_h, img_w):
    """
    Decompose GT mask into connected components and identify which ones are
    'remote' — i.e. do NOT contain the prompt point.

    Component detection and point lookup are done in original image space
    (avoids small-component loss from downsampling).
    The remote union mask is then preprocessed through SAM's pipeline to
    align with the 64×64 attention token grid.

    prompt_point: (x, y) in original image coordinates.

    Returns:
        remote_union_64: binary [64, 64] mask — union of all remote components
                         aligned to attn grid. None if no remote components.
        n_remote: int, number of remote components.
        n_total: int, total number of valid GT components.
    """
    gt_u8 = gt_mask.astype(np.uint8)
    n_labels, labels, stats_cv, _ = cv2.connectedComponentsWithStats(
        gt_u8, connectivity=8)

    # Prompt point in original image coordinates
    px = max(0, min(int(round(prompt_point[0])), img_w - 1))
    py = max(0, min(int(round(prompt_point[1])), img_h - 1))

    remote_union = np.zeros_like(gt_u8, dtype=np.uint8)
    n_remote = 0
    n_valid  = 0

    for lbl in range(1, n_labels):
        area = stats_cv[lbl, cv2.CC_STAT_AREA]
        if area < min_component_area:
            continue
        n_valid += 1
        comp_mask = (labels == lbl)
        # Component membership check in original image space — most accurate
        if comp_mask[py, px]:
            continue   # clicked component — skip
        remote_union |= comp_mask.astype(np.uint8)
        n_remote += 1

    if n_remote == 0:
        return None, 0, n_valid

    # Align to SAM's 64×64 attention grid via SAM's preprocessing pipeline
    remote_union_64 = sam_preprocess_mask(remote_union, img_h, img_w)
    return remote_union_64, n_remote, n_valid


def compute_remote_enrichment(attn_map_64, remote_mask_64, eps=1e-9):
    """
    Compute A_enrich_remote: how much more attention falls on remote GT
    components than expected by chance.

        A_enrich_remote = (sum of attn on remote pixels)
                          / (n_remote_pixels / N_TOKENS)

    Args:
        attn_map_64:    [64, 64] normalised attention weights (sum to 1)
        remote_mask_64: [64, 64] binary mask of remote GT components

    Returns float >= 0. Value of 1.0 = chance level.
    """
    n_remote_pixels = remote_mask_64.sum()
    if n_remote_pixels == 0:
        return None

    attn_on_remote   = attn_map_64[remote_mask_64].sum()
    expected_mass    = n_remote_pixels / N_TOKENS
    enrichment       = float(attn_on_remote) / (expected_mass + eps)
    return float(enrichment)


# ── GT component count ────────────────────────────────────────────────────────

def count_gt_components(gt_mask):
    gt_u8 = gt_mask.astype(np.uint8)
    n_labels, _, stats_cv, _ = cv2.connectedComponentsWithStats(
        gt_u8, connectivity=8)
    return sum(1 for lbl in range(1, n_labels)
               if stats_cv[lbl, cv2.CC_STAT_AREA] >= min_component_area)


# ================================ Section 2 (new) =============================
# token->image attentions of SAM1's TWT, in forward order (short name, module path)
TOKEN_TO_IMAGE = (("L0", "layers.0.cross_attn_token_to_image"),
                  ("L1", "layers.1.cross_attn_token_to_image"),
                  ("final", "final_attn_token_to_image"))
REGIONS = ("remote", "clicked")


def region_masks_64(gt_mask, prompt_point, img_h, img_w):
    """
    Same component decomposition and point lookup as get_remote_component_masks
    (Section 1), but also returns the clicked component.

    Returns dict:
      remote_64, clicked_64 : bool [64, 64] (all-False if empty)
      n_remote, n_valid      : ints (as Section 1)
      click_on_valid_component : bool — False if the click falls outside every
                                 valid (>= min_component_area) GT component
    """
    gt_u8 = gt_mask.astype(np.uint8)
    n_labels, labels, stats_cv, _ = cv2.connectedComponentsWithStats(gt_u8, connectivity=8)
    px = max(0, min(int(round(prompt_point[0])), img_w - 1))
    py = max(0, min(int(round(prompt_point[1])), img_h - 1))

    remote = np.zeros_like(gt_u8, dtype=np.uint8)
    clicked = np.zeros_like(gt_u8, dtype=np.uint8)
    n_remote, n_valid, on_valid = 0, 0, False
    for lbl in range(1, n_labels):
        if stats_cv[lbl, cv2.CC_STAT_AREA] < min_component_area:
            continue
        n_valid += 1
        comp = (labels == lbl)
        if comp[py, px]:
            clicked |= comp.astype(np.uint8)
            on_valid = True
            continue
        remote |= comp.astype(np.uint8)
        n_remote += 1
    empty = np.zeros((ATTN_GRID, ATTN_GRID), dtype=bool)
    return {
        "remote_64": sam_preprocess_mask(remote, img_h, img_w) if n_remote else empty,
        "clicked_64": sam_preprocess_mask(clicked, img_h, img_w) if on_valid else empty,
        "n_remote": n_remote, "n_valid": n_valid, "click_on_valid_component": on_valid,
    }


def point_attn_per_head(attn_module):
    """[H, 4096] float64 numpy: point token's attention over image tokens, batch 0."""
    a = attn_module.last_attn
    if a is None or a.size(-2) <= POINT_TOKEN_IDX:
        raise RuntimeError("no cached attention for the point token")
    return a[0, :, POINT_TOKEN_IDX, :].double().cpu().numpy()


def raw_attention_row(twt, regions):
    """
    Raw per-prompt readout. Columns:
      frac__{region}                 fraction of the 64x64 grid covered by the region
      mass__{layer}__h{h}__{region}  point-token attention mass of head h on the region
      ent__{layer}__h{h}             entropy (nats) of head h's point-token attention
                                     (per head; NOT the Section 5.1 entropy definition)
    Enrichment of any layer and head set S (head-averaged, as Section 1) is
      mean_{h in S} mass__{layer}__h{h}__{region} / frac__{region}
    (Section 1 renormalises the head average by its sum, which is 1 up to
    float error; compute_remote_enrichment also adds eps=1e-9 to frac.)
    """
    row = {}
    flat = {r: regions[f"{r}_64"].reshape(-1) for r in REGIONS}
    for r in REGIONS:
        row[f"frac__{r}"] = float(flat[r].sum()) / N_TOKENS
    for short, path in TOKEN_TO_IMAGE:
        A = point_attn_per_head(twt.get_submodule(path))           # [H, 4096]
        for h in range(A.shape[0]):
            p = A[h]
            for r in REGIONS:
                row[f"mass__{short}__h{h}__{r}"] = float(p[flat[r]].sum())
            row[f"ent__{short}__h{h}"] = float(-(p * np.log(p + 1e-12)).sum())
    return row


def enrichment_from_raw(df, layer="final", heads=None, region="remote", eps=1e-9):
    """Recompute head-averaged enrichment from saved raw columns (pandas DataFrame)."""
    cols = [c for c in df.columns if c.startswith(f"mass__{layer}__h") and c.endswith(f"__{region}")]
    if heads is not None:
        cols = [c for c in cols if int(c.split("__")[2][1:]) in set(heads)]
    if not cols:
        raise KeyError(f"no mass columns for layer={layer} region={region}")
    mass = df[cols].mean(axis=1)
    frac = df[f"frac__{region}"]
    return (mass / (frac + eps)).where(frac > 0)