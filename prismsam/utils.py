"""
I/O and prompt-sampling utilities, copied VERBATIM from the pre-revision
utils.py (read_image, polygons_to_mask, sample_points_safe) so that training
batches and any regenerated prompts are bit-identical to the old pipeline.
Do not edit the bodies; add new helpers below instead.
"""

import random
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw


# --- Image I/O utils ---
def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image 

def polygons_to_mask(
    width: int,
    height: int,
    outers: List[List[float]],
    inners: Optional[List[List[float]]] = None
) -> np.ndarray:
    """
    Rasterize OCHuman-style polygons to a binary mask (H, W), uint8 {0,1}.
    - outers: list of outer polygons, each [x1,y1,x2,y2,...].
    - inners: list of hole polygons, each [x1,y1,...]. Can be [] or None.
    """
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    # Fill all outer polygons with 1
    if outers:
        for poly in outers:
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            draw.polygon(pts, outline=1, fill=1)

    # Punch holes (inners) with 0
    if inners:
        for poly in inners:
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            draw.polygon(pts, outline=0, fill=0)

    return np.array(mask_img, dtype=np.uint8)  # (H, W) with {0, 1}

def sample_points_safe(
    gt_mask: np.ndarray,
    num_points: int = 5,
    dist_percentile: float = 0.6,
    min_pixels: int = 30,
    min_component_area: int = 150,  # new parameter
):
    gt_mask = gt_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gt_mask, connectivity=8)

    # Filter valid component labels by area
    valid_labels = [
        lbl for lbl in range(1, num_labels)
        if stats[lbl, cv2.CC_STAT_AREA] >= min_component_area
    ]

    # Fallback: if all components are too small, use the largest one
    if len(valid_labels) == 0:
        valid_labels = [1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])]

    chosen_points = []
    used_coords = set()
    trials = 0
    MAX_TRIALS = 100
    while len(chosen_points) < num_points:
        trials += 1
        if trials > MAX_TRIALS:
            break

        chosen_label = random.choice(valid_labels)  # fixed: only valid labels
        comp_mask = (labels == chosen_label).astype(np.uint8)

        dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
        if dist.max() == 0:
            continue

        thresh = np.percentile(dist[dist > 0], dist_percentile * 100)
        ys, xs = np.where(dist >= thresh)

        if len(xs) < min_pixels:
            ys, xs = np.where(comp_mask > 0)

        idx = np.random.randint(len(xs))
        coord = (int(xs[idx]), int(ys[idx]))
        if coord not in used_coords:
            chosen_points.append(list(coord))
            used_coords.add(coord)

    return np.stack(chosen_points)