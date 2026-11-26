"""
OVIS (train+valid pooled) helpers for the R3 OVIS diagnostic.

Section 1 is copied VERBATIM from the pre-revision EDA-OVIS/create_ovis_benchmark.py
(decode_rle, merge_ovis_splits, is_gt_eligible; file md5 d4803dd65bf5d73c585b269753b9fca1). Do not edit.
Pre-revision OVIS mining criterion: eligible = occlusion != no_occlusion and
>= 2 GT components of >= 150 px; hard = remote_part_recall(tau=0.5,
min_component_area=150) < 1.
"""

import json
import os
from typing import Dict, Optional

import numpy as np
from pycocotools import mask as maskUtils

from ..metrics import decompose_components

# =========================================================================
# Section 1 -- VERBATIM (pre-revision create_ovis_benchmark.py)
# =========================================================================

def decode_rle(seg, height: int, width: int) -> Optional[np.ndarray]:
    """Decode a single OVIS/COCO-style compressed RLE entry to a binary mask.
    Returns None if seg is None (instance absent/not annotated in this frame).
    """
    if seg is None:
        return None
    if isinstance(seg.get("counts"), str):
        seg = {"size": seg["size"], "counts": seg["counts"].encode("utf-8")}
    return maskUtils.decode(seg).astype(np.uint8)


def is_gt_eligible(
    gt_mask: Optional[np.ndarray],
    occlusion_label: Optional[str],
    occlusion_prefilter: bool,
    min_component_area: int,
) -> bool:
    """SAM-free structural check for a single (instance, frame): occlusion
    label != no_occlusion (if enabled) AND GT already has >=2 connected
    components. Guards against the RPR=0-on-oversegmented-single-component
    trap discussed in chat.
    """
    if gt_mask is None or gt_mask.sum() == 0:
        return False
    if occlusion_prefilter and occlusion_label == "no_occlusion":
        return False
    return len(decompose_components(gt_mask, min_component_area=min_component_area)) >= 2


def merge_ovis_splits(split_paths: Dict[str, str], id_block_size: int = 1_000_000) -> dict:
    """Merge multiple OVIS annotation files (e.g. train + valid_withgt) into
    one unified structure for mining across more video pool than a single
    split provides (see chat: no non-person category reaches anywhere near
    100 hard cases using train alone). Deliberately do NOT include the test
    split here -- it stays untouched, reserved entirely for evaluation, so
    there's no need to later remember to exclude anything.

    Each split's video/annotation ids independently start from a small
    range, so raw ids from different splits can silently collide if merged
    as-is. Every id is offset by `split_index * id_block_size` to guarantee
    disjointness (id_block_size=1e6 is far larger than OVIS's few-hundred
    videos / few-thousand annotations per split, so this is safe).

    Also prepends the split name to every file_names entry (e.g.
    "train/012b09a0/img_0000001.jpg" instead of "012b09a0/img_0000001.jpg"),
    so a single root_dir=ovis_root resolves images correctly regardless of
    which split they came from -- IMPORTANT: when using merged data, pass
    root_dir=ovis_root (not ovis_root/train) to the SAM predictor wrappers.

    split_paths: e.g. {"train": ".../annotations_train.json",
                        "valid": ".../annotations_valid_withgt.json"}
    """
    merged_videos = []
    merged_anns = []
    categories = None

    for i, (split_name, path) in enumerate(split_paths.items()):
        with open(path) as f:
            data = json.load(f)

        if categories is None:
            categories = data["categories"]
        else:
            this_names = {c["id"]: c["name"] for c in data["categories"]}
            ref_names = {c["id"]: c["name"] for c in categories}
            if this_names != ref_names:
                raise ValueError(
                    f"Category mapping in '{split_name}' ({path}) does not match "
                    f"the reference split -- check before merging blindly."
                )

        offset = i * id_block_size
        for v in data["videos"]:
            v = dict(v)
            v["id"] = v["id"] + offset
            v["file_names"] = [f"{split_name}/{fn}" for fn in v["file_names"]]
            merged_videos.append(v)
        for a in data["annotations"]:
            a = dict(a)
            a["id"] = a["id"] + offset
            a["video_id"] = a["video_id"] + offset
            merged_anns.append(a)

    return {"videos": merged_videos, "annotations": merged_anns, "categories": categories}


# =========================================================================
# Section 2 -- new
# =========================================================================

SPLITS = {"train": "annotations_train.json", "valid": "annotations_valid_withgt.json"}


def load_merged(ovis_root):
    return merge_ovis_splits({k: os.path.join(ovis_root, v) for k, v in SPLITS.items()})


class OvisIndex:
    """(merged) ann_id -> annotation, video_id -> video."""

    def __init__(self, merged):
        self.videos = {v["id"]: v for v in merged["videos"]}
        self.anns = {a["id"]: a for a in merged["annotations"]}
        self.categories = {c["id"]: c["name"] for c in merged["categories"]}

    def gt_mask(self, ann_id, frame_idx):
        a = self.anns[ann_id]
        v = self.videos[a["video_id"]]
        return decode_rle(a["segmentations"][frame_idx], v["height"], v["width"])

    def image_rel_path(self, ann_id, frame_idx):
        a = self.anns[ann_id]
        return self.videos[a["video_id"]]["file_names"][frame_idx]