"""
OCHuman adapter.

Record format (unchanged from the pre-revision benchmark json):
    {image_id, image_size: [H, W], bbox, segms: {outer, inner}, [points]}
few-shot files have no `points`; test files have K=5 eval `points`.
Test instances and their order are identical across test seeds; only the
points differ.

`OchumanIndex.instance_id(record)` is THE shared instance key for every driver
(PR, CE, attention, baselines): 'och:{image_id}:{ann_idx}', ann_idx = position
of the annotation in ochuman.json['images'][...]['annotations'] (None-segms
annotations included, so the index is the raw one). Resolved by exact bbox
match, segms as tie-breaker; anything but exactly one match raises.
"""

import json
import os

from torch.utils.data import Dataset

from ..utils import read_image, polygons_to_mask

DATASET = "och"
CATEGORY = "person"


def load_records(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def image_path(ochuman_root, record):
    return os.path.join(ochuman_root, "images", record["image_id"] + ".jpg")


def gt_mask(record):
    h, w = record["image_size"]
    return polygons_to_mask(width=w, height=h,
                            outers=record["segms"]["outer"],
                            inners=record["segms"]["inner"])


class OchumanIndex:
    def __init__(self, ochuman_json):
        with open(ochuman_json) as f:
            data = json.load(f)
        self._by_image = {
            img["image_id"]: [(i, a.get("bbox"), a.get("segms"))
                              for i, a in enumerate(img["annotations"])]
            for img in data["images"]
        }

    @classmethod
    def from_root(cls, ochuman_root):
        return cls(os.path.join(ochuman_root, "ochuman.json"))

    def ann_idx(self, record):
        anns = self._by_image.get(record["image_id"])
        if anns is None:
            raise KeyError(f"image_id {record['image_id']} not in ochuman.json")
        hits = [a for a in anns if a[1] == record["bbox"]]
        if len(hits) > 1:
            hits = [a for a in hits if a[2] == record["segms"]]
        if len(hits) != 1:
            raise ValueError(f"{len(hits)} annotations match record in image {record['image_id']}")
        return hits[0][0]

    def instance_id(self, record):
        return f"{DATASET}:{record['image_id']}:{self.ann_idx(record)}"


class OCHumanPromptDataset(Dataset):
    """Same behaviour as the pre-revision data.OCHumanPromptDataset."""

    def __init__(self, ochuman_root, fewshot_path, num_shots=100, num_prompts=3,
                 augment=None, device="cuda"):
        self.ochuman_root = ochuman_root
        self.num_prompts = num_prompts
        self.augment = augment
        self.device = device
        self.records = load_records(fewshot_path)[:num_shots]
        if len(self.records) < num_shots:
            raise ValueError(f"{fewshot_path} has {len(self.records)} records < num_shots={num_shots}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        sample = {
            "image": read_image(image_path(self.ochuman_root, rec)),
            "gt_mask": gt_mask(rec),
            "K": self.num_prompts,
            "device": self.device,
        }
        if self.augment is not None:
            sample = self.augment(sample)
        return sample

# ---------------------------------------------------------------- mining pool
TRAIN_SPLIT_JSON = "ochuman_coco_format_val_range_0.00_1.00.json"   # OCHuman 'val' = our train pool


def train_pool_records(ochuman_root):
    """Mining pool, built exactly as in pre-revision create_benchmark.py
    (same filtering and the same order, which the seeded shuffle relies on)."""
    with open(os.path.join(ochuman_root, "ochuman.json"), "r") as f:
        ochuman_all_data = json.load(f)
    with open(os.path.join(ochuman_root, TRAIN_SPLIT_JSON), "r") as f:
        data_train = json.load(f)
    train_image_files = [file["file_name"] for file in data_train["images"]]
    train_image_ids = [image_file.split(".")[0] for image_file in train_image_files]
    train_samples = [sample for sample in ochuman_all_data["images"]
                     if sample["image_id"] in train_image_ids]
    train_data = []
    for sample in train_samples:
        for ann in sample["annotations"]:
            if ann["segms"] is None:
                continue
            train_data.append({
                "image_id": sample["image_id"],
                "image_size": (sample["height"], sample["width"]),
                "bbox": ann["bbox"],
                "segms": ann["segms"],
            })
    return train_data
