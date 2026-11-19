"""
Per-instance result files (R7 spec) and aggregation.

Every PR evaluation writes two parquet files into <run>/eval/:

  pr__{bench}.instances.parquet   one row per (method, train_seed, test_seed, instance)
  pr__{bench}.prompts.parquet     one row per (... , instance, prompt)

Key columns (both files): method, train_seed, test_seed, dataset, category,
instance_id. `instance_id` comes from the dataset module's shared function and
is the pairing key across methods. train_seed is -1 for zero-shot baselines.

`aggregate_pr` recomputes the table numbers from the instance file. It must
reproduce the old eval_logs_seed*.txt to rounding (tests/test_regression_och.py).
"""

import json
import os

import numpy as np
import pandas as pd

KEY_COLS = ["method", "train_seed", "test_seed", "dataset", "category", "instance_id"]


class PRWriter:
    def __init__(self, method, train_seed, test_seed, dataset, category):
        self.keys = {"method": method, "train_seed": int(train_seed), "test_seed": int(test_seed),
                     "dataset": dataset, "category": category}
        self.instances, self.prompts = [], []
        self._seen = set()

    def add(self, instance_id, inst_row, prompt_rows, extra=None):
        if instance_id in self._seen:
            raise ValueError(f"duplicate instance_id {instance_id}")
        self._seen.add(instance_id)
        base = dict(self.keys, instance_id=instance_id)
        self.instances.append({**base, **inst_row, **(extra or {})})
        for pr in prompt_rows:
            self.prompts.append({**base, **pr})

    def save(self, out_dir, bench, meta=None):
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.join(out_dir, f"pr__{bench}")
        inst = pd.DataFrame(self.instances)
        pd.DataFrame(self.prompts).to_parquet(stem + ".prompts.parquet", index=False)
        inst.to_parquet(stem + ".instances.parquet", index=False)
        agg = aggregate_pr(inst)
        with open(stem + ".summary.json", "w") as f:
            json.dump({"keys": self.keys, "aggregate": agg, "meta": meta or {}}, f, indent=2)
        return stem, agg


def aggregate_pr(inst: pd.DataFrame, tau_remote=0.5):
    """Same definitions and averaging as the pre-revision test.py."""
    occ = inst.loc[inst["is_occluded"], "occ_dice"]
    return {
        "mIoU": float(np.mean(inst["iou"])),
        "Dice": float(np.mean(inst["dice"])),
        "MeanIoU_per_prompt": float(np.mean(inst["mean_iou_pp"])),
        "VarIoU_per_prompt": float(np.mean(inst["var_iou_pp"])),
        f"RemotePartRecall@{tau_remote:.2f}": float(np.mean(inst["rpr"])),
        "FragmentIoU": float(np.mean(inst["fragment_iou"])),
        "OccludedInstanceF1": float(np.mean(occ)) if len(occ) else float("nan"),
        "OccludedInstanceCount": int(len(occ)),
        "InstanceCount": int(len(inst)),
    }


def parse_legacy_log(path):
    """Old eval_logs_seed{s}.txt: 'key: value' lines, appended per run -> keep last."""
    out = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.rsplit(":", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out