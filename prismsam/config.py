"""
Experiment config resolution and naming (shared by train.py, batch0, eval).

Experiment name:
  {data}__{site}-{param}[-{lora spec}]__tok{0|1}__lr{lr}__tau{tau}-N{N}__steps{S}
config.json (inside the experiment folder) is the source of truth; analysis
code never parses folder names.
"""

import json

from . import adapt, paths
from .variants import VARIANTS, variant_name

TRAIN_DEFAULTS = {
    "lr": 1e-4, "total_steps": 5000, "batch_instances": 2, "K": 3,
    "weight_decay": 0.0, "max_grad_norm": 1.0, "amp": True,
    "focal_weight": 1.0, "dice_weight": 1.0, "score_weight": 1.0,
    "global_seed": 42,
}

# few-shot training files live under results_root/benchmarks (see plan Part 4)
FEWSHOT_TEMPLATE = "{results_root}/benchmarks/{data}/train/tau{tau}/s{seed}/fewshot_N{N}.json"


def fmt_lr(lr):
    m, e = f"{float(lr):.0e}".split("e")
    return f"{m}e{int(e)}"                   # 1e-4, 3e-5


def fmt_tau(tau):
    return f"{int(round(float(tau) * 100)):03d}"   # 0.35 -> 035


def bench_tag(b):
    """'035' for the pre-revision criterion (area 0); '050a150' with an area filter.
    Used in both the few-shot path (tau{tag}) and the experiment name."""
    tag = fmt_tau(b["tau"])
    area = int(b.get("mine_area", 0))
    return tag + (f"a{area}" if area else "")


def resolve_config(raw):
    """yaml/dict -> fully explicit config (what config.json stores)."""
    cfg = {"data": raw.get("data", "och"), "backbone": raw.get("backbone", "sam1"),
           "model_type": raw.get("model_type", "vit_l")}
    if "variant" in raw:
        if "adapt" in raw:
            raise ValueError("give either 'variant' or 'adapt', not both")
        cfg["variant"] = raw["variant"]
        cfg["adapt"] = json.loads(json.dumps(VARIANTS[raw["variant"]]))
    else:
        cfg["adapt"] = raw["adapt"]
    adapt.validate_adapt_cfg(cfg["adapt"])

    t = dict(TRAIN_DEFAULTS)
    unknown = set(raw.get("train", {})) - set(t)
    if unknown:
        raise ValueError(f"unknown train keys: {unknown}")
    t.update(raw.get("train", {}))
    t["lr"] = float(t["lr"])
    cfg["train"] = t

    b = raw["benchmark"]
    cfg["benchmark"] = {"tau": float(b["tau"]), "N": int(b["N"]),
                        "fewshot": b.get("fewshot", FEWSHOT_TEMPLATE)}
    # mining min_component_area (R3). 0 = pre-revision OCHuman criterion; the key is
    # only stored when non-zero so configs of existing experiments stay identical.
    if int(b.get("mine_area", 0)):
        cfg["benchmark"]["mine_area"] = int(b["mine_area"])

    cfg["exp_name"] = raw.get("exp_name") or "__".join([
        cfg["data"], variant_name(cfg["adapt"]), f"lr{fmt_lr(t['lr'])}",
        f"tau{bench_tag(cfg['benchmark'])}-N{cfg['benchmark']['N']}",
        f"steps{t['total_steps']}"])
    return cfg


def fewshot_path(cfg, seed):
    keys = dict(paths._load())
    b = cfg["benchmark"]
    return b["fewshot"].format(seed=seed, data=cfg["data"], tau=bench_tag(b), N=b["N"], **keys)


def test_path(data, test_seed):
    return f"{paths.get('results_root')}/benchmarks/{data}/test/test_s{test_seed}.json"