"""
R2 (attention enrichment across adaptation sites): experiment list and the
pre-revision reference numbers used as a regression check.

Experiment names are resolved exactly as in r1_table.py (R1 variants at the LR
selected by the pilot sweep) and batch0_och.py (re-registered legacy runs), so
the attention files land in the same folders as the PR / CE files.
"""

import json
import os

from .config import resolve_config

SEEDS = [0, 1, 2, 3, 4]
ZS = "zs__sam1"
MAIN = "och__twt-lora-scfull-qkvo-r8a8__tok1__lr1e-4__tau035-N100__steps5000"
DEC_FULL = "och__dec-full__tok1__lr3e-5__tau035-N100__steps5000"

# R1 variants rerun for R2 (nontwt-lora dropped, Nov 24; swept twt-lora row not used)
R2_VARIANTS = ["twt-full", "nontwt-full", "nontwt-full-tok0", "tok", "enc-lora"]

# Pre-revision LoRA ablation runs (Sec. 4.1.4 / S4.2): rerun only to reproduce
# the old S4.2 table with the new pipeline.
ABLATION = [("QK-only (multi-layer)", "min", "qk"), ("QK-only (final layer)", "final", "qk"),
            ("VO-only (multi-layer)", "min", "vo"), ("VO-only (final layer)", "final", "vo")]

# Old S4.2 table (bugfix_rerun_notes.md Part 7): mean ± std over seeds 0-4 with
# np.std (ddof=0); RPR on instances with >= 1 remote component; rho = Spearman
# of per-instance mean enrichment vs RPR, averaged over seeds.
S42_REFERENCE = {
    "SAM1 zero-shot":        {"mean_E": (4.666, 0.028), "rpr_occ": (0.5773, 0.0015), "rho": (0.311, 0.011)},
    "QK-only (multi-layer)": {"mean_E": (6.214, 0.398), "rpr_occ": (0.6852, 0.0074), "rho": (0.367, 0.018)},
    "QK-only (final layer)": {"mean_E": (5.253, 0.253), "rpr_occ": (0.6555, 0.0051), "rho": (0.370, 0.025)},
    "VO-only (multi-layer)": {"mean_E": (7.732, 0.122), "rpr_occ": (0.7388, 0.0049), "rho": (0.385, 0.019)},
    "VO-only (final layer)": {"mean_E": (4.940, 0.026), "rpr_occ": (0.6734, 0.0067), "rho": (0.366, 0.010)},
    "PRISM-SAM (twt-lora, main)": {"mean_E": (7.952, 0.246), "rpr_occ": (0.7518, 0.0042), "rho": (0.340, 0.017)},
}


def _raw(variant, lr):   # identical to sweep_lr / run_official / r1_table raw config
    return {"data": "och", "backbone": "sam1", "model_type": "vit_l", "variant": variant,
            "train": {"lr": lr, "total_steps": 5000}, "benchmark": {"tau": 0.35, "N": 100}}


def _legacy_lora_name(scope, proj):   # identical to batch0_och registration
    adapt = {"site": "twt", "param": "lora", "tok": True,
             "lora": {"rank": 8, "alpha": 8, "dropout": 0.05, "scope": scope, "proj": proj}}
    return resolve_config({"data": "och", "backbone": "sam1", "model_type": "vit_l",
                           "adapt": adapt, "train": {"lr": 1e-4, "total_steps": 5000},
                           "benchmark": {"tau": 0.35, "N": 100}})["exp_name"]


def experiments(results_root, include_ablation=True):
    """-> list of dicts: label, exp_name, group ('main' | 'ablation'), variant."""
    with open(os.path.join(results_root, "analysis", "R1", "lr_sweep_s10.json")) as f:
        selected = json.load(f)["selected"]
    rows = [{"label": "SAM1 zero-shot", "exp_name": ZS, "group": "main", "variant": None},
            {"label": "PRISM-SAM (twt-lora, main)", "exp_name": MAIN, "group": "main",
             "variant": "twt-lora"}]
    for v in R2_VARIANTS:
        lr = selected.get(v)
        if lr is None:
            raise ValueError(f"{v}: no selected LR in lr_sweep_s10.json")
        rows.append({"label": v, "exp_name": resolve_config(_raw(v, lr))["exp_name"],
                     "group": "main", "variant": v})
    rows.append({"label": "dec-full", "exp_name": DEC_FULL, "group": "main", "variant": "dec-full"})
    if include_ablation:
        for label, scope, proj in ABLATION:
            rows.append({"label": label, "exp_name": _legacy_lora_name(scope, proj),
                         "group": "ablation", "variant": f"twt-lora-sc{scope}-{proj}"})
    return rows


def attn_stem(exp_dir, seed, zero_shot):
    """eval file stem; zero-shot uses the test seed folder, fine-tuned the train seed (= test seed)."""
    return os.path.join(exp_dir, f"s{seed}", "eval", f"attn__och-test_s{seed}")