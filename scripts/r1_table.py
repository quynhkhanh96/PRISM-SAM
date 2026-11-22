"""
R1/R5 first read: mean ± std over seeds 0-4 for every variant, paired per-seed
differences vs PRISM-SAM (twt-lora, lr=1e-4), and a metric-vs-#params scatter.

    python scripts/r1_table.py
    -> <results_root>/analysis/R1/r1_table.md   (read this)
       <results_root>/analysis/R1/r1_per_seed.csv
       <results_root>/analysis/R1/r1_scatter.png

This is the descriptive layer only. Confidence intervals / significance come
from the shared R7 paired hierarchical bootstrap on the per-instance files;
until then, read "seeds won" and the paired mean difference together with the
seed std, and do not call anything "equivalent".

Pairing: within a seed s, every fine-tuned variant is trained on few-shot set
s and evaluated on test file s (same instances and prompts), so differences
are paired by seed. zs__sam1 uses test seed s.
"""

import csv
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from prismsam import paths  # noqa: E402
from prismsam.config import fmt_lr, resolve_config  # noqa: E402
from prismsam.variants import EXPECTED_TRAINABLE  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
PR = [("MeanIoU_per_prompt", "MeanIoU/P", True), ("VarIoU_per_prompt", "VarIoU/P", False),
      ("RemotePartRecall@0.50", "RPR@.50", True), ("FragmentIoU", "FragIoU", True),
      ("OccludedInstanceF1", "OccF1", True), ("mIoU", "mIoU", True), ("Dice", "Dice", True)]
CE = [("full", "pct_rpr_hit_on_click1", "CE RPR=1@c1 full", True),
      ("occluded", "pct_rpr_hit_on_click1", "CE RPR=1@c1 occ", True),
      ("full", "pct_iou_hit_on_click1", "CE IoU≥.9@c1 full", True)]
MAIN = "och__twt-lora-scfull-qkvo-r8a8__tok1__lr1e-4__tau035-N100__steps5000"


def raw(variant, lr):
    return {"data": "och", "backbone": "sam1", "model_type": "vit_l", "variant": variant,
            "train": {"lr": lr, "total_steps": 5000}, "benchmark": {"tau": 0.35, "N": 100}}


def rows_spec(selected):
    """(label, exp_name, variant-for-params, lr, zero_shot)"""
    spec = [("SAM1 zero-shot", "zs__sam1", None, None, True),
            ("PRISM-SAM (twt-lora, main)", MAIN, "twt-lora", 1e-4, False)]
    order = ["twt-lora", "twt-full", "nontwt-full", "nontwt-full-tok0", "nontwt-lora", "enc-lora", "tok"]
    for v in order:
        lr = selected.get(v)
        if lr is None:
            continue
        if v == "twt-lora":
            if abs(lr - 1e-4) < 1e-12:
                continue
            label = "twt-lora (best LR)"
        else:
            label = v
        spec.append((label, resolve_config(raw(v, lr))["exp_name"], v, lr, False))
    spec.append(("dec-full (existing)", "och__dec-full__tok1__lr3e-5__tau035-N100__steps5000",
                 "dec-full", 3e-5, False))
    return spec


def load(results, exp, s):
    d = os.path.join(results, exp, f"s{s}", "eval")
    out = {}
    p = os.path.join(d, f"pr__och-test_s{s}.summary.json")
    if os.path.exists(p):
        with open(p) as f:
            out.update(json.load(f)["aggregate"])
    p = os.path.join(d, f"ce__och-test_s{s}.summary.json")
    if os.path.exists(p):
        with open(p) as f:
            sm = json.load(f)["summary"]
        for sub, key, name, _ in CE:
            out[name] = sm[sub]["click1"][key]
    return out


CE_DETAIL = [  # (subset, path, column name, decimals)
    ("full", ("click1", "pct_rpr_hit_on_click1"), "c1 RPR=1 full %", 1),
    ("occluded", ("click1", "pct_rpr_hit_on_click1"), "c1 RPR=1 occ %", 1),
    ("full", ("click1", "pct_iou_hit_on_click1"), "c1 IoU≥.9 full %", 1),
    ("occluded", ("click1", "pct_iou_hit_on_click1"), "c1 IoU≥.9 occ %", 1),
    ("full", ("noc", "noc_at_rpr1", "mean"), "NoC-RPR1 (reached)", 2),
    ("full", ("noc", "noc_at_rpr1", "pct_censored"), "NoC-RPR1 cens %", 1),
    ("full", ("noc", "noc_at_iou_target", "mean"), "NoC@90 (reached)", 2),
    ("full", ("noc", "noc_at_iou_target", "pct_censored"), "NoC@90 cens %", 1),
    ("full", ("post_click1_trend", "pct_worsened_gt_1pt"), "post-c1 worsened %", 1),
    ("full", ("post_click1_trend", "mean_delta_click1_to_last"), "post-c1 mean Δ", 3),
]


def load_ce_detail(results, exp, s):
    p = os.path.join(results, exp, f"s{s}", "eval", f"ce__och-test_s{s}.summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        sm = json.load(f)["summary"]
    out = {}
    for sub, path, name, _ in CE_DETAIL:
        v = sm[sub]
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        out[name] = v
    return out


def fmt(vals, key):
    if not vals:
        return "—"
    v = np.array(vals, dtype=float)
    dec = 5 if key == "VarIoU_per_prompt" else (1 if key.startswith("CE") else 4)
    sd = v.std(ddof=1) if len(v) > 1 else float("nan")
    return f"{v.mean():.{dec}f} ± {sd:.{dec}f}" + ("" if len(v) == len(SEEDS) else f" (n={len(v)})")


def main():
    R = paths.get("results_root")
    results = os.path.join(R, "results")
    out_dir = os.path.join(R, "analysis", "R1")
    with open(os.path.join(out_dir, "lr_sweep_s10.json")) as f:
        selected = json.load(f)["selected"]

    spec = rows_spec(selected)
    data = {}   # label -> {seed -> metrics}
    for label, exp, _, _, _ in spec:
        data[label] = {s: load(results, exp, s) for s in SEEDS}

    metric_cols = [(k, n, hi) for k, n, hi in PR] + [(n, n, hi) for _, _, n, hi in CE]
    main_label = spec[1][0]

    lines = ["# R1/R5 — variants on OCHuman (PR primary, CE secondary)", "",
             "Mean ± std over seeds 0–4 (sample std). LRs selected on pilot seed 10 "
             "(MeanIoU/Prompt). Descriptive only — CIs/significance from R7.", "",
             "| variant | #params | lr | " + " | ".join(n for _, n, _ in metric_cols) + " |",
             "|---|---|---|" + "---|" * len(metric_cols)]
    for label, exp, v, lr, zs in spec:
        params = "0" if zs else f"{EXPECTED_TRAINABLE[v]:,}"
        cells = [fmt([data[label][s][k] for s in SEEDS if k in data[label][s]], k)
                 for k, _, _ in metric_cols]
        lines.append(f"| {label} | {params} | {fmt_lr(lr) if lr else '—'} | " + " | ".join(cells) + " |")

    lines += ["", f"## Paired by seed vs {main_label}", "",
              "Cell = mean paired difference (variant − main) [seeds where the variant is better / seeds "
              "available]. For VarIoU/P lower is better.", "",
              "| variant | " + " | ".join(n for _, n, _ in metric_cols) + " |",
              "|---|" + "---|" * len(metric_cols)]
    for label, *_ in spec:
        if label == main_label:
            continue
        cells = []
        for k, _, hi in metric_cols:
            pairs = [(data[label][s][k], data[main_label][s][k]) for s in SEEDS
                     if k in data[label][s] and k in data[main_label][s]]
            if not pairs:
                cells.append("—")
                continue
            d = np.array([a - b for a, b in pairs])
            won = int(np.sum(d > 0) if hi else np.sum(d < 0))
            dec = 5 if k == "VarIoU_per_prompt" else (1 if k.startswith("CE") else 4)
            cells.append(f"{d.mean():+.{dec}f} [{won}/{len(d)}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", "## Click efficiency (secondary) — full test set unless marked occ", "",
              "k_max = 5; NoC means are over reached instances only and must be read with "
              "%censored; post-c1 = censored instances (IoU@90 never reached) with ≥2 clicks.", "",
              "| variant | " + " | ".join(n for _, _, n, _ in CE_DETAIL) + " |",
              "|---|" + "---|" * len(CE_DETAIL)]
    for label, exp, *_ in spec:
        det = [load_ce_detail(results, exp, s) for s in SEEDS]
        det = [d for d in det if d is not None]
        cells = []
        for _, _, name, dec in CE_DETAIL:
            vals = [d[name] for d in det if d.get(name) is not None]
            if not vals:
                cells.append("—")
                continue
            v = np.array(vals, dtype=float)
            sd = v.std(ddof=1) if len(v) > 1 else float("nan")
            cells.append(f"{v.mean():.{dec}f} ± {sd:.{dec}f}" + ("" if len(v) == len(SEEDS) else f" (n={len(v)})"))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    missing = [(label, s) for label, *_ in spec for s in SEEDS
               if "MeanIoU_per_prompt" not in data[label][s]]
    if missing:
        lines += ["", "Missing PR results: " + ", ".join(f"{l} s{s}" for l, s in missing)]
    md = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "r1_table.md"), "w") as f:
        f.write(md)
    print(md)

    with open(os.path.join(out_dir, "r1_per_seed.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "exp_name", "lr", "seed"] + [n for _, n, _ in metric_cols])
        for label, exp, _, lr, _ in spec:
            for s in SEEDS:
                m = data[label][s]
                w.writerow([label, exp, lr, s] + [m.get(k, "") for k, _, _ in metric_cols])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: scatter skipped")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, name) in zip(axes, [("MeanIoU_per_prompt", "MeanIoU/Prompt ↑"),
                                     ("VarIoU_per_prompt", "VarIoU/Prompt ↓")]):
        for label, exp, v, lr, zs in spec:
            vals = [data[label][s][key] for s in SEEDS if key in data[label][s]]
            if not vals:
                continue
            x = 1 if zs else EXPECTED_TRAINABLE[v]
            ax.errorbar(x, np.mean(vals), yerr=np.std(vals, ddof=1) if len(vals) > 1 else 0,
                        fmt="o", capsize=3)
            ax.annotate(label, (x, np.mean(vals)), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("# trainable parameters (log; zero-shot plotted at 1)")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "r1_scatter.png")
    fig.savefig(p, dpi=160)
    print(f"scatter -> {p}")


if __name__ == "__main__":
    main()