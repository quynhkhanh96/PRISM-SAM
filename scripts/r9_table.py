"""
R9 first read (descriptive): PRISM-SAM trained on nested hard-case sets of size
N in {20, 50, 100, 200, 400}, OCHuman PR (primary) + CE (secondary), mean ± std
over seeds 0-4, paired per-seed differences between adjacent N, and a
metric-vs-N figure.

    python scripts/r9_table.py
    -> <results_root>/analysis/R9/r9_table.md, r9_per_seed.csv, r9_scaling.png

Pairing: within a seed s every row is evaluated on test file s, and the
training sets are nested (set N is a prefix of set 2N), so differences are
paired by seed. Rows differ in the training set only (fixed 5000 steps).
CIs / significance: scripts/r9_stats.py.
"""

import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np  # noqa: E402

import r1_table as T  # noqa: E402  (load, load_ce_detail, fmt, PR, CE, CE_DETAIL, MAIN)
from build_fewshot_r9 import NS, N_REUSED  # noqa: E402
from run_r9 import raw_config  # noqa: E402
from prismsam import paths  # noqa: E402
from prismsam.config import resolve_config  # noqa: E402

SEEDS = T.SEEDS
FIG_METRICS = [("MeanIoU_per_prompt", "MeanIoU/Prompt ↑"), ("VarIoU_per_prompt", "VarIoU/Prompt ↓"),
               ("RemotePartRecall@0.50", "RPR@0.50 ↑"), ("CE RPR=1@c1 full", "CE click-1 RPR=1 (%) ↑")]


def exp_name(N):
    name = resolve_config(raw_config(N))["exp_name"]
    if N == N_REUSED and name != T.MAIN:
        raise RuntimeError(f"N={N} resolves to {name}, expected the main run {T.MAIN}")
    return name


def spec():
    """[(label, exp_name, N or None)]"""
    return [("SAM1 zero-shot", "zs__sam1", None)] + \
        [(f"N={N}" + (" (main)" if N == N_REUSED else ""), exp_name(N), N) for N in NS]


def main():
    R = paths.get("results_root")
    results = os.path.join(R, "results")
    out_dir = os.path.join(R, "analysis", "R9")
    os.makedirs(out_dir, exist_ok=True)
    sp = spec()
    data = {label: {s: T.load(results, exp, s) for s in SEEDS} for label, exp, _ in sp}
    cols = [(k, n, hi) for k, n, hi in T.PR] + [(n, n, hi) for _, _, n, hi in T.CE]

    lines = ["# R9 — scaling with the number of hard cases, OCHuman (PR primary, CE secondary)", "",
             "PRISM-SAM recipe unchanged (twt-lora r8a8, lr 1e-4, 5000 total steps, cosine per "
             "epoch); nested training sets mined with (τ=0.35, area 0). Mean ± std over seeds "
             "0–4 (sample std). Descriptive only.", "",
             "| training set | " + " | ".join(n for _, n, _ in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for label, _, _ in sp:
        cells = [T.fmt([data[label][s][k] for s in SEEDS if k in data[label][s]], k) for k, _, _ in cols]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", "## Paired by seed, adjacent N (larger − smaller)", "",
              "Cell = mean paired difference [seeds where the larger N is better / seeds available]. "
              "VarIoU/P: lower is better.", "",
              "| comparison | " + " | ".join(n for _, n, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    fine = sp[1:]
    pairs_lab = [(fine[i + 1], fine[i]) for i in range(len(fine) - 1)]
    pairs_lab.append((fine[-1], next(x for x in fine if x[2] == N_REUSED)))   # largest N vs main
    for (la, _, Na), (lb, _, Nb) in pairs_lab:
        cells = []
        for k, _, hi in cols:
            pr = [(data[la][s][k], data[lb][s][k]) for s in SEEDS if k in data[la][s] and k in data[lb][s]]
            if not pr:
                cells.append("—")
                continue
            d = np.array([a - b for a, b in pr])
            won = int(np.sum(d > 0) if hi else np.sum(d < 0))
            dec = 5 if k == "VarIoU_per_prompt" else (1 if k.startswith("CE") else 4)
            cells.append(f"{d.mean():+.{dec}f} [{won}/{len(d)}]")
        lines.append(f"| N={Na} − N={Nb} | " + " | ".join(cells) + " |")

    lines += ["", "## Click efficiency detail", "",
              "| training set | " + " | ".join(n for _, _, n, _ in T.CE_DETAIL) + " |",
              "|---|" + "---|" * len(T.CE_DETAIL)]
    for label, exp, _ in sp:
        det = [d for d in (T.load_ce_detail(results, exp, s) for s in SEEDS) if d is not None]
        cells = []
        for _, _, name, dec in T.CE_DETAIL:
            v = np.array([d[name] for d in det if d.get(name) is not None], dtype=float)
            cells.append("—" if not len(v) else
                         f"{v.mean():.{dec}f} ± {(v.std(ddof=1) if len(v) > 1 else float('nan')):.{dec}f}"
                         + ("" if len(v) == len(SEEDS) else f" (n={len(v)})"))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    missing = [(l, s) for l, _, _ in sp for s in SEEDS if "MeanIoU_per_prompt" not in data[l][s]]
    if missing:
        lines += ["", "Missing PR results: " + ", ".join(f"{l} s{s}" for l, s in missing)]
    lines += ["", "Training-set composition and cross-seed overlap: r9_composition.md (same folder). "
              "CIs: r9_stats.md."]
    md = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "r9_table.md"), "w") as f:
        f.write(md)
    print(md)
    with open(os.path.join(out_dir, "r9_per_seed.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "exp_name", "N", "seed"] + [n for _, n, _ in cols])
        for label, exp, N in sp:
            for s in SEEDS:
                w.writerow([label, exp, N if N else 0, s] + [data[label][s].get(k, "") for k, _, _ in cols])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: figure skipped")
        return
    fig, axes = plt.subplots(1, len(FIG_METRICS), figsize=(4.0 * len(FIG_METRICS), 3.4))
    zs = sp[0][0]
    for ax, (key, name) in zip(axes, FIG_METRICS):
        xs, ms, sds = [], [], []
        for label, _, N in fine:
            v = [data[label][s][key] for s in SEEDS if key in data[label][s]]
            if v:
                xs.append(N)
                ms.append(np.mean(v))
                sds.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
                ax.scatter([N] * len(v), v, s=8, color="0.6", zorder=1)
        if xs:
            ax.errorbar(xs, ms, yerr=sds, fmt="-o", capsize=3, color="C0", zorder=2, label="PRISM-SAM")
        zv = [data[zs][s][key] for s in SEEDS if key in data[zs][s]]
        if zv:
            ax.axhline(np.mean(zv), ls="--", color="0.4", lw=1, label="SAM1 zero-shot")
        ax.set_xscale("log")
        ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        ax.minorticks_off()
        ax.set_xlabel("# hard cases N (log)")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "r9_scaling.png")
    fig.savefig(p, dpi=160)
    print(f"figure -> {p}")


if __name__ == "__main__":
    main()