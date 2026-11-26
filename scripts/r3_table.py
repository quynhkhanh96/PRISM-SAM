"""
R3 first read (descriptive): PRISM-SAM trained on few-shot sets mined with
different criteria, OCHuman PR (primary) + CE (secondary), mean ± std over
seeds 0-4, paired per-seed differences vs the main criterion (0.35, 0).

    python scripts/r3_table.py   -> <results_root>/analysis/R3/r3_table.md, r3_per_seed.csv

Pairing: within a seed s every row is evaluated on test file s (same instances
and prompts). Rows differ ONLY in the training set (recipe identical), and the
training sets of different criteria overlap (see och_composition.md), so the
paired differences are not between independent samples. CIs: R7 bootstrap.
"""

import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np  # noqa: E402

import r1_table as T  # noqa: E402  (load, load_ce_detail, fmt, PR, CE, CE_DETAIL)
from run_r3 import CRITERIA, raw_config  # noqa: E402
from prismsam import paths  # noqa: E402
from prismsam.config import resolve_config  # noqa: E402

SEEDS = T.SEEDS


def spec():
    rows = [("SAM1 zero-shot", "zs__sam1"), ("τ=0.35, area 0 (main)", T.MAIN)]
    for tau, area in CRITERIA:
        rows.append((f"τ={tau}, area {area}", resolve_config(raw_config(tau, area))["exp_name"]))
    return rows


def main():
    R = paths.get("results_root")
    results = os.path.join(R, "results")
    out_dir = os.path.join(R, "analysis", "R3")
    os.makedirs(out_dir, exist_ok=True)
    sp = spec()
    data = {label: {s: T.load(results, exp, s) for s in SEEDS} for label, exp in sp}
    cols = [(k, n, hi) for k, n, hi in T.PR] + [(n, n, hi) for _, _, n, hi in T.CE]
    main_label = sp[1][0]

    lines = ["# R3 — mining-criterion sensitivity, OCHuman (PR primary, CE secondary)", "",
             "PRISM-SAM recipe unchanged (twt-lora r8a8, lr 1e-4, 5000 steps, N=100); only the "
             "hard-case mining criterion differs. Evaluation identical for all rows (RPR@0.50 with "
             "150-px components). Mean ± std over seeds 0–4 (sample std). Descriptive only.", "",
             "| training set | " + " | ".join(n for _, n, _ in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for label, _ in sp:
        cells = [T.fmt([data[label][s][k] for s in SEEDS if k in data[label][s]], k) for k, _, _ in cols]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", f"## Paired by seed vs {main_label}", "",
              "Cell = mean paired difference (row − main) [seeds where the row is better / seeds "
              "available]. VarIoU/P: lower is better.", "",
              "| training set | " + " | ".join(n for _, n, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for label, _ in sp[2:]:
        cells = []
        for k, _, hi in cols:
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

    lines += ["", "## Click efficiency detail", "",
              "| training set | " + " | ".join(n for _, _, n, _ in T.CE_DETAIL) + " |",
              "|---|" + "---|" * len(T.CE_DETAIL)]
    for label, exp in sp:
        det = [d for d in (T.load_ce_detail(results, exp, s) for s in SEEDS) if d is not None]
        cells = []
        for _, _, name, dec in T.CE_DETAIL:
            v = np.array([d[name] for d in det if d.get(name) is not None], dtype=float)
            cells.append("—" if not len(v) else
                         f"{v.mean():.{dec}f} ± {(v.std(ddof=1) if len(v) > 1 else float('nan')):.{dec}f}"
                         + ("" if len(v) == len(SEEDS) else f" (n={len(v)})"))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    missing = [(l, s) for l, _ in sp for s in SEEDS if "MeanIoU_per_prompt" not in data[l][s]]
    if missing:
        lines += ["", "Missing PR results: " + ", ".join(f"{l} s{s}" for l, s in missing)]
    comp = os.path.join(out_dir, "och_composition.md")
    if os.path.exists(comp):
        lines += ["", "Training-set composition: see och_composition.md (same folder)."]
    md = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "r3_table.md"), "w") as f:
        f.write(md)
    print(md)
    with open(os.path.join(out_dir, "r3_per_seed.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "exp_name", "seed"] + [n for _, n, _ in cols])
        for label, exp in sp:
            for s in SEEDS:
                w.writerow([label, exp, s] + [data[label][s].get(k, "") for k, _, _ in cols])


if __name__ == "__main__":
    main()
