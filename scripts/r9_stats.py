"""
R9 statistics (R7 layer): paired hierarchical bootstrap on the per-instance
files for the scaling experiment.

    python scripts/r9_stats.py [--n_boot 2000]
    -> <results_root>/analysis/R9/r9_stats.md, r9_stats.json

Method (R7, agreed Nov 17):
  - resample seeds, then instances within each drawn seed; the statistic is
    computed per seed and averaged over seeds, exactly like the point estimate;
  - SHARED replicates: in replicate b the same seed draw and the same instance
    indices are used for every method, so every comparison of the replicate is
    paired (same instances, same prompts);
  - 95 % percentile CI; two-sided bootstrap p = min(1, 2 * min(P(Δ*<=0), P(Δ*>=0)));
  - Holm correction within each family: PR (all comparisons × PR metrics) and
    CE (all comparisons × CE click-1 metrics).
  - "% of main gain" = Δ expressed in units of the main result's gain over SAM1
    (N=100 − SAM1, point estimate), signed so that positive = the larger N is
    better (VarIoU/P: lower is better). Equivalence-type readings ("saturated",
    "≈") are made from this CI's width, never from p > 0.05.

Comparisons: adjacent N (larger − smaller), N=400 − N=100, and per N the
fraction of the main gain recovered, (N − SAM1) / (main − SAM1) (ratio computed
per replicate; descriptive, not Holm-corrected).

Metric definitions = records.aggregate_pr (PR) and ce_metrics.single_click_success_rate
(CE click 1: full set; occluded = num_gt_components >= 2).

NOTE: independent implementation of the agreed algorithm (r3_stats.py was not
available when this was written). If r3_stats.py is adopted as the shared R7
code, check that both give the same numbers on R3 before reporting.
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from build_fewshot_r9 import NS, N_REUSED  # noqa: E402
from r9_table import exp_name  # noqa: E402
from prismsam import paths  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
# (key, label, source column, higher_is_better, subset)  subset: None | "occ"
PR_METRICS = [("MeanIoU_per_prompt", "MeanIoU/P", "mean_iou_pp", True, None),
              ("VarIoU_per_prompt", "VarIoU/P", "var_iou_pp", False, None),
              ("RemotePartRecall@0.50", "RPR@.50", "rpr", True, None),
              ("FragmentIoU", "FragIoU", "fragment_iou", True, None),
              ("OccludedInstanceF1", "OccF1", "occ_dice", True, "occ"),
              ("mIoU", "mIoU", "iou", True, None),
              ("Dice", "Dice", "dice", True, None)]
CE_METRICS = [("ce_c1_rpr1_full", "CE c1 RPR=1 full %", "c1_rpr1", True, None),
              ("ce_c1_rpr1_occ", "CE c1 RPR=1 occ %", "c1_rpr1", True, "occ"),
              ("ce_c1_iou90_full", "CE c1 IoU≥.9 full %", "c1_iou90", True, None)]


# ------------------------------------------------------------------ loading

def load_pr(results, exp, s):
    p = os.path.join(results, exp, f"s{s}", "eval", f"pr__och-test_s{s}.instances.parquet")
    df = pd.read_parquet(p).set_index("instance_id")
    if df.index.duplicated().any():
        raise ValueError(f"duplicate instance_id in {p}")
    return df


def load_ce(results, exp, s):
    p = os.path.join(results, exp, f"s{s}", "eval", f"ce__och-test_s{s}.jsonl")
    if not os.path.exists(p):
        return None
    rows = []
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            ious, rprs = r["iou_per_click"], r.get("rpr_per_click") or []
            rows.append({"instance_id": r["instance_id"],
                         "c1_iou90": float(len(ious) >= 1 and ious[0] >= 0.90) * 100.0,
                         "c1_rpr1": float(len(rprs) >= 1 and rprs[0] >= 1.0) * 100.0,
                         "occ": r["num_gt_components"] >= 2})
    return pd.DataFrame(rows).set_index("instance_id")


def build_family(results, methods, metrics, loader, occ_col):
    """-> per seed: {"n": n, "occ": bool[n], "M": {method: float[n, n_metrics]}} with a
    common instance order for all methods (sets must be identical: pairing)."""
    out = []
    for s in SEEDS:
        dfs = {m: loader(results, e, s) for m, e in methods}
        if any(d is None for d in dfs.values()):
            missing = [m for m, d in dfs.items() if d is None]
            raise FileNotFoundError(f"s{s}: missing files for {missing}")
        ref_m = methods[0][0]
        order = dfs[ref_m].index
        for m, d in dfs.items():
            if set(d.index) != set(order):
                raise ValueError(f"s{s}: instance set of {m} differs from {ref_m} "
                                 f"({len(set(d.index) ^ set(order))} ids) — pairing broken")
        occ = dfs[ref_m].loc[order, occ_col].to_numpy(bool)
        for m, d in dfs.items():
            if not np.array_equal(d.loc[order, occ_col].to_numpy(bool), occ):
                raise ValueError(f"s{s}: occlusion flag of {m} differs from {ref_m}")
        M = {m: np.column_stack([d.loc[order, col].to_numpy(float) for _, _, col, _, _ in metrics])
             for m, d in dfs.items()}
        out.append({"n": len(order), "occ": occ, "M": M})
    return out


def metric_values(mat, occ, metrics):
    """mat: (n, n_metrics) rows of one (resampled) seed -> vector of metric values."""
    v = np.empty(len(metrics))
    for j, (_, _, _, _, subset) in enumerate(metrics):
        col = mat[:, j] if subset is None else mat[occ, j]
        v[j] = col.mean() if len(col) else np.nan
    return v


# ---------------------------------------------------------------- bootstrap

def bootstrap(family, methods, metrics, n_boot, rng):
    """-> point: {method: [n_metrics]}, reps: {method: [n_boot, n_metrics]}, per_seed."""
    names = [m for m, _ in methods]
    per_seed = {m: np.array([metric_values(fs["M"][m], fs["occ"], metrics) for fs in family])
                for m in names}
    point = {m: per_seed[m].mean(0) for m in names}
    reps = {m: np.empty((n_boot, len(metrics))) for m in names}
    S = len(family)
    for b in range(n_boot):
        draws = []
        for s in rng.integers(0, S, size=S):
            idx = rng.integers(0, family[s]["n"], size=family[s]["n"])
            draws.append((s, idx))
        for m in names:
            reps[m][b] = np.mean([metric_values(family[s]["M"][m][idx], family[s]["occ"][idx], metrics)
                                  for s, idx in draws], axis=0)
    return point, reps, per_seed


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def analyse(family_name, family, methods, metrics, comparisons, n_boot, rng):
    point, reps, per_seed = bootstrap(family, methods, metrics, n_boot, rng)
    zs, main = "SAM1", f"N={N_REUSED}"
    sign = np.array([1.0 if hi else -1.0 for _, _, _, hi, _ in metrics])
    gain = sign * (point[main] - point[zs])                  # main gain over SAM1 (> 0 expected)

    rows = []
    for a, b in comparisons:
        d_pt = point[a] - point[b]
        d_rep = reps[a] - reps[b]
        d_seed = per_seed[a] - per_seed[b]
        for j, (key, label, _, hi, _) in enumerate(metrics):
            r = d_rep[:, j][~np.isnan(d_rep[:, j])]
            lo, up = np.quantile(r, [0.025, 0.975])
            p = min(1.0, 2 * min(np.mean(r <= 0), np.mean(r >= 0)))
            pct = sorted([sign[j] * lo / gain[j] * 100, sign[j] * up / gain[j] * 100])
            won = int(np.sum(d_seed[:, j] > 0) if hi else np.sum(d_seed[:, j] < 0))
            rows.append({"family": family_name, "a": a, "b": b, "metric": key, "label": label,
                         "delta": float(d_pt[j]), "lo": float(lo), "hi": float(up), "p": float(p),
                         "pct_gain_est": float(sign[j] * d_pt[j] / gain[j] * 100),
                         "pct_gain_lo": float(pct[0]), "pct_gain_hi": float(pct[1]),
                         "seeds_won": won, "n_seeds": int(d_seed.shape[0])})
    for r, padj in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = float(padj)

    rec = []                                                   # fraction of main gain recovered
    den = sign * (reps[main] - reps[zs])
    for m, _ in methods:
        if m == zs:
            continue
        num = sign * (reps[m] - reps[zs])
        frac = num / den
        for j, (key, label, *_rest) in enumerate(metrics):
            f = frac[:, j][np.isfinite(frac[:, j])]
            lo, up = np.quantile(f, [0.025, 0.975]) if len(f) else (np.nan, np.nan)
            rec.append({"family": family_name, "method": m, "metric": key, "label": label,
                        "value": float(point[m][j]),
                        "frac_est": float(sign[j] * (point[m][j] - point[zs][j]) / gain[j]),
                        "frac_lo": float(lo), "frac_hi": float(up)})
    return rows, rec, {m: point[m].tolist() for m in point}, {k: float(g) for k, g in
                                                             zip([x[0] for x in metrics], gain)}


# --------------------------------------------------------------------- main

def fmt_num(x, key):
    dec = 5 if key == "VarIoU_per_prompt" else (2 if key.startswith("ce_") else 4)
    return f"{x:+.{dec}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--rng_seed", type=int, default=0)
    ap.add_argument("--no_ce", action="store_true")
    args = ap.parse_args()

    R = paths.get("results_root")
    results = os.path.join(R, "results")
    out_dir = os.path.join(R, "analysis", "R9")
    os.makedirs(out_dir, exist_ok=True)

    methods = [("SAM1", "zs__sam1")] + [(f"N={N}", exp_name(N)) for N in NS]
    ns = [f"N={N}" for N in NS]
    comparisons = [(ns[i + 1], ns[i]) for i in range(len(ns) - 1)] + [(ns[-1], f"N={N_REUSED}")]

    rng = np.random.default_rng(args.rng_seed)
    fams = [("PR", PR_METRICS, load_pr, "is_occluded")]
    if not args.no_ce:
        fams.append(("CE", CE_METRICS, load_ce, "occ"))
    all_rows, all_rec, points, gains = [], [], {}, {}
    for name, metrics, loader, occ_col in fams:
        fam = build_family(results, methods, metrics, loader, occ_col)
        rows, rec, pt, g = analyse(name, fam, methods, metrics, comparisons, args.n_boot, rng)
        all_rows += rows
        all_rec += rec
        points[name], gains[name] = pt, g

    with open(os.path.join(out_dir, "r9_stats.json"), "w") as f:
        json.dump({"n_boot": args.n_boot, "rng_seed": args.rng_seed, "methods": dict(methods),
                   "comparisons": comparisons, "points": points, "main_gain": gains,
                   "comparisons_rows": all_rows, "gain_recovered": all_rec}, f, indent=2)

    lines = ["# R9 — paired hierarchical bootstrap (seeds → instances)", "",
             f"n_boot = {args.n_boot}, shared replicates across methods; Holm within PR and within CE. "
             "Δ = larger N − smaller N. '% gain' = Δ (and its 95 % CI) in units of the main "
             "gain over SAM1 (N=100 − SAM1), positive = larger N better. Seeds = seeds where the "
             "larger N is better.", ""]
    for name, metrics, *_ in fams:
        lines += [f"## {name}: adjacent comparisons", "",
                  "| comparison | metric | Δ [95% CI] | % gain [CI] | p | p_Holm | seeds |",
                  "|---|---|---|---|---|---|---|"]
        for r in [r for r in all_rows if r["family"] == name]:
            star = " **" if r["p_holm"] < 0.05 else ""
            lines.append(
                f"| {r['a']} − {r['b']} | {r['label']} | {fmt_num(r['delta'], r['metric'])} "
                f"[{fmt_num(r['lo'], r['metric'])}, {fmt_num(r['hi'], r['metric'])}] | "
                f"{r['pct_gain_est']:+.1f} [{r['pct_gain_lo']:+.1f}, {r['pct_gain_hi']:+.1f}] | "
                f"{r['p']:.3f} | {r['p_holm']:.3f}{star} | {r['seeds_won']}/{r['n_seeds']} |")
        lines += ["", f"## {name}: fraction of the main gain recovered, (N − SAM1) / (N=100 − SAM1)", "",
                  "| metric | " + " | ".join(ns) + " |", "|---|" + "---|" * len(ns)]
        for key, label, *_ in metrics:
            cells = []
            for m in ns:
                r = next(x for x in all_rec if x["family"] == name and x["method"] == m and x["metric"] == key)
                cells.append(f"{r['frac_est']:.2f} [{r['frac_lo']:.2f}, {r['frac_hi']:.2f}]")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["Reading rule: 'saturated between N and 2N' only if the % gain CI of 2N − N is narrow "
              "and close to 0 (compare its width with the main gain); a wide CI is inconclusive."]
    md = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "r9_stats.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()