"""
R2 tables and figures from the attention files written by eval_attn.py.

    python scripts/r2_table.py                 # n_boot = 2000
    python scripts/r2_table.py --n_boot 0      # fast, no CIs

Reads <exp>/s{s}/eval/attn__och-test_s{s}.instances.parquet for every
experiment in prismsam/r2.py and writes <results_root>/analysis/R2/:

  r2_instances.parquet   one row per (label, seed, instance): this model's
                         mean_enrichment / rpr / n_remote + SAM1's on the same
                         instance and prompts (columns *_sam1). Everything below
                         is computed from this file; re-reporting needs only it
                         (plus the prompt-level files for other layers/heads).
  r2_per_seed.csv        per (label, seed) statistics
  r2_table.md            main table + regression vs the old S4.2 table + checks
  r2_bootstrap.json      every CI with its per-seed values
  r2_delta_scatter.png   ΔA_enrich vs ΔRPR(occ) relative to SAM1, per variant
  r2_scatter_grid.png    A_enrich vs RPR per variant (instances pooled over seeds)

Definitions (Section 5 / S4.2, causal_analysis.py):
  A_enrich   per-instance mean over prompts of A_enrich_remote (final layer,
             head-averaged); instances with no remote component are excluded.
  RPR(occ)   mean RPR@0.50 over instances with >= 1 remote component.
  ρ          Spearman(A_enrich, RPR) per seed, on instances with A_enrich defined.
  Seeds: mean ± std over seeds (std ddof given by --ddof; the S4.2 table used 0).
New columns (paired with SAM1 on the same instances and prompts):
  ΔA, ΔRPR   variant − SAM1, mean over instances, per seed then averaged.
  ρ_base     Spearman(A_enrich of SAM1, RPR of variant): how much of ρ is
             already predicted by the unadapted model's attention.
  ρ_Δ        Spearman(ΔA, ΔRPR): are instances whose attention moved more also
             the ones that improved more.
CIs: paired hierarchical bootstrap (seeds, then instances), 95% percentile.
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from prismsam import paths  # noqa: E402
from prismsam.bootstrap import hierarchical_bootstrap, spearman  # noqa: E402
from prismsam.r2 import S42_REFERENCE, SEEDS, attn_stem, experiments  # noqa: E402

TOL = {"mean_E": 0.02, "rpr_occ": 1e-3, "rho": 3e-3}   # regression vs S4.2 (cross-GPU noise)


def load_all(results, exps, seeds):
    frames, pr_check, missing = [], [], []
    for e in exps:
        exp_dir = os.path.join(results, e["exp_name"])
        zs = e["variant"] is None
        for s in seeds:
            p = attn_stem(exp_dir, s, zs) + ".instances.parquet"
            if not os.path.exists(p):
                missing.append(f"{e['label']} s{s}")
                continue
            d = pd.read_parquet(p)
            d["label"], d["group"], d["seed"] = e["label"], e["group"], s
            frames.append(d)
            prp = os.path.join(exp_dir, f"s{s}", "eval", f"pr__och-test_s{s}.instances.parquet")
            if os.path.exists(prp):
                pr = pd.read_parquet(prp)[["instance_id", "rpr"]]
                m = d[["instance_id", "rpr"]].merge(pr, on="instance_id", suffixes=("", "_pr"))
                pr_check.append({"label": e["label"], "seed": s, "n": len(m),
                                 "n_rpr_mismatch": int((np.abs(m["rpr"] - m["rpr_pr"]) > 1e-12).sum()),
                                 "n_missing": len(d) - len(m)})
            else:
                pr_check.append({"label": e["label"], "seed": s, "n": 0,
                                 "n_rpr_mismatch": None, "n_missing": None})
    if not frames:
        sys.exit("no attention files found — run scripts/run_r2.py first")
    return pd.concat(frames, ignore_index=True), pr_check, missing


def pair_with_sam1(df):
    sam = df[df["label"] == "SAM1 zero-shot"][["seed", "instance_id", "mean_enrichment", "rpr", "n_remote"]]
    sam = sam.rename(columns={"mean_enrichment": "mean_enrichment_sam1", "rpr": "rpr_sam1",
                              "n_remote": "n_remote_sam1"})
    out = df.merge(sam, on=["seed", "instance_id"], how="left")
    have = out["rpr_sam1"].notna()
    # validity of A_enrich depends only on GT and prompts -> must agree with SAM1
    bad = have & (out["mean_enrichment"].isna() != out["mean_enrichment_sam1"].isna())
    if bad.any():
        raise RuntimeError(f"{int(bad.sum())} instances valid for one model but not SAM1 — prompts differ?")
    return out


def seed_stats(d):
    v = d[d["mean_enrichment"].notna()]
    rho, p = stats.spearmanr(v["mean_enrichment"], v["rpr"])
    occ = d[d["n_remote"] >= 1]
    row = {"n_enrichment": len(v), "mean_E": v["mean_enrichment"].mean(),
           "median_E": v["mean_enrichment"].median(), "std_E_instances": v["mean_enrichment"].std(ddof=0),
           "rpr_all": d["rpr"].mean(), "rpr_occ": occ["rpr"].mean(), "rho": rho, "pval": p}
    if d["rpr_sam1"].notna().all():
        dE = v["mean_enrichment"] - v["mean_enrichment_sam1"]
        row.update({"dE": dE.mean(), "max_abs_dE": dE.abs().max(),
                    "dRPR_occ": (occ["rpr"] - occ["rpr_sam1"]).mean(),
                    "rho_base": spearman(v["mean_enrichment_sam1"], v["rpr"]),
                    "rho_delta": spearman(dE, v["rpr"] - v["rpr_sam1"])})
    return row


def tables_for_bootstrap(d_label):
    """per seed: arrays over instances with A_enrich defined (+ occ flag)."""
    out = []
    for s, d in sorted(d_label.groupby("seed")):
        v = d[d["mean_enrichment"].notna()]
        out.append({"E": v["mean_enrichment"].to_numpy(), "R": v["rpr"].to_numpy(),
                    "E0": v["mean_enrichment_sam1"].to_numpy(), "R0": v["rpr_sam1"].to_numpy(),
                    "occ": (v["n_remote"] >= 1).to_numpy()})
    return out


STATS = {
    "rho": lambda t: spearman(t["E"], t["R"]),
    "dE": lambda t: float(np.mean(t["E"] - t["E0"])),
    "rho_base": lambda t: spearman(t["E0"], t["R"]),
    "rho_delta": lambda t: spearman(t["E"] - t["E0"], t["R"] - t["R0"]),
}


def fmt_ms(vals, dec, ddof):
    v = np.asarray([x for x in vals if x is not None and not np.isnan(x)], float)
    if not len(v):
        return "—"
    sd = v.std(ddof=ddof) if len(v) > 1 else float("nan")
    return f"{v.mean():.{dec}f} ± {sd:.{dec}f}" + ("" if len(v) == len(SEEDS) else f" (n={len(v)})")


def fmt_ci(b, dec, signed=False):
    if b is None or np.isnan(b["est"]):
        return "—"
    f = f"{{:{'+' if signed else ''}.{dec}f}}"
    if np.isnan(b["lo"]):
        return f.format(b["est"])
    return f"{f.format(b['est'])} [{f.format(b['lo'])}, {f.format(b['hi'])}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--ddof", type=int, default=0, help="std over seeds; S4.2 used 0")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    R = paths.get("results_root")
    results = os.path.join(R, "results")
    out_dir = os.path.join(R, "analysis", "R2")
    os.makedirs(out_dir, exist_ok=True)
    exps = experiments(R, include_ablation=True)

    df, pr_check, missing = load_all(results, exps, seeds)
    df = pair_with_sam1(df)
    df.to_parquet(os.path.join(out_dir, "r2_instances.parquet"), index=False)

    per_seed = []
    for (label, s), d in df.groupby(["label", "seed"], sort=False):
        per_seed.append({"label": label, "seed": s, **seed_stats(d)})
    ps = pd.DataFrame(per_seed)
    ps.to_csv(os.path.join(out_dir, "r2_per_seed.csv"), index=False)

    labels = [e["label"] for e in exps if e["label"] in set(df["label"])]
    boots = {}
    for label in labels:
        dl = df[df["label"] == label]
        if dl["rpr_sam1"].isna().any():
            continue
        t = tables_for_bootstrap(dl)
        boots[label] = {k: hierarchical_bootstrap(t, f, n_boot=args.n_boot, rng_seed=i)
                        for i, (k, f) in enumerate(STATS.items())}
        occ_t = [{"R": x["R"][x["occ"]], "R0": x["R0"][x["occ"]]} for x in t]
        boots[label]["dRPR_occ"] = hierarchical_bootstrap(
            occ_t, lambda u: float(np.mean(u["R"] - u["R0"])), n_boot=args.n_boot, rng_seed=99)
    with open(os.path.join(out_dir, "r2_bootstrap.json"), "w") as f:
        json.dump({"n_boot": args.n_boot, "results": boots}, f, indent=1)

    def col(label, key):
        return ps.loc[ps["label"] == label, key].tolist()

    L = ["# R2 — point-token attention enrichment across adaptation sites (OCHuman, PR prompts)", "",
         f"Seeds {seeds}; mean ± std over seeds (ddof={args.ddof}). CIs: paired hierarchical "
         f"bootstrap, n_boot={args.n_boot}, 95% percentile. Definitions in scripts/r2_table.py.", "",
         "## Main table", "",
         "| Variant | A_enrich | RPR (occ.) | ρ(A, RPR) | worst-seed p | ΔA vs SAM1 [CI] | "
         "ΔRPR(occ) vs SAM1 [CI] | ρ_base [CI] | ρ_Δ [CI] |",
         "|---|---|---|---|---|---|---|---|---|"]
    for e in exps:
        label = e["label"]
        if e["group"] != "main" or label not in labels:
            continue
        b = boots.get(label, {})
        pv = col(label, "pval")
        worst = max(pv) if pv else float("nan")
        L.append(f"| {label} | {fmt_ms(col(label, 'mean_E'), 3, args.ddof)} | "
                 f"{fmt_ms(col(label, 'rpr_occ'), 4, args.ddof)} | {fmt_ms(col(label, 'rho'), 3, args.ddof)} | "
                 f"{'<1e-10' if worst < 1e-10 else f'{worst:.1e}'} | "
                 + " | ".join("—" if label == "SAM1 zero-shot" else fmt_ci(b.get(k), d, signed=sg)
                              for k, d, sg in [("dE", 3, True), ("dRPR_occ", 4, True),
                                               ("rho_base", 3, False), ("rho_delta", 3, False)])
                 + " |")

    L += ["", "## Regression vs the pre-revision S4.2 table", "",
          f"Tolerance on the mean over seeds: A_enrich {TOL['mean_E']}, RPR(occ) {TOL['rpr_occ']}, "
          f"ρ {TOL['rho']} (cross-GPU noise). std uses ddof=0 as in S4.2.", "",
          "| Variant | A_enrich new / old | RPR(occ) new / old | ρ new / old | verdict |",
          "|---|---|---|---|---|"]
    for label, ref in S42_REFERENCE.items():
        if label not in labels:
            L.append(f"| {label} | — | — | — | not run |")
            continue
        cells, ok = [], True
        for key, dec in [("mean_E", 3), ("rpr_occ", 4), ("rho", 3)]:
            v = np.array(col(label, key), float)
            ok &= abs(v.mean() - ref[key][0]) <= TOL[key] and len(v) == 5
            cells.append(f"{v.mean():.{dec}f} ± {v.std(ddof=0):.{dec}f} / "
                         f"{ref[key][0]:.{dec}f} ± {ref[key][1]:.{dec}f}")
        L.append(f"| {label} | " + " | ".join(cells) + f" | {'OK' if ok else 'CHECK'} |")

    L += ["", "## Checks", ""]
    if "nontwt-full-tok0" in labels:
        m = ps.loc[ps["label"] == "nontwt-full-tok0", "max_abs_dE"]
        L.append(f"- tok0: max |A_enrich(tok0) − A_enrich(SAM1)| over instances and seeds = "
                 f"{m.max():.3g} (must be 0 or float noise; see also analysis/R2/sanity_tok0/).")
    bad = [c for c in pr_check if c["n_rpr_mismatch"]]
    nopr = [c for c in pr_check if c["n_rpr_mismatch"] is None]
    L.append(f"- RPR in attention files vs PR files: {len(bad)} (label, seed) with mismatches"
             + ("" if not bad else ": " + ", ".join(f"{c['label']} s{c['seed']} ({c['n_rpr_mismatch']})"
                                                    for c in bad)))
    if nopr:
        L.append("- no PR file to cross-check: " + ", ".join(f"{c['label']} s{c['seed']}" for c in nopr))
    if missing:
        L.append("- missing attention files: " + ", ".join(missing))
    md = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "r2_table.md"), "w") as f:
        f.write(md)
    print(md)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: figures skipped")
        return
    main_labels = [e["label"] for e in exps if e["group"] == "main" and e["label"] in labels]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for i, label in enumerate(main_labels):
        d = ps[ps["label"] == label]
        if "dE" not in d or d["dE"].isna().all():
            continue
        c = f"C{i}"
        ax.scatter(d["dE"], d["dRPR_occ"], s=12, alpha=0.35, color=c)
        ax.errorbar(d["dE"].mean(), d["dRPR_occ"].mean(), xerr=d["dE"].std(ddof=args.ddof),
                    yerr=d["dRPR_occ"].std(ddof=args.ddof), fmt="o", color=c, capsize=3, label=label)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Δ A_enrich vs SAM1 (final layer)")
    ax.set_ylabel("Δ RPR@0.50 (occ.) vs SAM1")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "r2_delta_scatter.png"), dpi=170)
    plt.close(fig)

    n = len(main_labels)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.0 * rows), sharex=True, sharey=True,
                             squeeze=False)
    for ax, label in zip(axes.ravel(), main_labels):
        d = df[(df["label"] == label) & df["mean_enrichment"].notna()]
        ax.scatter(d["mean_enrichment"], d["rpr"], s=3, alpha=0.15)
        rho = ps.loc[ps["label"] == label, "rho"].mean()
        ax.set_title(f"{label}\nρ (mean over seeds) = {rho:.3f}", fontsize=8)
        ax.axvline(1.0, color="k", lw=0.6, ls=":")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("RPR@0.50")
    for ax in axes[-1, :]:
        ax.set_xlabel("A_enrich (final)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "r2_scatter_grid.png"), dpi=150)
    plt.close(fig)
    print(f"figures -> {out_dir}")


if __name__ == "__main__":
    main()