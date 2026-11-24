"""
Paired hierarchical bootstrap (R7 method agreed Nov 17): resample seeds, then
instances within each drawn seed; the statistic is computed per seed and
averaged over seeds, exactly like the point estimate.

Pairing is the caller's job: each per-seed table must already hold, row by
row, the paired values of the methods being compared (same instance, same
prompts), e.g. columns E_variant and E_sam1. Resampling rows then resamples
pairs.

First user: scripts/r2_table.py. Candidate for the shared R7 implementation —
review before reusing elsewhere.
"""

import numpy as np
from scipy.stats import rankdata


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def hierarchical_bootstrap(seed_tables, stat, n_boot=2000, rng_seed=0, alpha=0.05):
    """
    seed_tables: list (one per seed) of dict column -> 1-D np.ndarray (equal lengths)
    stat:        f(dict) -> float, computed on one seed's (resampled) table
    Returns dict: est (mean over seeds of stat), lo, hi (percentile CI), se,
                  n_boot, n_nan (replicates where stat was NaN, excluded).
    """
    per_seed = [stat(t) for t in seed_tables]
    est = float(np.nanmean(per_seed)) if not np.all(np.isnan(per_seed)) else float("nan")
    if n_boot <= 0:
        return {"est": est, "lo": float("nan"), "hi": float("nan"), "se": float("nan"),
                "n_boot": 0, "n_nan": 0, "per_seed": [float(v) for v in per_seed]}
    rng = np.random.default_rng(rng_seed)
    S = len(seed_tables)
    lens = [len(next(iter(t.values()))) for t in seed_tables]
    reps = np.empty(n_boot)
    for b in range(n_boot):
        vals = []
        for s in rng.integers(0, S, size=S):
            idx = rng.integers(0, lens[s], size=lens[s])
            vals.append(stat({k: v[idx] for k, v in seed_tables[s].items()}))
        reps[b] = np.mean(vals)
    ok = reps[~np.isnan(reps)]
    lo, hi = (np.quantile(ok, [alpha / 2, 1 - alpha / 2]) if len(ok) else (np.nan, np.nan))
    return {"est": est, "lo": float(lo), "hi": float(hi),
            "se": float(ok.std(ddof=1)) if len(ok) > 1 else float("nan"),
            "n_boot": int(n_boot), "n_nan": int(n_boot - len(ok)),
            "per_seed": [float(v) for v in per_seed]}