"""
R9: nested OCHuman few-shot sets N in {20, 50, 100, 200, 400} under the main
mining criterion (tau 0.35, area 0), seeds 0-4, derived from the full-pool scan
(scripts/mine_och.py). No new mining.

    python scripts/build_fewshot_r9.py --dry_run     # checks + stats only
    python scripts/build_fewshot_r9.py

Rule (= pre-revision early-stop mining, = build_fewshot_och.py): the set of size
N is the first N selected instances in the seed's visiting order, so for a seed
every smaller set is a prefix of every larger one (nested by construction).

Checks (abort before writing anything):
  - N=100 prefix == existing fewshot_N100.json, same instances, same order
    (exit 2) -> the main-results checkpoints are reused for N=100;
  - the pool holds >= max(N) selected instances for every seed (exit 3);
  - every N fits the fixed 5000-step schedule: ceil(N / batch) divides 5000 (exit 4).

Writes
  <R>/benchmarks/och/train/tau035/s{s}/fewshot_N{N}.json   (N != 100; never overwritten)
  <R>/benchmarks/och/train/tau035/s{s}/mining_meta_r9.json
  <R>/analysis/R9/r9_mining_check.json
  <R>/analysis/R9/r9_composition.csv / .md   per N x seed + cross-seed overlap
"""

import argparse
import csv
import itertools
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np  # noqa: E402

from build_fewshot_och import (MAIN, describe, fewshot_text, git_hash, load_scan,  # noqa: E402
                               select, tag)
from prismsam import paths  # noqa: E402
from prismsam.config import TRAIN_DEFAULTS  # noqa: E402
from prismsam.datasets import och  # noqa: E402

NS = [20, 50, 100, 200, 400]          # agreed Nov 27 (N=25 replaced by 20: integer epochs)
N_REUSED = 100                        # = main results (pre-revision set)
SEEDS = [0, 1, 2, 3, 4]


def steps_ok(N, total_steps=TRAIN_DEFAULTS["total_steps"], batch=TRAIN_DEFAULTS["batch_instances"]):
    spe = math.ceil(N / batch)        # DataLoader(drop_last=False)
    return total_steps % spe == 0, spe, total_steps // spe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", default=",".join(map(str, NS)))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    Ns = sorted(int(n) for n in args.Ns.split(","))
    seeds = [int(s) for s in args.seeds.split(",")]
    if N_REUSED not in Ns:
        sys.exit(f"N={N_REUSED} must be in --Ns (reuse check)")

    bad = [(N, steps_ok(N)) for N in Ns if not steps_ok(N)[0]]
    if bad:
        print(f"!! N not compatible with {TRAIN_DEFAULTS['total_steps']} steps: "
              + ", ".join(f"N={N} ({spe} steps/epoch)" for N, (_, spe, _) in bad))
        sys.exit(4)

    R = paths.get("results_root")
    bench = os.path.join(R, "benchmarks", "och", "train")
    scan_dir = os.path.join(bench, "scan")
    out = os.path.join(R, "analysis", "R9")
    os.makedirs(out, exist_ok=True)
    train_data = och.train_pool_records(paths.get("ochuman_root"))
    idx = och.OchumanIndex.from_root(paths.get("ochuman_root"))
    d_main = f"tau{tag(MAIN)}"

    check, comp_rows, ids_by, to_write, metas, short = {}, [], {}, [], [], []
    for s in seeds:
        rows, meta = load_scan(scan_dir, s)
        chosen_max, sel_all = select(rows, MAIN, max(Ns))
        if len(chosen_max) < max(Ns):
            short.append((s, len(sel_all)))

        # ---- N=100 prefix must be the existing pre-revision set
        existing = os.path.join(bench, d_main, f"s{s}", f"fewshot_N{N_REUSED}.json")
        with open(existing) as f:
            old_ids = [idx.instance_id(r) for r in json.load(f)]
        new_ids = [r["instance_id"] for r in chosen_max[:N_REUSED]]
        check[s] = {"same_instances_same_order": old_ids == new_ids,
                    "n_common": len(set(old_ids) & set(new_ids)),
                    "pool_selected": len(sel_all)}
        print(f"s{s}: N={N_REUSED} prefix == existing set: {check[s]['same_instances_same_order']} "
              f"(common {check[s]['n_common']}/{N_REUSED}); pool selected {len(sel_all)}")
        main_ids = set(old_ids)

        meta_out = {"criterion": {"tau": MAIN[0], "min_component_area": MAIN[1], "failure_mode": "C"},
                    "seed": s, "pool_selected": len(sel_all), "pool_instances": len(rows),
                    "scan_meta": meta, "git": git_hash(), "nested": True,
                    "rule": "first N in visiting order with remote_part_recall(tau, area) < 1 "
                            "and mode C; every N is a prefix of the largest", "sets": {}}
        for N in Ns:
            chosen = chosen_max[:N]
            ids_by[(s, N)] = [r["instance_id"] for r in chosen]
            attempts = (chosen[-1]["order"] + 1) if chosen else len(rows)
            _, spe, ep = steps_ok(N)
            comp_rows.append({"seed": s, "N": N, **describe(chosen, MAIN, main_ids),
                              "attempts_to_N": attempts, "steps_per_epoch": spe, "epochs": ep})
            meta_out["sets"][str(N)] = {"N": len(chosen), "attempts_to_N": attempts,
                                        "visit_orders": [r["order"] for r in chosen],
                                        "instance_ids": ids_by[(s, N)],
                                        "file": "existing (pre-revision)" if N == N_REUSED
                                        else f"fewshot_N{N}.json"}
            if N != N_REUSED:
                to_write.append((os.path.join(bench, d_main, f"s{s}", f"fewshot_N{N}.json"),
                                 fewshot_text(train_data, chosen)))
        metas.append((os.path.join(bench, d_main, f"s{s}", "mining_meta_r9.json"), meta_out))

    # ---- cross-seed overlap per N (mean over the 10 seed pairs)
    ov_rows = []
    for N in Ns:
        ov = [len(set(ids_by[(a, N)]) & set(ids_by[(b, N)]))
              for a, b in itertools.combinations(seeds, 2)]
        if ov:
            ov_rows.append({"N": N, "mean": float(np.mean(ov)), "min": int(min(ov)), "max": int(max(ov)),
                            "mean_frac": float(np.mean(ov)) / N})

    with open(os.path.join(out, "r9_mining_check.json"), "w") as f:
        json.dump(check, f, indent=2)
    with open(os.path.join(out, "r9_composition.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0]))
        w.writeheader()
        w.writerows(comp_rows)

    keys = [k for k in comp_rows[0] if k not in ("seed", "N")]
    lines = ["# R9 — nested OCHuman few-shot sets, criterion (0.35, 0)", "",
             "Mean over seeds (min–max). N=100 = existing pre-revision set (reused). "
             "overlap_with_main = instances shared with the seed's N=100 set "
             "(= min(N, 100) by nesting).", "",
             "| N | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for N in Ns:
        sub = [r for r in comp_rows if r["N"] == N]
        cells = []
        for k in keys:
            v = np.array([r[k] for r in sub], dtype=float)
            cells.append(f"{np.nanmean(v):.3g} ({np.nanmin(v):.3g}–{np.nanmax(v):.3g})")
        lines.append(f"| {N} | " + " | ".join(cells) + " |")
    lines += ["", "## Cross-seed overlap (instances shared by two seeds' sets, 10 pairs)", "",
              "| N | mean | min | max | mean / N |", "|---|---|---|---|---|"]
    lines += [f"| {r['N']} | {r['mean']:.1f} | {r['min']} | {r['max']} | {r['mean_frac']:.3f} |"
              for r in ov_rows]
    pool = [c["pool_selected"] for c in check.values()]
    lines += ["", f"Pool (selected under (0.35, 0), whole OCHuman train scan): mean {np.mean(pool):.0f} "
              f"(min {min(pool)}, max {max(pool)}) per seed."]
    md = "\n".join(lines) + "\n"
    with open(os.path.join(out, "r9_composition.md"), "w") as f:
        f.write(md)
    print(md)

    if not all(c["same_instances_same_order"] for c in check.values()):
        print("!! N=100 prefix does NOT reproduce the existing set — nothing written "
              "(see r9_mining_check.json).")
        sys.exit(2)
    if short:
        print(f"!! pool smaller than N={max(Ns)} for (seed, pool_selected) {short} — nothing written.")
        sys.exit(3)
    if args.dry_run:
        print("[dry run] no few-shot file written")
        return
    for p, text in to_write:
        if os.path.exists(p):
            with open(p) as f:
                if f.read() != text:
                    raise RuntimeError(f"{p} exists with different content")
            print(f"ok (exists) {p}")
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)
        print(f"wrote {p}")
    for p, m in metas:                     # descriptive; rewritten on every run
        with open(p, "w") as f:
            json.dump(m, f, indent=2)


if __name__ == "__main__":
    main()