"""
R3: derive OCHuman few-shot sets for several mining criteria from the full-pool
scan (scripts/mine_och.py), verify the pre-revision set is reproduced, and
describe the composition of every set.

    python scripts/build_fewshot_och.py                 # R3 criteria, N=100, seeds 0-4
    python scripts/build_fewshot_och.py --dry_run       # verify + stats, write no few-shot file

Criterion = (tau, area): an instance is selected iff
    remote_part_recall(gt, [SAM1 masks of the click], tau, min_component_area=area) < 1
    AND failure_mode == 'C'
and the set is the first N selected instances in the seed's visiting order
(= pre-revision early-stop mining). Pre-revision OCHuman criterion: (0.35, 0).

Writes
  <R>/benchmarks/och/train/tau{tag}/s{s}/fewshot_N{N}.json   same format as before
  <R>/benchmarks/och/train/tau{tag}/s{s}/mining_meta.json
  <R>/analysis/R3/och_mining_check.json      reproduction of the existing (0.35, 0) sets
  <R>/analysis/R3/och_composition.csv / .md  per criterion x seed
  <R>/analysis/R3/och_pool.csv               full-pool counts / hit rates per criterion
Existing few-shot files are never overwritten (md5 must match, else error).
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from prismsam import mining, paths  # noqa: E402
from prismsam.config import bench_tag  # noqa: E402
from prismsam.datasets import och  # noqa: E402

MAIN = (0.35, 0)                                    # pre-revision criterion
R3 = [(0.5, 0), (0.5, 150), (0.25, 150)]            # agreed Nov 26
EVAL = (0.5, 150)                                   # RPR@0.50 as evaluated (eval_pr.py)
SEEDS = [0, 1, 2, 3, 4]


def tag(c):
    return bench_tag({"tau": c[0], "mine_area": c[1]})


def md5_text(t):
    return hashlib.md5(t.encode()).hexdigest()


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_scan(scan_dir, s):
    with open(os.path.join(scan_dir, f"s{s}", "scan_meta.json")) as f:
        meta = json.load(f)
    if not (meta["complete_pool"] and meta["contiguous"] and meta["n_records"] == meta["pool_size"]):
        raise RuntimeError(f"s{s}: scan incomplete ({meta}) — rerun mine_och.py without --limit_images")
    with open(os.path.join(scan_dir, f"s{s}", "scan.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    return rows, meta


def select(rows, crit, N):
    sel = [r for r in rows if mining.is_selected(r, *crit)]
    return sel[:N], sel


def fewshot_text(train_data, chosen):
    recs = []
    for r in chosen:
        rec = dict(train_data[r["data_idx"]])
        rec["failure_mode"] = r["failure_mode"]     # = 'C', set exactly as before
        recs.append(rec)
    return json.dumps(recs)                          # pre-revision: json.dump(records, f)


def describe(chosen, crit, main_ids):
    """Composition of one few-shot set (descriptive)."""
    tau, area = crit
    n_comp150 = [sum(c["area"] >= 150 for c in r["comps"]) for r in chosen]
    n_comp_all = [len(r["comps"]) for r in chosen]
    miss_own = [mining.missed_components(r, tau, area) for r in chosen]
    miss_eval = [mining.missed_components(r, *EVAL) for r in chosen]
    only_small = [all(c["area"] < 150 for c in m) for m in miss_own]      # only meaningful for area 0
    max_iou_missed = [max(c[f"iou_p{area}"] for c in m) for m in miss_own if m]
    ids = [r["instance_id"] for r in chosen]
    return {
        "n": len(chosen),
        "n_images": len({i.rsplit(":", 1)[0] for i in ids}),
        "overlap_with_main": len(set(ids) & main_ids),
        "mean_gt_comp_ge150": float(np.mean(n_comp150)),
        "mean_gt_comp_all": float(np.mean(n_comp_all)),
        "mean_missed_own": float(np.mean([len(m) for m in miss_own])),
        "frac_missed_only_lt150px": float(np.mean(only_small)),
        "frac_hard_under_eval_rpr": float(np.mean([len(m) > 0 for m in miss_eval])),
        "mean_rpr_eval": float(np.mean([mining.rpr_from_record(r, *EVAL) for r in chosen])),
        "mean_best_iou_sam1": float(np.mean([r["best_iou"] for r in chosen])),
        "median_max_iou_missed": float(np.median(max_iou_missed)) if max_iou_missed else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--allow_main_mismatch", action="store_true",
                    help="write the new sets even if the (0.35, 0) set is not reproduced exactly")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    R = paths.get("results_root")
    bench = os.path.join(R, "benchmarks", "och", "train")
    scan_dir = os.path.join(bench, "scan")
    out = os.path.join(R, "analysis", "R3")
    os.makedirs(out, exist_ok=True)
    train_data = och.train_pool_records(paths.get("ochuman_root"))
    idx = och.OchumanIndex.from_root(paths.get("ochuman_root"))
    all_crit = [MAIN] + R3

    check, comp_rows, pool_rows, to_write, short = {}, [], [], [], []
    for s in seeds:
        rows, meta = load_scan(scan_dir, s)
        eligible = [r for r in rows if r["failure_mode"] == "C"]

        # ---- reproduction of the existing pre-revision set
        main_chosen, _ = select(rows, MAIN, args.N)
        existing = os.path.join(bench, f"tau{tag(MAIN)}", f"s{s}", f"fewshot_N{args.N}.json")
        new_text = fewshot_text(train_data, main_chosen)
        with open(existing) as f:
            old_text = f.read()
        old = json.loads(old_text)
        old_ids = [idx.instance_id(r) for r in old]
        new_ids = [r["instance_id"] for r in main_chosen]
        check[s] = {"md5_identical": md5_text(new_text) == hashlib.md5(old_text.encode()).hexdigest(),
                    "same_instances_same_order": old_ids == new_ids,
                    "n_common": len(set(old_ids) & set(new_ids)),
                    "only_in_existing": sorted(set(old_ids) - set(new_ids)),
                    "only_in_scan": sorted(set(new_ids) - set(old_ids))}
        print(f"s{s}: (0.35, 0) reproduction: {check[s]['same_instances_same_order']} "
              f"(md5 {check[s]['md5_identical']}, common {check[s]['n_common']}/{args.N})")
        main_ids = set(old_ids)

        for crit in all_crit:
            chosen, sel_all = select(rows, crit, args.N)
            if len(chosen) < args.N:
                print(f"  WARNING s{s} {crit}: only {len(chosen)} selected in the whole pool")
                short.append((s, crit, len(chosen)))
            attempts = (chosen[-1]["order"] + 1) if chosen else len(rows)
            pool_rows.append({"seed": s, "tau": crit[0], "area": crit[1],
                              "pool_instances": len(rows), "pool_eligible_C": len(eligible),
                              "pool_selected": len(sel_all),
                              "rate_selected_per_eligible": len(sel_all) / len(eligible),
                              "attempts_to_N": attempts,
                              "rate_selected_per_attempt": len(sel_all) / len(rows)})
            comp_rows.append({"seed": s, "tau": crit[0], "area": crit[1],
                              **describe(chosen, crit, main_ids), "attempts_to_N": attempts})
            if crit == MAIN:
                continue
            d = os.path.join(bench, f"tau{tag(crit)}", f"s{s}")
            to_write.append((d, fewshot_text(train_data, chosen), {
                "criterion": {"tau": crit[0], "min_component_area": crit[1], "failure_mode": "C"},
                "seed": s, "N": len(chosen), "attempts_to_N": attempts,
                "pool_selected": len(sel_all), "pool_eligible_C": len(eligible),
                "pool_instances": len(rows), "visit_orders": [r["order"] for r in chosen],
                "instance_ids": [r["instance_id"] for r in chosen],
                "scan_meta": meta, "git": git_hash(),
                "rule": "first N in visiting order with remote_part_recall(tau, area) < 1 and mode C",
            }))

    with open(os.path.join(out, "och_mining_check.json"), "w") as f:
        json.dump(check, f, indent=2)
    for name, rows_ in [("och_composition.csv", comp_rows), ("och_pool.csv", pool_rows)]:
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0]))
            w.writeheader()
            w.writerows(rows_)

    # ---- markdown summary: mean (min-max) over seeds per criterion
    keys = [k for k in comp_rows[0] if k not in ("seed", "tau", "area")]
    lines = ["# R3 — OCHuman few-shot sets per mining criterion (N=%d)" % args.N, "",
             "Mean over seeds (min–max). Criterion (tau, area): hard if RPR(tau, "
             "min_component_area=area) < 1 on SAM1's single click, plus failure mode C. "
             "Main = (0.35, 0). 'eval' = (0.5, 150), the RPR@0.50 used in evaluation.", "",
             "| tau | area | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for crit in all_crit:
        sub = [r for r in comp_rows if (r["tau"], r["area"]) == crit]
        cells = []
        for k in keys:
            v = np.array([r[k] for r in sub], dtype=float)
            cells.append(f"{np.nanmean(v):.3g} ({np.nanmin(v):.3g}–{np.nanmax(v):.3g})")
        lines.append(f"| {crit[0]} | {crit[1]} | " + " | ".join(cells) + " |")
    lines += ["", "## Full pool (no early stop)", "",
              "| tau | area | pool_eligible_C | pool_selected | selected/eligible |", "|---|---|---|---|---|"]
    for crit in all_crit:
        sub = [r for r in pool_rows if (r["tau"], r["area"]) == crit]
        lines.append(f"| {crit[0]} | {crit[1]} | {np.mean([r['pool_eligible_C'] for r in sub]):.0f} | "
                     f"{np.mean([r['pool_selected'] for r in sub]):.1f} | "
                     f"{np.mean([r['rate_selected_per_eligible'] for r in sub]):.3f} |")
    lines += ["", "Pool counts differ across seeds only through the click point (sampled per seed)."]
    md = "\n".join(lines) + "\n"
    with open(os.path.join(out, "och_composition.md"), "w") as f:
        f.write(md)
    print(md)

    ok = all(c["same_instances_same_order"] for c in check.values())
    if not ok:
        print("!! (0.35, 0) sets NOT reproduced exactly for seeds "
              f"{[s for s, c in check.items() if not c['same_instances_same_order']]} "
              "— see och_mining_check.json (likely GPU numerical noise on borderline components).")
        if not args.allow_main_mismatch:
            print("No few-shot file written. Inspect, then rerun with --allow_main_mismatch if acceptable.")
            sys.exit(2)
    if short:
        print(f"!! pool too small for N={args.N}: {short} — no few-shot file written (decide N / criterion first).")
        sys.exit(3)
    if args.dry_run:
        print("[dry run] no few-shot file written")
        return
    for d, text, meta in to_write:
        p = os.path.join(d, f"fewshot_N{args.N}.json")
        if os.path.exists(p):
            with open(p) as f:
                if f.read() != text:
                    raise RuntimeError(f"{p} exists with different content")
            print(f"ok (exists) {p}")
            continue
        os.makedirs(d, exist_ok=True)
        with open(p, "w") as f:
            f.write(text)
        with open(os.path.join(d, "mining_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
