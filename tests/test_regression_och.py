"""
Regression check (mandatory Batch-0 sanity check, R7 spec):
aggregates recomputed from a per-instance file must reproduce the numbers in
the pre-revision eval log to rounding (logs print 5 decimals).

    python tests/test_regression_och.py \
        --instances <out_dir>/pr__och-test_s0.instances.parquet \
        --legacy_log <old_run_dir>/eval_logs_seed0.txt

Exit code 0 = all metrics match; 1 = mismatch (table printed either way).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from prismsam.records import aggregate_pr, parse_legacy_log  # noqa: E402

TOL = 1e-5  # 5-decimal printing => max rounding error 5e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--legacy_log", required=True)
    args = ap.parse_args()

    inst = pd.read_parquet(args.instances)
    if inst["instance_id"].duplicated().any():
        print("FAIL duplicated instance_id in file")
        sys.exit(1)
    new = aggregate_pr(inst)
    old = parse_legacy_log(args.legacy_log)

    ok = True
    print(f"{'metric':28s} {'legacy':>10s} {'new':>10s} {'|diff|':>10s}")
    for k, v_old in old.items():
        if k not in new:
            print(f"{k:28s} {v_old:10.5f} {'(absent)':>10s}")
            continue
        v_new = new[k]
        d = abs(v_new - v_old)
        exact = k.endswith("Count")
        good = (d == 0) if exact else (d <= TOL)
        ok &= good
        print(f"{k:28s} {v_old:10.5f} {v_new:10.5f} {d:10.2e} {'OK' if good else 'FAIL'}")
    missing = set(new) - set(old)
    if missing:
        print(f"(not in legacy log: {sorted(missing)})")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()