"""
R1/R5 official runs: each variant at the LR selected by the pilot sweep,
seeds 0-4: train -> PR eval (-> CE eval with --ce).

    python scripts/run_official.py --dry_run
    python scripts/run_official.py --gpus 0,1,2            # PR only
    python scripts/run_official.py --gpus 0,1,2 --ce       # PR + CE
    python scripts/run_official.py --variants twt-full tok --gpus 0

Reads <results_root>/analysis/R1/lr_sweep_s10.json (written by sweep_lr.py).
Refuses to run a variant whose sweep is incomplete.

PRISM-SAM (twt-lora), decision (a) of Nov 20: main results stay at lr=1e-4
(re-registered pre-revision checkpoints). If the sweep selects 1e-4, nothing is
trained. If it selects another LR, the 5 seeds at that LR are trained into
their own folder and reported only in the R1 table (best-vs-best row); the
main method switches LR only if the gain over 1e-4 lies outside the R7
bootstrap CI (to be decided after the R7 analysis, not by this script).

Idempotent: existing checkpoints / eval files are not recomputed.
"""

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import yaml  # noqa: E402

from prismsam import paths  # noqa: E402
from prismsam.config import fmt_lr, resolve_config  # noqa: E402

VARIANTS = ["twt-lora", "twt-full", "nontwt-full", "nontwt-lora", "enc-lora", "tok"]
MAIN_LR = 1e-4
SEEDS = [0, 1, 2, 3, 4]


def raw_config(variant, lr):   # identical to sweep_lr.raw_config
    return {"data": "och", "backbone": "sam1", "model_type": "vit_l", "variant": variant,
            "train": {"lr": lr, "total_steps": 5000}, "benchmark": {"tau": 0.35, "N": 100}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--variants", nargs="*", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--ce", action="store_true", help="also run click-efficiency eval")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    R = paths.get("results_root")
    results = os.path.join(R, "results")
    sweep = os.path.join(R, "analysis", "R1", "lr_sweep_s10.json")
    if not os.path.exists(sweep):
        sys.exit(f"missing {sweep}: run scripts/sweep_lr.py first")
    with open(sweep) as f:
        selected = json.load(f)["selected"]

    gen_dir = os.path.join(REPO, "configs", "generated")
    plan, jobs = [], []
    for v in args.variants:
        lr = selected.get(v)
        if lr is None:
            sys.exit(f"{v}: sweep incomplete or missing in {sweep} — finish the sweep first")
        if v == "twt-lora" and abs(lr - MAIN_LR) < 1e-12:
            plan.append((v, lr, "selected = main LR -> reuse re-registered seeds 0-4, nothing to train"))
            continue
        raw = raw_config(v, lr)
        name = resolve_config(raw)["exp_name"]
        ypath = os.path.join(gen_dir, f"{name}.yaml")
        if not args.dry_run:
            os.makedirs(gen_dir, exist_ok=True)
            with open(ypath, "w") as f:
                yaml.safe_dump(raw, f, sort_keys=False)
        note = "best-vs-best row only (decision a)" if v == "twt-lora" else ""
        plan.append((v, lr, name + (f"  [{note}]" if note else "")))
        for s in seeds:
            jobs.append((v, name, ypath, s))

    print("Plan:")
    for v, lr, what in plan:
        print(f"  {v:12s} lr={fmt_lr(lr):<6s} {what}")
    if args.dry_run:
        print(f"{len(jobs)} jobs (train + PR{' + CE' if args.ce else ''}) over seeds {seeds}")
        return

    logdir = os.path.join(R, "analysis", "R1", "logs")
    os.makedirs(logdir, exist_ok=True)
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    status, lock = {}, threading.Lock()

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        while True:
            try:
                v, name, ypath, s = q.get_nowait()
            except queue.Empty:
                return
            exp_dir = os.path.join(results, name)
            sdir = os.path.join(exp_dir, f"s{s}")
            res = "ok"
            with open(os.path.join(logdir, f"{name}__s{s}.log"), "a") as lf:
                def call(cmd):
                    return subprocess.call([sys.executable] + cmd, stdout=lf,
                                           stderr=subprocess.STDOUT, env=env)
                if not os.path.exists(os.path.join(sdir, "ckpt", "final.pth")):
                    if call([os.path.join(REPO, "scripts", "train.py"), "--config", ypath,
                             "--seed", str(s)]) != 0:
                        res = "TRAIN_FAILED"
                if res == "ok" and not os.path.exists(
                        os.path.join(sdir, "eval", f"pr__och-test_s{s}.summary.json")):
                    if call([os.path.join(REPO, "scripts", "eval_pr.py"), "--exp_dir", exp_dir,
                             "--train_seed", str(s), "--test_seed", str(s)]) != 0:
                        res = "PR_FAILED"
                if res == "ok" and args.ce and not os.path.exists(
                        os.path.join(sdir, "eval", f"ce__och-test_s{s}.summary.json")):
                    if call([os.path.join(REPO, "scripts", "eval_ce.py"), "--exp_dir", exp_dir,
                             "--train_seed", str(s), "--test_seed", str(s)]) != 0:
                        res = "CE_FAILED"
            with lock:
                status[(name, s)] = res
                print(f"  [gpu {gpu}] {v} s{s}: {res}")

    print(f"{len(jobs)} jobs on GPUs {args.gpus}")
    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # registry rows for new experiments
    reg = os.path.join(R, "registry.csv")
    existing = set()
    if os.path.exists(reg):
        with open(reg) as f:
            existing = {row["exp_name"] for row in csv.DictReader(f)}
    new_file = not os.path.exists(reg)
    with open(reg, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["exp_name", "legacy_name", "r_items", "status", "notes"])
        for v, name, _, _ in {(j[0], j[1], None, None) for j in jobs}:
            if name not in existing:
                w.writerow([name, "", "R1;R5;R7", "trained",
                            "best-vs-best twt-lora row" if v == "twt-lora" else "R1 variant at swept LR"])

    bad = {k: v for k, v in status.items() if v != "ok"}
    print(f"done: {len(status) - len(bad)}/{len(status)} ok")
    for (name, s), v in sorted(bad.items()):
        print(f"  {v}: {name} s{s} (log: analysis/R1/logs/{name}__s{s}.log)")


if __name__ == "__main__":
    main()