"""
R3 official runs (OCHuman): PRISM-SAM recipe unchanged (twt-lora, lr 1e-4,
5000 steps), trained on the few-shot sets of each new mining criterion,
seeds 0-4: train -> PR eval -> CE eval.

    python scripts/run_r3.py --dry_run
    python scripts/run_r3.py --gpus 0,1,2
    python scripts/run_r3.py --gpus 0 --crit 0.5,150

Needs the few-shot files from scripts/build_fewshot_och.py. Idempotent.
The main row (tau 0.35, area 0) is the re-registered pre-revision run; nothing
is trained for it.
"""

import argparse
import csv
import os
import queue
import subprocess
import sys
import threading

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import yaml  # noqa: E402

from prismsam import paths  # noqa: E402
from prismsam.config import fewshot_path, resolve_config  # noqa: E402

CRITERIA = [(0.5, 0), (0.5, 150), (0.25, 150)]
SEEDS = [0, 1, 2, 3, 4]


def raw_config(tau, area):
    b = {"tau": tau, "N": 100}
    if area:
        b["mine_area"] = area
    return {"data": "och", "backbone": "sam1", "model_type": "vit_l", "variant": "twt-lora",
            "train": {"lr": 1e-4, "total_steps": 5000}, "benchmark": b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--crit", nargs="*", default=None, help="e.g. 0.5,0 0.25,150")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--no_ce", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    crits = ([(float(c.split(",")[0]), int(c.split(",")[1])) for c in args.crit]
             if args.crit else CRITERIA)

    R = paths.get("results_root")
    results = os.path.join(R, "results")
    gen_dir = os.path.join(REPO, "configs", "generated")
    jobs = []
    for tau, area in crits:
        raw = raw_config(tau, area)
        cfg = resolve_config(raw)
        name = cfg["exp_name"]
        ypath = os.path.join(gen_dir, f"{name}.yaml")
        missing = [s for s in seeds if not os.path.exists(fewshot_path(cfg, s))]
        print(f"  ({tau}, {area}) -> {name}" + (f"  MISSING few-shot for seeds {missing}" if missing else ""))
        if missing and not args.dry_run:
            sys.exit("run scripts/build_fewshot_och.py first")
        if not args.dry_run:
            os.makedirs(gen_dir, exist_ok=True)
            with open(ypath, "w") as f:
                yaml.safe_dump(raw, f, sort_keys=False)
        jobs += [(name, ypath, s) for s in seeds]
    if args.dry_run:
        print(f"[dry run] {len(jobs)} jobs (train + PR{'' if args.no_ce else ' + CE'})")
        return

    logdir = os.path.join(R, "analysis", "R3", "logs")
    os.makedirs(logdir, exist_ok=True)
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    status, lock = {}, threading.Lock()

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        while True:
            try:
                name, ypath, s = q.get_nowait()
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
                if res == "ok" and not args.no_ce and not os.path.exists(
                        os.path.join(sdir, "eval", f"ce__och-test_s{s}.summary.json")):
                    if call([os.path.join(REPO, "scripts", "eval_ce.py"), "--exp_dir", exp_dir,
                             "--train_seed", str(s), "--test_seed", str(s)]) != 0:
                        res = "CE_FAILED"
            with lock:
                status[(name, s)] = res
                print(f"  [gpu {gpu}] {name} s{s}: {res}")

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

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
        for name in sorted({j[0] for j in jobs} - existing):
            w.writerow([name, "", "R3;R7", "trained", "mining-criterion sensitivity"])

    bad = {k: v for k, v in status.items() if v != "ok"}
    print(f"done: {len(status) - len(bad)}/{len(status)} ok")
    for (name, s), v in sorted(bad.items()):
        print(f"  {v}: {name} s{s} (log: analysis/R3/logs/{name}__s{s}.log)")


if __name__ == "__main__":
    main()
