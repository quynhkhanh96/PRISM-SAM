"""
Run click-efficiency evaluation for several experiments x seeds over GPUs,
then regression-check against pre-revision CE files where they exist.

    python scripts/run_ce.py --gpus 0,1,2                     # default: Batch-0 CE set
    python scripts/run_ce.py --gpus 0,1,2 --exps och__twt-full__tok1__lr3e-5__tau035-N100__steps5000
    python scripts/run_ce.py --dry_run

Default experiment set = the two models with pre-revision CE results on OCH
(PRISM-SAM main, SAM1 zero-shot) plus dec-full. New R1 variants are passed
with --exps once trained. Idempotent: an existing ce__*.jsonl is not recomputed.

Legacy CE files are looked up in `legacy_ce_dir` (local_paths.yaml, optional):
  PRISM-SAM main -> noc_prismsam_seed{s}.jsonl, SAM1 zero-shot -> noc_sam1.jsonl (one file)
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

from prismsam import paths  # noqa: E402

MAIN = "och__twt-lora-scfull-qkvo-r8a8__tok1__lr1e-4__tau035-N100__steps5000"
DEFAULT_EXPS = [MAIN, "zs__sam1", "och__dec-full__tok1__lr3e-5__tau035-N100__steps5000"]
# SAM1 zero-shot has a single pre-revision CE file: the protocol is deterministic
# (clicks come from the GT) and the test instances are identical across seeds, so
# every zs seed is checked against the same file.
LEGACY_CE = {MAIN: "noc_prismsam_seed{s}.jsonl", "zs__sam1": "noc_sam1.jsonl"}


def legacy_file(exp, s):
    if exp not in LEGACY_CE:
        return None
    try:
        d = paths.get("legacy_ce_dir")
    except KeyError:
        return None
    p = os.path.join(d, LEGACY_CE[exp].format(s=s))
    return p if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--exps", nargs="*", default=DEFAULT_EXPS)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    R = paths.get("results_root")
    results = os.path.join(R, "results")
    jobs = []
    for exp in args.exps:
        exp_dir = os.path.join(results, exp)
        if not os.path.exists(os.path.join(exp_dir, "config.json")):
            print(f"  SKIP {exp}: no config.json")
            continue
        zs = exp.startswith("zs__")
        for s in seeds:
            if not zs and not os.path.lexists(os.path.join(exp_dir, f"s{s}", "ckpt", "final.pth")):
                print(f"  SKIP {exp} s{s}: no checkpoint")
                continue
            jobs.append((exp, s, legacy_file(exp, s)))

    if args.dry_run:
        for exp, s, leg in jobs:
            print(f"  {exp} s{s}  legacy={'yes' if leg else 'no'}")
        print(f"{len(jobs)} CE jobs")
        return

    logdir = os.path.join(R, "analysis", "ce", "logs")
    os.makedirs(logdir, exist_ok=True)
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    status, lock = {}, threading.Lock()

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        while True:
            try:
                exp, s, leg = q.get_nowait()
            except queue.Empty:
                return
            exp_dir = os.path.join(results, exp)
            out = os.path.join(exp_dir, f"s{s}", "eval", f"ce__och-test_s{s}.jsonl")
            log = os.path.join(logdir, f"{exp}__s{s}.log")
            with open(log, "a") as lf:
                rc = 0
                if not os.path.exists(os.path.join(exp_dir, f"s{s}", "eval",
                                                   f"ce__och-test_s{s}.summary.json")):
                    cmd = [sys.executable, os.path.join(REPO, "scripts", "eval_ce.py"),
                           "--exp_dir", exp_dir, "--test_seed", str(s), "--device", args.device]
                    if not exp.startswith("zs__"):
                        cmd += ["--train_seed", str(s)]
                    rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
                if rc != 0:
                    res = "EVAL_FAILED"
                elif leg is None:
                    res = "DONE (no legacy CE)"
                else:
                    rc = subprocess.call([sys.executable, os.path.join(REPO, "tests", "test_regression_ce.py"),
                                          "--new", out, "--legacy", leg],
                                         stdout=lf, stderr=subprocess.STDOUT)
                    res = "PASS" if rc == 0 else "REGRESSION_FAIL"
            with lock:
                status[(exp, s)] = res
                print(f"  [gpu {gpu}] {exp} s{s}: {res}")

    print(f"{len(jobs)} CE jobs on GPUs {args.gpus}")
    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summ = os.path.join(R, "analysis", "ce", "ce_runs.csv")
    new_file = not os.path.exists(summ)
    with open(summ, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["exp_name", "seed", "status"])
        for (exp, s), res in sorted(status.items()):
            w.writerow([exp, s, res])
    bad = {k: v for k, v in status.items() if v in ("EVAL_FAILED", "REGRESSION_FAIL")}
    print(f"done: {len(status) - len(bad)}/{len(status)} ok -> {summ}")
    for (exp, s), v in sorted(bad.items()):
        print(f"  {v}: {exp} s{s} (log: analysis/ce/logs/{exp}__s{s}.log)")


if __name__ == "__main__":
    main()