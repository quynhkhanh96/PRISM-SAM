"""
R2 runs: attention readouts for every R2 experiment x seed (eval_attn.py) over
GPUs, plus (with --sanity) the tok0 identity check (r2_sanity_tok0.py).

    python scripts/run_r2.py --dry_run
    python scripts/run_r2.py --gpus 0,1,2 --sanity          # everything
    python scripts/run_r2.py --gpus 0 --no_ablation         # skip the S4.2 reproduction runs
    python scripts/r2_table.py                              # afterwards

Experiment list: prismsam/r2.py (SAM1, PRISM-SAM main, twt-full, nontwt-full,
nontwt-full-tok0, tok, enc-lora, dec-full; + the 4 legacy QK/VO ablation runs to
reproduce the old S4.2 table). Idempotent: a finished attn summary is not
recomputed; the sanity check is skipped if its summary.json exists.
Logs: <results_root>/analysis/R2/logs/.
"""

import argparse
import os
import queue
import subprocess
import sys
import threading

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from prismsam import paths  # noqa: E402
from prismsam.r2 import attn_stem, experiments  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--no_ablation", action="store_true")
    ap.add_argument("--sanity", action="store_true", help="also run r2_sanity_tok0.py (one job)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    R = paths.get("results_root")
    results = os.path.join(R, "results")
    logdir = os.path.join(R, "analysis", "R2", "logs")
    jobs = []
    sanity_done = os.path.exists(os.path.join(R, "analysis", "R2", "sanity_tok0", "summary.json"))
    if args.sanity and not sanity_done:
        jobs.append(("sanity", None, [sys.executable, os.path.join(REPO, "scripts", "r2_sanity_tok0.py"),
                                      "--seeds", args.seeds, "--device", args.device]))
    for e in experiments(R, include_ablation=not args.no_ablation):
        exp_dir = os.path.join(results, e["exp_name"])
        if not os.path.exists(os.path.join(exp_dir, "config.json")):
            print(f"  SKIP {e['label']}: no {exp_dir}/config.json")
            continue
        zs = e["variant"] is None
        for s in seeds:
            if not zs and not os.path.lexists(os.path.join(exp_dir, f"s{s}", "ckpt", "final.pth")):
                print(f"  SKIP {e['label']} s{s}: no checkpoint")
                continue
            if os.path.exists(attn_stem(exp_dir, s, zs) + ".summary.json"):
                continue
            cmd = [sys.executable, os.path.join(REPO, "scripts", "eval_attn.py"),
                   "--exp_dir", exp_dir, "--test_seed", str(s), "--device", args.device]
            if not zs:
                cmd += ["--train_seed", str(s)]
            jobs.append((e["exp_name"], s, cmd))

    if args.dry_run:
        for name, s, _ in jobs:
            print(f"  {name}" + ("" if s is None else f" s{s}"))
        print(f"{len(jobs)} jobs")
        return

    os.makedirs(logdir, exist_ok=True)
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    status, lock = {}, threading.Lock()

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        while True:
            try:
                name, s, cmd = q.get_nowait()
            except queue.Empty:
                return
            log = os.path.join(logdir, f"{name}" + ("" if s is None else f"__s{s}") + ".log")
            with open(log, "a") as lf:
                rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
            with lock:
                status[(name, s)] = "OK" if rc == 0 else f"FAILED (see {log})"
                print(f"  [gpu {gpu}] {name}" + ("" if s is None else f" s{s}") + f": {status[(name, s)]}")

    print(f"{len(jobs)} jobs on GPUs {args.gpus}")
    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    bad = [k for k, v in status.items() if v != "OK"]
    print(f"done: {len(status) - len(bad)}/{len(status)} ok")
    if args.sanity:
        print("sanity verdict: see analysis/R2/sanity_tok0/summary.json ('overall')")


if __name__ == "__main__":
    main()