"""
Batch 0 — OCHuman: register pre-revision runs under canonical names and dump
per-instance PR results for all of them (no training).

    python scripts/batch0_och.py --gpus 0,1,2            # everything
    python scripts/batch0_och.py --gpus 0 --only zs__sam1 --seeds 0
    python scripts/batch0_och.py --dry_run               # show what would happen

Idempotent; safe to re-run. Steps:
  1. benchmarks  copy legacy test / few-shot json into
                 <results_root>/benchmarks/och/test/test_s{s}.json
                 <results_root>/benchmarks/och/train/tau035/s{s}/fewshot_N100.json
                 (existing targets are md5-checked, never overwritten)
  2. register    <results_root>/results/<canonical>/config.json (with a "legacy"
                 block) and s{s}/ckpt/final.pth -> symlink to the old final.pth
  3. evaluate    eval_pr.py --exp_dir ... then tests/test_regression_och.py
                 against the old eval_logs_seed{s}.txt, jobs queued over GPUs
  4. summary     <results_root>/analysis/batch0/och_summary.csv, registry.csv rows

A folder whose config.json has a "legacy" block holds re-registered old
checkpoints; train.py refuses to write into it.
"""

import argparse
import csv
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from prismsam import paths  # noqa: E402
from prismsam.config import resolve_config  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
OLD_LORA = ("OCH-scope{scope}-proj{proj}-r8-alpha8-hypermlpFalse-focal1.0-dice1.0-"
            "inst2-K3-ep100-shots100-fss{seed}-fixedsched")
OLD_DEC = "OCH-fullFT-focal1.0-dice1.0-inst2-K3-lr3e-05-ep100-shots100-fss{seed}-fixedsched"


def _lora(scope, proj):
    return {"site": "twt", "param": "lora", "tok": True,
            "lora": {"rank": 8, "alpha": 8, "dropout": 0.05, "scope": scope, "proj": proj}}


BENCH = {"tau": 0.35, "N": 100}
LEGACY_RUNS = [
    # (adapt, lr, old dir template, R-items served)
    (_lora("full", "qkvo"), 1e-4, OLD_LORA.replace("{scope}", "full").replace("{proj}", "qkvo"),
     "main;R1;R2;R3;R6;R7;R9;W2"),
    (_lora("min", "qk"), 1e-4, OLD_LORA.replace("{scope}", "min").replace("{proj}", "qk"), "ablation;R7"),
    (_lora("min", "vo"), 1e-4, OLD_LORA.replace("{scope}", "min").replace("{proj}", "vo"), "ablation;R7"),
    (_lora("final", "qk"), 1e-4, OLD_LORA.replace("{scope}", "final").replace("{proj}", "qk"), "ablation;R6;R7"),
    (_lora("final", "vo"), 1e-4, OLD_LORA.replace("{scope}", "final").replace("{proj}", "vo"), "ablation;R6;R7"),
    ({"site": "dec", "param": "full", "tok": True}, 3e-5, OLD_DEC, "R1;R2;R7"),
]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_checked(src, dst, dry):
    if not os.path.exists(src):
        return f"MISSING source {src}"
    if os.path.exists(dst):
        if md5(src) != md5(dst):
            raise RuntimeError(f"{dst} exists with different content than {src}")
        return "ok (exists)"
    if dry:
        return "would copy"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def write_config(exp_dir, cfg, dry):
    path = os.path.join(exp_dir, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old != cfg:
            raise RuntimeError(
                f"{path} exists with a different config.\n"
                f"  If it is the stopped reproduction run of train.py, rename that folder "
                f"(e.g. add '__repro') or delete it, then re-run.")
        return
    if not dry:
        os.makedirs(exp_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--only", nargs="*", default=None, help="canonical exp names to run")
    ap.add_argument("--model_type", default="vit_l")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    dry = args.dry_run

    L = paths.get("legacy_och_dir")
    R = paths.get("results_root")
    results = os.path.join(R, "results")

    # ---------------------------------------------------------- 1. benchmarks
    print("[1] benchmarks")
    for s in seeds:
        st = copy_checked(f"{L}/benchmark_data/ochuman_test_seed{s}.json",
                          f"{R}/benchmarks/och/test/test_s{s}.json", dry)
        sf = copy_checked(f"{L}/fewshot_data/ochuman_few-shot_seed{s}.json",
                          f"{R}/benchmarks/och/train/tau035/s{s}/fewshot_N100.json", dry)
        print(f"  s{s}: test {st}; fewshot {sf}")

    # ----------------------------------------------------------- 2. register
    print("[2] register")
    jobs, registry_rows = [], []
    for adapt_cfg, lr, old_tmpl, r_items in LEGACY_RUNS:
        cfg = resolve_config({"data": "och", "backbone": "sam1", "model_type": args.model_type,
                              "adapt": adapt_cfg, "train": {"lr": lr, "total_steps": 5000},
                              "benchmark": BENCH})
        cfg["legacy"] = {"run_dir": f"{L}/{old_tmpl}", "note": "pre-revision checkpoint, re-registered"}
        name = cfg["exp_name"]
        if args.only and name not in args.only:
            continue
        exp_dir = os.path.join(results, name)
        present = [s for s in seeds if os.path.exists(f"{L}/{old_tmpl.format(seed=s)}/final.pth")]
        if not present:
            print(f"  SKIP {name}: no legacy checkpoint found ({old_tmpl})")
            continue
        write_config(exp_dir, cfg, dry)
        registry_rows.append((name, old_tmpl, r_items))
        for s in seeds:
            old_dir = f"{L}/{old_tmpl.format(seed=s)}"
            old_ckpt, old_log = f"{old_dir}/final.pth", f"{old_dir}/eval_logs_seed{s}.txt"
            if s not in present:
                print(f"  SKIP {name} s{s}: missing {old_ckpt}")
                continue
            link = os.path.join(exp_dir, f"s{s}", "ckpt", "final.pth")
            if not os.path.lexists(link) and not dry:
                os.makedirs(os.path.dirname(link), exist_ok=True)
                os.symlink(old_ckpt, link)
            jobs.append((name, s, old_log))
        print(f"  {name}  <-  {old_tmpl}")

    zs_name = "zs__sam1"
    if not args.only or zs_name in args.only:
        zs_cfg = {"exp_name": zs_name, "data": "och", "backbone": "sam1",
                  "model_type": args.model_type, "adapt": None,
                  "legacy": {"run_dir": f"{L}/sam", "note": "zero-shot SAM1, pre-revision logs"}}
        write_config(os.path.join(results, zs_name), zs_cfg, dry)
        registry_rows.append((zs_name, "sam", "baseline;R7"))
        for s in seeds:
            jobs.append((zs_name, s, f"{L}/sam/eval_logs_seed{s}.txt"))
        print(f"  {zs_name}  <-  sam/")

    if dry:
        print(f"[dry run] {len(jobs)} eval jobs:")
        for j in jobs:
            print("  ", j)
        return

    # ----------------------------------------------------------- 3. evaluate
    print(f"[3] evaluate: {len(jobs)} jobs on GPUs {args.gpus}")
    logdir = os.path.join(R, "analysis", "batch0", "logs")
    os.makedirs(logdir, exist_ok=True)
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    status, lock = {}, threading.Lock()

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        while True:
            try:
                name, s, old_log = q.get_nowait()
            except queue.Empty:
                return
            exp_dir = os.path.join(results, name)
            zs = name.startswith("zs__")
            inst = os.path.join(exp_dir, f"s{s}", "eval", f"pr__och-test_s{s}.instances.parquet")
            log_path = os.path.join(logdir, f"{name}__s{s}.log")
            with open(log_path, "a") as lf:
                rc_eval = 0
                if not os.path.exists(inst):
                    cmd = [sys.executable, os.path.join(REPO, "scripts", "eval_pr.py"),
                           "--exp_dir", exp_dir, "--test_seed", str(s), "--device", args.device]
                    if not zs:
                        cmd += ["--train_seed", str(s)]
                    rc_eval = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
                if rc_eval != 0:
                    res = "EVAL_FAILED"
                elif not os.path.exists(old_log):
                    res = "NO_LEGACY_LOG"
                else:
                    rc = subprocess.call([sys.executable,
                                          os.path.join(REPO, "tests", "test_regression_och.py"),
                                          "--instances", inst, "--legacy_log", old_log],
                                         stdout=lf, stderr=subprocess.STDOUT)
                    res = "PASS" if rc == 0 else "REGRESSION_FAIL"
            with lock:
                status[(name, s)] = res
                print(f"  [gpu {gpu}] {name} s{s}: {res}")

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # ------------------------------------------------------------ 4. summary
    out = os.path.join(R, "analysis", "batch0", "och_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exp_name", "seed", "status"])
        for (name, s), res in sorted(status.items()):
            w.writerow([name, s, res])

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
        for name, old, r_items in registry_rows:
            if name not in existing:
                w.writerow([name, old, r_items, "legacy-registered", "batch0 per-instance dump"])

    bad = {k: v for k, v in status.items() if v != "PASS"}
    print(f"[4] summary -> {out}; {len(status) - len(bad)}/{len(status)} PASS")
    for (name, s), v in sorted(bad.items()):
        print(f"  {v}: {name} s{s}  (log: analysis/batch0/logs/{name}__s{s}.log)")


if __name__ == "__main__":
    main()