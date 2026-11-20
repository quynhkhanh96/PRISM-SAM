"""
R1/R5 learning-rate sweep for the control variants on the pilot seed.

Protocol (decided Nov 19, 2026; extended Nov 20; see R1R5_variants_and_code_notes.md):
  - all R1 variants incl. PRISM-SAM (twt-lora) are swept (best-vs-best);
    the twt-lora lr=1e-4 pilot run goes to '<canonical>__pilot' because the
    canonical folder holds the re-registered pre-revision checkpoints;
  - train on few-shot seed 10, evaluate on test seed 10 (test instances, so each
    control gets its most favourable LR — conservative for our claims);
  - selection metric fixed before looking: MeanIoU/Prompt (higher better);
    ties (|diff| < --tie_eps) broken by the number of the 7 metrics won;
  - official runs then use seeds 0-4 only. The s10 folder stays next to them
    in the experiment directory and is excluded from tables.

    python scripts/sweep_lr.py --gpus 0,1,2            # train + eval + table
    python scripts/sweep_lr.py --dry_run
    python scripts/sweep_lr.py --report_only           # rebuild the table

Each (variant, lr) is an ordinary experiment folder
  results/och__<variant>__lr<lr>__tau035-N100__steps5000/s10/
so the official seeds of the selected LR land in the same folder.
"""

import argparse
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

GRID = {
    # Nov 20: one extra step added beyond the grid edge for every variant whose
    # first-pass optimum sat on the edge of the grid (5 of 6). Second extension
    # (Nov 20 evening): nontwt-lora 1e-2, tok 3e-2 (optimum still on the edge).
    "twt-lora":         [3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    "twt-full":         [3e-6, 1e-5, 3e-5, 1e-4, 3e-4],
    "nontwt-full":      [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
    "nontwt-full-tok0": [3e-5, 1e-4, 3e-4, 1e-3],      # Nov 20: TWT computation untouched
    "nontwt-lora":      [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "enc-lora":         [3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    "tok":              [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
}
PILOT_SEED = 10
SELECT = "MeanIoU_per_prompt"
METRICS = [  # (name, higher_is_better)
    ("mIoU", True), ("Dice", True), ("MeanIoU_per_prompt", True),
    ("VarIoU_per_prompt", False), ("RemotePartRecall@0.50", True),
    ("FragmentIoU", True), ("OccludedInstanceF1", True),
]


def raw_config(variant, lr):
    return {"data": "och", "backbone": "sam1", "model_type": "vit_l", "variant": variant,
            "train": {"lr": lr, "total_steps": 5000}, "benchmark": {"tau": 0.35, "N": 100}}


def summary_path(results, exp_name, seed):
    return os.path.join(results, exp_name, f"s{seed}", "eval", f"pr__och-test_s{seed}.summary.json")


def build_jobs(variants, results, gen_dir, dry):
    jobs = []
    for v in variants:
        for lr in GRID[v]:
            raw = raw_config(v, lr)
            name = resolve_config(raw)["exp_name"]
            # The canonical folder may hold re-registered pre-revision checkpoints
            # (twt-lora lr=1e-4): never train into it; use a sibling pilot folder.
            cfg_path = os.path.join(results, name, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    if "legacy" in json.load(f):
                        raw["exp_name"] = name + "__pilot"
                        name = raw["exp_name"]
            ypath = os.path.join(gen_dir, f"{name}.yaml")
            if not dry:
                os.makedirs(gen_dir, exist_ok=True)
                with open(ypath, "w") as f:
                    yaml.safe_dump(raw, f, sort_keys=False)
            jobs.append((v, lr, name, ypath))
    return jobs


def run_jobs(jobs, gpus, results, logdir):
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    lock = threading.Lock()

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        while True:
            try:
                v, lr, name, ypath = q.get_nowait()
            except queue.Empty:
                return
            sdir = os.path.join(results, name, f"s{PILOT_SEED}")
            ckpt = os.path.join(sdir, "ckpt", "final.pth")
            log = os.path.join(logdir, f"{name}__s{PILOT_SEED}.log")
            status = "ok"
            with open(log, "a") as lf:
                if not os.path.exists(ckpt):
                    rc = subprocess.call([sys.executable, os.path.join(REPO, "scripts", "train.py"),
                                          "--config", ypath, "--seed", str(PILOT_SEED)],
                                         stdout=lf, stderr=subprocess.STDOUT, env=env)
                    if rc != 0:
                        status = "TRAIN_FAILED"
                if status == "ok" and not os.path.exists(summary_path(results, name, PILOT_SEED)):
                    rc = subprocess.call([sys.executable, os.path.join(REPO, "scripts", "eval_pr.py"),
                                          "--exp_dir", os.path.join(results, name),
                                          "--train_seed", str(PILOT_SEED), "--test_seed", str(PILOT_SEED)],
                                         stdout=lf, stderr=subprocess.STDOUT, env=env)
                    if rc != 0:
                        status = "EVAL_FAILED"
            with lock:
                print(f"  [gpu {gpu}] {v} lr={fmt_lr(lr)}: {status}")

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def report(jobs, results, out_dir, tie_eps):
    rows = []
    for v, lr, name, _ in jobs:
        p = summary_path(results, name, PILOT_SEED)
        if not os.path.exists(p):
            rows.append({"variant": v, "lr": lr, "exp_name": name, "missing": True})
            continue
        with open(p) as f:
            agg = json.load(f)["aggregate"]
        rows.append({"variant": v, "lr": lr, "exp_name": name, "missing": False,
                     **{m: agg[m] for m, _ in METRICS}})

    selected = {}
    for v in dict.fromkeys(r["variant"] for r in rows):
        cand = [r for r in rows if r["variant"] == v and not r["missing"]]
        if len(cand) < len(GRID[v]):
            selected[v] = None   # incomplete sweep: no selection
            continue
        best = max(r[SELECT] for r in cand)
        tied = [r for r in cand if best - r[SELECT] < tie_eps]
        if len(tied) > 1:
            best_of = {m: (max if hi else min)(c[m] for c in tied) for m, hi in METRICS}
            n_won = [sum(r[m] == best_of[m] for m, _ in METRICS) for r in tied]
            tied = [r for _, r in sorted(zip(n_won, tied), key=lambda t: -t[0])]
        selected[v] = tied[0]["lr"]

    os.makedirs(out_dir, exist_ok=True)
    lines = ["# LR sweep — pilot seed 10 (train s10, test s10)", "",
             f"Selection: max {SELECT}; ties (< {tie_eps:g}) broken by #metrics won.", "",
             "| variant | lr | " + " | ".join(m for m, _ in METRICS) + " | selected |",
             "|---|---|" + "---|" * len(METRICS) + "---|"]
    for r in rows:
        if r["missing"]:
            lines.append(f"| {r['variant']} | {fmt_lr(r['lr'])} | " + " | ".join("—" for _ in METRICS) + " | |")
            continue
        sel = "**✓**" if selected.get(r["variant"]) == r["lr"] else ""
        lines.append(f"| {r['variant']} | {fmt_lr(r['lr'])} | "
                     + " | ".join(f"{r[m]:.5f}" for m, _ in METRICS) + f" | {sel} |")
    lines += ["", "Selected LR per variant:"]
    for v, lr in selected.items():
        lines.append(f"- {v}: {fmt_lr(lr)}" if lr is not None else f"- {v}: incomplete sweep")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "lr_sweep_s10.md"), "w") as f:
        f.write(md)
    with open(os.path.join(out_dir, "lr_sweep_s10.json"), "w") as f:
        json.dump({"rows": rows, "selected": selected, "select_metric": SELECT,
                   "tie_eps": tie_eps}, f, indent=2)
    print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--variants", nargs="*", default=list(GRID), choices=list(GRID))
    ap.add_argument("--tie_eps", type=float, default=1e-3,
                    help="MeanIoU/Prompt differences below this count as ties")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()

    R = paths.get("results_root")
    results = os.path.join(R, "results")
    for need in (f"{R}/benchmarks/och/test/test_s{PILOT_SEED}.json",
                 f"{R}/benchmarks/och/train/tau035/s{PILOT_SEED}/fewshot_N100.json"):
        if not os.path.exists(need):
            sys.exit(f"missing {need} — run: python scripts/batch0_och.py --seeds {PILOT_SEED}")

    gen_dir = os.path.join(REPO, "configs", "generated")
    jobs = build_jobs(args.variants, results, gen_dir, args.dry_run)
    if args.dry_run:
        for v, lr, name, _ in jobs:
            print(f"  {v:12s} lr={fmt_lr(lr):<6s} {name}")
        print(f"{len(jobs)} runs (train s{PILOT_SEED} + eval s{PILOT_SEED})")
        return
    if not args.report_only:
        logdir = os.path.join(R, "analysis", "R1", "logs")
        os.makedirs(logdir, exist_ok=True)
        print(f"{len(jobs)} runs on GPUs {args.gpus}")
        run_jobs(jobs, args.gpus.split(","), results, logdir)
    # the table always covers the whole grid, whatever subset was run now
    all_jobs = build_jobs(list(GRID), results, gen_dir, True)
    report(all_jobs, results, os.path.join(R, "analysis", "R1"), args.tie_eps)


if __name__ == "__main__":
    main()