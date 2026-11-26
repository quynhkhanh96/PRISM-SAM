"""
R3 (and R9): full-pool hard-case scan of the OCHuman train pool, one record per
(seed, instance), no early stop and no mining criterion applied.

    python scripts/mine_och.py --seeds 0,1,2,3,4            # GPU, resumable
    python scripts/mine_och.py --seeds 0 --limit_images 20  # quick check

Why one scan suffices: in the pre-revision miner (create_benchmark.py) the
random stream is consumed only by the seeded shuffle and by exactly one
np.random.randint per visited instance (sample_point_largest_component_safe,
every branch). Nothing depends on SAM's output, so for a given seed the visiting
order and every click point are the same for any (tau, area, N). The few-shot set
of any criterion is the first N selected instances of this scan
(scripts/build_fewshot_och.py). SAM's output does not depend on which seed asked
for it, so every image is encoded once for all seeds.

Phase 1 (CPU): replay the seeded RNG per seed -> (order, data_idx, point).
Phase 2 (GPU): per image: set_image once, predict every (seed, instance) point
               (multimask, 4 masks, as before), store mining.scan_record.
Output: <results_root>/benchmarks/och/train/scan/
          parts.jsonl                 append-only progress (resume)
          s{s}/scan.jsonl             one line per visited instance, in visiting order
          s{s}/scan_meta.json
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from prismsam import mining, paths  # noqa: E402
from prismsam.datasets import och  # noqa: E402
from prismsam.models import build_sam1  # noqa: E402
from prismsam.utils import read_image  # noqa: E402


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def replay_points(train_data, seed):
    """Exact replay of collect_hard_cases_ochuman's RNG use (no early stop)."""
    random.seed(seed)
    np.random.seed(seed)
    indices = [ind for ind in range(len(train_data))]
    random.shuffle(indices)
    out = []
    for order, index in enumerate(indices):
        rec = train_data[index]
        px, py = mining.sample_point_largest_component_safe(och.gt_mask(rec))
        out.append((order, index, [int(px), int(py)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--model_type", default="vit_l")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default=None, help="default <results_root>/benchmarks/och/train/scan")
    ap.add_argument("--limit_images", type=int, default=None, help="debug: first K images only")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    root = paths.get("ochuman_root")
    out_dir = args.out_dir or os.path.join(paths.get("results_root"), "benchmarks", "och", "train", "scan")
    os.makedirs(out_dir, exist_ok=True)
    train_data = och.train_pool_records(root)
    index = och.OchumanIndex.from_root(root)
    print(f"train pool: {len(train_data)} instances")

    # ------------------------------------------------------------ phase 1
    jobs = {}   # image_id -> list of (seed, order, data_idx, point)
    for s in seeds:
        for order, idx, pt in tqdm(replay_points(train_data, s), desc=f"replay s{s}", leave=False):
            jobs.setdefault(train_data[idx]["image_id"], []).append((s, order, idx, pt))
    images = sorted(jobs)
    if args.limit_images:
        images = images[:args.limit_images]

    # ------------------------------------------------------------ phase 2
    parts = os.path.join(out_dir, "parts.jsonl")
    done = set()
    if os.path.exists(parts):
        with open(parts) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["seed"], r["order"]))
    todo = [im for im in images if any((s, o) not in done for s, o, _, _ in jobs[im])]
    print(f"{len(images)} images, {len(todo)} to process ({len(done)} records already done)")

    if todo:
        from segment_anything.predictor import SamPredictor
        sam = build_sam1(args.model_type, paths.sam1_checkpoint(args.model_type))
        sam.to(device=args.device)
        sam.eval()   # no dropout / batch-norm in SAM1: identical to the pre-revision (train-mode) call
        predictor = SamPredictor(sam)
        t0 = time.time()
        with open(parts, "a") as pf:
            for im in tqdm(todo, desc="images"):
                image = read_image(och.image_path(root, train_data[jobs[im][0][2]]))
                predictor.set_image(image)
                for s, order, idx, pt in jobs[im]:
                    if (s, order) in done:
                        continue
                    rec = train_data[idx]
                    gt = och.gt_mask(rec)
                    with torch.no_grad():
                        masks = predictor.predict(point_coords=np.array([pt]),
                                                  point_labels=np.array([1]))[0]
                    row = {"seed": s, "order": order, "data_idx": idx,
                           "instance_id": index.instance_id(rec), "point": pt,
                           **mining.scan_record(gt, list(masks)),
                           # direct cross-checks against the verbatim metric
                           "rpr_check": {"t035a0": mining.M.remote_part_recall(gt, [masks], tau=0.35),
                                         "t050a150": mining.M.remote_part_recall(
                                             gt, [masks], tau=0.5, min_component_area=150)}}
                    pf.write(json.dumps(row) + "\n")
                pf.flush()
        print(f"phase 2: {time.time() - t0:.0f}s")

    # ------------------------------------------------------------ split per seed
    by_seed = {s: {} for s in seeds}
    with open(parts) as f:
        for line in f:
            r = json.loads(line)
            if r["seed"] in by_seed:
                by_seed[r["seed"]][r["order"]] = r
    expected = {s: sum(1 for im in images for js, *_ in jobs[im] if js == s) for s in seeds}
    meta_common = {
        "git": git_hash(), "model_type": args.model_type,
        "ochuman_json_md5": md5_file(os.path.join(root, "ochuman.json")),
        "train_split_json": och.TRAIN_SPLIT_JSON, "pool_size": len(train_data),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "complete_pool": args.limit_images is None,
    }
    for s in seeds:
        rows = [by_seed[s][o] for o in sorted(by_seed[s])]
        # sanity: the stored components reproduce the verbatim metric
        bad = [r["order"] for r in rows
               if abs(mining.rpr_from_record(r, 0.35, 0) - r["rpr_check"]["t035a0"]) > 1e-12
               or abs(mining.rpr_from_record(r, 0.5, 150) - r["rpr_check"]["t050a150"]) > 1e-12]
        if bad:
            raise RuntimeError(f"s{s}: rpr_from_record disagrees with remote_part_recall at orders {bad[:10]}")
        sd = os.path.join(out_dir, f"s{s}")
        os.makedirs(sd, exist_ok=True)
        with open(os.path.join(sd, "scan.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        meta = {**meta_common, "seed": s, "n_records": len(rows), "n_expected": expected[s],
                "contiguous": sorted(by_seed[s]) == list(range(len(rows)))}
        with open(os.path.join(sd, "scan_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"s{s}: {len(rows)} records (expected {expected[s]}), contiguous={meta['contiguous']}")


if __name__ == "__main__":
    main()
