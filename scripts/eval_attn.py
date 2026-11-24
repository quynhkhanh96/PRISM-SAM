"""
R2 — point-token attention on OCHuman PR prompts, per instance x prompt.

Same model loading, test file, prompts and predictor call as eval_pr.py, so the
rows pair with the PR files on (instance_id, prompt_id) and the masks (hence
RPR) are the same ones the PR benchmark scores.

    python scripts/eval_attn.py --exp_dir <results>/<exp> --train_seed 0 --test_seed 0
    python scripts/eval_attn.py --exp_dir <results>/zs__sam1 --test_seed 0

Writes <exp>/s{seed}/eval/:
  attn__och-test_s{t}.prompts.parquet    one row per (instance, prompt):
      enrich_remote      A_enrich_remote, Section 5 definition (final token->image
                         attention, head-averaged; prismsam.attention Section 1);
                         NaN if the prompt has no remote component
      n_remote, n_valid, click_on_valid_component
      frac__*, mass__{L0,L1,final}__h{0..7}__{remote,clicked}, ent__*   raw readouts
  attn__och-test_s{t}.instances.parquet  one row per instance:
      mean_enrichment    mean of enrich_remote over prompts (Section 5 definition)
      n_remote           max over prompts (Section 5 definition)
      num_components     count_gt_components (Section 5 definition)
      rpr, n_gt_components, is_occluded, mean_iou_pp, var_iou_pp   (pr_instance_values)
  attn__och-test_s{t}.summary.json       meta + one-seed summary (old S4.2 statistics)

Raw columns let any other layer / head aggregation / region be recomputed later
(prismsam.attention.enrichment_from_raw) without rerunning inference.
"""

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402
from tqdm import tqdm  # noqa: E402

from prismsam import attention as A  # noqa: E402
from prismsam import paths  # noqa: E402
from prismsam.config import test_path  # noqa: E402
from prismsam.datasets import och  # noqa: E402
from prismsam.metrics import pr_instance_values  # noqa: E402
from prismsam.models import build_model, load_finetuned  # noqa: E402
from prismsam.utils import read_image  # noqa: E402
from segment_anything.predictor import SamPredictor  # noqa: E402

TAU_REMOTE = 0.5            # = eval_pr.py
MIN_COMPONENT_AREA = 150    # = eval_pr.py = attention.min_component_area
OCCLUSION_MIN_COMPONENTS = 2
assert MIN_COMPONENT_AREA == A.min_component_area


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def one_seed_summary(inst):
    """Old S4.2 statistics for one seed (causal_analysis.summarize)."""
    valid = inst[~inst["mean_enrichment"].isna()]
    E, R = valid["mean_enrichment"].to_numpy(), valid["rpr"].to_numpy()
    rho, p = stats.spearmanr(E, R) if len(E) > 2 else (float("nan"), float("nan"))
    return {"n_enrichment": int(len(E)), "mean_E": float(E.mean()), "median_E": float(np.median(E)),
            "std_E": float(E.std()), "mean_RPR": float(inst["rpr"].mean()),
            "mean_RPR_occ": float(inst.loc[inst["n_remote"] >= 1, "rpr"].mean()),
            "rho": float(rho), "pval": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--train_seed", type=int, default=-1)
    ap.add_argument("--test_seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N instances only")
    args = ap.parse_args()

    with open(os.path.join(args.exp_dir, "config.json")) as f:
        cfg = json.load(f)
    method = cfg.get("exp_name", os.path.basename(os.path.normpath(args.exp_dir)))
    zero_shot = cfg.get("adapt") is None
    if zero_shot:
        args.train_seed, ckpt = -1, None
        sdir = os.path.join(args.exp_dir, f"s{args.test_seed}")
    else:
        if args.train_seed < 0:
            sys.exit("--train_seed is required for a fine-tuned experiment")
        sdir = os.path.join(args.exp_dir, f"s{args.train_seed}")
        ckpt = os.path.join(sdir, "ckpt", "final.pth")
    out_dir = os.path.join(sdir, "eval")
    stem = os.path.join(out_dir, f"attn__och-test_s{args.test_seed}")
    test_file = test_path(och.DATASET, args.test_seed)

    model = build_model(cfg, sam_checkpoint=paths.sam1_checkpoint(cfg.get("model_type", "vit_l")))
    ckpt_kind = load_finetuned(model, ckpt) if ckpt else "zero-shot"
    model.to(args.device).eval()
    predictor = SamPredictor(model)
    twt = model.mask_decoder.transformer
    final_attn = twt.final_attn_token_to_image
    print(f"[eval_attn] {method} ckpt={ckpt} ({ckpt_kind}) test={test_file}")

    root = paths.get("ochuman_root")
    index = och.OchumanIndex.from_root(root)
    records = och.load_records(test_file)
    if args.limit:
        records = records[:args.limit]

    keys = {"method": method, "train_seed": args.train_seed, "test_seed": args.test_seed,
            "dataset": och.DATASET, "category": och.CATEGORY}
    prompt_rows, inst_rows, seen = [], [], set()
    with torch.no_grad():
        for rec in tqdm(records):
            iid = index.instance_id(rec)
            if iid in seen:
                raise ValueError(f"duplicate instance_id {iid}")
            seen.add(iid)
            predictor.set_image(read_image(och.image_path(root, rec)))
            img_h, img_w = rec["image_size"]
            gt = och.gt_mask(rec)
            preds_pp, enr, n_rem = [], [], []
            for pid, point in enumerate(rec["points"]):
                for _, path in A.TOKEN_TO_IMAGE:
                    twt.get_submodule(path).last_attn = None
                masks, _, _ = predictor.predict(point_coords=np.array([point]),
                                                point_labels=np.array([1]))   # = eval_pr.py
                preds_pp.append(masks)

                # --- Section 5 definition (verbatim path, attention_alignment_analysis) ---
                attn_map_64 = A.get_point_attn_map(final_attn)
                if attn_map_64 is None:
                    raise RuntimeError("final attention not cached")
                remote_64, n_remote, _ = A.get_remote_component_masks(gt, point, img_h, img_w)
                e = (A.compute_remote_enrichment(attn_map_64, remote_64)
                     if remote_64 is not None else None)
                enr.append(e)
                n_rem.append(n_remote)

                # --- raw readouts (new) ---
                regions = A.region_masks_64(gt, point, img_h, img_w)
                if regions["n_remote"] != n_remote or (
                        remote_64 is not None and not np.array_equal(regions["remote_64"], remote_64)):
                    raise RuntimeError(f"region_masks_64 disagrees with Section 1 on {iid} prompt {pid}")
                row = dict(keys, instance_id=iid, prompt_id=pid,
                           enrich_remote=float(e) if e is not None else float("nan"),
                           n_remote=int(n_remote), n_valid=int(regions["n_valid"]),
                           click_on_valid_component=bool(regions["click_on_valid_component"]))
                row.update(A.raw_attention_row(twt, regions))
                prompt_rows.append(row)

            valid = [x for x in enr if x is not None]
            inst, _ = pr_instance_values(gt, preds_pp, tau_remote=TAU_REMOTE,
                                         min_component_area=MIN_COMPONENT_AREA,
                                         occlusion_min_components=OCCLUSION_MIN_COMPONENTS)
            inst_rows.append(dict(
                keys, instance_id=iid,
                mean_enrichment=float(np.mean(valid)) if valid else float("nan"),
                n_remote=int(max(n_rem)) if n_rem else 0,
                num_components=int(A.count_gt_components(gt)),
                rpr=inst["rpr"], n_gt_components=inst["n_gt_components"],
                is_occluded=inst["is_occluded"], mean_iou_pp=inst["mean_iou_pp"],
                var_iou_pp=inst["var_iou_pp"]))

    os.makedirs(out_dir, exist_ok=True)
    inst_df = pd.DataFrame(inst_rows)
    pd.DataFrame(prompt_rows).to_parquet(stem + ".prompts.parquet", index=False)
    inst_df.to_parquet(stem + ".instances.parquet", index=False)
    summ = one_seed_summary(inst_df)
    meta = {"config": cfg, "ckpt": ckpt, "ckpt_kind": ckpt_kind,
            "test_file": os.path.abspath(test_file), "git": git_hash(), "limit": args.limit,
            "tau_remote": TAU_REMOTE, "min_component_area": MIN_COMPONENT_AREA,
            "point_token_idx": A.POINT_TOKEN_IDX, "layers": [p for _, p in A.TOKEN_TO_IMAGE],
            "device_name": (torch.cuda.get_device_name() if args.device.startswith("cuda")
                            and torch.cuda.is_available() else args.device),
            "torch": torch.__version__}
    with open(stem + ".summary.json", "w") as f:
        json.dump({"keys": keys, "summary": summ, "meta": meta}, f, indent=2)
    print(f"[eval_attn] saved {stem}.*")
    for k, v in summ.items():
        print(f"  {k}: {v:.5g}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()