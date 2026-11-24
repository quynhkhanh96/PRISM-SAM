"""
R2 condition 0 — `nontwt-full-tok0` must leave SAM1's TWT computation untouched.

Claim being checked (framing_after_R1R5.md, "Conditions to lock" #2):
tok0 freezes the image encoder, the prompt encoder, the TWT weights and every
TWT input (iou_token, mask tokens), so everything up to and including the TWT
-- in particular every attention map -- must be identical to zero-shot SAM1.
Only the readout (output_upscaling + output_hypernetworks_mlps) may differ.

Two checks per train seed (train seed s is evaluated on test file s, as in PR):

  1. static   every parameter/buffer of the tok0 model equals SAM1's, except
              mask_decoder.output_upscaling.* and
              mask_decoder.output_hypernetworks_mlps.*  (which MUST differ,
              otherwise the checkpoint was not loaded).
  2. runtime  on the PR test prompts (same instances, same K=5 points, same
              predictor call as eval_pr.py), compare tok0 vs SAM1:
                enc.features                       (once per image)
                twt.in.{image_embedding,image_pe,point_embedding}
                attn.<module>                      every Attention in the TWT
                twt.out.{queries,keys}
              and, as a positive control that MUST differ somewhere,
                readout.low_res_logits

Verdict: PASS_EXACT if every checked tensor is bit-identical, PASS_TOL if the
largest |diff| <= --atol, otherwise FAIL. The control must differ on at least
one prompt (else CONTROL_FAIL: the readout checkpoint is probably not loaded).

    python scripts/r2_sanity_tok0.py                       # seeds 0-4, all instances
    python scripts/r2_sanity_tok0.py --seeds 0 --limit 50  # quick look

Outputs (<results_root>/analysis/R2/sanity_tok0/):
    s{s}.static.json       per-key result of check 1
    s{s}.runtime.parquet   one row per (instance, prompt, tensor):
                           instance_id, prompt_id (-1 = per-image tensor),
                           tensor, equal, max_abs_diff, numel
    summary.json           verdict per seed + overall
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from prismsam import paths  # noqa: E402
from prismsam.config import resolve_config, test_path  # noqa: E402
from prismsam.datasets import och  # noqa: E402
from prismsam.models import build_model, load_finetuned  # noqa: E402
from prismsam.utils import read_image  # noqa: E402
from segment_anything.modeling.transformer import Attention  # noqa: E402
from segment_anything.predictor import SamPredictor  # noqa: E402

VARIANT = "nontwt-full-tok0"
ALLOWED_DIFF_PREFIXES = ("mask_decoder.output_upscaling.",
                         "mask_decoder.output_hypernetworks_mlps.")
CONTROL = "readout.low_res_logits"


def raw_config(variant, lr):   # identical to sweep_lr.raw_config / run_official.raw_config
    return {"data": "och", "backbone": "sam1", "model_type": "vit_l", "variant": variant,
            "train": {"lr": lr, "total_steps": 5000}, "benchmark": {"tau": 0.35, "N": 100}}


def default_exp_dir(results_root):
    sweep = os.path.join(results_root, "analysis", "R1", "lr_sweep_s10.json")
    with open(sweep) as f:
        lr = json.load(f)["selected"][VARIANT]
    if lr is None:
        sys.exit(f"{VARIANT}: no selected LR in {sweep}")
    return os.path.join(results_root, "results", resolve_config(raw_config(VARIANT, lr))["exp_name"])


# ------------------------------------------------------------------ capture

class TWTProbe:
    """Records TWT inputs/outputs and every Attention.last_attn of one model."""

    def __init__(self, model):
        self.twt = model.mask_decoder.transformer
        self.attn = [(n, m) for n, m in self.twt.named_modules() if isinstance(m, Attention)]
        if not self.attn:
            raise RuntimeError("no Attention modules found in the TWT")
        self.data = {}
        self.twt.register_forward_pre_hook(self._pre)
        self.twt.register_forward_hook(self._post)

    def _pre(self, module, args):
        img, pe, pts = args
        self.data["twt.in.image_embedding"] = img.detach()
        self.data["twt.in.image_pe"] = pe.detach()
        self.data["twt.in.point_embedding"] = pts.detach()

    def _post(self, module, args, out):
        q, k = out
        self.data["twt.out.queries"] = q.detach()
        self.data["twt.out.keys"] = k.detach()

    def reset(self):
        self.data = {}
        for _, m in self.attn:
            m.last_attn = None

    def collect(self):
        out = dict(self.data)
        for n, m in self.attn:
            if m.last_attn is None:
                raise RuntimeError(f"attention {n} did not run")
            out[f"attn.{n}"] = m.last_attn
        return out


def compare(a, b):
    if a.shape != b.shape:
        raise RuntimeError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    equal = bool(torch.equal(a, b))
    diff = 0.0 if equal else float((a.float() - b.float()).abs().max().item())
    return equal, diff, int(a.numel())


# ------------------------------------------------------------------ checks

def static_check(m_tok0, m_sam):
    sd0, sd1 = m_tok0.state_dict(), m_sam.state_dict()
    if set(sd0) != set(sd1):
        raise RuntimeError(f"state_dict keys differ: {sorted(set(sd0) ^ set(sd1))[:10]}")
    rows = []
    for k in sorted(sd0):
        eq, diff, n = compare(sd0[k].cpu(), sd1[k].cpu())
        rows.append({"key": k, "allowed_to_differ": k.startswith(ALLOWED_DIFF_PREFIXES),
                     "equal": eq, "max_abs_diff": diff, "numel": n})
    bad = [r["key"] for r in rows if not r["equal"] and not r["allowed_to_differ"]]
    readout_changed = [r["key"] for r in rows if not r["equal"] and r["allowed_to_differ"]]
    ok = not bad and bool(readout_changed)
    return ok, {"ok": ok, "unexpected_diffs": bad, "n_readout_changed": len(readout_changed),
                "n_readout_keys": sum(r["allowed_to_differ"] for r in rows), "rows": rows}


def runtime_check(pred_tok0, pred_sam, probe_tok0, probe_sam, records, index, root):
    rows = []
    with torch.no_grad():
        for rec in tqdm(records):
            iid = index.instance_id(rec)
            img = read_image(och.image_path(root, rec))
            pred_tok0.set_image(img)
            pred_sam.set_image(img)
            eq, d, n = compare(pred_tok0.features, pred_sam.features)
            rows.append({"instance_id": iid, "prompt_id": -1, "tensor": "enc.features",
                         "equal": eq, "max_abs_diff": d, "numel": n})
            for pid, point in enumerate(rec["points"]):
                outs = []
                for pred, probe in ((pred_tok0, probe_tok0), (pred_sam, probe_sam)):
                    probe.reset()
                    # identical call to eval_pr.py (multimask_output=True default)
                    _, _, low = pred.predict(point_coords=np.array([point]),
                                             point_labels=np.array([1]))
                    t = probe.collect()
                    t[CONTROL] = torch.from_numpy(low)
                    outs.append(t)
                a, b = outs
                for name in a:
                    eq, d, n = compare(a[name], b[name])
                    rows.append({"instance_id": iid, "prompt_id": pid, "tensor": name,
                                 "equal": eq, "max_abs_diff": d, "numel": n})
    return pd.DataFrame(rows)


def verdict(df, atol):
    chk = df[df["tensor"] != CONTROL]
    ctl = df[df["tensor"] == CONTROL]
    per_tensor = (chk.groupby("tensor")
                  .agg(n=("equal", "size"), n_not_equal=("equal", lambda s: int((~s).sum())),
                       max_abs_diff=("max_abs_diff", "max"))
                  .reset_index().to_dict(orient="records"))
    max_diff = float(chk["max_abs_diff"].max())
    if bool(chk["equal"].all()):
        v = "PASS_EXACT"
    elif max_diff <= atol:
        v = "PASS_TOL"
    else:
        v = "FAIL"
    control_frac = float((~ctl["equal"]).mean()) if len(ctl) else 0.0
    if control_frac == 0.0:
        v = "CONTROL_FAIL"
    return v, {"verdict": v, "max_abs_diff": max_diff, "atol": atol,
               "control_fraction_differs": control_frac,
               "control_median_max_abs_diff": float(ctl["max_abs_diff"].median()) if len(ctl) else None,
               "n_instances": int(df["instance_id"].nunique()), "per_tensor": per_tensor}


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default=None, help="default: tok0 folder at the swept LR")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--limit", type=int, default=None, help="first N test instances only")
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    R = paths.get("results_root")
    exp_dir = args.exp_dir or default_exp_dir(R)
    out_dir = args.out_dir or os.path.join(R, "analysis", "R2", "sanity_tok0")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json")) as f:
        cfg = json.load(f)
    if cfg["adapt"].get("site") != "nontwt" or cfg["adapt"].get("tok"):
        sys.exit(f"{exp_dir} is not a nontwt tok0 experiment: adapt={cfg['adapt']}")
    sam_ckpt = paths.sam1_checkpoint(cfg.get("model_type", "vit_l"))
    zs_cfg = {"backbone": "sam1", "model_type": cfg.get("model_type", "vit_l"), "adapt": None}

    root = paths.get("ochuman_root")
    index = och.OchumanIndex.from_root(root)

    m_sam = build_model(zs_cfg, sam_checkpoint=sam_ckpt).to(args.device).eval()
    pred_sam = SamPredictor(m_sam)
    probe_sam = TWTProbe(m_sam)

    summary = {"exp_dir": exp_dir, "limit": args.limit, "seeds": {}}
    for s in [int(x) for x in args.seeds.split(",")]:
        ckpt = os.path.join(exp_dir, f"s{s}", "ckpt", "final.pth")
        m_tok0 = build_model(cfg, sam_checkpoint=sam_ckpt)
        load_finetuned(m_tok0, ckpt)
        m_tok0.to(args.device).eval()

        ok_static, st = static_check(m_tok0, m_sam)
        with open(os.path.join(out_dir, f"s{s}.static.json"), "w") as f:
            json.dump(st, f, indent=1)
        print(f"[s{s}] static: {'OK' if ok_static else 'FAIL'} "
              f"(readout keys changed {st['n_readout_changed']}/{st['n_readout_keys']}, "
              f"unexpected diffs {st['unexpected_diffs'][:5]})")

        records = och.load_records(test_path(och.DATASET, s))
        if args.limit:
            records = records[:args.limit]
        df = runtime_check(SamPredictor(m_tok0), pred_sam, TWTProbe(m_tok0), probe_sam,
                           records, index, root)
        df.insert(0, "train_seed", s)
        df.to_parquet(os.path.join(out_dir, f"s{s}.runtime.parquet"), index=False)
        v, info = verdict(df, args.atol)
        info["static_ok"] = ok_static
        summary["seeds"][s] = info
        print(f"[s{s}] runtime: {v}  max|diff|={info['max_abs_diff']:.3g}  "
              f"control differs on {100 * info['control_fraction_differs']:.1f}% of prompts")
        del m_tok0
        torch.cuda.empty_cache()

    verdicts = [x["verdict"] for x in summary["seeds"].values()]
    statics = [x["static_ok"] for x in summary["seeds"].values()]
    summary["overall"] = ("PASS" if all(statics) and all(v.startswith("PASS") for v in verdicts)
                          else "FAIL")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"overall: {summary['overall']}  -> {out_dir}/summary.json")


if __name__ == "__main__":
    main()