"""
R3 mining: rpr_from_record(scan_record(gt, preds), tau, area) must equal the
verbatim remote_part_recall(gt, [preds], tau, min_component_area=area) for any
tau / area, on random masks with partial overlaps and tiny fragments. CPU only.

    python tests/test_mining.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from prismsam import mining  # noqa: E402
from prismsam.config import resolve_config, bench_tag  # noqa: E402


def rand_mask(rng, H=96, W=128, n=3):
    m = np.zeros((H, W), np.uint8)
    for _ in range(rng.integers(1, n + 1)):
        h, w = rng.integers(3, 30, size=2)
        y, x = rng.integers(0, H - h), rng.integers(0, W - w)
        m[y:y + h, x:x + w] = 1
    return m


def perturb(rng, gt):
    p = gt.copy()
    ys, xs = np.nonzero(gt)
    if len(ys) and rng.random() < 0.8:       # erase a random box -> partial coverage
        y, x = ys[rng.integers(len(ys))], xs[rng.integers(len(xs))]
        p[max(0, y - rng.integers(2, 20)):y + rng.integers(2, 20), max(0, x - rng.integers(2, 25)):x + rng.integers(2, 25)] = 0
    return np.maximum(p, rand_mask(rng, n=2) * (rng.random() < 0.5))


def main():
    rng = np.random.default_rng(0)
    taus = [0.0, 0.1, 0.25, 0.35, 0.5, 0.75, 0.9, 1.0]
    n_cmp, n_diff_crit = 0, 0
    for _ in range(400):
        gt = rand_mask(rng, n=4)
        preds = [perturb(rng, gt) for _ in range(4)]
        rec = mining.scan_record(gt, preds)
        verdicts = set()
        for area in (0, 150):
            for tau in taus:
                ref = mining.M.remote_part_recall(gt, [preds], tau=tau, min_component_area=area)
                got = mining.rpr_from_record(rec, tau, area)
                assert abs(ref - got) < 1e-12, (tau, area, ref, got)
                n_cmp += 1
                verdicts.add(got < 1.0)
        n_diff_crit += len(verdicts) > 1
    print(f"rpr_from_record == remote_part_recall on {n_cmp} (instance, tau, area) cases; "
          f"{n_diff_crit}/400 instances change verdict across criteria")

    # naming: existing experiments keep their names and configs; new ones get the area tag
    base = {"data": "och", "variant": "twt-lora", "train": {"lr": 1e-4, "total_steps": 5000}}
    old = resolve_config({**base, "benchmark": {"tau": 0.35, "N": 100}})
    assert old["exp_name"] == "och__twt-lora-scfull-qkvo-r8a8__tok1__lr1e-4__tau035-N100__steps5000"
    assert "mine_area" not in old["benchmark"]
    a0 = resolve_config({**base, "benchmark": {"tau": 0.5, "N": 100, "mine_area": 0}})
    assert a0["exp_name"].endswith("tau050-N100__steps5000") and "mine_area" not in a0["benchmark"]
    a150 = resolve_config({**base, "benchmark": {"tau": 0.25, "N": 100, "mine_area": 150}})
    assert a150["exp_name"].endswith("tau025a150-N100__steps5000"), a150["exp_name"]
    assert bench_tag(a150["benchmark"]) == "025a150"
    print("naming OK:", a0["exp_name"], "|", a150["exp_name"])


if __name__ == "__main__":
    main()
