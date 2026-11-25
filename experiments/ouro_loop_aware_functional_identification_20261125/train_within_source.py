"""Use full equal-exposure arms to test within-source functional weighting."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import rankdata

import minimal
import train_identification as shared


def transition(row):
    return (int(row["native_direct"]), int(row["native_loop"]))


def weights(rows, probes, policy):
    strata = defaultdict(list)
    for k, row in rows.items():
        cell = (*transition(probes[k]), int(np.log2(max(1, row["input_tokens"]))))
        strata[cell].append(k)
    out = {}
    for cohort in strata.values():
        slopes = [probes[k]["slopes"]["t1"] for k in cohort]
        effect = [min(x["excess"], x["loop"]) for x in slopes]
        rank = rankdata(effect, method="average")
        for key, value in zip(cohort, rank, strict=True):
            centered = float(value / (len(cohort) + 1) - .5)
            out[key] = 1.0 + {"uniform": 0, "high": 1, "low": -1}[policy] * centered
    return out


def run(a):
    policy = a.policy
    lock = {"data_value_policy": policy, "training_source": a.source, "same_full_fitting_folds": [0, 1, 2],
            "matching": "same native K3 to K4 correctness cell and logarithmic input length",
            "selector_sha256": minimal.sha(Path(__file__))}
    a.out.mkdir(parents=True, exist_ok=True)
    receipt = a.out / "WITHIN_SOURCE.json"
    if receipt.exists():
        if json.loads(receipt.read_text(encoding="utf-8")) != lock:
            raise ValueError("Within-source weighting protocol changed")
    else:
        minimal.dump(receipt, lock)
    a.include = [a.source]

    def apply(rows, probes, _forecast):
        return {"loop": weights(rows, probes, policy)}

    with patch.object(shared, "selection", apply):
        shared.fit(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "previous", "model", "data", "moments", "forecast", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--policy", choices=("uniform", "high", "low"), required=True)
    p.add_argument("--seed", type=int, default=261125137)
    p.set_defaults(arm="loop", limit=None, max_work=None, initialize=None)
    run(p.parse_args())
