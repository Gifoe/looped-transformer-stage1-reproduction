"""Verify matched optimizer schedules and summarize the trajectory of the common group objective."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

import minimal
from train_advantage import load_fit, schedule


def reading(path):
    return json.loads(path.read_text())


def load_factors(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)["state"]
    factors = [payload[key].flatten().double() for key in ("out.weight", "encode.weight")]
    if sum(x.numel() for x in factors) != 4096:
        raise ValueError(f"Invalid parameter count {path}")
    return factors


def analyze(root, data):
    input_data = load_fit(data)
    directions = {}
    results = {}
    events = {}
    for path in sorted(root.glob("*/TRAIN.json")):
        train, lock = reading(path), reading(path.with_name("TRAIN_LOCK.json"))
        run = path.parent.name
        source = path.with_name("controller.pt")
        if train["checkpoint_sha256"] != minimal.sha(source):
            raise RuntimeError(f"Fit checkpoint differs {source}")
        expected = {}
        for step, (k, rows) in enumerate(schedule(input_data, lock["batch"], lock["seed"]), 1):
            if step > lock["planned_steps"]:
                break
            ds = rows[0]["dataset"]
            if run == "ranking" and ds not in ("arc_challenge", "csqa"):
                continue
            expected[step] = (k, ds, [row["sample_id"] for row in rows])
        actual = {r["step"]: (r["depth"], r["dataset"], r["ids"]) for r in train["logs"]}
        if actual != expected or len(actual) != train["number_updates"]:
            raise RuntimeError(f"Optimizer schedule differs {run}")
        events[run] = actual
        grouped = defaultdict(list)
        for item in train["logs"]:
            for group, value in item["group_gain"].items():
                grouped[f"{group}/k{item['depth']}"].append(value)
        gains = {key: dict(n=len(v), initial_mean=float(np.mean(v[:12])), final_mean=float(np.mean(v[-12:])))
                 for key, v in grouped.items()}
        factors = load_factors(source)
        product = float(factors[0].norm() * factors[1].norm())
        directions[run] = [x / x.norm().clamp_min(1e-12) for x in factors]
        results[run] = dict(updates=len(actual), unique_ids=len({value for entry in actual.values() for value in entry[2]}),
                           domain_updates=dict(Counter(group for _, group, _ in actual.values())),
                           initial_sha256=lock["initial_sha256"], controller_sha256=train["checkpoint_sha256"],
                           policy_prefixes=sum(row.get("off_policy_prefixes", 0) for row in train["logs"]),
                           product_norm=product, seconds=train["seconds"], cuda_peak=train["cuda_peak"],
                           padded_token_loops=train["padded_token_loops"], group_gains=gains,
                           group_dual_weights=train["group_dual_weights"] if lock["objective"] == "worst_advantage" else None,
                           end_softmin_diagnostic=train["group_dual_weights"] if lock["objective"] == "mean_advantage" else None)
    score = {d: {e: float((v[0]@w[0])*(v[1]@w[1])) for e, w in directions.items()} for d, v in directions.items()}
    matched = "teacher_resume" in events and "on_policy_resume" in events
    if matched:
        same = events["teacher_resume"] == events["on_policy_resume"]
        same_init = results["teacher_resume"]["initial_sha256"] == results["on_policy_resume"]["initial_sha256"]
        if not same or not same_init or results["on_policy_resume"]["policy_prefixes"] == 0:
            raise RuntimeError("The paired continuation arm did not match its experimental design")
    return dict(runs=results, rank_one_update_cosines=score, continuation_matched=matched)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("fits", "data", "out"):
        p.add_argument(f"--{key}", type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.fits, a.data)
    minimal.dump(a.out, result)
    print({name: (row["updates"], row["unique_ids"], row["policy_prefixes"]) for name, row in result["runs"].items()},
          "continuation_matched", result["continuation_matched"], flush=True)


if __name__ == "__main__":
    main()
