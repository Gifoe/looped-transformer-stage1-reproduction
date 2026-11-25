"""Refit an established candidate-weighting arm on every eligible training fold."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import minimal
from task_data import hash_order, load
import train_identification as shared
import train_warm_tangent as tangent


def schedule(sources, data, seed, limit):
    if limit is not None:
        raise ValueError("Full-coverage refits require each selected eligible question")
    rows = {}
    for source in sources:
        file = data / f"{source}.jsonl"
        receipt = json.loads(file.with_suffix(".json").read_text(encoding="utf-8"))
        if receipt["output_sha256"] != minimal.sha(file):
            raise ValueError(f"Changed fit population for {source}")
        items = load(file)
        if len(items) != receipt["retained"]:
            raise ValueError(f"Wrong eligible population for {source}")
        for item in items:
            identifier = (source, item["sample_id"])
            if identifier in rows:
                raise ValueError(f"Duplicate training record {identifier}")
            rows[identifier] = item
    order = sorted(rows, key=lambda key: hash_order(str(key), f"loop-identification/training/{seed}"))
    for first in range(0, len(order), 128):
        order[first:first + 128] = sorted(
            order[first:first + 128],
            key=lambda key: (rows[key]["input_tokens"], hash_order(str(key), "loop-bucket-tie")),
        )
    return rows, [rows[key] for key in order]


def run(a):
    audit = a.forecast / "AUDIT.json"
    observed = json.loads(audit.read_text(encoding="utf-8"))
    if observed["forecast_sha256"] != minimal.sha(a.forecast / "FORECAST.json"):
        raise ValueError("Full coverage requires a completed frozen cohort audit")
    if a.arm == "tangent" and a.initialize is None:
        raise ValueError("Tangent weighting requires the historical T1 initialization")
    lock = {"closed_audit_sha256": minimal.sha(audit), "schedule_sha256": minimal.sha(Path(__file__)),
            "policy": a.arm, "same_independent_development_for_selection": True, "folds": [0, 1, 2, 3]}
    a.out.mkdir(exist_ok=True, parents=True)
    path = a.out / "FULL_COVERAGE.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != lock:
            raise ValueError("Full-coverage training protocol changed")
    else:
        minimal.dump(path, lock)
    with patch.object(shared, "frozen_schedule", schedule):
        if a.arm == "tangent":
            a.arm = "loop"
            tangent.main(a)
        else:
            shared.fit(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "previous", "model", "data", "moments", "forecast", "out"):
        p.add_argument(f"--{key}", required=True, type=Path)
    p.add_argument("--initialize", type=Path)
    p.add_argument("--include", nargs="+", required=True)
    p.add_argument("--arm", choices=(*shared.ARMS, "tangent"), required=True)
    p.add_argument("--seed", type=int, default=261125137)
    p.set_defaults(limit=None, max_work=None)
    run(p.parse_args())
