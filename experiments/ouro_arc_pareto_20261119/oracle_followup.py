"""Evaluate the frozen Q/K/V intervention on untouched validation IDs."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from analyze_arc import gate_oracle, load, paired, summary, vectors
from diagnostics import oracle
from run_arc import dump, load_runtime, read_data, write_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    freeze = json.loads((args.out / "freeze.json").read_text())
    path = args.out / "oracle_predictions.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == freeze["oracle_scores_sha256"]
    pilot = {row["id"] for row in load(path)}
    remaining = sorted((row for row in read_data(args.data, "validation") if row["id"] not in pilot),
                       key=lambda row: row["id"])
    assert (len(pilot), len(remaining)) == (128, 171)
    model, tokenizer, _ = load_runtime(args.model)
    trials, execution = [], []
    with TemporaryDirectory() as scratch:
        folder = Path(scratch)
        for start in range(0, len(remaining), 128):
            oracle(model, tokenizer, remaining[start:start+128], folder, 16)
            trials.extend(load(folder / "oracle_predictions.jsonl"))
            execution.append(json.loads((folder / "oracle_execution.json").read_text()))
    assert {t["id"] for t in trials} == {r["id"] for r in remaining}
    write_predictions(args.out / "oracle_followup_predictions.jsonl", trials)
    holdout = summary(trials)
    validation = {row["id"]: row for row in load(args.out / "validation_predictions.jsonl")}
    validation = [validation[row["id"]] for row in trials]
    gold = np.array([row["gold"] for row in trials])
    comparison = freeze["strong_baseline_by_depth"]["2"]
    base = vectors(validation, comparison, "acc_norm")[0] == gold
    report = {}
    for field in ("q", "k", "v"):
        trial = vectors(trials, f"{field}_teacher", "acc_norm")[0] == gold
        report[field] = {
            "frozen_k2_reference": comparison, "reference_correct": int(base.sum()),
            **paired(base, trial),
        }
    dump(args.out / "oracle_followup_metrics.json", dict(
        metrics=holdout, same_pilot_gate=gate_oracle(holdout),
        vs_frozen_k2_baseline=report, execution=execution))


if __name__ == "__main__":
    main()
