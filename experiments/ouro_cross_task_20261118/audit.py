"""Independently check delivered result populations and scalar consistency."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from analyze import digest, load_lines, write_json


def followup_audit(study, manifest):
    exclusive = set(manifest)
    source_ids = {g["source"] for g in manifest.values() if g["task"] == "prontoqa_derived"}
    inspected = {}
    for experiment, expected_groups, expected_rows, source in (
        ("prefix_followup", 90, 360, "prefix_followup.py"),
        ("record_followup", 30, 60, "record_followup.py"),
    ):
        folder = study / experiment
        path = folder / "semantic_interventions.jsonl"
        if not path.exists():
            continue
        meta = json.loads((folder/"completed.json").read_text(encoding="utf-8"))
        assert meta["inputs_sha256"] == digest(path)
        assert meta["script_sha256"] == digest(Path(__file__).with_name(source))
        inputs = {g["id"]:g for g in load_lines(path)}
        rows = load_lines(folder/"rows.jsonl")
        assert len(rows) == expected_rows and len(inputs) == expected_groups
        assert set(inputs).isdisjoint(exclusive)
        assert {g["source"] for g in inputs.values() if g["task"] == "prontoqa_derived"}.isdisjoint(source_ids)
        exclusive.update(inputs)
        assert {r["id"] for r in rows} == set(inputs)
        for row in rows:
            group = inputs[row["id"]]
            donor = row.get("same_answer_donor") or row["donor"]
            world = group["worlds"]
            assert world[donor]["answer"] == world[row["receive"]]["answer"] == row["original"]
            assert world[donor]["middle_answer"] != world[row["receive"]]["middle_answer"]
            assert row["conditions"]["self"]["scores"] == row["natural"]["scores"]
        inspected[experiment] = {"groups": expected_groups, "directions_by_site": expected_rows}
    return inspected


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("study", type=Path)
    args = cli.parse_args()
    study = args.study
    inputs = study / "semantic_interventions.jsonl"
    manifest = {x["id"]: x for x in load_lines(inputs)}
    counts = dict(Counter(g["task"] for g in manifest.values()))
    assert counts == {"multihop": 80, "arithmetic": 80, "prontoqa_derived": 60,
                      "prontoqa": 90, "arc": 100, "hellaswag": 50}
    data = load_lines(study / "trajectory_results" / "predictions.jsonl")
    natural = {(r["id"], r["world"]): r for r in data}
    assert len(natural) == len(data) == 1560
    assert set(natural) == {(g["id"], name) for g in manifest.values() for name in g["worlds"]}
    frozen = json.loads((study / "site_freeze.json").read_text(encoding="utf-8"))
    assert frozen["inputs_sha256"] == digest(inputs)
    for phase in ("pilot", "dev", "confirmation", "trajectory"):
        receipt = json.loads((study / phase / "completed.json").read_text(encoding="utf-8"))
        assert receipt["input_sha256"] == frozen["inputs_sha256"]
        assert receipt["runner_sha256"] == frozen["runner_sha256"]
    assert frozen["dev_rows_sha256"] == digest(study / "dev" / "causal_rows.jsonl")
    dev = load_lines(study / "dev" / "causal_rows.jsonl")
    confirm = load_lines(study / "confirmation" / "causal_rows.jsonl")
    assert len(dev) == 576 and len(confirm) == 180
    assert {x["id"] for x in dev}.isdisjoint({x["id"] for x in confirm})
    for row in dev + confirm:
        assert row["status"] == "ok"
        key = row["id"], row["receive"]
        linked = natural[key]
        donor = natural[(row["id"], row["same_answer_donor"])]
        g = manifest[row["id"]]
        expected_split = "development" if "intermediate_query" not in row["conditions"] else "confirmation"
        assert g["split"] == expected_split
        assert row["natural"]["scores"] == linked["scores"]
        assert row["same_answer_donor_native_scores"] == donor["scores"]
        assert linked["gold"] == donor["gold"] == row["original"]
        assert linked["anchor"] == donor["anchor"] == row["anchor"]
        assert row["conditions"]["self"]["scores"] == linked["scores"]
        for detail in row["conditions"].values():
            if "scores" in detail and row["site"][0] == 2:
                assert detail["scores"][0] == linked["scores"][0]
    geometry = np.load(study / "trajectory_results" / "geometry.npz")
    assert geometry["rms"].shape == (1560, 4, 25, 2)
    assert geometry["cosine"].shape == (1560, 3, 25, 2)
    assert np.isfinite(geometry["relative_l2"]).all()
    pairs = np.load(study / "trajectory_results" / "paired_geometry.npz")
    assert pairs["cosine"].shape == (220, 4, 4, 5, 2)
    assert (pairs["relative_l2"][:, 2, :, :, 0] == 0).all()
    payload = {"status": "passed", "manifest_counts": counts, "natural_worlds": len(data),
               "development_site_directions": len(dev), "confirmation_directions": len(confirm),
               "exact_natural_score_parity_from_trajectory": len(dev)+len(confirm),
               "map_only_anchor_geometry_identity_groups": len(pairs["ids"]),
               "followups": followup_audit(study, manifest)}
    write_json(study / "execution_audit.json", payload)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
