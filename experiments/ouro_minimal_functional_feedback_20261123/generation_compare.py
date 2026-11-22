"""Paired greedy results, including question-clustered intervals across depths."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from minimal import DEPTHS, dump, load_jsonl, sha


def transitions(old, new):
    names = ("invalid", "wrong", "correct")

    def state(row):
        return names[0 if not row["valid"] else 2 if row["correct"] else 1]

    cells = Counter(
        f"{state(previous)}_to_{state(current)}"
        for previous, current in zip(old, new, strict=True)
    )
    return {f"{i}_to_{j}": cells[f"{i}_to_{j}"] for i in names for j in names}


def paired_status(a, b, field):
    differences = np.array(
        [int(new[field]) - int(old[field]) for old, new in zip(a, b, strict=True)],
        dtype=np.int8,
    )
    return differences


def summarize(values, *, seed):
    # Each row is one question, with one or several depths occupying its columns.
    n = len(values)
    rng = np.random.default_rng(seed)
    by_question = values.mean(axis=1)
    blocks = [
        by_question[rng.integers(n, size=(500, n))].mean(axis=1)
        for _ in range(20)
    ]
    lower, upper = 100 * np.quantile(np.concatenate(blocks), (0.025, 0.975))
    return dict(
        n_questions=n,
        n_outcomes=values.size,
        rescued=int((values > 0).sum()),
        damaged=int((values < 0).sum()),
        delta_pp=100 * float(by_question.mean()),
        ci95_pp=[float(lower), float(upper)],
    )


def main(results):
    config = json.loads((results / "FREEZE.json").read_text(encoding="utf-8"))
    folder = results / "generation"
    path = folder / "GENERATION_RESULTS.jsonl"
    grouped = {}
    for row in load_jsonl(path):
        key = (row["split"], row["method"], row["depth"])
        batch = grouped.setdefault(key, {})
        if row["id"] in batch:
            raise RuntimeError(f"Duplicate generation record for {key}, {row['id']}")
        batch[row["id"]] = row
    output = {}
    baselines = ("native", "original_feedback", "original_joint")
    comparisons = [(new, old) for new in config["generation_arms"] for old in baselines]
    t2, constrained = config["kl_pair"]
    comparisons.append((constrained, t2))
    for split, expected in (("arc_validation", 299), ("openbookqa_test", 500)):
        ids = sorted(grouped[split, "native", 1])
        if len(ids) != expected:
            raise RuntimeError(f"{split} has {len(ids)} unique generation IDs")
        collection = output[split] = {}
        for newer, older in comparisons:
            point = {}
            paired = {field: [] for field in ("valid", "correct")}
            for depth in DEPTHS:
                new = grouped[split, newer, depth]
                old = grouped[split, older, depth]
                if new.keys() != old.keys() or set(ids) != new.keys():
                    raise RuntimeError(f"ID mismatch for {split}, {newer}, {older}, K{depth}")
                a = [old[key] for key in ids]
                b = [new[key] for key in ids]
                if any(x["gold"] != y["gold"] or x["prompt"] != y["prompt"]
                       for x, y in zip(a, b, strict=True)):
                    raise RuntimeError("Generation prompts or answers disagree")
                result = point[f"k{depth}"] = {"transitions": transitions(a, b)}
                for field in paired:
                    d = paired_status(a, b, field)
                    paired[field].append(d)
                    result[field] = summarize(d[:, None], seed=261123278)
            point["pooled_k2_k4"] = {
                field: summarize(np.stack(array[1:], axis=1), seed=261123279)
                for field, array in paired.items()
            }
            collection[f"{newer}_vs_{older}"] = point
    dump(folder / "GENERATION_PAIRED.json", dict(
        source_sha256=sha(path),
        ci="Percentile bootstrap over whole question IDs, 10,000 replicates. K2–K4 pooling preserves the three outcomes per sampled question.",
        splits=output,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    main(parser.parse_args().results)
