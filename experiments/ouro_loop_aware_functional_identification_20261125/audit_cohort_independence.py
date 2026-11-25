"""Compare frozen development stems with prepared controller-training data."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from prepare_external import Corpus, normalize_question
from task_data import hash_order, identity


def records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def check(a):
    mmlu = Corpus()
    for p in sorted(a.mmlu.glob("*/dev-*.parquet")):
        for i, row in enumerate(pq.read_table(p).to_pylist()):
            mmlu.add(normalize_question(row), f"{p.parent.name}/{i}")
    if len(mmlu.samples) == 0:
        raise ValueError("No separate MMLU development split")
    targets = {}
    for p in sorted(a.confirmation.glob("*_k4.json")):
        plan = json.loads(p.read_text(encoding="utf-8"))["plan"]
        source, ids = plan["target"], set(plan["ids"])
        targets[source] = [r for r in records(a.pools / f"{source}.jsonl") if r["sample_id"] in ids]
        if len(targets[source]) != len(ids):
            raise ValueError(f"Expected confirmation cohort does not match for {source}")
    if not targets:
        raise ValueError("No sealed confirmation cohorts")

    result = {}
    for manifest in sorted(a.pools.glob("*.json")):
        source = manifest.stem
        foreign = Corpus()
        for key, block in targets.items():
            if key != source:
                for row in block:
                    foreign.add(identity(row["source_question"]), f"{key}/{row['sample_id']}")
        hits = {"mmlu_dev": [], "frozen_forecast_other_source": []}
        for row in records(manifest.with_suffix(".jsonl")):
            stem = identity(row["source_question"])
            direct = mmlu.match(stem)
            if direct[0] is not None:
                hits["mmlu_dev"].append({"source_id": row["sample_id"], "target": direct[0], "similarity": direct[1]})
            if int(hash_order(row["fold_key"], "loop-identification/fold"), 16) % 4 == 3:
                continue
            cross = foreign.match(stem)
            if cross[0] is not None:
                hits["frozen_forecast_other_source"].append(
                    {"source_id": row["sample_id"], "target": cross[0], "similarity": cross[1]}
                )
        result[source] = hits
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for opt in ("pools", "mmlu", "confirmation", "out"):
        parser.add_argument(f"--{opt}", type=Path, required=True)
    check(parser.parse_args())
