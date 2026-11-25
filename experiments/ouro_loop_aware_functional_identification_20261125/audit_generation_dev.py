"""Compare generation accuracy, format, and termination on the fixed GSM8K development cohort."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from aggregate_full import pair
from audit_protocol_gap import cohort
import minimal


def run(a):
    receipt = a.dev / "DEV_PROTOCOL.json"
    plan = json.loads(receipt.read_text(encoding="utf-8"))
    ids = plan["cohorts"]["gsm8k/gsm_process"]["ids"]
    freeze = minimal.sha(receipt)
    methods = ["native", "t1", "t2", "mixed"]
    methods.extend(name for name in plan["checkpoint_sha256"] if name not in methods)
    fields = ("correct", "strict", "valid", "process")
    assessed = {}
    for name in methods:
        if not (a.dev / f"gsm8k__gsm_process__{name}.json").is_file():
            continue
        cases = cohort(a.dev, "gsm8k", name, "gsm_process", freeze)
        if set(cases) != set(ids):
            raise ValueError(f"Wrong GSM8K development identities for {name}")
        scores = {field: [cases[i]["metrics"][field] for i in ids] for field in fields}
        assessed[name] = {
            "metrics": {field: statistics.mean(values) for field, values in scores.items()},
            "stops": dict(Counter(r["stop"] for r in cases.values())),
            "mean_answer_tokens": statistics.mean(r["answer_tokens"] for r in cases.values()),
            "vectors": scores,
        }
    refs = ["native", "t1"]
    if not all(name in assessed for name in refs):
        raise ValueError("The historical comparison arms are incomplete")
    results = {}
    for name, data in assessed.items():
        d = {key: value for key, value in data.items() if key != "vectors"}
        d["paired"] = {base: {metric: pair(assessed[base]["vectors"][metric], data["vectors"][metric], None)
                             for metric in fields} for base in refs}
        results[name] = d
    minimal.dump(a.out, {"dev_protocol_sha256": freeze, "n": len(ids), "completed_arms": results})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())
