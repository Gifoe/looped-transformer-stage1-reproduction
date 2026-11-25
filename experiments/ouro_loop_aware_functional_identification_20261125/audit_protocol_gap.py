"""Measure ranking and free-generation outcomes on the same development IDs."""

import argparse
import json
from pathlib import Path
import statistics

from aggregate_full import pair
import minimal
from task_data import load


def cohort(folder, task, method, mode, protocol_sha):
    p = folder / f"{task}__{mode}__{method}.jsonl"
    stamp = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    if stamp["protocol_sha256"] != protocol_sha or stamp["sha256"] != minimal.sha(p):
        raise ValueError(f"Development receipt changed at {p}")
    rows = load(p)
    return {row["id"]: row for row in rows}


def main(a):
    register = a.dev / "DEV_PROTOCOL.json"
    plan = json.loads(register.read_text(encoding="utf-8"))["cohorts"]
    protocol_sha = minimal.sha(register)
    out = {"protocol_sha256": protocol_sha, "tasks": {}}
    for task in ("arc_challenge", "csqa"):
        ids = plan[f"{task}/content"]["ids"]
        if not set(ids) <= set(plan[f"{task}/candidate"]["ids"]):
            raise ValueError("Generation prompts are expected on a subset of candidate cohort identities")
        outcomes = {}
        for name in a.methods:
            c = cohort(a.dev, task, name, "candidate", protocol_sha)
            g = cohort(a.dev, task, name, "content", protocol_sha)
            d = {"candidate": [c[i]["metrics"]["acc_norm"] for i in ids]}
            d.update({key: [g[i]["metrics"][key] for i in ids]
                      for key in ("correct", "strict", "valid", "loose_correct")})
            outcomes[name] = d
        differences = {}
        for name, d in outcomes.items():
            differences[name] = {
                "means": {key: statistics.mean(value) for key, value in d.items()},
                "candidate_correct_generation_invalid": sum(ok and not valid for ok, valid in zip(d["candidate"], d["valid"], strict=True)),
                "candidate_correct_generation_wrong_valid": sum(ok and valid and not correct
                    for ok, valid, correct in zip(d["candidate"], d["valid"], d["correct"], strict=True)),
                "paired_vs_native": {key: pair(outcomes["native"][key], value, None) for key, value in d.items()},
            }
        out["tasks"][task] = {"n": len(ids), "by_method": differences}
    minimal.dump(a.out, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=("native", "t1", "t2", "mixed"))
    main(p.parse_args())
