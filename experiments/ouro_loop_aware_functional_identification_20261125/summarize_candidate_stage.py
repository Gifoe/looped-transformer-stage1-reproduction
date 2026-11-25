"""Summarize only the completed independent candidate endpoints."""

import argparse
from pathlib import Path
import statistics

from aggregate_full import pair
import minimal
from score_source_training import compare, read
from task_data import load


def run(a):
    protocol = a.development / "DEV_PROTOCOL.json"
    sha = minimal.sha(protocol)
    lock = read(protocol)
    modes = [key for key in lock["cohorts"] if key.endswith(("/candidate", "/mmlu"))]
    models = {}
    optional = {"t2", "mixed"}
    for name in lock["checkpoint_sha256"]:
        metrics = {}
        for key in modes:
            task, mode = key.split("/")
            receipt = a.development / f"{task}__{mode}__{name}.json"
            if not receipt.is_file() and name in optional:
                break
            raw = receipt.with_suffix(".jsonl")
            line = read(receipt)
            if line["protocol_sha256"] != sha or line["sha256"] != minimal.sha(raw):
                raise ValueError(f"Changed development receipt: {receipt}")
            ids = lock["cohorts"][key]["ids"]
            metric = lock["cohorts"][key]["primary"]
            items = load(raw)
            if [r["id"] for r in items] != ids or statistics.mean(r["metrics"][metric] for r in items) != line["value"]:
                raise ValueError(f"Development score or IDs no longer match: {raw}")
            metrics[key] = {"value": line["value"], "n": line["n"], "row_hash": line["sha256"]}
        if len(metrics) == len(modes):
            models[name] = metrics
    forecast = compare(read(a.forecast), {"all_runs": models}, False)
    table = {method: {key: result["value"] for key, result in outcomes.items()} for method, outcomes in models.items()}
    minimal.dump(a.out, dict(development_protocol_sha256=sha, stage="independent candidate development only",
                 endpoints=modes, outcomes=models, accuracy=table, pretraining_forecast_audit=forecast))
    if a.contrasts:
        comparisons={}
        for reference, method in (('uniform','loop'),('ordinary','loop'),('warm_uniform','warm_loop'),
                                  ('t1','loop'),('t1','ordinary')):
            scores={}
            for key in modes:
                task, mode=key.split('/')
                metric=lock['cohorts'][key]['primary']
                published=[load(a.development/f'{task}__{mode}__{arm}.jsonl') for arm in (reference,method)]
                values=[[row['metrics'][metric] for row in group] for group in published]
                scores[task]=pair(*values,None)
            comparisons[f'{method}_minus_{reference}']=scores
        minimal.dump(a.contrasts,comparisons)
    for method, values in table.items():
        print(method, " ".join(f"{100*values[key]:.2f}" for key in modes))
    for scope, fit in forecast["prospective_training_rewards"].items():
        print(scope, {name: result["aggregate"]["spearman_mean"] for name, result in fit["methods"].items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("development", "forecast", "out"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--contrasts",type=Path)
    run(p.parse_args())
