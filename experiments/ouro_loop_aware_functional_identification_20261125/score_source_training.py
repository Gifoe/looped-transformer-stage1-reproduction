"""Measure whether sealed pretraining ranks predict independently scored fits."""

import argparse
import json
from pathlib import Path
import statistics

from score_forecast import evaluate


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(forecast, development, work_matched):
    roles = development["all_runs"]
    suffix = "_work" if work_matched else ""
    required = {f"source_{s}{suffix}" for s in forecast["source_order"]}
    if not required <= roles.keys():
        raise ValueError(f"Missing matched training controls: {sorted(required - roles.keys())}")
    targets = {}
    for source in forecast["source_order"]:
        candidate = f"{source}/candidate"
        if candidate in roles["native"]:
            targets[source] = candidate
    output = {}
    for scope in ("leave_source_out", "leave_task_type_out"):
        rows = []
        for sealed in forecast["records"]:
            source, target = sealed["source"], sealed["target"]
            if sealed["scope"] != scope or target not in targets:
                continue
            endpoint = targets[target]
            baseline = roles["native"][endpoint]
            fitted = roles[f"source_{source}{suffix}"][endpoint]
            if baseline["n"] != fitted["n"]:
                raise ValueError(f"Unmatched developer cohort for {source}/{target}")
            gain = fitted["value"] - baseline["value"]
            rows.append(dict(**sealed, actual=dict(scaled_loop=gain, accuracy_delta=gain, interaction=None)))
        if rows:
            scores = {metric: evaluate(rows, metric) for metric, value in rows[0]["predicted"].items() if value is not None}
            output[scope] = dict(methods=scores, records=rows)
    cross_task = {}
    for source in forecast["source_order"]:
        fit = roles[f"source_{source}{suffix}"]
        gaps = {endpoint: result["value"] - roles["native"][endpoint]["value"]
                for endpoint, result in fit.items() if not endpoint.startswith(source + "/")}
        cross_task[source] = dict(gains=gaps, average=statistics.mean(gaps.values()), weakest=min(gaps.values()))
    return {"matched_by": "padded_token_loops" if work_matched else "question_count",
            "endpoints": targets, "prospective_training_rewards": output, "cross_task": cross_task}


def main(a):
    forecast = read(a.forecast)
    original = compare(forecast, read(a.development), False)
    outputs = [original]
    if a.work_development:
        outputs.append(compare(forecast, read(a.work_development), True))
    a.out.write_text(json.dumps(outputs, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", required=True, type=Path)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--work-development", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    main(parser.parse_args())
