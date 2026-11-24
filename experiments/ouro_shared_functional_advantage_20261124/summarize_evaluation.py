"""Verify result receipts and compute paired statistics for each fixed evaluation cohort."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_controller import annotate, rows_for
from evaluate_generation import evaluate, summarize as metrics_for
from freeze_eval import BASELINES
import minimal
import paired_analysis
from task_data import load


paired_analysis.METRICS.update({"premature_eos", "eos_before_16", "process_correct"})
REFERENCES = ("native", "historical_rank", "historical_retention", "historical_mixed", "mixed", "teacher_resume")


def verify_rows(outputs, receipt, source):
    plan = receipt["plan"]
    mode = plan["mode"]
    if mode == "candidate":
        for index, row in enumerate(outputs):
            scores = np.asarray(row["scores"])
            chars = np.asarray(row["chars"])
            if scores.shape != chars.shape or np.any(chars < 1) or not np.isfinite(scores).all():
                raise ValueError(f"Invalid candidate scores at {row['id']}")
            match = dict(acc=int(scores.argmax()) == row["gold"],
                         acc_norm=int((scores / chars).argmax()) == row["gold"])
            if match != row["metrics"]:
                raise ValueError(f"Candidate scores and metrics differ at {row['id']}")
            if source is not None:
                question = source[index]["candidate"]
                choices = question["choices"]
                if row["labels"] != choices["label"] or row["chars"] != [len(text) for text in choices["text"]]:
                    raise ValueError(f"Candidate order or lengths differ at {row['id']}")
                if row["labels"][row["gold"]] != question["answerKey"]:
                    raise ValueError(f"Candidate gold answer differs at {row['id']}")
    else:
        if any(len(row["generated_tokens"]) != row["answer_tokens"] or row["answer_tokens"] > plan["max_new"]
               for row in outputs):
            raise ValueError(f"Invalid generation lengths at {plan}")
        if receipt["cost"]["generated_tokens"] != sum(row["answer_tokens"] for row in outputs):
            raise ValueError(f"Generation cost and token counts differ at {plan}")
        if source is not None:
            for row, original in zip(outputs, source, strict=True):
                rescored = dict(row, metrics=evaluate(original, mode, row["prediction"], row["stop"]))
                if mode == "gsm_process":
                    annotate([rescored])
                if rescored["metrics"] != row["metrics"] or row["label"] != original.get("label"):
                    raise ValueError(f"Generated answer and metrics differ at {row['id']}")
    if metrics_for(outputs) != receipt["metrics"]:
        raise ValueError(f"Aggregate metrics differ from the raw outputs at {plan}")


def checked(phase, protocol, allow_shortfall):
    plan = json.loads(protocol.read_text()) if protocol else None
    plan_sha = minimal.sha(protocol) if protocol else None
    expected = {}
    if plan:
        for case in plan["cases"]:
            for arm in case["arms"]:
                stem = f"{case['dataset']}__{arm}__k{case['depth']}__{case['mode']}"
                expected[stem] = case
    observed, sources = {}, {}
    common_runtime = None
    for receipt in sorted(phase.glob("*.json")):
        path = receipt.with_suffix(".jsonl")
        doc = json.loads(receipt.read_text())
        if doc["rows_sha256"] != minimal.sha(path):
            raise RuntimeError(f"Torn result at {path}")
        outputs = load(path)
        info = doc["plan"]
        if len(outputs) != info["n"] or len(outputs) != len({r["id"] for r in outputs}):
            raise RuntimeError(f"Unexpected row count {path}")
        # Reservable VRAM varies across server images on the same GPU model.
        runtime = {key: value for key, value in doc["runtime"].items() if key != "capacity_bytes"}
        if common_runtime is None:
            common_runtime = runtime
        elif runtime != common_runtime:
            raise RuntimeError(f"Model or numerical environment differs at {path}")
        source = None
        if plan:
            case = expected[receipt.stem]
            dataset, arm, depth, mode = receipt.stem.split("__")
            key = dataset, depth, mode
            if key not in sources:
                sources[key] = rows_for(case)
            source = sources[key]
            frozen = dict(case, arm=arm, n=len(source), checkpoint_sha256=plan["arm_sha256"][arm], protocol_sha256=plan_sha)
            if info != frozen or [r["id"] for r in outputs] != [r["sample_id"] for r in source]:
                raise RuntimeError(f"Wrong frozen controller, settings or query set {path}")
        verify_rows(outputs, doc, source)
        observed[receipt.stem] = outputs, doc
    missing = sorted(set(expected) - set(observed))
    if missing and not allow_shortfall:
        raise RuntimeError(f"Incomplete sealed evaluation with {len(missing)} missing cells")
    return observed, dict(cells=len(observed), expected=len(expected) if plan else None, missing=missing)


def summarize(observed, references=REFERENCES):
    family = defaultdict(dict)
    for label, (rows, receipt) in observed.items():
        ds, arm, k, mode = label.split("__")
        family[ds, k, mode][arm] = (rows, receipt)
    scores = {}
    for (ds, k, mode), data in family.items():
        cells = {}
        for arm, (results, receipt) in data.items():
            peers = {name: paired_analysis.compare(data[name][0], results) for name in references if name in data}
            stops = Counter(row.get("stop", "candidate") for row in results)
            output_lengths = [row["answer_tokens"] for row in results if "answer_tokens" in row]
            cells[arm] = dict(metrics=receipt["metrics"], n=len(results), paired=peers, cost=receipt["cost"],
                              median_output_tokens=float(np.median(output_lengths)) if output_lengths else None,
                              stop_counts=dict(stops))
        scores[f"{ds}/{k}/{mode}"] = cells
    return scores


def write_csv(path, scores):
    columns = ("dataset", "depth", "mode", "controller", "metric", "n", "score", "count", "comparison",
               "reference", "delta", "improved", "harmed", "ci_low", "ci_high", "mcnemar", "total_seconds", "ms_per_item",
               "model_calls", "padded_token_loops", "cuda_peak_bytes", "output_token_median")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for cell, arms in scores.items():
            ds, k, mode = cell.split("/")
            for arm, row in arms.items():
                c = row["cost"]
                n = row["n"]
                for metric, value in row["metrics"].items():
                    base = [ds, k, mode, arm, metric, value["n"], value["value"], value["count"]]
                    cost = [c["seconds"], 1e3*c["seconds"]/n, c["model_calls"], c["backbone_padded_token_loops"],
                            c["cuda_peak"], row["median_output_tokens"]]
                    used = False
                    for reference, paired in row["paired"].items():
                        if metric not in paired:
                            continue
                        detail = paired[metric]
                        writer.writerow([*base, reference, detail["reference"], detail["delta"], detail["rescued"],
                                         detail["harmed"], *detail["paired_ci95"], detail["exact_mcnemar"], *cost])
                        used = True
                    if not used:
                        writer.writerow([*base, *([None]*8), *cost])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--protocol", type=Path)
    p.add_argument("--allow-shortfall", action="store_true")
    a = p.parse_args()
    observed, coverage = checked(a.phase_dir, a.protocol, a.allow_shortfall)
    references = tuple(dict.fromkeys((*REFERENCES, *BASELINES))) if a.protocol else REFERENCES
    scores = summarize(observed, references)
    minimal.dump(a.out, dict(coverage=coverage, results=scores))
    write_csv(a.out.with_suffix(".csv"), scores)
    print(coverage, flush=True)


if __name__ == "__main__":
    main()
