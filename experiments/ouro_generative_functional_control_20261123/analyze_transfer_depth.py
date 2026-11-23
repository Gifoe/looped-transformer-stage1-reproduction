"""Compare K2–K6 on the same IDs and inspect generation/score disagreements."""

import argparse
from collections import Counter
import json
from pathlib import Path

from paired_analysis import compare, read
from task_data import load


GENERATIVE_METRICS = {
    "content": "correct", "letter": "strict", "candidate": "acc_norm", "span": "correct",
    "gsm_final": "correct", "other": "rouge_l",
}


def rows(result, dataset, arm, depth, mode):
    return {r["id"]: r for r in load(result / f"{dataset}__{arm}__k{depth}__{mode}.jsonl")}


def transfer(config, result):
    comparisons = {}
    for case in config["cases"]:
        ds, mode = case["dataset"], case["mode"]
        depths = case["depths"]
        if len(depths) < 2:
            continue
        arms = config["allowed_run_order"]
        if any(not (result / f"{ds}__{arm}__k{k}__{mode}.json").is_file() for arm in arms for k in depths):
            continue
        common = set.intersection(*(set(rows(result, ds, "native", k, mode)) for k in depths))
        ids = sorted(common)
        key = f"{ds}/{mode}"
        comparisons[key] = {"ids": ids, "depths": {}}
        for depth in depths:
            aligned = {}
            for arm in arms:
                arm_rows = rows(result, ds, arm, depth, mode)
                aligned[arm] = [arm_rows[q] for q in ids]
            metric = GENERATIVE_METRICS[mode]
            comparisons[key]["depths"][f"k{depth}"] = {
                arm: {base: compare(aligned[base], aligned[arm]).get(metric) for base in ("native", "rank_only")}
                for arm in arms
            }
    return comparisons


def error_audit(config, result, sample_cap=5):
    datasets = sorted({case["dataset"] for case in config["cases"] if case["mode"] == "candidate"})
    summary = {}
    for dataset in datasets:
        for arm in config["allowed_run_order"]:
            if any(not (result / f"{dataset}__{arm}__k4__{mode}.json").is_file()
                   for mode in ("content", "letter", "candidate")):
                continue
            content = rows(result, dataset, arm, 4, "content")
            letters = rows(result, dataset, arm, 4, "letter")
            scored = rows(result, dataset, arm, 4, "candidate")
            if set(content) != set(letters) or set(content) != set(scored):
                raise ValueError(f"Mismatched scoring populations for {dataset} / {arm}")
            counts = Counter()
            samples = {}
            for key, c in content.items():
                letter = letters[key]
                right = scored[key]["metrics"]["acc_norm"]
                judged = c["metrics"]["correct"]
                counts[f"candidate_{int(right)}_content_{int(judged)}"] += 1
                reasons = []
                if right and not judged:
                    reasons.append("scores_correct_content_wrong")
                if judged and not right:
                    reasons.append("generation_correct_candidate_wrong")
                if letter["metrics"]["loose_correct"] and not letter["metrics"]["strict"]:
                    reasons.append("letter_only_loose")
                if letter["metrics"]["failed"] == "lowercase_option":
                    reasons.append("letter_lowercase")
                if c["metrics"]["label_conflict"]:
                    reasons.append("content_label_conflict")
                if c["stop"] == "token_limit":
                    reasons.append("content_hit_limit")
                for reason in reasons:
                    counts[reason] += 1
                    examples = samples.setdefault(reason, [])
                    if len(examples) < sample_cap:
                        examples.append({"id": key, "content": c["prediction"], "letter": letter["prediction"],
                                         "gold_letter": letter["label"], "content_stop": c["stop"]})
            summary[f"{dataset}/{arm}"] = dict(n=len(content), counts=dict(counts), examples=samples)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results"))
    a = p.parse_args()
    data = a.results / "confirmation"
    config = read(a.results / "FINAL_PROTOCOL.json")
    for name, value in (("MATCHED_DEPTH_TRANSFER", transfer(config, data)), ("K4_FAILURE_AUDIT", error_audit(config, data))):
        path = a.results / f"{name}.json"
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
