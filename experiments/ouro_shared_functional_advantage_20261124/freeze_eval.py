"""Choose one development-locked deployment before evaluating any official test rows."""

import argparse
import json
from pathlib import Path

import minimal
from evaluate_controller import jobs, __file__ as evaluator
from task_data import load


BASELINES = ("native", "historical_rank", "historical_retention", "historical_mixed", "ranking", "generation", "mixed", "linear_weak")
ADVANTAGE = ("mean_advantage", "worst_advantage", "worst_rank_init", "worst_mixed_init")
ENDPOINTS = (("arc_challenge", "candidate", "acc_norm"), ("csqa", "candidate", "acc_norm"),
             ("arc_challenge", "content", "correct"), ("csqa", "content", "correct"),
             ("squad", "span", "correct"), ("gsm8k", "gsm_process", "strict"),
             ("gsm8k", "gsm_process", "process_correct"))


def receipt(path):
    data = json.loads(path.read_text())
    if minimal.sha(path.with_suffix(".jsonl")) != data["rows_sha256"]:
        raise RuntimeError(f"Torn development result {path}")
    return data


def select(development, baselines, proposed):
    exclusion = json.loads((development.parent / "DEVELOPMENT_EXCLUSIONS.json").read_text())["squad"]
    removed = set(exclusion["excluded_ids"])
    frontier = {}
    values = {arm: {} for arm in (*baselines, *proposed)}
    for ds, mode, metric in ENDPOINTS:
        for k in (2, 3, 4):
            name = f"{ds}/k{k}/{mode}/{metric}"
            for arm in values:
                path = development / f"{ds}__{arm}__k{k}__{mode}.json"
                data = receipt(path)["metrics"][metric]
                if ds == "squad":
                    subset = [row["metrics"][metric] for row in load(path.with_suffix(".jsonl")) if row["id"] not in removed]
                    if len(subset) != exclusion["independent_n"]:
                        raise ValueError("Locked SQuAD passage exclusion differs across arms")
                    values[arm][name] = sum(subset) / len(subset)
                else:
                    values[arm][name] = data["value"]
            ref = max(baselines, key=lambda arm: values[arm][name])
            frontier[name] = dict(arm=ref, value=values[ref][name])
    result = {}
    for arm, group in values.items():
        differences = {k: v-frontier[k]["value"] for k, v in group.items()}
        result[arm] = dict(worst_delta=min(differences.values()), mean_delta=sum(differences.values())/len(differences),
                           endpoints=group, deltas=differences)
    selected = max(proposed, key=lambda arm: (result[arm]["worst_delta"], result[arm]["mean_delta"], arm))
    return selected, result, frontier


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("results", "data", "historical_rank", "historical_retention", "historical_mixed", "out"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--extra", action="append", default=[])
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("The official confirmation is already locked")
    proposed = (*ADVANTAGE, *a.extra)
    methods = (*BASELINES, *proposed)
    selected, evidence, reference = select(a.results / "development", BASELINES, proposed)
    arms = {x: ("native" if x == "native" else str(getattr(a, x)) if x.startswith("historical_")
               else str(a.results / "fits" / x / "controller.pt")) for x in methods}
    fingerprints = {name: ("native" if path == "native" else minimal.sha(path)) for name, path in arms.items()}
    transfer = list(dict.fromkeys((selected, "native", "historical_rank", "historical_mixed", "mixed")))
    cases = list(jobs(a.data, "confirmation", (2, 3, 4, 5, 6), 10000, 96))
    for spec in cases:
        primary = spec["depth"] == 4
        spec["arms"] = list(methods) if primary else transfer
        if spec["mode"] == "gsm_process":
            spec["limit"] = 96 if primary else 48
    order = list(dict.fromkeys((selected, "native", "historical_rank", "historical_retention", "historical_mixed", *methods)))
    lock = dict(selected=selected, validation=evidence, reference_envelope=reference, arm_path=arms,
                arm_sha256=fingerprints, run_order=order, transfer_arms=transfer, cases=cases,
                evaluator_sha256=minimal.sha(evaluator), contamination_sha256=minimal.sha(a.data/"CONTAMINATION.json"),
                development_exclusions_sha256=minimal.sha(a.results/"DEVELOPMENT_EXCLUSIONS.json"),
                selection="max worst endpoint difference to best baseline on development, then mean difference",
                model="Ouro-1.4B Base frozen", parameters=4096)
    minimal.dump(a.out, lock)
    print(selected, {k: (v["worst_delta"], v["mean_delta"]) for k, v in evidence.items()}, flush=True)


if __name__ == "__main__":
    main()
