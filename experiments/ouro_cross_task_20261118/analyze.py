"""Freeze site selection on development and summarize held-out evidence."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from design import TASKS

PRIMARY = {
    "semantic_minus_natural": ("semantic_same_answer", "natural"),
    "semantic_minus_irrelevant": ("semantic_same_answer", "irrelevant"),
    "semantic_minus_random": ("semantic_same_answer", "random"),
}
COMPARISONS = {
    **PRIMARY,
    "different_answer_minus_natural": ("semantic_different_answer", "natural"),
    "map_only_minus_natural": ("map_only", "natural"),
    "self_minus_natural": ("self", "natural"),
    "irrelevant_minus_natural": ("irrelevant", "natural"),
    "random_minus_natural": ("random", "natural"),
    "full_minus_blocked": ("semantic_same_answer", "block_readout"),
    "blocked_minus_natural": ("block_readout", "natural"),
    "block_only_minus_natural": ("block_only", "natural"),
    "intermediate_minus_natural": ("probe_semantic", "probe_natural"),
    "intermediate_minus_random": ("probe_semantic", "probe_random"),
}


def load_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, item):
    path.write_text(json.dumps(item, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, entries):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)


def flatten(row):
    series = {"natural": row["natural"]["margin"]}
    for condition, details in row["conditions"].items():
        if condition == "intermediate_query":
            series.update({f"probe_{k}": details[k] for k in ("natural", "semantic", "random")})
        else:
            series[condition] = details["margin"]
    return series


def clustered(rows, left, right, depth):
    clusters = defaultdict(list)
    for row in rows:
        series = flatten(row)
        if left in series and right in series:
            clusters[row["id"]].append(series[left][depth] - series[right][depth])
    if any(len(values) != 2 for values in clusters.values()):
        raise RuntimeError("Missing reciprocal intervention direction")
    return np.array([np.mean(v) for v in clusters.values()])


def interval(values, confidence):
    rnd = np.random.default_rng(20261118)
    boot = values[rnd.integers(0, len(values), (30000, len(values)))].mean(axis=1)
    q = (1 - confidence) / 2
    lo, hi = np.quantile(boot, (q, 1-q)).tolist()
    return float(values.mean()), lo, hi


def freeze(study, runner):
    rows_path = study / "dev" / "causal_rows.jsonl"
    metadata = json.loads((rows_path.parent / "completed.json").read_text())
    rows = load_lines(rows_path)
    assert metadata["runner_sha256"] == digest(runner)
    selections, development = {}, []
    for task in TASKS:
        sites = sorted({tuple(r["site"]) for r in rows if r["task"] == task})
        records = []
        for site in sites:
            current = [r for r in rows if r["status"] == "ok" and r["task"] == task and tuple(r["site"]) == site]
            means = {name: float(np.mean(clustered(current, *conditions, 3)))
                     for name, conditions in PRIMARY.items()}
            values = {"task": task, "site": list(site), "directions": len(current),
                      **means, "minimum_primary_mean": min(means.values())}
            records.append(values)
        best = max(records, key=lambda r: r["minimum_primary_mean"])
        selections[task] = best["site"]
        development.extend(records)
    payload = {
        "selected": selections,
        "rule": "maximize minimum of the three group-mean K4 contrasts, earliest site breaks ties",
        "primary_confirmation": "K4 same-answer donor versus natural, irrelevant direction and random direction",
        "confirmation_simultaneous_level": 1-0.05/3,
        "confirmation_gate": "all three lower bounds exceed zero within a task",
        "rationale": "intersection-union within task, Bonferroni across three tasks",
        "inputs_sha256": metadata["input_sha256"],
        "runner_sha256": metadata["runner_sha256"],
        "dev_rows_sha256": digest(rows_path),
    }
    write_json(study / "site_freeze.json", payload)
    write_csv(study / "development_selection.csv", development)
    print(json.dumps(payload), flush=True)


def summarize_behavior(entries):
    result = []
    names = sorted({entry["task"] for entry in entries})
    for task in names:
        task_rows = [r for r in entries if r["task"] == task and r["world"] == "base"]
        probe = [r for r in task_rows if "probe_gold" in r]
        for k in range(4):
            row = {"task": task, "K": k+1, "base_count": len(task_rows),
                   "correct": sum(r["predictions"][k] == r["gold"] for r in task_rows),
                   "tied": sum(r["predictions"][k] == -1 for r in task_rows),
                   "normalized_candidate_margin_mean": float(np.mean([
                       r["scores"][k][r["gold"]] - max(s for i, s in enumerate(r["scores"][k]) if i != r["gold"])
                       for r in task_rows
                   ])), "middle_count": len(probe),
                   "middle_correct": sum(r["probe_predictions"][k] == r["probe_gold"] for r in probe),
                   "middle_tied": sum(r["probe_predictions"][k] == -1 for r in probe),
                   "rescues_from_k1": sum(r["predictions"][k] == r["gold"] != r["predictions"][0] for r in task_rows),
                   "losses_from_k1": sum(r["predictions"][0] == r["gold"] != r["predictions"][k] for r in task_rows)}
            result.append(row)
    return result


def summarize_geometry(study, predictions):
    geometry = np.load(study / "trajectory_results" / "geometry.npz")
    assert geometry["ids"].tolist() == [f"{r['id']}/{r['world']}" for r in predictions]
    relative, cosine, rms = geometry["relative_l2"], geometry["cosine"], geometry["rms"]
    result, association = [], []
    for task in sorted({r["task"] for r in predictions}):
        indices = [i for i, r in enumerate(predictions) if r["task"] == task and r["world"] == "base"]
        batch = [predictions[i] for i in indices]
        for role_index, role in enumerate(("anchor", "readout")):
            for k in range(3):
                for event in range(25):
                    result.append({"task": task, "role": role, "transition": f"K{k+1}->K{k+2}",
                                   "event": "post_norm" if event == 24 else f"L{event}",
                                   "n": len(indices),
                                   "median_cosine": float(np.median(cosine[indices, k, event, role_index])),
                                   "median_relative_l2": float(np.median(relative[indices, k, event, role_index])),
                                   "median_rms_pre": float(np.median(rms[indices, k, event, role_index])),
                                   "median_rms_post": float(np.median(rms[indices, k+1, event, role_index]))})
            # Position is fixed by role, depth by transition. Regress the continuous
            # angle-change endpoint on decision changes with length, norm and
            # baseline top-2 margin as nuisance covariates. This is descriptive.
            for k in range(3):
                switches = [r["predictions"][k] != r["predictions"][k+1] for r in batch]
                conf = [np.subtract(*sorted(r["scores"][k], reverse=True)[:2]) for r in batch]
                lens = [len(r["ids"]) for r in batch]
                for event in (11, 23, 24):
                    end = 1 - cosine[indices, k, event, role_index]
                    norm = rms[indices, k, event, role_index]
                    x = np.column_stack((np.ones(len(batch)), switches, lens, conf, norm))
                    defined = np.linalg.matrix_rank(x) == x.shape[1]
                    association.append({"task": task, "role": role, "from_K": k+1, "event": event,
                                        "n": len(batch), "switched": int(sum(switches)),
                                        "mean_changed": float(end[switches].mean()) if any(switches) else None,
                                        "mean_stable": float(end[np.logical_not(switches)].mean()) if not all(switches) else None,
                                        "adjusted_change_coefficient": float(np.linalg.lstsq(x, end, rcond=None)[0][1]) if defined else None})
    write_csv(study / "trajectory_results" / "layer_aggregates.csv", result)
    write_csv(study / "trajectory_results" / "transition_associations.csv", association)


def summarize_semantics(study):
    pairs = np.load(study / "trajectory_results" / "paired_geometry.npz")
    ids, events, contrasts = pairs["ids"], pairs["events"], pairs["contrasts"]
    rows = []
    for task in TASKS:
        keep = np.char.startswith(ids, task + "-")
        for j, contrast in enumerate(contrasts.tolist()):
            for k in range(4):
                for l, event in enumerate(events.tolist()):
                    for r, role in enumerate(("anchor", "readout")):
                        rows.append({"task": task, "contrast": contrast, "K": k+1, "event": event,
                                     "role": role, "groups": int(keep.sum()),
                                     "median_cosine": float(np.median(pairs["cosine"][keep, j, k, l, r])),
                                     "median_relative_l2": float(np.median(pairs["relative_l2"][keep, j, k, l, r]))})
    write_csv(study / "trajectory_results" / "semantic_geometry.csv", rows)


def merge_followup(study, entries):
    prefix_path = study / "prefix_followup" / "cross_task_mechanisms.csv"
    if not prefix_path.is_file():
        return
    with prefix_path.open(encoding="utf-8", newline="") as file:
        prefixes = {r["task"]: r for r in csv.DictReader(file)}
    record_path = study / "record_followup" / "causal_effects.csv"
    record = None
    if record_path.is_file():
        with record_path.open(encoding="utf-8", newline="") as file:
            record = next(r for r in csv.DictReader(file)
                          if r["readout_K"] == "4" and r["contrast"] == "target_minus_unqueried")
    for entry in entries:
        follow = prefixes[entry["task"]]
        entry.update(
            prefix_fresh_groups=follow["groups"],
            prefix_K2_vs_native_nats=follow["K2_full_vs_native_nats"],
            prefix_K2_vs_random_nats=follow["K2_full_vs_random_nats"],
            prefix_K2_passes=follow["K2_passes_three_primary_intervals"],
            prefix_K2_minus_K1_nats=follow["K2_minus_K1_full_patch_nats"],
        )
        if record:
            relevant = entry["task"] == "multihop"
            entry.update(
                queried_record_vs_unqueried_nats=record["mean_nats"] if relevant else "",
                queried_record_vs_unqueried_lower=record["lower"] if relevant else "",
                queried_record_vs_unqueried_upper=record["upper"] if relevant else "",
            )

def summarize_causal(study, behavior):
    path = study / "confirmation" / "causal_rows.jsonl"
    rows = load_lines(path)
    metadata = json.loads((path.parent / "completed.json").read_text())
    frozen = json.loads((study / "site_freeze.json").read_text())
    assert metadata["runner_sha256"] == frozen["runner_sha256"]
    assert metadata["input_sha256"] == frozen["inputs_sha256"]
    assert metadata["valid_causal_rows"] == len(rows) == metadata["causal_rows"]
    final, cross, cases = [], [], []
    for task in TASKS:
        current = [r for r in rows if r["task"] == task and r["status"] == "ok"]
        assert len(current) == 60 and all(r["site"] == frozen["selected"][task] for r in current)
        k_site, l_site = current[0]["site"]
        for k in range(4):
            for name, (a, b) in COMPARISONS.items():
                grouped = clustered(current, a, b, k)
                level = frozen["confirmation_simultaneous_level"] if name in PRIMARY and k == 3 else 0.95
                effect, lower, upper = interval(grouped, level)
                final.append({"task": task, "site_K": k_site, "site_layer_zero_index": l_site,
                              "readout_K": k+1, "comparison": name, "groups": len(grouped),
                              "directions": len(current), "effect_nats": effect, "lower": lower,
                              "upper": upper, "interval_level": level})
        primary = {r["comparison"]: r for r in final if r["task"] == task
                   and r["readout_K"] == 4 and r["comparison"] in PRIMARY}
        probe = next(r for r in final if r["task"] == task and r["readout_K"] == 4
                     and r["comparison"] == "intermediate_minus_random")
        bal = [b for b in behavior if b["task"] == task and b["K"] == 4][0]
        natural_success = [r for r in current if r["natural"]["pred"][3] == r["original"]
                           and r["same_answer_donor_native_predictions"][3] == r["original"]]
        adopted = [r for r in current if r["conditions"]["semantic_same_answer"]["pred"][3] == r["counterfactual"]
                   and r["natural"]["pred"][3] != r["counterfactual"]]
        dose_error = [(r["conditions"][c]["dose"] - r["dose"]) / r["dose"]
                      for r in current for c in ("irrelevant", "random") if r["dose"] > 0]
        cross.append({
            "task": task, "site_K": k_site, "site_layer_zero_index": l_site, "groups": 30,
            "native_k4_correct": bal["correct"], "native_n": bal["base_count"],
            "intermediate_k4_correct": bal["middle_correct"], "intermediate_n": bal["middle_count"],
            "semantic_effect_nats": primary["semantic_minus_natural"]["effect_nats"],
            "sem_minus_irrelevant_nats": primary["semantic_minus_irrelevant"]["effect_nats"],
            "sem_minus_random_nats": primary["semantic_minus_random"]["effect_nats"],
            "passes_three_primary_intervals": all(v["lower"] > 0 for v in primary.values()),
            "intermediate_query_sem_minus_random_nats": probe["effect_nats"],
            "intermediate_query_lower_95": probe["lower"],
            "intermediate_query_upper_95": probe["upper"],
            "strongly_aligned_natural_directions": len(natural_success),
            "adopted_cf_directions": len(adopted),
            "adopted_cf_among_naturally_correct_directions": sum(r in adopted for r in natural_success),
            "max_abs_control_dose_relative_error": max(map(abs, dose_error), default=0),
        })
        # Show a typical magnitude case and, when possible, a naturally correct
        # hard flip. Keep selection criteria explicit and do not infer prevalence.
        key = lambda r: r["conditions"]["semantic_same_answer"]["margin"][3] - r["natural"]["margin"][3]
        ordering = sorted(current, key=key)
        cases.append({"task": task, "kind": "median_ordered_semantic_margin_effect",
                      "row": ordering[len(ordering)//2]})
        diagnostic = [r for r in natural_success if r in adopted]
        if diagnostic:
            cases.append({"task": task, "kind": "largest_effect_with_natural_correctness_and_cf_flip",
                          "row": max(diagnostic, key=key)})
    write_csv(study / "causal_effects.csv", final)
    merge_followup(study, cross)
    write_csv(study / "cross_task_mechanisms.csv", cross)
    write_json(study / "case_witnesses.json", cases)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("freeze", "results"))
    p.add_argument("study", type=Path)
    p.add_argument("--runner", type=Path, default=Path(__file__).with_name("run.py"))
    args = p.parse_args()
    if args.mode == "freeze":
        freeze(args.study, args.runner)
        return
    study = args.study
    natural = study / "trajectory_results" / "predictions.jsonl"
    predictions = load_lines(natural)
    behavior = summarize_behavior(predictions)
    write_csv(study / "trajectory_results" / "behavior.csv", behavior)
    summarize_geometry(study, predictions)
    summarize_semantics(study)
    summarize_causal(study, behavior)


if __name__ == "__main__":
    main()
