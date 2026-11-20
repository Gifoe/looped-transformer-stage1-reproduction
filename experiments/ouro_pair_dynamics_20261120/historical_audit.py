"""Recount the historical open new-structure K1-to-K4 contrast without a model."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess


REVISION = "bfca32210a0d08d6bc03f070f69e94212ffb88f8"
ARCHIVE = "experiments/ouro_hidden_computation_20261120/results/mechanism/"
DIAGONALS = ((1, 1), (2, 2))
ARMS = ("tableau", "random", "same_y_mapping_control", "oracle_joint_mapping", "whole_boundary", "natural")


def read_blob(repo, name):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{REVISION}:{ARCHIVE}{name}"])


def unique_argmax(values):
    high = max(values)
    return values.index(high) if values.count(high) == 1 else -1


def bootstrap_delta(group_deltas):
    rng = random.Random(20261120)
    estimates = [statistics.mean(rng.choices(group_deltas, k=len(group_deltas))) for _ in range(20000)]
    quantiles = statistics.quantiles(estimates, n=40, method="inclusive")
    return [statistics.mean(group_deltas), quantiles[0], quantiles[-1]]


def audit(repo):
    names = ("causal.jsonl", "natural.jsonl", "mechanism_tableaux.jsonl", "mechanism_summary.json")
    blobs = {name: read_blob(repo, name) for name in names}
    records = {name: list(map(json.loads, blobs[name].splitlines())) for name in names[:-1]}
    groups = {row["id"]: row for row in records["mechanism_tableaux.jsonl"] if row["split"] == "new_structure"}
    assert len(groups) == 48 and all(row["structure"] for row in groups.values())

    def select(rows, kind, loop):
        selected = {}
        for row in rows:
            if (row["id"] not in groups or row["fork"] != "open" or row["kind"] != kind
                    or row["loop"] != loop):
                continue
            key = (row["id"], row["s"], row["m"], row["arm"] or "natural")
            assert key not in selected
            assert row["y"] == (row["s"] + row["m"]) % 3
            assert [unique_argmax(d[0]) for d in row["scores"]] == row["pred_content"]
            assert row["pred_content"] == row["pred"]
            selected[key] = row
        return selected

    outcomes = Counter()
    greedy = Counter()
    ties = Counter()
    paired_tables = {arm: Counter() for arm in ARMS[1:]}
    differences = {arm: [] for arm in ARMS[1:]}
    margin_differences = {name: [] for name in ("base", "random", "same_y_mapping_control", "source_only", "rule_only")}
    whole_layout_discrepancy = Counter()
    roles = Counter()
    indexes = select(records["causal.jsonl"], "interchange", 1) | select(records["natural.jsonl"], "natural", None)
    for group_id in sorted(groups):
        group_deltas = Counter()
        group_margin_deltas = Counter()
        receiver = indexes[group_id, 0, 0, "natural"]
        source, mapping = (receiver["roles"][name] for name in ("source", "mapping"))
        roles[len(source), len(mapping)] += 1
        assert not (set(source) & set(mapping))
        worlds = {(world["s"], world["m"]): world["y"] for world in groups[group_id]["worlds"]}
        for s, m in DIAGONALS:
            gold = worlds[s, m]
            assert gold not in {worlds[0, 0], worlds[s, 0], worlds[0, m]}
            natural = indexes[group_id, s, m, "natural"]
            assert natural["roles"] == receiver["roles"]
            assert len(natural["ids"]) == len(receiver["ids"])
            assert all(a == b or i in source or i in mapping
                       for i, (a, b) in enumerate(zip(natural["ids"], receiver["ids"], strict=True)))
            hits = {}
            competitor = worlds[s, 0]

            def margin(row):
                scores = row["scores"][3][0]
                return scores[gold] - (scores[worlds[0, 0]] + scores[competitor]) / 2

            for arm in ARMS:
                row = indexes[group_id, s, m, arm]
                assert row["y"] == gold
                assert row["content_tokens"] == receiver["content_tokens"]
                scores = row["scores"][3][0]
                assert len(scores) == 5
                pred = unique_argmax(scores)
                hits[arm] = pred == gold
                outcomes[arm] += hits[arm]
                ties[arm] += pred == -1
                greedy[arm] += row["top_token"][3] == row["content_tokens"][gold]
            joint_margin = margin(indexes[group_id, s, m, "tableau"])
            comparators = {
                "base": indexes[group_id, 0, 0, "self"],
                "random": indexes[group_id, s, m, "random"],
                "same_y_mapping_control": indexes[group_id, s, m, "same_y_mapping_control"],
                "source_only": indexes[group_id, s, 0, "tableau"],
                "rule_only": indexes[group_id, 0, m, "tableau"],
            }
            for name, row in comparators.items():
                group_margin_deltas[name] += joint_margin - margin(row)
            whole = indexes[group_id, s, m, "whole_boundary"]
            for depth in range(4):
                whole_layout_discrepancy[f"K{depth+1}_different_scores"] += whole["scores"][depth][0] != natural["scores"][depth][0]
                whole_layout_discrepancy[f"K{depth+1}_different_pred"] += whole["pred"][depth] != natural["pred"][depth]
            for arm in ARMS[1:]:
                group_deltas[arm] += int(hits["tableau"]) - int(hits[arm])
                paired_tables[arm][f"joint_{int(hits['tableau'])}_other_{int(hits[arm])}"] += 1
        for arm in ARMS[1:]:
            differences[arm].append(group_deltas[arm] / 2)
        for name, delta in group_margin_deltas.items():
            margin_differences[name].append(delta / 2)

    summary = json.loads(blobs["mechanism_summary.json"])["interchanges"]
    assert outcomes["tableau"] == 63 and outcomes["natural"] == 56
    for arm in ARMS:
        expected = summary[f"new_structure/open/1/{arm}_hit"]
        assert expected["n"] == len(groups) and outcomes[arm] == expected["mean"][3] * 96
    for name, deltas in margin_differences.items():
        expected = summary[f"new_structure/open/1/tableau_vs_{name}"]["mean"][3]
        assert abs(statistics.mean(deltas) - expected) < 1e-12
    return {
        "revision": REVISION,
        "sha256": {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()},
        "groups": len(groups),
        "targets": len(groups) * len(DIAGONALS),
        "roles": {f"source_{s}_mapping_{m}": count for (s, m), count in roles.items()},
        "five_candidate_hits": outcomes,
        "five_candidate_ties": ties,
        "full_vocab_greedy_hits": greedy,
        "whole_boundary_vs_natural": dict(whole_layout_discrepancy),
        "joint_vs_control_paired_tables": paired_tables,
        "joint_minus_control_mean_lo_hi": {arm: bootstrap_delta(diff) for arm, diff in differences.items()},
        "joint_margin_minus_control_mean_nats": {name: statistics.mean(diff) for name, diff in margin_differences.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path, nargs="?", default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(audit(args.repository), indent=2))
