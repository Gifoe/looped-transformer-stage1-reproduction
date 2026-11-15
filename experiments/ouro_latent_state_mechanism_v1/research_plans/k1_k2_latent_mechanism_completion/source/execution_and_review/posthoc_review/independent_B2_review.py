"""Supplementary CPU review of complete, already audited B2 splits.

No model imports or intervention selection. Reconstructs the published analysis
from hash-bound original scalar metadata, then reports descriptive binary paths.
The fixed bootstrap is replayed to verify arithmetic, not a new statistical test.
"""
import argparse
import hashlib
import json
import pathlib
import statistics
from fractions import Fraction

import numpy as np


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def checked(root, relative, digest):
    path = (root / relative).resolve()
    assert path.is_relative_to(root) and sha(path) == digest
    return read(path)


def contrasts():
    definitions = {
        "attention": {
            "loss": {"N": 1, "T": -1},
            "restore": {"U_plus": 1, "U": -1},
            "norm_loss": {"N": 1, "upstream_norm": -1},
            "norm_restore": {"U_plus_norm_preserve": 1, "U": -1},
            "specific_loss": {"upstream_unrelated": 1, "T": -1},
            "specific_restore": {"U_plus": 1, "U_plus_unrelated": -1},
        },
        "L18_branch": {},
        "MLP_secondary": {
            "loss": {"N": 1, "M": -1},
            "restore": {"U_M_plus": 1, "U": -1},
            "norm_loss": {"N": 1, "M_norm_preserve": -1},
            "norm_restore": {"U_M_plus_norm_preserve": 1, "U": -1},
            "specific_loss": {"M_unrelated": 1, "M": -1},
            "specific_restore": {"U_M_plus": 1, "U_M_plus_unrelated": -1},
        },
    }
    for branch, restored, removed in (("X", "TX", "TA"), ("A", "TA", "TX")):
        cells = definitions["L18_branch"]
        cells[branch + "_restore"] = {restored: 1, "T": -1}
        cells[branch + "_loss"] = {"N": 1, removed: -1}
        for control in ("norm", "unrelated"):
            cells[branch + "_restore_vs_" + control] = {
                restored: 1, f"T_{branch}_{control}": -1}
            cells[branch + "_loss_vs_" + control] = {
                f"N_{branch}_{control}": 1, removed: -1}
    return definitions


def gaps(values, gold, wrong):
    other = [v for i, v in enumerate(values) if i != gold]
    return {"meanwrong": values[gold] - sum(other) / len(other),
            "fixedwrong": values[gold] - values[wrong],
            "dynamic": values[gold] - max(other)}


def close(actual, expected):
    assert np.isclose(actual, expected, rtol=1e-12, atol=1e-13), (actual, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--stage", required=True,
                        choices=["train64", "train1030", "validation299"])
    args = parser.parse_args()
    root = pathlib.Path(args.root).resolve()
    stage = args.stage
    data = root / "b2_data/treatments" / stage
    analysis = root / "b2_analysis" / stage
    pending = read(data / "PENDING.json")
    audit = read(data / "INDEPENDENT_COVERAGE_AUDIT.json")
    completion = read(analysis / "COMPLETE.json")
    freeze = root / "B2_FULL_SOURCE_FREEZE.json"
    assert pending["status"] == "COMPLETE" and audit["status"] == "PASS"
    assert completion["status"] == "COMPLETE"
    assert audit["pending_sha256"] == completion["pending_sha256"] == sha(data / "PENDING.json")
    assert completion["audit_sha256"] == sha(data / "INDEPENDENT_COVERAGE_AUDIT.json")
    assert pending["source_freeze_sha256"] == sha(freeze)
    frozen = read(freeze)
    assert all(sha(root / p) == h for p, h in frozen["sources"].items())
    labels_path = root / "analysis_inputs/analysis_labels.jsonl"
    labels = {(r["split"], r["id"]): r["gold"] for line in
              labels_path.read_text(encoding="utf-8").splitlines() if (r := json.loads(line))}
    published = [json.loads(line) for line in
                 (analysis / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in published] == [r["id"] for r in pending["cases"]]
    assert len(published) == pending["n"] == len({r["id"] for r in published})
    definitions = contrasts()
    reconstructed = []
    scalar_checks = 0
    for item, existing in zip(pending["cases"], published):
        metadata = checked(data, item["metadata_file"], item["metadata_sha256"])
        ab = checked(root, metadata["AB_metadata_path"], metadata["AB_metadata_sha256"])
        reference = checked(root, metadata["reference_metadata_path"],
                            metadata["reference_metadata_sha256"])
        gold = labels[(metadata["split"], metadata["id"])]
        candidates = {"N": {"status": "OK", "scores": ab["identity"]["N_scores"]},
                      "U": ab["conditions"]["H01"],
                      "K1": {"status": "OK", "scores": ab["identity"]["K1_scores"]},
                      **reference["conditions"], **metadata["conditions"]}
        assert set(candidates) == set(existing["conditions"])
        row = {"id": item["id"], "gold": gold, "conditions": {}, "families": {}}
        for name, value in candidates.items():
            assert value["status"] == existing["conditions"][name]["status"]
            if value["status"] != "OK":
                row["conditions"][name] = {"status": value["status"]}
                continue
            scores = value["scores"]
            raw = scores["candidate_logits"]
            assert len(raw) == len(metadata["targets"]) and len(raw) >= 2
            ties = [i for i, v in enumerate(raw) if v == max(raw)]
            assert ties == scores["ties"] and ties[0] == scores["prediction"]
            assert ties == [i for i, v in enumerate(scores["scores"]) if v == max(scores["scores"])]
            row["conditions"][name] = {
                "status": "OK", "prediction": ties[0], "ties": ties,
                "correct": ties[0] == gold, "candidate_logits": raw,
                "scores": scores["scores"], "full_logits_sha256": value.get("full_logits_sha256")}
            for k in ("prediction", "ties", "correct"):
                assert row["conditions"][name][k] == existing["conditions"][name][k]
        k1, n = row["conditions"]["K1"], row["conditions"]["N"]
        group = ("rescue" if not k1["correct"] and n["correct"] else "damage"
                 if k1["correct"] and not n["correct"] else "stable_correct"
                 if k1["correct"] else "stable_wrong")
        no_tie = len(k1["ties"]) == len(n["ties"]) == 1
        assert (group, no_tie, gold) == (existing["natural_group"], existing["natural_no_tie"], existing["gold"])
        row.update(natural_group=group, natural_no_tie=no_tie)
        wrong = max((i for i in range(len(k1["scores"])) if i != gold),
                    key=lambda i: k1["scores"][i])
        for name, value in row["conditions"].items():
            if value["status"] != "OK":
                continue
            production = gaps(value["scores"], gold, wrong)
            exact = gaps([Fraction(x) for x in value["candidate_logits"]], gold, wrong)
            value["gaps"] = production
            value["fraction_gaps"] = {k: str(v) for k, v in exact.items()}
            assert value["fraction_gaps"] == existing["conditions"][name]["fraction_gaps"]
            for key, val in production.items():
                close(val, existing["conditions"][name]["gaps"][key]); scalar_checks += 1
        for family, contrasts_in_family in definitions.items():
            row["families"][family] = {}
            for name, coefficients in contrasts_in_family.items():
                if any(row["conditions"][k]["status"] != "OK" for k in coefficients):
                    row["families"][family][name] = None
                    assert existing["families"][family][name] is None
                    continue
                values = {}
                for form, key in (("production", "gaps"), ("fraction", "fraction_gaps")):
                    values[form] = {}
                    for metric in ("meanwrong", "fixedwrong", "dynamic"):
                        vals = [(coefficient, row["conditions"][k][key][metric])
                                for k, coefficient in coefficients.items()]
                        total = sum(coef * (Fraction(v) if form == "fraction" else v)
                                    for coef, v in vals)
                        if form == "fraction":
                            assert Fraction(existing["families"][family][name][form][metric]) == total
                            total = str(total)
                        else:
                            close(total, existing["families"][family][name][form][metric]); scalar_checks += 1
                        values[form][metric] = total
                row["families"][family][name] = values
        reconstructed.append(row)
    summary = read(analysis / "summary.json")
    rng = np.random.default_rng(20261114)
    descriptions = {}
    interval_checks = 0
    for group in ("all", "rescue", "damage", "stable_correct", "stable_wrong"):
        for no_tie in (False, True):
            key = group + ("_no_tie" if no_tie else "")
            rr = [r for r in reconstructed if (group == "all" or r["natural_group"] == group)
                  and (not no_tie or r["natural_no_tie"])]
            assert summary[key]["n"] == len(rr)
            for family, cells in definitions.items():
                passes = []
                for name in cells:
                    valid = [r for r in rr if r["families"][family][name] is not None]
                    arr = np.array([r["families"][family][name]["production"]["meanwrong"] for r in valid])
                    cell = summary[key]["families"][family]["contrasts"][name]
                    assert cell["n_total"] == len(rr) and cell["n_defined"] == len(valid)
                    assert cell["full_defined_coverage"] == (len(valid) == len(rr))
                    for label, predicate in (("positive_ids", lambda x: x > 0),
                                             ("strict_counterexample_ids", lambda x: x < 0),
                                             ("zero_ids", lambda x: x == 0)):
                        assert cell[label] == [r["id"] for r in valid if predicate(
                            Fraction(r["families"][family][name]["fraction"]["meanwrong"]))]
                    if len(arr):
                        # Identical predeclared RNG stream and row order. One block
                        # instead of producer's 50 blocks checks batching invariance.
                        boot = arr[rng.integers(0, len(arr), size=(5000, len(arr)))].mean(axis=1)
                        lower = float(np.quantile(boot, .05 / len(cells)))
                        close(cell["mean"], float(arr.mean())); close(cell["lower"], lower)
                        for x, y in zip(cell["ci95"], np.quantile(boot, [.025, .975])):
                            close(x, y)
                        interval_checks += 1
                        nv = [r["families"][family][name]["production"]["meanwrong"]
                              for r in valid if r["natural_no_tie"]]
                        passes.append(bool(len(valid) == len(rr) and arr.mean() > 0
                                           and lower > 0 and nv and statistics.mean(nv) >= 0))
                    else:
                        passes.append(False)
                    assert passes[-1] == cell["positive_pass"]
                assert all(passes) == summary[key]["families"][family]["all_positive_pass"]
            descriptive = {"n": len(rr), "conditions": {}, "binary_paths": {}, "transitions": {}}
            for condition in reconstructed[0]["conditions"]:
                valid = [r for r in rr if r["conditions"][condition]["status"] == "OK"]
                cell = summary[key]["conditions"][condition]
                assert cell["n_defined"] == len(valid)
                for label, predicate in (
                    ("correct_ids", lambda r: r["conditions"][condition]["correct"]),
                    ("tie_ids", lambda r: len(r["conditions"][condition]["ties"]) > 1),
                    ("prediction_changed_from_N_ids", lambda r: r["conditions"][condition]["prediction"] != r["conditions"]["N"]["prediction"]),
                    ("rescued_from_N_ids", lambda r: not r["conditions"]["N"]["correct"] and r["conditions"][condition]["correct"]),
                    ("damaged_from_N_ids", lambda r: r["conditions"]["N"]["correct"] and not r["conditions"][condition]["correct"])):
                    assert cell[label] == [r["id"] for r in valid if predicate(r)]
                descriptive["conditions"][condition] = {"n_defined": len(valid),
                    "n_correct": len(cell["correct_ids"]), "n_tied": len(cell["tie_ids"])}
            transitions = [("T", "TX"), ("T", "TA"), ("N", "TA"), ("N", "TX"),
                           ("N", "T"), ("U", "U_plus"), ("N", "M"), ("U", "U_M_plus")]
            transitions += [("T", f"T_{branch}_{control}") for branch in ("X", "A")
                            for control in ("norm", "unrelated")]
            transitions += [("N", f"N_{branch}_{control}") for branch in ("X", "A")
                            for control in ("norm", "unrelated")]
            for before, after in transitions:
                valid = [r for r in rr if all(r["conditions"][c]["status"] == "OK"
                                              for c in (before, after))]
                descriptive["transitions"][f"{before}->{after}"] = {
                    "n_defined": len(valid),
                    "wrong_to_correct_ids": [r["id"] for r in valid if not r["conditions"][before]["correct"]
                                             and r["conditions"][after]["correct"]],
                    "correct_to_wrong_ids": [r["id"] for r in valid if r["conditions"][before]["correct"]
                                             and not r["conditions"][after]["correct"]],
                    "before_tie_ids": [r["id"] for r in valid if len(r["conditions"][before]["ties"]) > 1],
                    "after_tie_ids": [r["id"] for r in valid if len(r["conditions"][after]["ties"]) > 1]}
            for base, perturb, endpoint in (("N", "T", "TX"), ("N", "T", "TA"),
                                            ("N", "T", "TXA")):
                name = f"{base}_correct__{perturb}_wrong__{endpoint}_correct"
                valid = [r for r in rr if all(r["conditions"][c]["status"] == "OK"
                                              for c in (base, perturb, endpoint))]
                susceptible = [r for r in valid if r["conditions"][base]["correct"]
                               and not r["conditions"][perturb]["correct"]]
                recovered = [r for r in susceptible if r["conditions"][endpoint]["correct"]]
                descriptive["binary_paths"][name] = {
                    "n_defined": len(valid), "susceptible_ids": [r["id"] for r in susceptible],
                    "recovered_ids": [r["id"] for r in recovered],
                    "unrecovered_ids": [r["id"] for r in susceptible if not r["conditions"][endpoint]["correct"]],
                    "endpoint_tie_ids": [r["id"] for r in susceptible if len(r["conditions"][endpoint]["ties"]) > 1],
                    "interpretation": "TXA is structural identity; U endpoints use a different background. Descriptive joint events, not an explanatory fraction or a new confirmatory test."}
            descriptions[key] = descriptive
    output = root / "posthoc_review" / stage
    output.mkdir(parents=True, exist_ok=False)
    (output / "compact_scalar_evidence.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in reconstructed), encoding="utf-8")
    (output / "BINARY_PATHS.json").write_text(json.dumps(descriptions, indent=2), encoding="utf-8")
    receipt = {"status": "PASS", "stage": stage, "n": len(reconstructed),
               "scalar_checks": scalar_checks, "bootstrap_interval_checks": interval_checks,
               "source_sha256": {str(p.relative_to(root)): sha(p) for p in
                    (data / "PENDING.json", data / "INDEPENDENT_COVERAGE_AUDIT.json",
                     analysis / "rows.jsonl", analysis / "summary.json", analysis / "COMPLETE.json",
                     labels_path, freeze)}, "script_sha256": sha(__file__),
               "scope": "Independent scalar metadata, all-group arithmetic and original bootstrap replay. Prior full tensor audit required; no second model execution. Binary events descriptive; TXA identity is not mediation percentage."}
    assert all(sha(root / p) == h for p, h in receipt["source_sha256"].items())
    (output / "INDEPENDENT_REVIEW.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
