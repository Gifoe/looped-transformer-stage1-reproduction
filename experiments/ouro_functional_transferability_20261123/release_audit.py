"""Check declared primary coverage, forecast locks, raw rows and the GPU cap."""

from hashlib import file_digest
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def rows(path):
    return [json.loads(r) for r in path.read_text(encoding="utf-8").splitlines()]


def check_ids(path, n):
    data = rows(path)
    ids = {r["id"] for r in data}
    assert len(data) == len(ids) == n, path
    return ids


def check_ledger(path):
    started = None
    elapsed = 0
    stages = {}
    last = 0
    for line in path.read_text().splitlines():
        action, stage, raw_start, *extra = line.split("\t")
        start = int(raw_start)
        if action == "START":
            assert started is None and start >= last and int(extra[0]) == 28800 - elapsed
            started = stage, start
        else:
            end, duration, status = map(int, extra)
            assert action == "END" and started == (stage, start) and end - start == duration >= 0
            elapsed += duration
            stages.setdefault(stage, []).append(dict(seconds=duration, return_code=status))
            last, started = end, None
    assert started is None and elapsed <= 28800
    for stage in ("preflight", "arc_moments", "csqa_moments", "depth_train", "position_train", "csqa_train",
                  "forecasts", "depth_confirm", "arc_csqa_confirm", "source_confirm", "position_confirm",
                  "state_horizon_confirm", "transport_confirm", "r2_forecasts", "gamma_confirm", "r2_greedy"):
        assert any(item["return_code"] == 0 for item in stages.get(stage, [])), stage
    return dict(seconds=elapsed, hard_cap_seconds=28800, stages=stages)


def main(root):
    results = root / "results"
    assert len(read(results/"PREFLIGHT.json")["cells"]) == 6
    forecast_file = results/"forecast/FORECAST_LOCK.json"
    forecast = read(forecast_file)
    assert sha(results/"forecast/PREDICTIONS.json") == forecast["predictions_sha256"]
    assert read(results/"forecast/PREDICTIONS.json")["protocol"]["source_code_sha256"] == sha(root/"theory_prediction_round1.py")
    round2_file = results/"forecast_round2/FORECAST_LOCK.json"
    round2 = read(round2_file)
    assert sha(results/"forecast_round2/PREDICTIONS.json") == round2["predictions_sha256"]
    assert read(results/"forecast_round2/PREDICTIONS.json")["protocol"]["source_code_sha256"] == sha(root/"theory_prediction.py")
    single=read(results/"forecast_round2/SINGLE_CONTROLLER.json")
    assert single["single_checkpoint"] and single["parameters"]==4096 and not single["selection_uses_confirmation"]
    assert single["checkpoint_sha256"] in round2["controller_sha256s"]
    assert sha(root/single["checkpoint"]) == single["checkpoint_sha256"]
    core = ("arc_train", "csqa_train", "arc_discovery", "csqa_discovery", "arc_confirm", "csqa_confirm")
    sets = {name:check_ids(root/"data"/f"{name}.jsonl", 1119 if name.endswith("train") else
                          200 if name.endswith("discovery") else 299) for name in core}
    for i, a in enumerate(core):
        for b in core[i+1:]:
            assert not (sets[a] & sets[b]), (a, b)
    for name in ("arc_confirm", "csqa_confirm"):
        assert sha(root / "data" / f"{name}.jsonl") == forecast["evaluation_sha256s"][name]
    for name, size in (("arc_round2_confirm", 70), ("csqa_round2_confirm", 299)):
        path = root / "data" / "optional" / f"{name}.jsonl"
        assert sha(path) == round2["evaluation_sha256s"][name]
        sets[name] = check_ids(path, size)
        assert all(not (sets[name] & sets[other]) for other in sets if name != other)
    trained = {
        "arc/depth": {"k2": {2}, "k3": {3}, "k4": {4}, "k23": {2,3}, "k24": {2,4}, "k234": {2,3,4}},
        "arc/position": {f"p{k}": {4} for k in (1, 2, 3)},
        "csqa/source": {"k234": {2,3,4}},
    }
    base_root = results/"HOLDOUT"
    training_summary = {}
    for relative, groups in trained.items():
        here = base_root/relative/"s261123137"
        initial = set()
        for name, admitted in groups.items():
            directory = here/name
            receipt = read(directory/"TRAIN.json")
            assert {r["depth"] for r in receipt["step_records"]} == admitted
            assert len(receipt["step_records"]) == receipt["steps"] == 282
            assert receipt["checkpoint_sha256"] == sha(directory/"controller.pt")
            assert receipt["checkpoint_sha256"] in forecast["controller_sha256s"]
            assert receipt["rank_one_parameters"] == 4096 and receipt["actual_unique_questions"] == 1119
            initial.add(receipt["initial_state_sha256"])
        assert len(initial) == 1
        training_summary[relative] = next(iter(initial))
    assert len(set(training_summary.values())) == 1
    checks = {}
    for report in sorted(base_root.glob("**/confirm/**/SUMMARY.json")):
        details = read(report)
        lock = details["forecast"]
        assert len(lock["predictions_sha256"]) == 64 and report.stat().st_mtime >= forecast_file.stat().st_mtime
        if report.parent.name == "r2":
            assert lock["predictions_sha256"] == round2["predictions_sha256"]
            assert lock["lock_sha256"] == sha(round2_file) and report.stat().st_mtime >= round2_file.stat().st_mtime
        else:
            assert lock["predictions_sha256"] == forecast["predictions_sha256"] and lock["lock_sha256"] == sha(forecast_file)
        for name, endpoint in details["evaluation"].items():
            raw = report.parent/f"{name}_PREDICTIONS.jsonl"
            assert sha(raw) == endpoint["complete_candidate_scores_sha256"]
            raw_ids = check_ids(raw, endpoint["n"])
            if name in sets:
                assert raw_ids == sets[name]
        checks[str(report.relative_to(results))] = {name:len(x["results"]) for name,x in details["evaluation"].items()}
    assert len(checks) == 14, sorted(checks)
    horizon = results / "state_horizon/arc_confirm"
    for cell, item in read(horizon/"STATE_HORIZON.json")["conditions"].items():
        raw = horizon/f"{cell}.jsonl"
        assert sha(raw) == item["rows_sha256"] and check_ids(raw, 96) <= sets["arc_confirm"]
    transport = results / "D_TRANSPORT/arc_confirm"
    assert check_ids(transport/"PREDICTIONS.jsonl", 299) == sets["arc_confirm"]
    assert (transport/"TRANSPORT_SUMMARY.json").stat().st_mtime >= forecast_file.stat().st_mtime
    assert read(results/"PREDICTION_AUDIT/AUDIT.json")["all"]["n_conditions"] >= 61
    assert read(results/"ROUND2_AUDIT/RESULTS.json")["cell_count"] == 100
    gamma = results/"GAMMA"
    probe = read(gamma/"PROBE_LOCK.json")
    assert sha(gamma/"GAMMA.pt") == probe["controller_sha256"]
    assert sha(root/"gamma_intervention.py") == probe["code_sha256"]
    assert read(gamma/"SUMMARY.json")["lock_sha256"] == sha(gamma/"PROBE_LOCK.json")
    for name in ("arc_round2_confirm", "csqa_round2_confirm"):
        for k in range(1, 7):
            raw = gamma/name/f"K{k}.jsonl"
            assert sha(raw) == read(raw.with_suffix(".json"))["sha256"]
            assert check_ids(raw, len(sets[name])) == sets[name]
    generation = results/"TRANSFER_GENERATION"
    g = read(generation/"GENERATION_SUMMARY.json")
    assert g["all_cohorts_complete"] and g["lock_sha256"] == sha(generation/"GENERATION_LOCK.json")
    assert len(list(generation.glob("*__k4.jsonl"))) == 8
    for dataset in ("arc_round2_confirm", "csqa_round2_confirm"):
        for raw in generation.glob(f"{dataset}__*__k4.jsonl"):
            assert sha(raw) == read(raw.with_suffix(".json"))["prediction_sha256"]
            assert check_ids(raw, len(sets[dataset])) == sets[dataset]
    evidence = dict(ledger=check_ledger(root/"GPU_LEDGER.tsv"), seed_initial_state_sha256=training_summary,
                    forecast_sha256=sha(forecast_file), core_prediction_files=checks, primary_data_population=sorted(sets))
    (root/"RELEASE_AUDIT.json").write_text(json.dumps(evidence, indent=2)+"\n", encoding="utf-8")
    print("PASS", len(checks), "confirmation receipts and", evidence["ledger"]["seconds"], "GPU seconds", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
