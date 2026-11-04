"""Independent postrun R25 score/statistic/provenance review; never loads a model."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def lines(path):
    return [json.loads(s) for s in (ROOT / path).read_text(encoding="utf-8").splitlines() if s.strip()]


def paired(new, old, gold):
    good_new, good_old = np.asarray(new) == gold, np.asarray(old) == gold
    rescue = int((good_new & ~good_old).sum())
    damage = int((~good_new & good_old).sum())
    n = rescue + damage
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(rescue, damage) + 1)) / 2 ** n) if n else 1.0
    return dict(rescues=rescue, damages=damage, net=rescue-damage, exact_mcnemar_p=p)


def main():
    lock_path = "configs/r25_HELLINGER_MEMORY_LOCK.json"
    lock = read(lock_path)
    assert sha(lock_path) == "a2ba1d236eb1ace9d60af16e2b574dfb018c494a7496c87199f8e41f5c9d8641"
    checked = {}
    for key in ["source_sha256", "inherited_source_sha256", "input_sha256"]:
        for path, expected in lock[key].items():
            path = "scripts/" + path if key != "input_sha256" else path
            assert sha(path) == expected, path
            checked[path] = expected
    complete = read("results/r25_COMPLETE.json")
    supervisor = read("logs/r25_SUPERVISOR.json")
    book = read("logs/r25_gpu_budget.json")
    analysis = read("results/r25_ANALYSIS.json")
    integrity = read("results/r25_LABEL_FREE_INTEGRITY.json")
    input_audit = read("results/r25_INPUT_AUDIT.json")
    key_audit = read("results/r25_KEY_AUDIT.json")
    assert complete["status"] == supervisor["status"] == "complete"
    assert complete["n"] == 24 and supervisor["exit_code"] == 0
    assert not supervisor["overrun"] and not supervisor.get("timed_out", False)
    assert 0 < supervisor["gpu_worker_seconds"] <= 300
    assert complete["all_parameter_hashes_before"] and complete["all_parameter_hashes_before"] == complete["all_parameter_hashes_after"]
    assert not complete["weights_modified"]
    for key, path in [("protocol_sha256", lock_path), ("score_file_sha256", "results/r25_pilot.jsonl"), ("key_audit_sha256", "results/r25_KEY_AUDIT.json"), ("input_audit_sha256", "results/r25_INPUT_AUDIT.json")]:
        assert complete[key] == sha(path)
    assert analysis["protocol_sha256"] == sha(lock_path) and analysis["scores_sha256"] == sha("results/r25_pilot.jsonl")
    assert analysis["label_free_integrity"] == integrity and integrity["status"] == "passed_complete_label_free_integrity"
    assert input_audit["n"] == 24 and input_audit["all143value_ids_offsets_predecessors_exact"] and not input_audit["model_or_key_or_query_loaded"]
    assert key_audit["key_input_ids_sha256"] == input_audit["key_input_ids_sha256"]
    assert key_audit["selected_vs_full_native4_BF16_states_exact"] and key_audit["selected_vs_full_native4_head_fullV_exact"]
    inherited = lock["budget"]["inherited_ledger_sha256"]
    assert len(inherited) == 13 and all(sha(path) == expected for path, expected in inherited.items())
    old_seconds = sum(read(path)["used_seconds"] for path in inherited)
    assert abs(old_seconds - 18641.79952767305) < 1e-9
    assert book["used_seconds"] == supervisor["gpu_worker_seconds"]
    assert book["total_used_seconds"] == old_seconds + book["used_seconds"] <= 21600
    rows = lines("results/r25_pilot.jsonl")
    assert [r["id"] for r in rows] == [r["id"] for r in lock["cohort"]] and len(rows) == 24
    memory = read("configs/r25_MEMORY_MANIFEST.json")
    values = np.asarray(memory["values"], dtype=np.int64)
    vocab = memory["vocabulary_size"]
    assert len(values) == 143 and vocab == 49152
    raw = np.random.default_rng(2021).permutation(143)
    counts = np.bincount(values, minlength=vocab)
    tau = np.arange(vocab)
    rng = np.random.default_rng(2021)
    for frequency in sorted(set(counts[counts > 0].tolist())):
        ids = np.flatnonzero(counts == frequency)
        tau[ids] = rng.permutation(ids)
    assert raw.tolist() == memory["nulls"]["raw_occurrence_permutation"]
    assert tau.tolist() == memory["nulls"]["tau"]
    assert np.array_equal(np.sort(tau), np.arange(vocab))
    assert np.array_equal(np.bincount(values[raw], minlength=vocab), counts)
    assert np.array_equal(np.bincount(tau[values], minlength=vocab), counts)
    authority = read("configs/r25_PARITY_AUTHORITY.json")["rows"]
    readouts = []; distance_checks = []; total_positions = 0; zeros = 0; unmatched = 0; norm_error = 0.0
    raw_control_zeros = 0
    for row in rows:
        historical = authority[row["id"]]
        assert row["original_baseline_vectors_parity_error"] == 0
        for j, candidate in enumerate(row["candidates"]):
            n = candidate["tokens"]; total_positions += n
            assert candidate["choice"] == j and candidate["baseline_parity_error"] == 0
            assert candidate["characters"] > 0 and n > 0
            assert all(math.isfinite(x) for x in candidate["scores"].values())
            for name, observed in candidate["scores"].items():
                assert observed == candidate["sums"][name] / candidate["characters"] + candidate["late"]
            for name, old in [("M2", "M2_N0_token"), ("M3", "M3_multi_token")]:
                assert candidate["scores"][name] == historical["method_scores"][old][j]
                assert candidate["metrics"][name + "_alpha"] == historical["token_diagnostics"][j][old]["alpha"]
            assert candidate["scores"]["frequency"] == candidate["scores"]["M3"]
            for name, number in candidate["original_sums"].items():
                assert number == historical["original_methods"][name]["sum_scores"][j]
            retrieved = candidate["retrieval"]
            assert [r["context"] for r in retrieved] == ["Q", "N0", "N1", "N2"]
            neighbors = np.asarray([r["neighbor_occurrences"] for r in retrieved], dtype=int)
            assert neighbors.shape == (4, n) and (neighbors >= 0).all() and (neighbors < 143).all()
            observed_values = values[neighbors]
            diag = candidate["metrics"]
            for t in range(n):
                coefficients = {}
                for token, weight in zip(observed_values[:, t], [3, -1, -1, -1]):
                    coefficients[int(token)] = coefficients.get(int(token), 0) + weight
                nz = sum(weight != 0 for weight in coefficients.values())
                assert diag["c_nonzero_coordinates"][t] == nz
                assert diag["c_zero"][t] == diag["tau_c_zero"][t] == (nz == 0)
                assert diag["query_reference_same_value"][t] == int((observed_values[1:, t] == observed_values[0, t]).sum())
                shuffled_values = values[raw][neighbors[:, t]]
                raw_coefficients = {}
                for token, weight in zip(shuffled_values, [3, -1, -1, -1]):
                    raw_coefficients[int(token)] = raw_coefficients.get(int(token), 0) + weight
                assert diag["raw_c_zero"][t] == all(x == 0 for x in raw_coefficients.values())
            zeros += sum(diag["c_zero"]); raw_control_zeros += sum(diag["raw_c_zero"]); unmatched += sum(diag["unmatched"])
            valid = ~np.asarray(diag["unmatched"], dtype=bool)
            errors = np.abs(np.asarray(diag["main_P_norm2"]) - diag["matched_P_norm2"])
            norm_error = max(norm_error, float(errors[valid].max(initial=0)))
            for retrieval in retrieved:
                if "first_numpy_check" in retrieval:
                    check = retrieval["first_numpy_check"]
                    a, b = np.asarray(check["distances_Torch"]), np.asarray(check["distances_NumPy"])
                    assert a.shape == b.shape == (143,) and np.isfinite(a).all() and np.isfinite(b).all()
                    assert np.max(np.abs(a-b)) <= 1e-12
                    assert int(np.argmin(a)) == int(np.argmin(b)) == retrieval["neighbor_occurrences"][0]
                    distance_checks.append(dict(context=retrieval["context"], max_error=float(np.max(np.abs(a-b)))))
            readouts.extend(candidate["readout_parity"])
    assert len(readouts) == len(distance_checks) == 4 and norm_error <= 1e-10
    assert zeros == integrity["zero_c_positions"] and total_positions == integrity["positions"] and unmatched == integrity["unmatched_positions"]
    # Labels enter this independent review only after the checks above.
    history = {r["id"]: r for split in ["discovery", "selection"] for r in lines("results/r7_" + split + "_development.jsonl")}
    names = list(rows[0]["candidates"][0]["scores"]) + ["R6", "Fixed", "A1"]
    predictions = {name: [] for name in names}; gold = []; cases = []
    for row in rows:
        old = history[row["id"]]; gold.append(old["gold"])
        scores = {name: [c["scores"][name] for c in row["candidates"]] for name in names if name not in ["R6", "Fixed", "A1"]}
        for name, old_name in [("R6", "M0_R6"), ("Fixed", "fixed"), ("A1", "A1")]:
            scores[name] = old["baseline_scores"][old_name]
        for name in names:
            predictions[name].append(int(np.argmax(scores[name])))
        p, r, y = predictions["primary"][-1], predictions["R6"][-1], gold[-1]
        if p != r:
            cases.append(dict(id=row["id"], gold=y, primary_prediction=p, R6_prediction=r,
                              rescue=p == y and r != y, damage=p != y and r == y))
    y = np.asarray(gold)
    correct = {k: int((np.asarray(v) == y).sum()) for k, v in predictions.items()}
    comparisons = {k: paired(predictions["primary"], v, y) for k, v in predictions.items() if k != "primary"}
    assert correct == analysis["counts"]
    for name, p in comparisons.items():
        for field in ["rescues", "damages", "net"]:
            assert p[field] == analysis["paired"][name][field]
    unique = (np.asarray(predictions["R6"]) == y) & (np.asarray(predictions["Fixed"]) != y)
    retained = int((unique & (np.asarray(predictions["primary"]) == y)).sum())
    assert retained == analysis["R6_unique_retained"] and int(unique.sum()) == analysis["R6_unique_total"]
    strong = all(analysis["gates"].values())
    cost = unmatched == 0 and comparisons["R6"]["net"] >= 0 and retained == int(unique.sum())
    assert strong == analysis["original_strong_screen_passed"] == analysis["eligible_for_feasibility_review"]
    assert cost == analysis["cost_review_eligible"] and not analysis["automatic_extension_authorized"]
    result = dict(status="passed_independent_postrun_review", protocol_sha256=sha(lock_path),
                  n=24, checked_source_input_hashes=len(checked), counts=correct, paired=comparisons,
                  changed_vs_R6_cases=cases, prediction_disagreements_vs_primary={k: int((np.asarray(v) != predictions["primary"]).sum()) for k, v in predictions.items()},
                  R6_unique_total=int(unique.sum()), R6_unique_retained=retained,
                  total_answer_positions=total_positions, zero_c_positions=zeros, raw_control_zero_positions=raw_control_zeros,
                  unmatched_positions=unmatched, P_norm_max_error=norm_error, readout_checks=len(readouts), independent_recorded_distance_rechecks=distance_checks,
                  null_map_and_all_position_sparse_direction_reconstruction=True,
                  original_strong_screen_passed=strong, cost_review_eligible=cost,
                  gpu_worker_seconds=book["used_seconds"], prior_gpu_worker_seconds=old_seconds,
                  total_gpu_worker_seconds=book["total_used_seconds"], remaining_seconds=21600-book["total_used_seconds"],
                  component_seconds=complete["component_seconds"], model_load_seconds=complete["model_load_seconds"],
                  frozen_parameter_count=len(complete["all_parameter_hashes_before"]),
                  scope="Independent local provenance, raw-score arithmetic, null maps, sparse direction and statistics. No local model run or independent regeneration of fullV key/query vectors.",
                  heldout_accessed=False, new_gpu_jobs_authorized=False)
    path = ROOT / "results/r25_ROOT_VERIFICATION.json"
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "n", "counts", "original_strong_screen_passed", "cost_review_eligible", "gpu_worker_seconds", "total_gpu_worker_seconds"]}))


if __name__ == "__main__":
    main()
