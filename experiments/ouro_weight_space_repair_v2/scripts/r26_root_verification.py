"""Independent NumPy review of all R26 saved vectors and fixed comparison domains."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "03ef1e4369ef74b04c75b65bd30f8b2528603e1d21dec0828184d26433b1dcbb"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def category(items):
    def all_exact(names):
        subset = [x for x in items if x["variant"] in names]
        assert subset
        return all(x["state_exact"] and x["head_exact"] for x in subset)
    repeat = all_exact(["A_repeat"])
    content = all_exact(["B1", "B2"])
    length = all_exact(["C1", "C2"])
    result = ("inconclusive_repeat" if not repeat else
              "content_invariance_not_established" if not content else
              "length_bitwise_sensitive" if not length else "finite_config_prefix_evidence")
    return dict(category=result, A_repeat_all_exact=repeat, A_B_all_exact=content, A_C_all_exact=length)


def main():
    lock_path = "configs/r26_CAUSAL_ACCEPTANCE_LOCK.json"
    assert sha(lock_path) == PROTOCOL
    lock = read(lock_path)
    for path, expected in lock["source_sha256"].items():
        assert "\\" not in path and sha(path) == expected
    ledgers = lock["budget"]["inherited_ledger_sha256"]
    assert len(ledgers) == 14 and all(sha(p) == h for p, h in ledgers.items())
    old = sum(read(p)["used_seconds"] for p in ledgers)
    assert old == 18819.051603254396
    complete = read("results/r26_COMPLETE.json")
    supervisor = read("logs/r26_SUPERVISOR.json")
    book = read("logs/r26_gpu_budget.json")
    acceptance = read("results/r26_ACCEPTANCE.json")
    assert complete["status"] == supervisor["status"] == "complete"
    assert supervisor["exit_code"] == 0 and not supervisor["overrun"] and not supervisor.get("timed_out", False)
    actual = supervisor["gpu_worker_seconds"]
    assert 0 < actual <= 120 and complete["contexts"] == 5 and complete["forwards"] == 30
    assert complete["protocol_sha256"] == PROTOCOL
    assert complete["all_parameter_hashes_before"] and complete["all_parameter_hashes_before"] == complete["all_parameter_hashes_after"]
    assert complete["all_parameter_hashes_before"] == read("results/r25_COMPLETE.json")["all_parameter_hashes_before"]
    assert book["used_seconds"] == actual and book["inherited_total_seconds"] == old
    assert book["total_used_seconds"] == old + actual <= 21600
    for field, path in [("raw_sha256", "results/r26_RAW.json"), ("vectors_sha256", "results/r26_VECTORS.npz"), ("input_audit_sha256", "results/r26_INPUT_AUDIT.json")]:
        assert complete[field] == sha(path)
    audit = read("results/r26_INPUT_AUDIT.json")
    assert audit == read("results/r26_CPU_INPUT_PREFLIGHT.json")
    rows = read("results/r26_RAW.json")["records"]
    assert [r["context"] for r in rows] == ["Q", "N0", "N1", "N2", "key"]
    measurements = []; used = set(); vectors = {}
    with np.load(ROOT / "results/r26_VECTORS.npz", allow_pickle=False) as container:
        for row, plan in zip(rows, audit["plan"]["contexts"]):
            depth = 5 if row["context"] == "Q" else 4
            cuts = plan["cuts"]; tokens = plan["tokens"]
            assert row["complete"] and row["depth"] == depth and row["cuts"] == cuts
            assert len(row["inputs"]) == 6 and 0 < cuts[0] < cuts[1] < len(tokens)-1
            expected_inputs = [
                ("A", tokens, cuts),
                ("B1", tokens[:cuts[0]+1] + [(x+1) % 49152 for x in tokens[cuts[0]+1:]], cuts),
                ("B2", tokens[:cuts[1]+1] + [(x+1) % 49152 for x in tokens[cuts[1]+1:]], cuts),
                ("A_repeat", tokens, cuts),
                ("C1", tokens[:cuts[0]+1], [cuts[0]]),
                ("C2", tokens[:cuts[1]+1], [cuts[1]])]
            for observed, (name, ids, positions) in zip(row["inputs"], expected_inputs):
                assert observed == dict(variant=name, input_ids=ids, input_sha256=digest(ids), shape=[1, len(ids)], positions=positions)
            expected_domain = {(name, k, pos) for name, positions in [("A_repeat", cuts), ("B1", cuts[:1]), ("B2", cuts[1:]), ("C1", cuts[:1]), ("C2", cuts[1:])] for k in range(1, depth+1) for pos in positions}
            observed_domain = [(x["variant"], x["depth"], x["position"]) for x in row["comparisons"]]
            assert len(observed_domain) == len(set(observed_domain)) == len(expected_domain) == 6*depth
            assert set(observed_domain) == expected_domain
            for pair in row["comparisons"]:
                item = dict(context=row["context"], variant=pair["variant"], depth=pair["depth"], position=pair["position"])
                for domain in ["state", "head"]:
                    metric = pair[domain]; shape = [1, 1, 2048 if domain == "state" else 49152]
                    dtype = "torch.bfloat16" if domain == "state" else "torch.float32"
                    assert metric["shape"] == shape and metric["dtype"] == dtype
                    arrays = []
                    for side in ["A", "B"]:
                        key = metric["vector_" + side]; used.add(key)
                        if key not in vectors:
                            vectors[key] = container[key]
                        a = vectors[key]
                        assert a.shape == tuple(shape) and a.dtype == np.dtype("uint16" if domain == "state" else "float32")
                        assert hashlib.sha256((dtype + str(tuple(shape))).encode() + a.tobytes()).hexdigest() == key
                        assert hashlib.sha256(a.tobytes()).hexdigest() == metric["sha256_" + side]
                        arrays.append(a)
                    a, b = arrays
                    bits_a = a if domain == "state" else a.view(np.uint32)
                    bits_b = b if domain == "state" else b.view(np.uint32)
                    unequal = int(np.count_nonzero(bits_a != bits_b))
                    if domain == "state":
                        a = (a.astype(np.uint32) << 16).view(np.float32)
                        b = (b.astype(np.uint32) << 16).view(np.float32)
                    assert np.isfinite(a).all() and np.isfinite(b).all()
                    delta = a.astype(np.float64) - b.astype(np.float64)
                    maximum = float(np.max(np.abs(delta)))
                    rms = float(np.sqrt(np.mean(delta*delta, dtype=np.float64)))
                    exact = unequal == 0
                    assert metric["exact"] == exact and metric["unequal_elements"] == unequal and metric["max_abs"] == maximum
                    assert np.isfinite(metric["RMS"]) and 0 <= metric["RMS"] <= maximum
                    item[domain + "_exact"] = exact
                    item[domain + "_max_abs"] = maximum
                    item[domain + "_RMS_CPU_FP64"] = rms
                    item[domain + "_RMS_worker_difference"] = abs(rms - metric["RMS"])
                    item[domain + "_unequal_elements"] = unequal
                measurements.append(item)
        assert used == set(container.files) and len(used) == complete["unique_vectors"]
    assert len(measurements) == 126
    overall = category(measurements)
    for k, v in overall.items():
        assert acceptance[k] == v
    by_context = {name: category([x for x in measurements if x["context"] == name]) for name in ["Q", "N0", "N1", "N2", "key"]}
    for name, result in by_context.items():
        for k, v in result.items():
            assert acceptance["per_context"][name][k] == v
    summary = {}
    for group, variants in [("repeat", ["A_repeat"]), ("content", ["B1", "B2"]), ("length", ["C1", "C2"])]:
        subset = [x for x in measurements if x["variant"] in variants]
        assert len(subset) == 42
        summary[group] = dict(pairs=len(subset), state_exact_pairs=sum(x["state_exact"] for x in subset), fullV_exact_pairs=sum(x["head_exact"] for x in subset), max_state_difference=max(x["state_max_abs"] for x in subset), max_fullV_difference=max(x["head_max_abs"] for x in subset))
    out = dict(status="passed_independent_saved_vector_review", protocol_sha256=PROTOCOL,
               category=overall, per_context=by_context, summary=summary, comparisons=126,
               unique_vectors=len(used), measurements=measurements,
               vectors_sha256=complete["vectors_sha256"], all_269_parameter_hashes_unchanged_since_R25=len(complete["all_parameter_hashes_before"]) == 269,
               actual_gpu_worker_seconds=actual, old_gpu_worker_seconds=old, total_gpu_worker_seconds=old+actual, remaining_seconds=21600-old-actual,
               source_and_runtime_gate_scope="Frozen worker actual-source assertions and server preflight; Root does not reload the model or intercept SDPA.",
               RMS_scope="Independent NumPy FP64 values; no CPU/GPU reduction equality threshold.",
               inference_method_changed=False, accuracy_computed=False, new_gpu_job_authorized=False,
               limitation="Finite five-context/two-cut configurations only. No universal causal proof, semantic leakage attribution, ARC benefit, or deployment-method promotion.")
    path = ROOT / "results/r26_ROOT_VERIFICATION.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps({k: out[k] for k in ["status", "category", "summary", "comparisons", "unique_vectors", "actual_gpu_worker_seconds", "total_gpu_worker_seconds"]}))


if __name__ == "__main__":
    main()
