"""Independent raw-vector and fixed-stage audit of R27; never loads the model."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "9de2154c5d6ba207b6a2ad864410d2c555f4107ce34a4721ec5ba3c62cf1ea20"
PATCHES = [("V_identity", "V", "B", False), ("V_clean", "V", "A", False),
           ("V_shuffle", "V", "A", True), ("K_clean", "K", "A", False),
           ("K_shuffle", "K", "A", True), ("K_identity", "K", "B", False)]


def read(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def main():
    assert sha("configs/r27_EXECUTION_LOCK.json") == PROTOCOL
    spec = read("configs/r27_EXECUTION_LOCK.json")
    assert all(sha(p) == h for p, h in spec["source_sha256"].items())
    old_ledgers = spec["budget"]["inherited_ledger_sha256"]
    assert len(old_ledgers) == 15 and all(sha(p) == h for p, h in old_ledgers.items())
    old_used = sum(read(p)["used_seconds"] for p in old_ledgers)
    assert old_used == 18874.135196831077
    sup = read("logs/r27_EXEC_SUPERVISOR.json")
    book = read("logs/r27_EXEC_gpu_budget.json")
    complete = read("results/r27_EXEC_COMPLETE.json")
    analysis = read("results/r27_EXEC_ANALYSIS.json")
    assert sup["status"] == "complete" and sup["exit_code"] == 0 and not sup["overrun"] and not sup.get("timed_out", False)
    actual = sup["gpu_worker_seconds"]
    assert 0 < actual <= 180 and book["used_seconds"] == actual
    assert book["inherited_total_seconds"] == old_used and book["total_used_seconds"] == old_used + actual <= 21600
    assert len(complete["all_parameter_hashes_before"]) == 269
    assert complete["all_parameter_hashes_before"] == complete["all_parameter_hashes_after"] == read("results/r26_COMPLETE.json")["all_parameter_hashes_after"]
    assert complete["protocol_sha256"] == PROTOCOL and not complete["invalid_reasons"]
    assert complete["status"] in ["complete_phaseB", "complete_phaseA_not_admitted"]
    for field, path in [("raw_sha256", "results/r27_EXEC_RAW.json"), ("vectors_sha256", "results/r27_EXEC_VECTORS.npz"), ("input_audit_sha256", "results/r27_EXEC_INPUT_AUDIT.json")]:
        assert complete[field] == sha(path)
    assert read("results/r27_EXEC_INPUT_AUDIT.json") == read("results/r27_EXEC_INPUT_PREFLIGHT.json")
    raw = read("results/r27_EXEC_RAW.json")
    assert not raw["invalid_reasons"] and raw["cleanup"] == complete["cleanup"]
    assert raw["cleanup"]["handles_removed"] and raw["cleanup"]["original_hooks_restored"] and raw["cleanup"]["all_model_hooks_empty"] and not raw["cleanup"]["errors"]
    manifest = read("configs/r27_CPU_STRUCTURE_MANIFEST.json")
    assert sha("configs/r27_CPU_STRUCTURE_MANIFEST.json") == "c7e59ff12fc7b70fe74e16b1f49361cd96f390066f168a19f3c9d6cb9cdf942a"
    rows = [dict(table=t["table"], query_row=q["query_row"], **v) for t in manifest["tables"] for q in t["instances"] for v in q["variants"]]
    assert len(rows) == len(raw["baselines"]) == 8
    refs = {}
    def collect(x):
        if isinstance(x, dict):
            if set(x) == {"key", "dtype", "shape", "raw_sha256"}:
                if x["key"] in refs:
                    assert refs[x["key"]] == x
                refs[x["key"]] = x
            else:
                for v in x.values():
                    collect(v)
        elif isinstance(x, list):
            for v in x:
                collect(v)
    collect(raw)
    arrays = {}
    with np.load(ROOT / "results/r27_EXEC_VECTORS.npz", allow_pickle=False) as container:
        assert set(container.files) == set(refs)
        for key, r in refs.items():
            a = container[key]
            assert r["dtype"] in ["torch.bfloat16", "torch.float32"]
            assert a.dtype == np.dtype("uint16" if r["dtype"] == "torch.bfloat16" else "float32") and list(a.shape) == r["shape"]
            data = a.tobytes()
            assert hashlib.sha256(data).hexdigest() == r["raw_sha256"]
            assert hashlib.sha256((r["dtype"] + str(tuple(r["shape"]))).encode() + data).hexdigest() == key
            expanded = (a.astype(np.uint32) << 16).view(np.float32) if r["dtype"] == "torch.bfloat16" else a
            assert np.isfinite(expanded).all()
            arrays[key] = a
    def same(a, b):
        assert a["dtype"] == b["dtype"] and a["shape"] == b["shape"]
        assert arrays[a["key"]].tobytes() == arrays[b["key"]].tobytes()
    def shape(r, dimensions, dtype):
        assert r["shape"] == dimensions and r["dtype"] == dtype
    event_order = [[loop, layer, kind] for loop in range(4) for layer in range(24) for kind in ["K", "V"]]
    def clock(record, row, mode, kind=None, shuffle=False):
        c = record["clock"]
        assert c["mode"] == mode and c["kind"] == kind and c["calls_per_module"] == [4]*48 and c["events"] == event_order
        assert c["source_positions"] == row["source_positions"] and c["captured_loop"] == 4
        assert c["permutation"] == ([1, 2, 3, 0] if shuffle else [0, 1, 2, 3])
        assert c["other_rows_exact_verified"] == (mode == "patch")
    def validate_record(record, row):
        for k in ["table", "query_row", "version", "input_ids_sha256", "input_shape", "source_positions", "expected_label"]:
            assert record[k] == row[k]
        assert record["label_ids"] == {s["label"]: s["token_id"] for s in row["sources"]}
        assert len(record["query_states"]) == 4
        for r in record["query_states"]:
            shape(r, [1, 1, 2048], "torch.bfloat16")
        shape(record["fullV"], [1, 1, 49152], "torch.float32")
    validate_record(raw["no_hook"], rows[0])
    for baseline, row in zip(raw["baselines"], rows):
        validate_record(baseline, row); clock(baseline, row, "capture")
        assert set(baseline["donors"]) == {"K", "V"}
        for domain in ["K", "V"]:
            assert len(baseline["donors"][domain]) == 24
            for r in baseline["donors"][domain]:
                shape(r, [1, 4, 2048], "torch.bfloat16")
        if row["version"] == "B":
            assert len(baseline["first3_full_states"]) == 3
            for r, query in zip(baseline["first3_full_states"], baseline["query_states"][:3]):
                shape(r, [1, row["input_shape"][1], 2048], "torch.bfloat16")
                assert np.array_equal(arrays[r["key"]][:, -1:, :], arrays[query["key"]])
    for a, b in zip(raw["no_hook"]["query_states"], raw["baselines"][0]["query_states"]):
        same(a, b)
    same(raw["no_hook"]["fullV"], raw["baselines"][0]["fullV"])
    assert raw["no_op_exact"]
    source_domain = {(t, v, k, l) for t in [1, 2] for v in ["A", "B"] for k in ["K", "V"] for l in range(24)}
    observed_source = [(p["table"], p["version"], p["kind"], p["layer"]) for p in raw["source_pairs"]]
    assert len(observed_source) == len(set(observed_source)) == 192 and set(observed_source) == source_domain
    for pair in raw["source_pairs"]:
        a, b = [r for r in raw["baselines"] if r["table"] == pair["table"] and r["version"] == pair["version"]]
        assert pair["A"] == a["donors"][pair["kind"]][pair["layer"]] and pair["B"] == b["donors"][pair["kind"]][pair["layer"]]
        same(pair["A"], pair["B"]); assert pair["exact"]
    phaseB = complete["status"] == "complete_phaseB"
    expected_forwards = 33 if phaseB else 9
    assert complete["forwards"] == expected_forwards and complete["baselines"] == 8 and complete["patches"] == (24 if phaseB else 0)
    assert len(raw["forward_records"]) == expected_forwards and [x["number"] for x in raw["forward_records"]] == list(range(1, expected_forwards+1))
    assert raw["forward_records"][0] == dict(number=1, phase="no_hook", table=rows[0]["table"], query_row=rows[0]["query_row"], version="A")
    for call, baseline in zip(raw["forward_records"][1:9], raw["baselines"]):
        assert call["phase"] == "A" and all(call[k] == baseline[k] for k in ["table", "query_row", "version", "clock"])
    assert len(raw["patches"]) == (24 if phaseB else 0)
    identity_pairs = 0; first3_pairs = 0
    if phaseB:
        for i, row in enumerate([r for r in rows if r["version"] == "B"]):
            base = raw["baselines"][2*i+1]
            first_query = next(r["query_row"] for r in rows if r["table"] == row["table"])
            for j, (name, kind, version, shuffle) in enumerate(PATCHES):
                r = raw["patches"][6*i+j]
                validate_record(r, row); clock(r, row, "patch", kind, shuffle)
                assert r["patch"] == name and (r["donor_table"], r["donor_query_row"], r["donor_version"], r["donor_kind"]) == (row["table"], first_query, version, kind)
                donor = next(b for b in raw["baselines"] if b["table"] == row["table"] and b["query_row"] == first_query and b["version"] == version)
                assert r["donor_vectors"] == donor["donors"][kind]
                assert r["first3_full_exact"] and r["baseline_first3_full_states"] == base["first3_full_states"]
                for full, expected, query in zip(r["first3_full_states"], base["first3_full_states"], r["query_states"][:3]):
                    same(full, expected); first3_pairs += 1
                    assert np.array_equal(arrays[full["key"]][:, -1:, :], arrays[query["key"]])
                if name.endswith("identity"):
                    assert r["identity_exact"]
                    for a, b in zip(r["query_states"] + [r["fullV"]], base["query_states"] + [base["fullV"]]):
                        same(a, b); identity_pairs += 1
                call = raw["forward_records"][9+6*i+j]
                assert call["phase"] == "B" and all(call[k] == r[k] for k in ["table", "query_row", "patch", "clock"])
        assert first3_pairs == 72 and identity_pairs == 40
    # Labels enter this independent effect review only after the entire raw-vector audit.
    masses = []; descriptive_error = 0.; strictly_correct = 0
    for r in raw["baselines"] + raw["patches"]:
        z = arrays[r["fullV"]["key"]].astype(np.float64).reshape(-1)
        ids = r["label_ids"]; y = r["expected_label"]
        scores = {k: float(z[v]) for k, v in ids.items()}
        strict = all(scores[y] > scores[k] for k in ids if k != y)
        assert scores == r["metrics"]["label_logits"] and strict == r["metrics"]["expected_strictly_highest"]
        shifted = z-z.max(); p = np.exp(shifted); p /= p.sum(dtype=np.float64)
        mass = float(sum(p[v] for v in ids.values())); assert np.isfinite(mass)
        descriptive_error = max(descriptive_error, abs(mass-r["metrics"]["label_probability_mass"]))
        masses.append(mass)
        if r in raw["baselines"]:
            strictly_correct += strict
    phaseA_pass = strictly_correct == 8
    assert phaseA_pass == phaseB == raw["phaseA_pass"] == analysis["summary"]["phaseA_pass"]
    paths = {}
    if phaseB:
        for kind in ["V", "K"]:
            effects = []
            for i in range(4):
                a, b = raw["baselines"][2*i:2*i+2]; ya, yb = a["expected_label"], b["expected_label"]
                def margin(r):
                    z = arrays[r["fullV"]["key"]].reshape(-1)
                    return float(np.float64(z[r["label_ids"][ya]])-np.float64(z[r["label_ids"][yb]]))
                group = raw["patches"][6*i:6*i+6]
                clean = next(r for r in group if r["patch"] == kind+"_clean")
                shuffled = next(r for r in group if r["patch"] == kind+"_shuffle")
                mb, mc, ms = margin(b), margin(clean), margin(shuffled)
                assert mb < 0 and margin(a) > 0
                effects.append(dict(instance=i, baseline_B_margin=mb, clean_margin=mc, shuffle_margin=ms, transport_delta=mc-mb, specificity=mc-ms, crossed_zero=mc>0))
            passed = all(e["transport_delta"] > 0 and e["specificity"] > 0 for e in effects) and sum(e["crossed_zero"] for e in effects) >= 3
            assert analysis["summary"]["paths"][kind]["rows"] == effects and analysis["summary"]["paths"][kind]["fixed_screen_pass"] == passed
            paths[kind] = dict(effects=effects, fixed_screen_pass=passed, primary=kind=="V")
    else:
        assert not analysis["summary"]["phaseB_performed"]
    result = dict(status="passed_independent_raw_vector_execution_review", protocol_sha256=PROTOCOL,
                  execution_status=complete["status"], forwards=expected_forwards, no_op_pairs=5, source_query_pairs=192,
                  full_first3_sequence_pairs=first3_pairs, identity_state_head_pairs=identity_pairs,
                  vectors=len(arrays), raw_vector_bytes=sum(a.nbytes for a in arrays.values()), phaseA_strict_correct=strictly_correct,
                  phaseA_total=8, phaseB_performed=phaseB, paths=paths, label_mass_min=min(masses), label_mass_max=max(masses),
                  descriptive_FP64_mass_max_difference_from_worker=descriptive_error,
                  total_gpu_worker_seconds=book["total_used_seconds"], actual_gpu_worker_seconds=actual, remaining_seconds=21600-book["total_used_seconds"],
                  all_269_parameters_unchanged_since_R26=True, vectors_sha256=complete["vectors_sha256"],
                  scope="Independent original saved tensors, all declared domains, exact gates, raw-logit effects and accounting. Off-source projection preservation is a frozen-hook runtime assertion; full intermediate projection outputs are not independently regenerated. No local model, SDPA interception, ARC scores or deployment candidate.",
                  limitation="Fixed synthetic four-label subspace; no unrestricted generation, reliability, compact-circuit, or ARC benefit claim.",
                  new_model_job_authorized=False)
    with (ROOT / "results/r27_ROOT_EXEC_VERIFICATION.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps({k: result[k] for k in ["status", "execution_status", "forwards", "phaseA_strict_correct", "phaseB_performed", "paths", "actual_gpu_worker_seconds", "total_gpu_worker_seconds"]}))


if __name__ == "__main__":
    main()
