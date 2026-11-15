"""Additional independent provenance check after original scientific audits."""
import hashlib
import json
import pathlib
import time

root = pathlib.Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
amendment_path = root / "async_io/EXECUTION_AMENDMENT.json"
amendment = json.loads(amendment_path.read_text())
transition = json.loads((root / "async_io/TRANSITION_RECEIPT.json").read_text())
retained = {r["id"]: r for r in transition["completed_metadata"]}
amendment_sha = sha(amendment_path)
results = []
for stage in ("train64", "train1030", "validation299"):
    directory = root / "b2_data/treatments" / stage
    while not (root / "b2_analysis" / stage / "COMPLETE.json").exists():
        state = json.loads((root / "B2_QUEUE_STATE.json").read_text())
        assert state["status"] != "FAILED_STOPPED", state
        time.sleep(30)
    audit = json.loads((directory / "INDEPENDENT_COVERAGE_AUDIT.json").read_text())
    pending = json.loads((directory / "PENDING.json").read_text())
    assert audit["status"] == "PASS" and pending["status"] == "COMPLETE"
    assert audit["pending_sha256"] == sha(directory / "PENDING.json")
    count_original, count_amended, count_preserved = 0, 0, 0
    for item in pending["cases"]:
        path = directory / item["metadata_file"]
        assert sha(path) == item["metadata_sha256"]
        metadata = json.loads(path.read_text())
        if "execution_amendment_sha256" in metadata:
            assert metadata["execution_amendment_sha256"] == amendment_sha
            assert metadata["execution_revision"] == "async_cpu_writer_v1"
            count_amended += 1
        else:
            assert stage == "train64" or (stage == "train1030" and item["id"] in retained)
            count_original += 1
        if stage == "train1030" and item["id"] in retained:
            original = retained[item["id"]]
            assert item["metadata_sha256"] == original["metadata_sha256"]
            assert item["tensor_raw_sha256"] == original["tensor_raw_sha256"]
            count_preserved += 1
        assert metadata["source_freeze_sha256"] == amendment["original_science_freeze_sha256"]
        assert metadata["producer_identity"]["source_freeze_sha256"] == amendment["original_science_freeze_sha256"]
    if stage == "train1030":
        assert count_preserved == len(retained)
    assert all(sha(root / p) == h for p, h in amendment["sources"].items())
    frozen = json.loads((root / "B2_FULL_SOURCE_FREEZE.json").read_text())
    assert sha(root / "B2_FULL_SOURCE_FREEZE.json") == amendment["original_science_freeze_sha256"]
    assert all(sha(root / p) == h for p, h in frozen["sources"].items())
    result = {"status": "PASS", "stage": stage, "n_original": count_original,
              "n_async": count_amended, "n_original_preserved": count_preserved,
              "pending_sha256": sha(directory / "PENDING.json"),
              "audit_sha256": sha(directory / "INDEPENDENT_COVERAGE_AUDIT.json"),
              "execution_amendment_sha256": amendment_sha,
              "script_sha256": sha(__file__),
              "scope": "Source/metadata lineage; original tensor, score and effect audits remain separate."}
    (root / "async_io" / (stage + "_EXECUTION_LINEAGE.json")).write_text(json.dumps(result, indent=2))
    results.append(result)
(root / "async_io/ALL_EXECUTION_LINEAGE_COMPLETE.json").write_text(json.dumps({"status": "PASS", "stages": results}, indent=2))
print("ALL_EXECUTION_LINEAGE_PASS")
