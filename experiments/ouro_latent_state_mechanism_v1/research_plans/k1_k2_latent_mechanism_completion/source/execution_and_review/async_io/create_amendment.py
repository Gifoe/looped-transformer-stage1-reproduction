"""Freeze execution-only lineage before changing any live processes."""
import hashlib
import json
import pathlib
import time

root = pathlib.Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
science = root / "B2_FULL_SOURCE_FREEZE.json"
frozen = json.loads(science.read_text())
assert all(sha(root / p) == h for p, h in frozen["sources"].items())
gate_path = root / "async_io/REAL_STORAGE_EQUIVALENCE.json"
gate = json.loads(gate_path.read_text())
assert gate["status"] == "PASS" and gate["n_real_cases"] == 3
assert gate["storage_worker_sha256"] == sha(root / "async_io/async_storage.py")
assert gate["script_sha256"] == sha(root / "async_io/verify_real_storage.py")
sources = ["async_io/async_storage.py", "async_io/b2_collect_async_v1.py",
           "async_io/run_queue_async_v1.py", "async_io/create_amendment.py",
           "async_io/verify_real_storage.py", "async_io/REAL_STORAGE_EQUIVALENCE.json",
           "async_io/transition_and_launch.py"]
amendment = {"status": "CPU_ASYNC_STORAGE_ONLY", "utc_epoch": time.time(),
    "original_science_freeze_sha256": sha(science),
    "sources": {p: sha(root / p) for p in sources},
    "change": "One CPU evidence writer overlaps already detached CPU evidence serialization with next sequential GPU question. Original codec and every full audit unchanged. No batching, parallel GPU models, source edits, model, kernels, dtype, interventions, cohorts, donors, directions or metrics changed.",
    "guards": "One outstanding write; CPU-only tensors; ordered commit and final join; original immutable source guard plus execution guard; disk reserve includes two cases. Writer errors fail queue. All pre-switch committed records reused with original hashes; later records identify this amendment.",
    "scientific_scope": "Original prospective Final25 criteria remain. Storage equivalence is not a new independent GPU replay. Subsequent original full tensor and score audits remain mandatory.",
    "chronology": "After train64 complete/audited, during original Train1030 capture; no new effects-based scientific selection. All sources/results preserved."}
with (root / "async_io/EXECUTION_AMENDMENT.json").open("x") as handle:
    json.dump(amendment, handle, indent=2)
print("EXECUTION_AMENDMENT", sha(root / "async_io/EXECUTION_AMENDMENT.json"))
