"""Explicit saved-case-preserving process transition after CPU equivalence gate."""
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import time

root = pathlib.Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
amendment_path = root / "async_io/EXECUTION_AMENDMENT.json"
amendment = json.loads(amendment_path.read_text())
assert amendment["status"] == "CPU_ASYNC_STORAGE_ONLY"
assert all(sha(root / p) == h for p, h in amendment["sources"].items())
frozen_path = root / "B2_FULL_SOURCE_FREEZE.json"
frozen = json.loads(frozen_path.read_text())
assert sha(frozen_path) == amendment["original_science_freeze_sha256"]
assert all(sha(root / p) == h for p, h in frozen["sources"].items())
state_path = root / "B2_QUEUE_STATE.json"
before = json.loads(state_path.read_text())
assert before["status"] == "RUNNING" and before["step"] == "treatments_train1030"
assert not (root / "b2_data/treatments/train1030/PENDING.json").exists()
assert not (root / "async_io/TRANSITION_RECEIPT.json").exists()


def matches(pid, expected):
    try:
        command = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    return [b.decode() for b in command if b] == expected


python = "/root/autodl-tmp/ouro-e6a/venv/bin/python"
parent, collector = 9190, 11673
assert (matches(parent, [python, "-u", str(root / "b2_run_queue.py")])
        or matches(parent, [python, str(root / "b2_run_queue.py")]))
assert matches(collector, [python, str(root / "bridge/b2_collect.py"), "--root", str(root),
                           "--kind", "treatment", "--stage", "train1030"])
intent = {"status": "AUTHORIZED_EXECUTION_TRANSITION_STARTED", "utc_epoch": time.time(),
          "old_queue_pid": parent, "old_collector_pid": collector,
          "old_queue_state": before, "execution_amendment_sha256": sha(amendment_path)}
(root / "async_io/TRANSITION_INTENT.json").write_text(json.dumps(intent, indent=2))
os.kill(parent, signal.SIGTERM)
os.kill(collector, signal.SIGINT)
for _ in range(60):
    process = pathlib.Path(f"/proc/{collector}/stat")
    if not process.exists() or process.read_text().split()[2] == "Z":
        break
    time.sleep(.5)
else:
    raise RuntimeError("Original collector did not exit; no second collector launched")
for _ in range(20):
    jobs = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    if not jobs.strip():
        break
    time.sleep(.5)
assert not jobs.strip(), jobs
data = (root / "b2_data/treatments/train1030").resolve()
assert data.is_relative_to(root.resolve())
retained = []
interrupted = []
expected = frozen["cohorts"]["train1030"]
for ident in expected:
    path = data / (hashlib.sha256(ident.encode()).hexdigest() + ".json")
    if not path.exists():
        continue
    try:
        metadata = json.loads(path.read_text())
    except json.JSONDecodeError:
        destination = root / "async_io/interrupted_metadata" / path.name
        destination.parent.mkdir(exist_ok=True)
        assert not destination.exists()
        interrupted.append({"file": str(path.relative_to(root)), "sha256": sha(path),
                            "retained_at": str(destination.relative_to(root))})
        path.rename(destination)
        continue
    assert metadata["id"] == ident and metadata["source_freeze_sha256"] == sha(frozen_path)
    retained.append({"id": ident, "metadata_file": path.name, "metadata_sha256": sha(path),
                     "tensor_raw_sha256": metadata["raw_tensor_sha256"]})
receipt = {**intent, "status": "ORIGINAL_WORKERS_STOPPED_EVIDENCE_RETAINED",
           "completed_metadata": retained, "n_retained": len(retained),
           "interrupted_metadata_preserved": interrupted,
           "scope": "All chunks retained, including orphans; only a currently uncommitted question may be recomputed. Original scalar/tensor audits unchanged."}
(root / "async_io/TRANSITION_RECEIPT.json").write_text(json.dumps(receipt, indent=2))
environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
logfile = (root / "async_io/supervisor.log").open("x")
child = subprocess.Popen([python, str(root / "async_io/run_queue_async_v1.py")], cwd=root,
    env=environment, stdin=subprocess.DEVNULL, stdout=logfile, stderr=subprocess.STDOUT,
    start_new_session=True)
print(json.dumps({"status": "VERSIONED_QUEUE_LAUNCHED", "new_queue_pid": child.pid,
                  "n_retained": len(retained), "interrupted_metadata_preserved": interrupted,
                  "execution_amendment_sha256": sha(amendment_path)}))
