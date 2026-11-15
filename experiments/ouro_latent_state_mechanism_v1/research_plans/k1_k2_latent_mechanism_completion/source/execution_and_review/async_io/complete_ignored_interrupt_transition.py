"""Complete explicit transition after inherited SIGINT-ignore was observed.

Original collector and frozen scripts remain untouched. SIGTERM may interrupt
one uncommitted case; every completed file and orphan chunk is retained.
"""
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
assert all(sha(root / p) == h for p, h in amendment["sources"].items())
science = root / "B2_FULL_SOURCE_FREEZE.json"
frozen = json.loads(science.read_text())
assert sha(science) == amendment["original_science_freeze_sha256"]
assert all(sha(root / p) == h for p, h in frozen["sources"].items())
intent = json.loads((root / "async_io/TRANSITION_INTENT.json").read_text())
assert intent["execution_amendment_sha256"] == sha(amendment_path)
assert not (root / "async_io/TRANSITION_RECEIPT.json").exists()
assert not pathlib.Path("/proc/9190").exists()
assert not (root / "b2_data/treatments/train1030/PENDING.json").exists()
python = "/root/autodl-tmp/ouro-e6a/venv/bin/python"
collector = intent["old_collector_pid"]
command = [x.decode() for x in pathlib.Path(f"/proc/{collector}/cmdline").read_bytes().split(b"\0") if x]
assert command == [python, str(root / "bridge/b2_collect.py"), "--root", str(root),
                   "--kind", "treatment", "--stage", "train1030"]
signal_status = pathlib.Path(f"/proc/{collector}/status").read_text()
ignored = next(line.split()[1] for line in signal_status.splitlines() if line.startswith("SigIgn:"))
assert int(ignored, 16) & (1 << (signal.SIGINT - 1))
supplement = {"status": "OBSERVED_INHERITED_SIGINT_IGNORE_USE_SIGTERM",
              "utc_epoch": time.time(), "script_sha256": sha(__file__),
              "execution_amendment_sha256": sha(amendment_path),
              "intent_sha256": sha(root / "async_io/TRANSITION_INTENT.json"),
              "collector_pid": collector, "signal_ignore_mask": ignored,
              "retention": "Completed metadata/tensors retained; invalid partial metadata quarantined unchanged; no old source overwritten."}
supplement_path = root / "async_io/TRANSITION_SUPPLEMENT.json"
with supplement_path.open("x") as handle:
    json.dump(supplement, handle, indent=2)
os.kill(collector, signal.SIGTERM)
for _ in range(60):
    process = pathlib.Path(f"/proc/{collector}/stat")
    if not process.exists() or process.read_text().split()[2] == "Z":
        break
    time.sleep(.5)
else:
    raise RuntimeError("Original collector still alive; no new collector launched")
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
for ident in frozen["cohorts"]["train1030"]:
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
    assert metadata["id"] == ident and metadata["source_freeze_sha256"] == sha(science)
    retained.append({"id": ident, "metadata_file": path.name, "metadata_sha256": sha(path),
                     "tensor_raw_sha256": metadata["raw_tensor_sha256"]})
receipt = {**intent, "status": "ORIGINAL_WORKERS_STOPPED_EVIDENCE_RETAINED",
           "completed_metadata": retained, "n_retained": len(retained),
           "interrupted_metadata_preserved": interrupted,
           "transition_supplement_sha256": sha(supplement_path),
           "scope": "All chunks retained, including orphans; currently uncommitted question recomputed. Original scientific sources/audits unchanged."}
(root / "async_io/TRANSITION_RECEIPT.json").write_text(json.dumps(receipt, indent=2))
environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
with (root / "async_io/supervisor.log").open("x") as logfile:
    child = subprocess.Popen([python, str(root / "async_io/run_queue_async_v1.py")], cwd=root,
        env=environment, stdin=subprocess.DEVNULL, stdout=logfile, stderr=subprocess.STDOUT,
        start_new_session=True)
print(json.dumps({"status": "VERSIONED_QUEUE_LAUNCHED", "new_queue_pid": child.pid,
                  "n_retained": len(retained), "interrupted_metadata_preserved": interrupted,
                  "execution_amendment_sha256": sha(amendment_path)}))
