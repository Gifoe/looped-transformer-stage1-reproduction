"""CPU-only real-evidence equivalence gate before any asynchronous collection."""
import argparse
import hashlib
import json
import pathlib
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
args = parser.parse_args()
root = pathlib.Path(args.root).resolve()
sys.path[:0] = [str(root / "code"), str(root / "deps"), str(root / "async_io")]
import torch
from ab1_prefix_codec import load
from async_storage import SingleWriter
torch.set_num_threads(1)
torch.set_grad_enabled(False)
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
source = root / "b2_data/treatments/train64"
pending = json.loads((source / "PENDING.json").read_text())
audit = json.loads((source / "INDEPENDENT_COVERAGE_AUDIT.json").read_text())
assert pending["status"] == "COMPLETE" and audit["status"] == "PASS"
assert audit["pending_sha256"] == sha(source / "PENDING.json")
destination = root / "async_io/real_storage_gate"
assert not destination.exists()
destination.mkdir()
records = []
writer = SingleWriter()
started = time.perf_counter()
for entry in pending["cases"][:3]:
    path = source / entry["metadata_file"]
    assert sha(path) == entry["metadata_sha256"]
    metadata = json.loads(path.read_text())
    tensors = load(metadata, source)
    previous = writer.finish()
    if previous:
        records.append(previous)
    writer.submit(destination, metadata["id"], metadata, tensors, metadata["prefix_len"])
last = writer.close()
if last:
    records.append(last)
assert len(records) == 3
for original, copied in zip(pending["cases"][:3], records):
    assert copied["id"] == original["id"]
    assert (source / original["metadata_file"]).read_bytes() == (destination / copied["metadata_file"]).read_bytes()
    assert copied["metadata_sha256"] == original["metadata_sha256"]
    assert copied["tensor_raw_sha256"] == original["tensor_raw_sha256"]
    # Original codec verifies every archive/raw/canonical hash on reconstruction.
    load(destination / copied["metadata_file"], destination)
receipt = {"status": "PASS", "n_real_cases": 3, "cases": records,
           "seconds": time.perf_counter() - started,
           "source_pending_sha256": sha(source / "PENDING.json"),
           "script_sha256": sha(__file__),
           "storage_worker_sha256": sha(root / "async_io/async_storage.py"),
           "scope": "CPU-only already-audited real tensors. Async storage metadata and canonical bytes identical; no GPU/model/mathematical change tested or claimed."}
(root / "async_io/REAL_STORAGE_EQUIVALENCE.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
