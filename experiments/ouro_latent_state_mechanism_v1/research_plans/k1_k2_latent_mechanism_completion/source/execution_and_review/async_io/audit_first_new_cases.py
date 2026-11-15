"""Fixed first three new cases: original independent saved-tensor auditor only."""
import hashlib
import json
import pathlib
import sys
import time

root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "code"), str(root / "deps"), str(root / "bridge")]
import torch
from ab1_prefix_codec import load
from b2_audit import audit_case, score
torch.set_num_threads(1)
torch.set_grad_enabled(False)
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
science_path = root / "B2_FULL_SOURCE_FREEZE.json"
frozen = json.loads(science_path.read_text())
amendment_path = root / "async_io/EXECUTION_AMENDMENT.json"
amendment = json.loads(amendment_path.read_text())
transition = json.loads((root / "async_io/TRANSITION_RECEIPT.json").read_text())
start = transition["n_retained"]
ids = frozen["cohorts"]["train1030"][start:start + 3]
assert len(ids) == 3
records = []
directory = root / "b2_data/treatments/train1030"


def checked(relative, digest):
    path = (root / relative).resolve()
    assert path.is_relative_to(root) and sha(path) == digest
    metadata = json.loads(path.read_text())
    return metadata, load(metadata, path.parent)


for ident in ids:
    path = directory / (hashlib.sha256(ident.encode()).hexdigest() + ".json")
    while not path.exists():
        state = json.loads((root / "B2_QUEUE_STATE.json").read_text())
        assert state["status"] != "FAILED_STOPPED", state
        time.sleep(5)
    # Collector prints a case only after exclusive metadata write/fsync is done.
    while True:
        log = (root / "b2_queue_treatments_train1030_async_v1.log").read_text()
        if any(line.startswith("B2_CASE ") and line.split()[-1] == ident for line in log.splitlines()):
            break
        time.sleep(2)
    before = sha(path)
    metadata = json.loads(path.read_text())
    assert metadata["id"] == ident and metadata["execution_amendment_sha256"] == sha(amendment_path)
    tensors = load(metadata, directory)
    ab, abt = checked(metadata["AB_metadata_path"], metadata["AB_metadata_sha256"])
    reference, rt = checked(metadata["reference_metadata_path"], metadata["reference_metadata_sha256"])
    donor_ab, dat = checked(metadata["donor_AB_metadata_path"], metadata["donor_AB_metadata_sha256"])
    donor_ref, drt = checked(metadata["donor_reference_metadata_path"], metadata["donor_reference_metadata_sha256"])
    assert donor_ab["id"] == donor_ref["id"] == metadata["donor_id"]
    for key, condition in (("native_K1_logits", ab["identity"]["K1_scores"]),
                           ("native_K2_logits", ab["identity"]["N_scores"]),
                           ("H01_logits", ab["conditions"]["H01"]["scores"])):
        score(abt[key], condition, metadata["targets"])
    proof = audit_case(metadata, tensors, abt, reference, rt, dat, drt, False)
    assert len(proof["conditions"]) == 22
    assert all(value in ("PASS", "UNAVAILABLE_MATH_PROVEN") for value in proof["conditions"].values())
    assert sha(path) == before
    records.append({"id": ident, "metadata_sha256": before, "audit": proof})
assert all(sha(root / p) == h for p, h in frozen["sources"].items())
assert all(sha(root / p) == h for p, h in amendment["sources"].items())
receipt = {"status": "PASS", "cases": records, "script_sha256": sha(__file__),
           "original_auditor_sha256": sha(root / "bridge/b2_audit.py"),
           "source_freeze_sha256": sha(science_path), "execution_amendment_sha256": sha(amendment_path),
           "scope": "Original full saved-tensor/score/control audit on fixed first three new captures; not an independent GPU replay or full-split completion."}
(root / "async_io/FIRST_NEW3_TENSOR_AUDIT.json").write_text(json.dumps(receipt, indent=2))
print("FIRST_NEW3_TENSOR_AUDIT_PASS", ids, flush=True)
