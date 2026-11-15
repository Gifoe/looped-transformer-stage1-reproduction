"""Versioned supervisor; same fixed remaining stages and original auditors."""
import hashlib
import json
import os
import pathlib
import subprocess
import time
import traceback

R = pathlib.Path(__file__).resolve().parents[1]
PY = "/root/autodl-tmp/ouro-e6a/venv/bin/python"
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           PYTHONPATH=str(R / "code") + ":" + str(R / "deps") + ":" + str(R / "bridge"))
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
amendment_sha = sha(R / "async_io/EXECUTION_AMENDMENT.json")


def state(status, **fields):
    path = R / "B2_QUEUE_STATE.json"
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps({"status": status, "epoch": time.time(),
        "execution_amendment_sha256": amendment_sha, **fields}, indent=2))
    os.replace(partial, path)


def verify():
    amendment = json.loads((R / "async_io/EXECUTION_AMENDMENT.json").read_text())
    assert sha(R / "async_io/EXECUTION_AMENDMENT.json") == amendment_sha
    assert sha(R / "B2_FULL_SOURCE_FREEZE.json") == amendment["original_science_freeze_sha256"]
    frozen = json.loads((R / "B2_FULL_SOURCE_FREEZE.json").read_text())
    assert all(sha(R / p) == h for p, h in frozen["sources"].items())
    assert all(sha(R / p) == h for p, h in amendment["sources"].items())


def step(name, args):
    verify()
    state("RUNNING", step=name + "_async_v1")
    with (R / ("b2_queue_" + name + "_async_v1.log")).open("x") as log:
        subprocess.run([PY, *map(str, args)], env=ENV, cwd=R,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def phase(kind, stage):
    collector = R / ("async_io/b2_collect_async_v1.py" if kind == "treatments"
                     else "bridge/b2_collect.py")
    collector_stage = "validation299" if kind == "references" else stage
    step(kind + "_" + stage, [collector, "--root", R, "--kind",
         "treatment" if kind == "treatments" else "reference", "--stage", collector_stage])
    directory = R / "b2_data" / kind / stage
    pending = json.loads((directory / "PENDING.json").read_text())
    freeze = json.loads((R / "B2_FULL_SOURCE_FREEZE.json").read_text())
    ids = freeze["cohorts"][stage] if kind == "treatments" else freeze["reference_ids"][stage]
    assert pending["status"] == "COMPLETE" and pending["n"] == len(ids)
    assert [v["id"] for v in pending["cases"]] == ids and len(set(ids)) == len(ids)
    assert pending["source_freeze_sha256"] == sha(R / "B2_FULL_SOURCE_FREEZE.json")
    with (directory / "MANIFEST_COVERAGE.json").open("x") as handle:
        json.dump({"status": "PASS", "n": len(ids),
                   "pending_sha256": sha(directory / "PENDING.json"),
                   "expected_ID_list_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                   "execution_amendment_sha256": amendment_sha}, handle, indent=2)
    step("audit_" + kind + "_" + stage,
         [R / "bridge/b2_audit.py", "--root", R, "--kind", kind, "--stage", stage])
    if kind == "treatments":
        step("analyze_" + stage, [R / "bridge/b2_analyze.py", "--root", R, "--stage", stage])


try:
    verify()
    assert json.loads((R / "async_io/TRANSITION_RECEIPT.json").read_text())["status"] == "ORIGINAL_WORKERS_STOPPED_EVIDENCE_RETAINED"
    assert not (R / "b2_data/treatments/train1030/PENDING.json").exists()
    phase("treatments", "train1030")
    phase("references", "validation")
    phase("treatments", "validation299")
    verify()
    state("COMPLETE", scope="Original Final25 collection and original audits/analyses complete; final scientific synthesis/publication remain.")
except BaseException as error:
    state("FAILED_STOPPED", error=repr(error), traceback=traceback.format_exc())
    raise
