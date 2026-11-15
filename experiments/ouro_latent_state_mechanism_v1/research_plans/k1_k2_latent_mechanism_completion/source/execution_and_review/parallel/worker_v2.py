"""Isolated model/hook process; prospective assignments only, original B2 math."""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--spec", required=True)
parser.add_argument("--index", required=True, type=int)
args = parser.parse_args()
root = pathlib.Path(args.root).resolve()
spec_path = pathlib.Path(args.spec).resolve()
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(pathlib.Path(p).read_text())
spec = read(spec_path)
assert spec_path.is_relative_to(root)
assert args.index < len(spec["assignments"])
control = pathlib.Path(spec["control_dir"])
assert control.resolve().is_relative_to(root)
report_path = control / f"worker{args.index}_RESULT.json"
sys.path[:0] = [str(root / "parallel"), str(root / "bridge"), str(root / "code"), str(root / "deps")]


def check():
    assert sha(root / "B2_FULL_SOURCE_FREEZE.json") == spec["science_freeze_sha256"]
    frozen = read(root / "B2_FULL_SOURCE_FREEZE.json")
    assert all(sha(root / p) == h for p, h in frozen["sources"].items())
    assert all(sha(root / p) == h for p, h in spec["sources"].items())
    assert sha(spec_path) == spec_sha


spec_sha = sha(spec_path)
records = []
try:
    import torch
    from ab1_loader_draft import load_model
    from ab1_prefix_codec import load
    from b2_kernel import treatments
    from parallel_storage import Writer
    torch.set_num_threads(1); torch.set_grad_enabled(False); torch.manual_seed(2021)
    torch.backends.cuda.matmul.allow_tf32 = False
    check()
    frozen = read(root / "B2_FULL_SOURCE_FREEZE.json")
    rows = {row["id"]: row for line in (root / "inputs" / (spec["split"] + "_model_inputs.jsonl")).read_text().splitlines()
            if (row := json.loads(line))}
    prefix = read(root / "inputs/CPU_PREPARATION_MANIFEST.json")["prefix_length"]
    ab_index = {}
    for stage in (["train64", "train1030"] if spec["split"] == "train" else ["validation299"]):
        directory = root / "data" / stage
        pending = read(directory / "PENDING.json")
        audit = read(directory / "INDEPENDENT_COVERAGE_AUDIT.json")
        assert pending["status"] == "COMPLETE" and audit["status"] == "PASS"
        assert audit["pending_sha256"] == sha(directory / "PENDING.json")
        ab_index.update({r["id"]: (directory / r["metadata_file"], r["metadata_sha256"]) for r in pending["cases"]})
    ref_directory = root / "b2_data/references" / spec["split"]
    ref_pending = read(ref_directory / "PENDING.json")
    ref_audit = read(ref_directory / "INDEPENDENT_COVERAGE_AUDIT.json")
    assert ref_pending["status"] == "COMPLETE" and ref_audit["status"] == "PASS"
    assert ref_audit["pending_sha256"] == sha(ref_directory / "PENDING.json")
    ref_index = {r["id"]: (ref_directory / r["metadata_file"], r["metadata_sha256"]) for r in ref_pending["cases"]}

    def evidence(index, ident):
        path, digest = index[ident]
        assert sha(path) == digest
        metadata = read(path)
        return path, metadata, load(metadata, path.parent)

    model, identity = load_model(frozen["model_path"], authorized=True, device="cuda:0")
    producer = dict(identity, source_freeze_sha256=spec["science_freeze_sha256"])
    ready = {"pid": os.getpid(), "identity": identity, "spec_sha256": spec_sha,
             "uuid": str(torch.cuda.get_device_properties(0).uuid) if hasattr(torch.cuda.get_device_properties(0), "uuid") else None}
    (control / f"worker{args.index}_READY.json").write_text(json.dumps(ready))
    deadline = time.monotonic() + 240
    while not (control / "GO.json").exists():
        assert time.monotonic() < deadline, "Start barrier timeout"
        time.sleep(.05)
    assert read(control / "GO.json")["spec_sha256"] == spec_sha
    started = time.perf_counter()
    writer = Writer()
    output = pathlib.Path(spec["output_dir"])
    namespace = spec["namespace_prefix"] + f"/worker{args.index}"
    for number, ident in enumerate(spec["assignments"][args.index]):
        if number % 16 == 0:
            check()
        amp, am, at = evidence(ab_index, ident)
        rmp, rm, rt = evidence(ref_index, ident)
        donor = frozen["donors"][spec["split"]][ident]
        damp, dam, dat = evidence(ab_index, donor)
        drmp, drm, drt = evidence(ref_index, donor)
        assert identity == {k: v for k, v in am["source_identity"].items() if k != "source_freeze_sha256"}
        q = dict(rows[ident], prefix_len=prefix)
        m, tensors = treatments(model, q, am, at, rm, rt,
            dict(id=donor, split=spec["split"], source_identity=dam["source_identity"], tensors=dat),
            dict(id=donor, split=spec["split"], source_identity=drm["source_identity"], tensors=drt))
        if spec["mode"] == "benchmark":
            expected_dir = root / "b2_data/treatments/train64"
            expected_pending = read(expected_dir / "PENDING.json")
            entry = next(v for v in expected_pending["cases"] if v["id"] == ident)
            expected_path = expected_dir / entry["metadata_file"]
            assert sha(expected_path) == entry["metadata_sha256"]
            expected_meta = read(expected_path)
            expected_tensors = load(expected_meta, expected_dir)
            assert set(tensors) == set(expected_tensors)
            for name, value in tensors.items():
                assert torch.equal(value.reshape(-1).view(torch.uint8), expected_tensors[name].contiguous().reshape(-1).view(torch.uint8)), (ident, name)
            normalized = json.loads(json.dumps(m))
            expected_subset = {k: expected_meta[k] for k in m}
            if normalized != expected_subset:
                differences = {k: {"actual": normalized[k], "expected": expected_subset[k]} for k in normalized if normalized[k] != expected_subset[k]}
                (control / ("worker" + str(args.index) + "_METADATA_DIFFERENCES.json")).write_text(json.dumps(differences, indent=2))
                raise AssertionError((ident, list(differences)))
            del expected_tensors
        else:
            assert spec["mode"] == "science"
        m["kernel_status"] = m["status"]; m["execution_status"] = "COLLECTED_AWAITING_AUDIT"
        m.update(R=q["R"], prefix_len=prefix, targets=q["targets"], input_ids_sha256=q["input_ids_sha256"],
            AB_metadata_path=str(amp.relative_to(root)), AB_metadata_sha256=sha(amp),
            reference_metadata_path=str(rmp.relative_to(root)), reference_metadata_sha256=sha(rmp),
            donor_id=donor, donor_AB_metadata_path=str(damp.relative_to(root)), donor_AB_metadata_sha256=sha(damp),
            donor_reference_metadata_path=str(drmp.relative_to(root)), donor_reference_metadata_sha256=sha(drmp),
            producer_identity=producer, source_freeze_sha256=spec["science_freeze_sha256"],
            execution_revision="isolated_parallel_v1", parallel_spec_sha256=spec_sha,
            parallel_spec_path=str(spec_path.relative_to(root)), worker_index=args.index)
        if spec["mode"] == "science":
            m["parallel_execution_amendment_sha256"] = spec["execution_amendment_sha256"]
        previous = writer.finish()
        if previous:
            records.append(previous)
        writer.submit(output, namespace, ident, m, tensors, prefix)
        del tensors, at, rt, dat, drt
        progress = {"status": "RUNNING", "pid": os.getpid(), "records": records,
                    "submitted": number + 1, "spec_sha256": spec_sha}
        partial = control / f"worker{args.index}_PROGRESS.partial"
        partial.write_text(json.dumps(progress)); os.replace(partial, control / f"worker{args.index}_PROGRESS.json")
    last = writer.close()
    if last:
        records.append(last)
    check()
    assert [r["id"] for r in records] == spec["assignments"][args.index]
    report = {"status": "PASS", "pid": os.getpid(), "n": len(records), "records": records,
              "wall_seconds": time.perf_counter() - started, "spec_sha256": spec_sha,
              "scope": "Full tensors/meta match saved serial evidence" if spec["mode"] == "benchmark" else "Collection awaiting original independent tensor audit"}
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)
except BaseException as error:
    report_path.write_text(json.dumps({"status": "FAILED", "pid": os.getpid(), "error": repr(error),
                                      "traceback": traceback.format_exc(), "records": records}, indent=2))
    raise
