"""Two isolated workers, original scientific cohorts/operators and original audits."""
import fcntl
import gzip
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time
import traceback

R = pathlib.Path(__file__).resolve().parents[1]
PY = "/root/autodl-tmp/ouro-e6a/venv/bin/python"
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(pathlib.Path(p).read_text())
AMENDMENT = R / "parallel/EXECUTION_AMENDMENT.json"
amendment_sha = sha(AMENDMENT)
science = read(R / "B2_FULL_SOURCE_FREEZE.json")
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           PYTHONPATH=str(R / "code") + ":" + str(R / "deps") + ":" + str(R / "bridge"))


def verify():
    amendment = read(AMENDMENT)
    assert sha(AMENDMENT) == amendment_sha
    assert amendment["status"] == "ISOLATED_PROCESS_SCIENCE_EXECUTION"
    assert sha(R / "B2_FULL_SOURCE_FREEZE.json") == amendment["science_freeze_sha256"]
    assert all(sha(R / p) == h for p, h in science["sources"].items())
    assert all(sha(R / p) == h for p, h in amendment["sources"].items())


def state(status, **fields):
    path = R / "B2_QUEUE_STATE.json"
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps({"status": status, "epoch": time.time(),
                                    "parallel_execution_amendment_sha256": amendment_sha, **fields}, indent=2))
    os.replace(temporary, path)


def gpu_jobs():
    return {int(v) for v in subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                        "--format=csv,noheader,nounits"], text=True).split()}


def step(name, arguments):
    verify(); state("RUNNING", step=name + "_parallel_v1")
    with (R / ("b2_queue_" + name + "_parallel_v1.log")).open("x") as logfile:
        subprocess.run([PY, *map(str, arguments)], check=True, cwd=R, env=ENV,
                       stdout=logfile, stderr=subprocess.STDOUT)


def archive_duplicate_internal_metadata(output):
    """Keep a verified gzip of private duplicate manifests; final manifests stay unchanged."""
    receipts = []
    base = output / "parallel_v1"
    for internal in sorted(base.glob("worker*/*.json")):
        assert internal.resolve().is_relative_to(base.resolve())
        root_manifest = output / internal.name
        assert root_manifest.exists()
        raw = internal.read_bytes()
        original = json.loads(raw)
        rebased = read(root_manifest)
        namespace = rebased.pop("storage_namespace")
        assert internal.parent == output / namespace
        prefix = namespace + "/"
        for tensor in rebased["tensor_manifest"].values():
            for chunk in tensor["chunks"]:
                assert chunk["file"].startswith(prefix)
                chunk["file"] = chunk["file"][len(prefix):]
        assert rebased == original
        packed = gzip.compress(raw, compresslevel=6, mtime=0)
        archive = internal.with_suffix(".json.gz")
        with archive.open("xb") as handle:
            handle.write(packed); handle.flush(); os.fsync(handle.fileno())
        assert gzip.decompress(archive.read_bytes()) == raw
        receipts.append({"internal_file": str(internal.relative_to(R)), "raw_sha256": hashlib.sha256(raw).hexdigest(),
                         "archive_file": str(archive.relative_to(R)), "archive_sha256": sha(archive),
                         "root_metadata_sha256": sha(root_manifest), "saved_bytes": len(raw) - len(packed)})
        assert internal.read_bytes() == raw
        internal.unlink()
    (output / "DUPLICATE_INTERNAL_METADATA_LOSSLESS_ARCHIVE.json").write_text(json.dumps({"status": "PASS", "records": receipts}, indent=2))


def capture(stage):
    verify()
    ids = science["cohorts"][stage]
    output = R / "b2_data/treatments" / stage
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "PENDING.json").exists()
    retained = []
    for ident in ids:
        path = output / (hashlib.sha256(ident.encode()).hexdigest() + ".json")
        if path.exists():
            metadata = read(path)
            assert metadata["id"] == ident and metadata["source_freeze_sha256"] == sha(R / "B2_FULL_SOURCE_FREEZE.json")
            retained.append({"id": ident, "metadata_file": path.name, "metadata_sha256": sha(path),
                             "tensor_raw_sha256": metadata["raw_tensor_sha256"]})
    retained_ids = {r["id"] for r in retained}
    todo = [ident for ident in ids if ident not in retained_ids]
    control = R / "parallel/science" / stage
    control.mkdir(parents=True, exist_ok=False)
    count = min(read(AMENDMENT)["workers"], len(todo))
    assert count > 0
    workers = []
    writer_reports = []
    with open(science["lock_path"], "a") as project_lock:
        fcntl.flock(project_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not gpu_jobs(), "Nonproject CUDA process active"
        hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.free", "--format=csv,noheader,nounits"], text=True).strip().split(",")
        assert hardware[0].strip() == science["gpu_uuid"] and hardware[1].strip() == "NVIDIA GeForce RTX 4090"
        assert int(hardware[2]) >= 20480
        assert shutil.disk_usage(R).free > 8 * 1024**3 + 512 * 1024**2
        state("RUNNING", step="treatments_" + stage + "_parallel_v1", n_retained=len(retained),
              n_committed=len(retained), n_total=len(ids), workers=count)
        spec = {"mode": "science", "split": "validation" if stage == "validation299" else "train",
                "science_freeze_sha256": sha(R / "B2_FULL_SOURCE_FREEZE.json"),
                "sources": {**read(AMENDMENT)["sources"], "parallel/EXECUTION_AMENDMENT.json": amendment_sha},
                "assignments": [todo[i::count] for i in range(count)], "control_dir": str(control),
                "output_dir": str(output), "namespace_prefix": "parallel_v1",
                "execution_amendment_sha256": amendment_sha}
        spec_path = control / "SPEC.json"
        spec_path.write_text(json.dumps(spec, indent=2))
        try:
            for i in range(count):
                with (control / f"worker{i}.log").open("x") as logfile:
                    child = subprocess.Popen([PY, str(R / "parallel/worker_v3.py"), "--root", str(R),
                                              "--spec", str(spec_path), "--index", str(i)],
                                             cwd=R, env=ENV, stdin=subprocess.DEVNULL,
                                             stdout=logfile, stderr=subprocess.STDOUT)
                workers.append(child)
            deadline = time.monotonic() + 180
            while not all((control / f"worker{i}_READY.json").exists() for i in range(count)):
                assert all(child.poll() is None for child in workers), "Worker loading failed"
                assert time.monotonic() < deadline
                time.sleep(.25)
            allowed = {child.pid for child in workers}
            assert gpu_jobs() <= allowed
            (control / "GO.json").write_text(json.dumps({"spec_sha256": sha(spec_path)}))
            while any(child.poll() is None for child in workers):
                completed_reports = [control / f"worker{i}_RESULT.json" for i in range(count)]
                if all(path.exists() for path in completed_reports):
                    checked_reports = [read(path) for path in completed_reports]
                    assert all(report["status"] == "PASS" and report["spec_sha256"] == sha(spec_path) for report in checked_reports)
                    break
                verify()
                assert all(child.poll() in (None, 0) for child in workers), "Isolated worker failed"
                assert gpu_jobs() <= allowed, "Unexpected CUDA process"
                free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
                assert free >= 2048, "Parallel GPU memory reserve crossed"
                assert shutil.disk_usage(R).free > 8 * 1024**3 + 256 * 1024**2, "Parallel disk reserve crossed"
                committed = len(retained)
                for i in range(count):
                    path = control / f"worker{i}_PROGRESS.json"
                    if path.exists():
                        committed += len(read(path)["records"])
                state("RUNNING", step="treatments_" + stage + "_parallel_v1", n_retained=len(retained),
                      n_committed=committed, n_total=len(ids), workers=count, worker_pids=sorted(allowed))
                time.sleep(3)
            assert all(child.poll() in (None, 0) for child in workers)
            writer_reports = [read(control / f"worker{i}_RESULT.json") for i in range(count)]
            assert all(r["status"] == "PASS" and r["spec_sha256"] == sha(spec_path) for r in writer_reports)
        finally:
            for child in workers:
                if child.poll() is None:
                    child.terminate()
            for child in workers:
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
            for _ in range(30):
                if not gpu_jobs():
                    break
                time.sleep(.2)
            assert not gpu_jobs()
    entries = retained + [v for r in writer_reports for v in r["records"]]
    assert len(entries) == len(ids) == len({v["id"] for v in entries})
    by_id = {v["id"]: v for v in entries}
    ordered = [by_id[ident] for ident in ids]
    assert all(sha(output / v["metadata_file"]) == v["metadata_sha256"] for v in ordered)
    producer = dict(read(control / "worker0_READY.json")["identity"], source_freeze_sha256=sha(R / "B2_FULL_SOURCE_FREEZE.json"))
    pending = {"status": "COMPLETE", "n": len(ids), "split": spec["split"], "stage": stage,
               "kind": "treatment", "cases": ordered, "producer_identity": producer,
               "source_freeze_sha256": sha(R / "B2_FULL_SOURCE_FREEZE.json"),
               "parallel_execution_amendment_sha256": amendment_sha, "parallel_spec_sha256": sha(spec_path)}
    with (output / "PENDING.json").open("x") as handle:
        json.dump(pending, handle, indent=2)
    archive_duplicate_internal_metadata(output)
    manifest_and_audit("treatments", stage)


def manifest_and_audit(kind, stage):
    directory = R / "b2_data" / kind / stage
    pending = read(directory / "PENDING.json")
    ids = science["cohorts"][stage] if kind == "treatments" else science["reference_ids"][stage]
    assert pending["status"] == "COMPLETE" and [v["id"] for v in pending["cases"]] == ids
    with (directory / "MANIFEST_COVERAGE.json").open("x") as handle:
        json.dump({"status": "PASS", "n": len(ids), "pending_sha256": sha(directory / "PENDING.json"),
                   "expected_ID_list_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}, handle, indent=2)
    step("audit_" + kind + "_" + stage, [R / "bridge/b2_audit.py", "--root", R, "--kind", kind, "--stage", stage])
    if kind == "treatments":
        step("analyze_" + stage, [R / "bridge/b2_analyze.py", "--root", R, "--stage", stage])


try:
    verify()
    assert read(R / "parallel/TRANSITION_RECEIPT.json")["status"] == "ORIGINAL_STOPPED_RETAINED"
    capture("train1030")
    step("references_validation", [R / "bridge/b2_collect.py", "--root", R, "--kind", "reference", "--stage", "validation299"])
    manifest_and_audit("references", "validation")
    capture("validation299")
    verify()
    state("COMPLETE", scope="Fixed original experiments and original audits/analysis complete. Scientific synthesis and GitHub publication remain.")
except BaseException as error:
    state("FAILED_STOPPED", error=repr(error), traceback=traceback.format_exc())
    raise
