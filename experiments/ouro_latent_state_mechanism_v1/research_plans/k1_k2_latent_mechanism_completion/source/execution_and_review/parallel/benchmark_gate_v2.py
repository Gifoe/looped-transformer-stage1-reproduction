"""Bounded matched-case 1-vs-2 isolated-process test. Original job resumes finally."""
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import time
import traceback

root = pathlib.Path(__file__).resolve().parents[1]
PY = "/root/autodl-tmp/ouro-e6a/venv/bin/python"
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(pathlib.Path(p).read_text())
frozen_path = root / "B2_FULL_SOURCE_FREEZE.json"
frozen = read(frozen_path)
assert all(sha(root / p) == h for p, h in frozen["sources"].items())
current = 12922
command = [v.decode() for v in pathlib.Path(f"/proc/{current}/cmdline").read_bytes().split(b"\0") if v]
assert command == [PY, str(root / "async_io/b2_collect_async_v1.py"), "--root", str(root),
                   "--kind", "treatment", "--stage", "train1030"]
assert read(root / "B2_QUEUE_STATE.json")["step"] == "treatments_train1030_async_v1"
lock = (root / "parallel/benchmark.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert shutil.disk_usage(root).free > 8 * 1024**3 + 512 * 1024**2
gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.free", "--format=csv,noheader,nounits"], text=True).strip().split(",")
assert gpu[0].strip() == frozen["gpu_uuid"] and gpu[1].strip() == "NVIDIA GeForce RTX 4090"
assert int(gpu[2]) >= 12288, "Two additional technical models plus stopped collector require12GiB free"


def jobs():
    return {int(v) for v in subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).split()}


assert jobs() == {current}
ids = frozen["cohorts"]["train64"][:6]
assert len(ids) == 6
sources = {p: sha(root / p) for p in ("parallel/worker_v2.py", "parallel/parallel_storage.py", "parallel/benchmark_gate_v2.py")}
tech = {"status": "FROZEN_BEFORE_PARALLEL_TEST", "epoch": time.time(),
        "science_freeze_sha256": sha(frozen_path), "sources": sources, "ids": ids,
        "protocol": "Same6 completed original Train64 questions: one worker then two disjoint workers. Full2380 tensor byte checks plus entire kernel metadata, same22 original treatment slots, original AB/reference/donors. Isolated model/hook processes. Original scientific process SIGSTOP only during bounded test, always SIGCONT finally; no scientific result overwritten.",
        "criterion": "All6 PASS in both modes and >5% matched-block throughput gain for considering production. No output-dependent question or condition selection."}
tech_path = root / "parallel/TECH_FREEZE_v2.json"
with tech_path.open("x") as handle:
    json.dump(tech, handle, indent=2)
sources["parallel/TECH_FREEZE_v2.json"] = sha(tech_path)
children = []
phases = []
paused = False
environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
try:
    os.kill(current, signal.SIGSTOP); paused = True
    time.sleep(1)
    assert pathlib.Path(f"/proc/{current}/stat").read_text().split()[2] in ("T", "t")
    for count in (1, 2):
        directory = root / "parallel/benchmark_v2" / ("serial" if count == 1 else "two_workers")
        directory.mkdir(parents=True, exist_ok=False)
        output = directory / "evidence"
        output.mkdir()
        spec = {"mode": "benchmark", "split": "train", "science_freeze_sha256": sha(frozen_path),
                "sources": sources, "assignments": [ids[i::count] for i in range(count)],
                "control_dir": str(directory), "output_dir": str(output), "namespace_prefix": "worker_CAS"}
        spec_path = directory / "SPEC.json"
        spec_path.write_text(json.dumps(spec, indent=2))
        active = []
        for i in range(count):
            logfile = (directory / f"worker{i}.log").open("x")
            child = subprocess.Popen([PY, str(root / "parallel/worker_v2.py"), "--root", str(root),
                                      "--spec", str(spec_path), "--index", str(i)], env=environment,
                                     stdin=subprocess.DEVNULL, stdout=logfile, stderr=subprocess.STDOUT)
            logfile.close(); active.append(child); children.append(child)
        deadline = time.monotonic() + 150
        while not all((directory / f"worker{i}_READY.json").exists() for i in range(count)):
            assert all(child.poll() is None for child in active), "Worker failed during loading; read logs"
            assert time.monotonic() < deadline, "Worker readiness timeout"
            time.sleep(.25)
        allowed = {current, *[child.pid for child in active]}
        assert jobs() <= allowed
        (directory / "GO.json").write_text(json.dumps({"spec_sha256": sha(spec_path)}))
        start = time.perf_counter()
        samples = []
        while any(child.poll() is None for child in active):
            assert time.perf_counter() - start < 180, "Matched-block phase timeout"
            assert all(child.poll() in (None, 0) for child in active), "Worker failed; no production switch"
            assert jobs() <= allowed
            sample = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.free", "--format=csv,noheader,nounits"], text=True).strip()
            samples.append([int(value.strip()) for value in sample.split(",")])
            assert samples[-1][2] >= 1024, "Technical memory reserve crossed"
            time.sleep(.5)
        assert all(child.returncode == 0 for child in active)
        reports = [read(directory / f"worker{i}_RESULT.json") for i in range(count)]
        assert all(r["status"] == "PASS" and r["spec_sha256"] == sha(spec_path) for r in reports)
        assert sorted(v["id"] for r in reports for v in r["records"]) == sorted(ids)
        phase = {"workers": count, "n": len(ids), "coordinator_wall_seconds": time.perf_counter() - start,
                 "steady_worker_wall_seconds": max(r["wall_seconds"] for r in reports),
                 "gpu_mean_percent": sum(s[0] for s in samples) / len(samples),
                 "peak_memory_MiB_including_paused_original": max(s[1] for s in samples),
                 "minimum_free_memory_MiB": min(s[2] for s in samples), "samples": samples,
                 "worker_report_sha256": {f"worker{i}_RESULT.json": sha(directory / f"worker{i}_RESULT.json") for i in range(count)}}
        phases.append(phase)
        print("PARALLEL_PHASE_PASS", json.dumps({k: v for k, v in phase.items() if k != "samples"}), flush=True)
    ratio = phases[0]["steady_worker_wall_seconds"] / phases[1]["steady_worker_wall_seconds"]
    result = {"status": "PASS", "n_matched_cases": 6, "phases": phases,
              "observed_speedup_two_workers": ratio, "production_candidate": ratio > 1.05,
              "technical_freeze_sha256": sha(tech_path),
              "scope": "Matched6-case execution benchmark; full endpoint/capture/metadata equality to saved original serial evidence. Not new mechanism effects, full-split inference or a universal speed guarantee."}
    (root / "parallel/BENCHMARK_RESULT_v2.json").write_text(json.dumps(result, indent=2))
except BaseException as error:
    (root / "parallel/BENCHMARK_FAILURE_v2.json").write_text(json.dumps({"status": "FAILED", "error": repr(error),
             "traceback": traceback.format_exc(), "completed_phases": phases}, indent=2))
    raise
finally:
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill(); child.wait()
    for _ in range(30):
        if jobs() <= {current}:
            break
        time.sleep(.2)
    if paused:
        os.kill(current, signal.SIGCONT)
        (root / "parallel/ORIGINAL_RESUMED_v2.json").write_text(json.dumps({"status": "SIGCONT_SENT", "pid": current, "epoch": time.time()}))
