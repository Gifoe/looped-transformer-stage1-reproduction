"""Observational timing only; compares different consecutive real question blocks."""
import hashlib
import json
import pathlib
import re
import statistics
import subprocess
import time

root = pathlib.Path(__file__).resolve().parents[1]
receipt = json.loads((root / "async_io/TRANSITION_RECEIPT.json").read_text())
frozen = json.loads((root / "B2_FULL_SOURCE_FREEZE.json").read_text())
retained = receipt["completed_metadata"]
ids = frozen["cohorts"]["train1030"]
assert [v["id"] for v in retained] == ids[:len(retained)]
data = root / "b2_data/treatments/train1030"
times_old = [(data / item["metadata_file"]).stat().st_mtime for item in retained[-21:]]
log = root / "b2_queue_treatments_train1030_async_v1.log"
end = time.time() + 1200
notified = False
while time.time() < end:
    state = json.loads((root / "B2_QUEUE_STATE.json").read_text())
    assert state["status"] != "FAILED_STOPPED", state
    text = log.read_text() if log.exists() else ""
    counts = re.findall(r"B2_CASE treatment train1030 (\d+) 1030", text)
    count = int(counts[-1]) if counts else 0
    if count > len(retained) and not notified:
        print("FIRST_NEW_ASYNC_CASE_COMMITTED", count, flush=True)
        notified = True
    if count >= len(retained) + 12:
        break
    time.sleep(10)
else:
    raise RuntimeError("No sufficient new case timing within20minutes")
times_new = [(data / (hashlib.sha256(ident.encode()).hexdigest() + ".json")).stat().st_mtime
             for ident in ids[len(retained):len(retained) + 12]]
old_intervals = [b - a for a, b in zip(times_old, times_old[1:])]
new_intervals = [b - a for a, b in zip(times_new, times_new[1:])]
gpu = subprocess.check_output(["nvidia-smi", "dmon", "-s", "pucm", "-d", "1", "-c", "20"], text=True)
samples = [int(line.split()[4]) for line in gpu.splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
result = {"status": "MEASURED_OBSERVATIONAL_THROUGHPUT", "epoch": time.time(),
          "old_block_n_intervals": len(old_intervals), "new_block_n_intervals": len(new_intervals),
          "old_mean_seconds_per_case": statistics.mean(old_intervals),
          "old_median_seconds_per_case": statistics.median(old_intervals),
          "new_mean_seconds_per_case": statistics.mean(new_intervals),
          "new_median_seconds_per_case": statistics.median(new_intervals),
          "observed_throughput_ratio": statistics.mean(old_intervals) / statistics.mean(new_intervals),
          "gpu_utilization_mean_percent_20s": statistics.mean(samples),
          "gpu_utilization_range_percent_20s": [min(samples), max(samples)],
          "old_ids": [r["id"] for r in retained[-21:]],
          "new_ids": ids[len(retained):len(retained) + 12],
          "scope": "Different consecutive case blocks, not paired controlled speed benchmark. Resume revalidation excluded. Scientific effects not inspected or used for timing selection."}
(root / "async_io/THROUGHPUT_OBSERVATION.json").write_text(json.dumps(result, indent=2))
(root / "async_io/GPU_DMON_AFTER_SWITCH.txt").write_text(gpu)
print(json.dumps(result), flush=True)
