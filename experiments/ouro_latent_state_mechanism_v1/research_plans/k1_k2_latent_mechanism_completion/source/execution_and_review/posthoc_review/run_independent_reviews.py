"""Read-only review queue; waits for each existing scientific stage to finish."""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback

root = pathlib.Path(__file__).resolve().parents[1]
output = root / "posthoc_review"
script = output / "independent_B2_review.py"
script_sha = hashlib.sha256(script.read_bytes()).hexdigest()


def state(status, **fields):
    path = output / "QUEUE_STATE.json"
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps({"status": status, "epoch": time.time(),
                                  "review_script_sha256": script_sha, **fields}, indent=2))
    os.replace(partial, path)


try:
    assert not (output / "QUEUE_STATE.json").exists()
    for stage in ("train64", "train1030", "validation299"):
        state("WAITING_FOR_ORIGINAL_STAGE", stage=stage)
        while not (root / "b2_analysis" / stage / "COMPLETE.json").exists():
            original = json.loads((root / "B2_QUEUE_STATE.json").read_text())
            assert original["status"] != "FAILED_STOPPED", original
            time.sleep(30)
        assert hashlib.sha256(script.read_bytes()).hexdigest() == script_sha
        state("REVIEWING", stage=stage)
        with (output / (stage + ".log")).open("x") as logfile:
            subprocess.run([sys.executable, str(script), "--root", str(root), "--stage", stage],
                           check=True, stdout=logfile, stderr=subprocess.STDOUT)
    state("COMPLETE", scope="Scalar review only; no whole-research completion claim")
except BaseException as error:
    state("FAILED_STOPPED", error=repr(error), traceback=traceback.format_exc())
    raise
