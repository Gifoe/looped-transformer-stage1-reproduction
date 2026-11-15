"""Launch supplementary CPU checks, no GPU model or live-source mutations."""
import os
import pathlib
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
for script in ("verify_execution_lineage.py", "audit_first_new_cases.py"):
    log_path = root / "async_io" / (script + ".log")
    assert not log_path.exists()
    with log_path.open("x") as logfile:
        child = subprocess.Popen(["/root/autodl-tmp/ouro-e6a/venv/bin/python",
                                  str(root / "async_io" / script)], env=environment,
                                  cwd=root, stdin=subprocess.DEVNULL, stdout=logfile,
                                  stderr=subprocess.STDOUT, start_new_session=True)
    print(script, child.pid)
