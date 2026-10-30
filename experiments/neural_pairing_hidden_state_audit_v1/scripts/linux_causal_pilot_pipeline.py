"""Detached bounded inference-only causal pilot; original training remains deferred."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from linux_runtime import ROOT,configure,save

configure()
lock=(ROOT/'logs/causal_pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=dict(status='checking_instrumentation',pid=os.getpid(),started_unix=time.time(),gpu_worker_seconds=0)
receipt=ROOT/'logs/CAUSAL_PILOT_STATE.json';save(receipt,state)
env=dict(os.environ,NP_UPSTREAM=str(ROOT/'upstream'),NP_MODEL_ROOT=str(ROOT/'models'))
previous=ROOT/'logs/CAUSAL_TRACE_CHECK.json'
if previous.exists() and not (ROOT/'logs/CAUSAL_TRACE_CHECK_server_v1.json').exists():
    (ROOT/'logs/CAUSAL_TRACE_CHECK_server_v1.json').write_bytes(previous.read_bytes())
with (ROOT/'logs/causal_instrument_check.out').open('xb') as out,(ROOT/'logs/causal_instrument_check.err').open('xb') as err:
    check=subprocess.run([sys.executable,str(ROOT/'scripts/check_causal_trace.py')],cwd=ROOT,env=env,stdout=out,stderr=err,timeout=180)
if check.returncode:
    state.update(status='failed_instrumentation',exit_code=check.returncode);save(receipt,state);raise SystemExit(check.returncode)
start=time.time()
try:
    with (ROOT/'logs/causal_pilot.out').open('xb') as out,(ROOT/'logs/causal_pilot.err').open('xb') as err:
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/run_causal_pilot.py'),'--bs','2'],cwd=ROOT,env=env,
            stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
        state.update(status='running',worker_pid=p.pid);save(receipt,state)
        try:code=p.wait(timeout=2400)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM);code=p.wait(timeout=60)
    state.update(exit_code=code,status='complete' if code==0 and (ROOT/'fresh/causal_pilot_16/COMPLETE.json').exists() else 'failed')
finally:
    state.update(finished_unix=time.time(),gpu_worker_seconds=time.time()-start);save(receipt,state)
