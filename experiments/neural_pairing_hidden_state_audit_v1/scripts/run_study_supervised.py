"""Bounded detached inference study with separate CPU instrumentation gate."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from linux_runtime import ROOT,configure,save

ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);ap.add_argument('--seconds',type=int,default=3600)
ap.add_argument('--worker-script',default='run_causal_pilot.py')
ap.add_argument('arguments',nargs=argparse.REMAINDER);args=ap.parse_args()
assert args.study.replace('_','').isalnum()
assert args.worker_script in ['run_causal_pilot.py','run_feedback_validation.py']
configure()
lock=(ROOT/'logs'/(args.study+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=dict(status='checking_instrument',pid=os.getpid(),started_unix=time.time(),gpu_worker_seconds=0)
receipt=ROOT/'logs'/(args.study+'_STATE.json');save(receipt,state)
env=dict(os.environ,NP_UPSTREAM=str(ROOT/'upstream'),NP_MODEL_ROOT=str(ROOT/'models'))
check_path=ROOT/'logs/CAUSAL_TRACE_CHECK.json'
if check_path.exists():
    snapshot=ROOT/'logs'/('CAUSAL_TRACE_BEFORE_'+args.study+'.json')
    if not snapshot.exists():snapshot.write_bytes(check_path.read_bytes())
with (ROOT/'logs'/(args.study+'_check.out')).open('xb') as out,(ROOT/'logs'/(args.study+'_check.err')).open('xb') as err:
    check=subprocess.run([sys.executable,str(ROOT/'scripts/check_causal_trace.py')],cwd=ROOT,env=env,stdout=out,stderr=err,timeout=180)
if check.returncode:
    state.update(status='failed_check',exit_code=check.returncode);save(receipt,state);raise SystemExit(check.returncode)
start=time.time()
arguments=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
try:
    with (ROOT/'logs'/(args.study+'.out')).open('xb') as out,(ROOT/'logs'/(args.study+'.err')).open('xb') as err:
        worker=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts'/args.worker_script),*arguments],cwd=ROOT,env=env,
            stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
        state.update(status='running',worker_pid=worker.pid);save(receipt,state)
        try:code=worker.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            os.killpg(worker.pid,signal.SIGTERM);code=worker.wait(timeout=60)
    state.update(status='complete' if code==0 and (ROOT/'fresh'/args.study/'COMPLETE.json').exists() else 'failed',exit_code=code)
finally:
    state.update(finished_unix=time.time(),gpu_worker_seconds=time.time()-start);save(receipt,state)
