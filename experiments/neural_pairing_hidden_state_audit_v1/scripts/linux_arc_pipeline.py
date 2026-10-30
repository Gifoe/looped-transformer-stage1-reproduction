"""Detached ARC priority worker with durable receipts and owned-process timeout."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from linux_runtime import ROOT,configure,save

configure()
lock=(ROOT/'logs/arc_pipeline.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status=dict(status='waiting_for_pilot',pid=os.getpid(),started_unix=time.time(),gpu_worker_seconds=0)
path=ROOT/'logs/ARC_PIPELINE_STATE.json'
save(path,status)
wait=time.time()
while True:
    pilot=ROOT/'fresh/pilot_01/receipt.json'
    if pilot.exists():
        d=json.loads(pilot.read_text())
        if d['status']=='failed':
            status.update(status='blocked_by_failed_pilot',pilot_error=d.get('error'));save(path,status);raise SystemExit(1)
        if d['status']=='passed':break
    assert time.time()-wait<1200,'pilot wait deadline; inspect existing pilot before relaunch'
    time.sleep(2)
start=time.time()
try:
    with (ROOT/'logs/arc_first.out').open('xb') as out,(ROOT/'logs/arc_first.err').open('xb') as err:
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/run_arc_first.py'),'--bs','16','--seconds','3600'],
            cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
        status.update(status='running',worker_pid=p.pid);save(path,status)
        try:code=p.wait(timeout=3900)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM);code=p.wait(timeout=60)
    status.update(worker_exit_code=code,gpu_worker_seconds=time.time()-start)
    if code:raise RuntimeError('ARC worker exited '+str(code))
    if not (ROOT/'fresh/arc_first/COMPLETE.json').exists():
        status.update(status='time_limited_incomplete');save(path,status);raise SystemExit(2)
    subprocess.run([sys.executable,str(ROOT/'scripts/analyze_arc_first.py')],cwd=ROOT,check=True,timeout=120)
    status.update(status='complete')
except BaseException as exc:
    status.update(status='failed' if status['status']!='time_limited_incomplete' else status['status'],error=repr(exc))
    raise
finally:
    status.update(finished_unix=time.time(),gpu_worker_seconds=time.time()-start);save(path,status)
