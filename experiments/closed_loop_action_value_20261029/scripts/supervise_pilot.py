"""Bound one unchanged worker attempt and preserve its stdout/stderr and receipt."""
import subprocess, sys, time, json, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
assert not (ROOT/'results/pilot.jsonl').exists(), 'Inspect partial data before any retry'
start=time.time()
with (ROOT/'results/worker.out').open('xb') as out, (ROOT/'results/worker.err').open('xb') as err:
    child=subprocess.Popen([sys.executable,'-X','utf8','-X','faulthandler','-u',str(ROOT/'scripts/pilot.py')],stdout=out,stderr=err)
    (ROOT/'results/ACTIVE.json').write_text(json.dumps(dict(pid=child.pid,parent_pid=os.getpid(),start_unix=start)),encoding='utf8')
    try:code=child.wait(timeout=1200)
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill.exe','/PID',str(child.pid),'/T','/F'],stdout=err,stderr=err,check=True)
        code=child.wait()
receipt=dict(exit_code=code,start_unix=start,end_unix=time.time(),charged_seconds=time.time()-start,pid=child.pid)
(ROOT/'results/SUPERVISOR_EXIT.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
print(json.dumps(receipt),flush=True)
