"""Own the diagnostic child handle and record its exact native return code."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import time

root=Path(__file__).resolve().parents[1]
backup=root/'logs/recovery_validation1p4_20261026'
for name in ['results/recovery_boundary_replay.json','results/recovery_next_case_probe.json','logs/recovery_probe_exit.json']:
    path=root/name;target=backup/(path.stem+'_attempt1'+path.suffix)
    if target.exists():assert target.read_bytes()==path.read_bytes(),'Attempt1 preservation differs'
    else:shutil.copyfile(path,target)
started=time.time();timed_out=False
with (root/'logs/recovery_probe_attempt2_stdout.log').open('w',encoding='utf-8') as out,(root/'logs/recovery_probe_attempt2_stderr.log').open('w',encoding='utf-8') as err:
    child=subprocess.Popen(['D:/loopbase_env_20261025/Scripts/python.exe','-u','-X','utf8',str(root/'scripts/recovery_boundary_replay.py')],cwd=root,stdout=out,stderr=err,env=dict(os.environ,PYTHONUTF8='1',PYTHONFAULTHANDLER='1'))
    try:code=child.wait(timeout=600)
    except subprocess.TimeoutExpired:
        timed_out=True
        subprocess.run(['taskkill.exe','/PID',str(child.pid),'/T','/F'],stdout=err,stderr=err)
        child.wait();code=124
result=dict(controller_pid=os.getpid(),child_redirector_pid=child.pid,started_unix=started,finished_unix=time.time(),exit_code=code,timeout=timed_out)
(root/'logs/recovery_probe_exit_attempt2.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
raise SystemExit(code)
