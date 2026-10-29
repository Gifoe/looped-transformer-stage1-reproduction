import subprocess,sys,time,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
assert not (ROOT/'results/development.jsonl').exists(),'Inspect before retry'
start=time.time()
with (ROOT/'results/worker.out').open('xb') as out,(ROOT/'results/worker.err').open('xb') as err:
    p=subprocess.Popen([sys.executable,'-X','utf8','-X','faulthandler','-u',str(ROOT/'scripts/run_development.py')],stdout=out,stderr=err)
    (ROOT/'results/ACTIVE.json').write_text(json.dumps(dict(pid=p.pid,parent_pid=os.getpid(),start_unix=start)),encoding='utf8')
    try:code=p.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill.exe','/PID',str(p.pid),'/T','/F'],check=True,stdout=err,stderr=err);code=p.wait()
record=dict(pid=p.pid,exit_code=code,start_unix=start,end_unix=time.time(),charged_seconds=time.time()-start)
(ROOT/'results/EXIT.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps(record),flush=True)
