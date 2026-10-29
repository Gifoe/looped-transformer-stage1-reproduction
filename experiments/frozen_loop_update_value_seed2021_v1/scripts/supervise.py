"""Own-worker wall-time supervisor; conservative charge even on native failure."""
import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/protocol.json').read_text())
SHARED=ROOT.parent/'looped_stage1_trajectory_audit_seed2021_v1/extensions/baseline_suite_20261025_seed2021/logs/gpu_budget.json'
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2),encoding='utf8');tmp.replace(p)
def main():
 stage=sys.argv[1];ledger=ROOT/'logs/round_budget.json';lock=ROOT/'logs/supervisor.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 try:
  prior=json.loads(SHARED.read_text());roundlog=json.loads(ledger.read_text()) if ledger.exists() else dict(used_seconds=0,runs=[],historical_start_seconds=prior['used_seconds'])
  available=min(CFG['new_gpu_seconds']-roundlog['used_seconds'],CFG['cumulative_gpu_seconds']-prior['used_seconds'])-60
  assert available>120
  start=time.time();env=dict(os.environ,WORKER_DEADLINE=str(start+available))
  with (ROOT/'logs'/(stage+'.out')).open('wb') as out,(ROOT/'logs'/(stage+'.err')).open('wb') as err:
   p=subprocess.Popen([sys.executable,'-X','utf8',str(ROOT/'scripts/score.py'),stage],stdout=out,stderr=err,env=env)
   save(ROOT/'logs/active.json',dict(stage=stage,pid=p.pid,parent_pid=os.getpid(),started_unix=start,deadline=start+available))
   try:code=p.wait(timeout=available)
   except subprocess.TimeoutExpired:
    if os.name=='nt':subprocess.run(['taskkill.exe','/PID',str(p.pid),'/T','/F'],check=True,stdout=err,stderr=err)
    else:p.kill()
    code=p.wait()
  seconds=time.time()-start
  record=dict(stage=stage,pid=p.pid,started_unix=start,finished_unix=time.time(),seconds=seconds,exit_code=code,accounting='Entire subprocess interval including startup, parity, all views and failure')
  # Re-read, preserving any valid external ledger addition.
  latest=json.loads(SHARED.read_text());latest['used_seconds']+=seconds;latest['runs'].append(dict(record,tag='update_value_'+stage));latest['extension_used_seconds']=latest['used_seconds']-latest['stage1_used_seconds'];save(SHARED,latest)
  roundlog['used_seconds']+=seconds;roundlog['runs'].append(record);save(ledger,roundlog)
  save(ROOT/'logs'/(stage+'_exit.json'),record);save(ROOT/'logs/active.json',dict(record,finished=True));print(json.dumps(record),flush=True)
 finally:lock.unlink()
if __name__=='__main__':main()
