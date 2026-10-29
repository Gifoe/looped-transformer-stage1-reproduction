"""Durable sequential collection, then bounded CPU heads and policy lock."""
import json,sys,subprocess,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def save(x): (ROOT/'logs/development_state.json').write_text(json.dumps(x,indent=2),encoding='utf8')
def main():
 discovery=json.loads((ROOT/'results/discovery_analysis.json').read_text())
 assert discovery['n']==400 and discovery['training_free_first']
 assert json.loads((ROOT/'logs/discovery_exit.json').read_text())['exit_code']==0
 for stage in ['head_train','validation']:
  assert not (ROOT/'logs'/(stage+'_exit.json')).exists()
  save(dict(pid=os.getpid(),stage=stage,started_unix=time.time()))
  result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/supervise.py'),stage])
  assert result.returncode==0
  assert json.loads((ROOT/'logs'/(stage+'_exit.json')).read_text())['exit_code']==0
  assert json.loads((ROOT/'results'/(stage+'_COMPLETE.json')).read_text())['passed']
 save(dict(pid=os.getpid(),stage='cpu_heads_and_lock',started_unix=time.time()))
 with (ROOT/'logs/head_analysis.out').open('w',encoding='utf8') as out:
  result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/analyze.py'),'train'],stdout=out,stderr=subprocess.STDOUT)
 assert result.returncode==0
 save(dict(pid=os.getpid(),stage='development_complete',finished_unix=time.time(),final_lock_exists=(ROOT/'configs/FINAL_LOCK.json').exists()))
if __name__=='__main__':main()
