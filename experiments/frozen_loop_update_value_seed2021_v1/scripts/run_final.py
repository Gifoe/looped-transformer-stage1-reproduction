"""Run untouched final tasks only after the full policy/source lock is verified."""
import json,sys,subprocess,time,hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def save(x):(ROOT/'logs/final_state.json').write_text(json.dumps(x,indent=2),encoding='utf8')
def main():
 lock=json.loads((ROOT/'configs/FINAL_LOCK.json').read_text())
 assert json.loads((ROOT/'logs/development_state.json').read_text())['stage']=='development_complete'
 for rel,h in lock['frozen_files'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
 for stage in ['final_sciq','final_piqa']:
  assert not (ROOT/'logs'/(stage+'_exit.json')).exists()
  save(dict(stage=stage,pid=os.getpid(),started_unix=time.time(),lock_frozen_unix=lock['frozen_unix']))
  result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/supervise.py'),stage]);assert result.returncode==0
  receipt=json.loads((ROOT/'logs'/(stage+'_exit.json')).read_text());assert receipt['exit_code']==0 and receipt['started_unix']>lock['frozen_unix']
  assert json.loads((ROOT/'results'/(stage+'_COMPLETE.json')).read_text())['passed']
 for script in ['finalize_analysis.py','audit_results.py','diagnose_signals.py','render_report.py','render_appendix.py']:
  save(dict(stage=script,pid=os.getpid(),started_unix=time.time()))
  with (ROOT/'logs'/(script+'.out')).open('w',encoding='utf8') as out:
   result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts'/script)],stdout=out,stderr=subprocess.STDOUT)
  assert result.returncode==0,script
 save(dict(stage='final_complete',pid=os.getpid(),finished_unix=time.time()))
if __name__=='__main__':main()
