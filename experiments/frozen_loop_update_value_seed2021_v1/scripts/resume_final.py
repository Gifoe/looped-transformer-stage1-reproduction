"""Continue only the audited PIQA operational recovery, then unchanged analyses."""
import json,sys,subprocess,time,hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def save(stage):(ROOT/'logs/final_state.json').write_text(json.dumps(dict(stage=stage,pid=os.getpid(),recorded_unix=time.time()),indent=2),encoding='utf8')
lock=json.loads((ROOT/'configs/FINAL_LOCK.json').read_text())
for rel,h in lock['frozen_files'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
assert json.loads((ROOT/'logs/final_sciq_exit.json').read_text())['exit_code']==0
assert json.loads((ROOT/'results/PIQA_OPERATIONAL_RECOVERY.json').read_text())['passed']
save('final_piqa_resume')
result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/supervise.py'),'final_piqa']);assert result.returncode==0
assert json.loads((ROOT/'logs/final_piqa_exit.json').read_text())['exit_code']==0
assert json.loads((ROOT/'results/final_piqa_COMPLETE.json').read_text())['n']==1802
for script in ['finalize_analysis.py','audit_results.py','diagnose_signals.py','render_report.py','render_appendix.py']:
 save(script)
 with (ROOT/'logs'/(script+'.out')).open('w',encoding='utf8') as out:result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts'/script)],stdout=out,stderr=subprocess.STDOUT)
 assert result.returncode==0,script
save('final_complete')
