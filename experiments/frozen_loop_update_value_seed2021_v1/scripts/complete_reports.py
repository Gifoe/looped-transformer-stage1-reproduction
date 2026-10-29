"""CPU-only recovery after completed inference and primary statistical audits."""
import json,subprocess,sys,time,shutil,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
assert json.loads((ROOT/'results/INDEPENDENT_FINAL_AUDIT.json').read_text())['passed']
assert (ROOT/'results/FINAL_ANALYSIS.json').exists()
for name in ['diagnose_signals.py.out','final_exit.json']:
 p=ROOT/'logs'/name;dest=p.with_name('cpu_attempt1_'+name)
 assert not dest.exists();shutil.copy2(p,dest)
receipts=[]
for script in ['diagnose_signals.py','render_report.py','render_appendix.py']:
 (ROOT/'logs/final_state.json').write_text(json.dumps(dict(stage=script,pid=os.getpid(),cpu_recovery=True,recorded_unix=time.time()),indent=2),encoding='utf8')
 begin=time.time()
 with (ROOT/'logs'/(script+'.out')).open('w',encoding='utf8') as out:
  result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/report_worker.py'),script],stdout=out,stderr=subprocess.STDOUT)
 receipts.append(dict(script=script,returncode=result.returncode,seconds=time.time()-begin))
 (ROOT/'logs/CPU_REPORT_RECOVERY.json').write_text(json.dumps(receipts,indent=2),encoding='utf8')
 assert result.returncode==0,receipts[-1]
(ROOT/'logs/final_state.json').write_text(json.dumps(dict(stage='final_complete',pid=os.getpid(),finished_unix=time.time(),cpu_recovery=True),indent=2),encoding='utf8')
