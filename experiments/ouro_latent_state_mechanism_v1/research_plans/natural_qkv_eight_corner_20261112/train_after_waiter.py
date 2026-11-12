"""Wait for train-only collector; perform one audit then one train analysis."""
import sys,time,subprocess
from pathlib import Path
BASE=Path('/home/luansiyu/loop-latent/research_plans/natural_qkv_eight_corner_20261112');unit='ouro-qkv-eight-train-20261112.service'
for _ in range(2880):
 p=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','Result'],capture_output=True,text=True,check=True);s=dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line)
 if s.get('ActiveState') in ('inactive','failed'):
  if s.get('Result')!='success' or not (BASE/'train/receipt.json').is_file():raise RuntimeError('train collector did not complete')
  for name in ('train_independent_audit.py','train_factorial_analysis.py'):
   print('START',name,flush=True);r=subprocess.run([sys.executable,'-u',str(BASE/name)])
   if r.returncode:print('STOP',name,r.returncode,flush=True);sys.exit(r.returncode)
  print('TRAIN_AUDIT_AND_ANALYSIS_COMPLETE',flush=True);sys.exit(0)
 time.sleep(30)
raise TimeoutError('train wait limit')
