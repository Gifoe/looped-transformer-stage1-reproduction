"""Evaluate frozen criteria only after each independent split analysis finishes."""
import time,subprocess,sys
from pathlib import Path
b=Path('/home/luansiyu/loop-latent/research_plans/natural_qkv_eight_corner_20261112')
for split in ('validation','test'):
 for i in range(2880):
  if (b/(split+'_analysis/summary.json')).is_file():break
  p=subprocess.run(['systemctl','--user','show','ouro-qkv-eight-split-after-20261112.service','-p','ActiveState','-p','Result'],capture_output=True,text=True,check=True)
  if 'ActiveState=failed' in p.stdout or ('ActiveState=inactive' in p.stdout and 'Result=success' not in p.stdout):raise RuntimeError('split chain failed')
  time.sleep(30)
 else:raise TimeoutError(split)
 subprocess.run([sys.executable,'-u',str(b/'evaluate_prospective.py'),split],check=True)
print('PROSPECTIVE_EVALUATION_COMPLETE',flush=True)
