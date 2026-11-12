"""One-shot independent replay after the collector has finished successfully."""
import sys,time,subprocess
from pathlib import Path
run=Path('/home/luansiyu/loop-latent/research_plans/natural_qkv_full_20261112')
unit='ouro-natural-qkv-full-20261112.service'
for tick in range(1440):
 p=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','Result'],capture_output=True,text=True,check=True);state=dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line)
 if state.get('ActiveState') in ('failed','inactive'):
  if state.get('Result')!='success' or not (run/'receipt.json').is_file():raise RuntimeError('collector ended without successful full receipt')
  print('COLLECTOR_COMPLETE_START_CUDA_REPLAY',flush=True);r=subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('gpu_qkv_audit.py'))]);sys.exit(r.returncode)
 time.sleep(30)
raise TimeoutError('collector wait limit reached')
