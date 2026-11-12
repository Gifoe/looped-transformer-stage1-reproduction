"""Wait for all train metadata, then run approved train CPU audit once."""
import os,time,subprocess,sys
from pathlib import Path
run=Path('/home/luansiyu/loop-latent/research_plans/natural_qkv_full_20261112')
for tick in range(1440):
 n=len(list(run.glob('train_*.json')))
 if n==1119:
  print('TRAIN_COMPLETE_START_CPU_AUDIT',flush=True)
  result=subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('train_qkv_audit.py'))]);sys.exit(result.returncode)
 if n>1119:raise RuntimeError('unexpected train count')
 time.sleep(30)
raise TimeoutError('train waiting limit reached')
