"""Wait for approved train CPU audit then perform train-only analysis once."""
import sys,time,subprocess,json
from pathlib import Path
cov=Path('/home/luansiyu/loop-latent/research_plans/natural_qkv_train_preliminary_20261112/coverage.json')
unit='ouro-natural-qkv-train-audit-wait-20261112.service'
for tick in range(1440):
 p=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','Result'],capture_output=True,text=True,check=True);state=dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line)
 if state.get('ActiveState') in ('failed','inactive'):
  if state.get('Result')!='success' or not cov.is_file() or json.loads(cov.read_text())['status']!='CPU_SOURCE_ENDPOINT_PASS_INDEPENDENT_CUDA_PENDING':raise RuntimeError('train audit did not pass')
  print('TRAIN_AUDIT_COMPLETE_START_ANALYSIS',flush=True);r=subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('train_qkv_analysis.py'))]);sys.exit(r.returncode)
 time.sleep(30)
raise TimeoutError('train audit wait limit reached')
