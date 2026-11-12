"""One-way E1 continuation: independently audited train -> validation -> test."""
import os,time,json,subprocess,sys
from pathlib import Path
R=Path('/home/luansiyu/loop-latent'); B=R/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112'; E=B/'E1'; py=str(R/'conda-env/bin/python')
if not __debug__:raise RuntimeError('optimized python forbidden')
a=E/'remainder_cuda_audit.json';receipt=E/'train/remainder_receipt.json'
print('WAITING_TRAIN_INDEPENDENT_AUDIT',flush=True)
receipt_seen_at=None
while not a.exists():
 if receipt.exists():
  if receipt_seen_at is None:receipt_seen_at=time.monotonic()
  q=subprocess.run(['pgrep','-f','[e]1_train_remainder_(collector|audit).py'],capture_output=True,text=True)
  if time.monotonic()-receipt_seen_at>60 and q.returncode!=0:
   raise RuntimeError('Train receipt exists but independent audit stopped without success; no split launched')
 time.sleep(15)
r=json.loads(a.read_text())
if r.get('status')!='TRAIN_REMAINDER_INDEPENDENT_CUDA_SCORE_AND_SOURCE_AUDIT_PASSED' or r.get('count')!=1055:raise RuntimeError('train audit failed')
for name,args in [('e1_train_remainder_analysis.py',[]),('e1_train_remainder_fraction.py',[]),('e1_split_collector.py',['validation']),('e1_split_audit.py',['validation']),('e1_split_analysis.py',['validation']),('e1_split_fraction.py',['validation']),('e1_split_collector.py',['test']),('e1_split_audit.py',['test']),('e1_split_analysis.py',['test']),('e1_split_fraction.py',['test'])]:
 print('START',name,*args,flush=True); subprocess.run([py,'-u',str(E/name),*args],check=True);print('FINISHED',name,*args,flush=True)
print('E1_THREE_SPLIT_AUDITS_COMPLETE',flush=True)
