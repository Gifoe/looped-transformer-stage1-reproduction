"""Freeze pre-effect discovery64 from natural train groups only."""
import json,hashlib,gzip,datetime,os
from pathlib import Path
R=Path('/home/luansiyu/loop-latent');O=R/'research_plans/natural_qkv_eight_corner_20261112';B=R/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';p=O/'train_analysis/rows.jsonl.gz';c=O/'train_independent_cuda_coverage.json';s=O/'train_analysis/summary.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
summary=json.loads(s.read_text());cov=json.loads(c.read_text());assert sha(c)=='d257487676f31d0d122a913caad2c85369629ce1acca21bdabb5b2de0603c381' and summary['coverage_sha256']==sha(c) and summary['rows_sha256']==sha(p) and len(cov['questions'])==1119
with gzip.open(p,'rt') as f:r=[json.loads(x) for x in f]
assert len(r)==1119 and sum(x['demo'] for x in r)==25 and len({x['id'] for x in r})==1119
out={};seed='e1-e3-discovery64-20261112'
for group in ('rescue','damage','stable_correct','stable_wrong'):
 cand=[]
 for x in r:
  if x['demo']:continue
  y=x['gold'];pred=[max(range(len(a)),key=lambda j:a[j]) for a in x['natural_scores']];g={(False,True):'rescue',(True,False):'damage',(True,True):'stable_correct',(False,False):'stable_wrong'}[(pred[0]==y,pred[1]==y)];assert g==x['group']
  if g==group:cand.append((hashlib.sha256((seed+'\0'+x['id']).encode()).hexdigest(),x['id']))
 assert len(cand)>=16;out[group]=[{'id':i,'selection_sha256':h} for h,i in sorted(cand)[:16]]
assert len({x['id'] for rows in out.values() for x in rows})==64
result={'status':'PRE_EFFECT_DISCOVERY64_FROZEN','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seed':seed,'count':64,'selection':'train non-demo natural groups, each 16 smallest sha256(seed+NUL+id); independent of new ADD effects; natural groups recomputed from original K1/K2','train_coverage_sha256':sha(c),'train_rows_sha256':sha(p),'train_summary_sha256':sha(s),'groups':out,'interpretation':'stratified equal group counts; not representative train average; old ARC outcomes exposed; train remainder 1030 cannot be claimed independent blind'};dest=B/'orders/E1_DISCOVERY64_FROZEN.json'
with dest.open('x') as f:json.dump(result,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print('E1_DISCOVERY64_FROZEN',sha(dest),flush=True)
