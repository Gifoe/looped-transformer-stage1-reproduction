"""Describe cached E1 internal vector and fixed-head diagnostic projections."""
import gzip,json,statistics,hashlib,os
from pathlib import Path
import numpy as np
B=Path('/home/luansiyu/loop-latent/repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/E1');source=B/'E1_existing_corner_projection.jsonl.gz';receipt=json.loads((B/'E1_existing_corner_projection.receipt.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)==receipt['output_sha256'] and receipt['count']==2590
with gzip.open(source,'rt') as f:r=[json.loads(x) for x in f]
assert len(r)==2590 and len({(x['split'],x['id']) for x in r})==2590
out={'status':'E1_CPU_DIAGNOSTIC_SUMMARY','scope':'FP64 linear o_proj and Wgamma projection; actual a is post-localnorm; difference includes GPU o_proj rounding and localnorm, not localnorm alone. Wgamma excludes final RMS denominator and is not native score. Attention weights not observed.','source_sha256':sha(source),'groups':{}}
for split in ('train','validation','test'):
 groups={'all':[x for x in r if x['split']==split],'natural_no_tie':[x for x in r if x['split']==split and x['natural_no_top_tie']]}
 if split=='train':groups.update({'non_demo':[x for x in r if x['split']==split and not x['demo']],'demo':[x for x in r if x['split']==split and x['demo']]})
 for g in ('rescue','damage','stable_correct','stable_wrong'):groups[g]=[x for x in r if x['split']==split and x['group']==g]
 out['groups'][split]={}
 for group,rows in groups.items():
  if not rows:continue
  z={'N':len(rows),'query':{}}
  for i in ('1','2'):
   rr=[x['Q'][i] for x in rows];k={}
   for name in ('sdpa_interaction_l2','oproj_linear_interaction_l2','actual_a_interaction_l2','linear_vs_actual_cos','linear_wgamma_allwrong','actual_a_wgamma_allwrong','linear_wgamma_fixedwrong','actual_a_wgamma_fixedwrong','score_gap_interaction'):
    a=np.array([v[name] for v in rr if v[name] is not None]);k[name]={'N':len(a),'mean':float(a.mean()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'q05':float(np.quantile(a,.05)) if len(a) else None,'q95':float(np.quantile(a,.95)) if len(a) else None}
   z['query'][i]=k
  out['groups'][split][group]=z
p=B/'E1_existing_corner_summary.json'
with p.open('x') as f:json.dump(out,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
print('E1_SUMMARY_COMPLETE',sha(p),flush=True)
