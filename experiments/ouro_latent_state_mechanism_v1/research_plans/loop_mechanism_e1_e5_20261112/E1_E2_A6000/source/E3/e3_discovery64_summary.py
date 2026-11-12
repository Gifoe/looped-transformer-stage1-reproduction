"""Summarize audited discovery64 value-position components by natural group only."""
import os,json,gzip,hashlib,datetime
from pathlib import Path
import numpy as np
B=Path('/home/luansiyu/loop-latent/repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112/E3/discovery64');A=B/'independent_audit.json';R=B/'audited_rows.jsonl.gz'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads(A.read_text());assert a['status']=='E3_DISCOVERY64_INDEPENDENT_SAVED_PARTITION_AUDIT_PASSED' and a['count']==64 and a['audited_rows_sha256']==sha(R)
with gzip.open(R,'rt') as f:rows=[json.loads(x) for x in f]
assert len(rows)==64 and len({r['id'] for r in rows})==64
out={'status':'E3_DISCOVERY64_VALUE_POSITION_EXPLORATORY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'audit_sha256':sha(A),'rows_sha256':sha(R),'scope':'16 per natural group, not population-weighted. Components allocate ideal fixed-query attention interaction by value-change position. Key changes affect the full softmax, so a P/Q/R value contribution does not identify key source or semantic origin. O-proj/Wgamma are fixed linear diagnostics, not model scores; residual includes BF16 and ADD rounding. Full QKV was streamed, not archived.','groups':{}}
for g in ('rescue','damage','stable_correct','stable_wrong'):
 rr=[r for r in rows if r['group']==g];assert len(rr)==16;out['groups'][g]={}
 for q in ('1','2'):
  out['groups'][g][q]={}
  for block in ('P','Q','R','numeric_residual'):
   z=[r['q'][q][block] for r in rr];out['groups'][g][q][block]={}
   for k in ('vector_l2','oproj_linear_l2','wgamma_allwrong','wgamma_fixedwrong'):
    v=np.array([x[k] for x in z]);out['groups'][g][q][block][k]={'N':len(v),'mean':float(np.mean(v)),'median':float(np.median(v)),'q05':float(np.quantile(v,.05)),'q95':float(np.quantile(v,.95)),'positive':int((v>0).sum()),'negative':int((v<0).sum()),'zero':int((v==0).sum())}
p=B/'summary.json'
with p.open('x') as f:json.dump(out,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
print('E3_DISCOVERY64_SUMMARY_COMPLETE',sha(p),flush=True)
