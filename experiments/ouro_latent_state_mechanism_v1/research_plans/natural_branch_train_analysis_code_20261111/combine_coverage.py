"""Join all CPU state audit ledgers to independent CUDA replay by exact identity and file hashes."""
import json,hashlib,os
from pathlib import Path
R=Path('/home/luansiyu/loop-latent/research_plans'); GPU=R/'natural_branch_gpu_replay_20261111.json';OUT=R/'NATURAL_BRANCH_INDEPENDENT_COVERAGE_20261111.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def require(x,msg):
 if not x:raise ValueError(msg)
def main():
 gpu=json.loads(GPU.read_text());require(gpu['status']=='GPU_LOGIT_REPLAY_PASSED_NOT_FULL_STATE_COVERAGE' and gpu['count']==2590,'GPU status')
 gm={(x['split'],x['id']):x for x in gpu['questions']};require(len(gm)==2590,'GPU duplicate')
 result={};expected_total=set()
 for split,n in [('train',1119),('validation',299),('test',1172)]:
  d=R/('natural_branch_'+split+'_exploration_20261111');cp=d/(split+'_coverage_preliminary.json');c=json.loads(cp.read_text());rp=d/(split+'_rows.json');sp=d/(split+'_summary.json')
  require(c['status']=='PRELIMINARY_CPU_ANCHORS_PASS_CUDA_REPLAY_PENDING' and c['count']==n and len(c['questions'])==n and sha(rp)==c['rows_sha256'],'CPU coverage')
  summary=json.loads(sp.read_text());require(summary['coverage_sha256']==sha(cp) and summary['rows_sha256']==sha(rp),'summary binding')
  ids=set()
  for q in c['questions']:
   key=(split,q['id']);require(key not in ids and key in gm,'ID missing or duplicate');ids.add(key);g=gm[key]
   for field in ('natural_metadata_sha256','natural_tensor_sha256','branch_metadata_sha256','branch_tensor_sha256'):require(q[field]==g[field],'hash mismatch '+field)
  require(len(ids)==n,'ID count');expected_total|=ids
  result[split]={'count':n,'cpu_coverage_sha256':sha(cp),'rows_sha256':sha(rp),'summary_sha256':sha(sp)}
 require(expected_total==set(gm),'unmatched GPU identities')
 evidence={'status':'INDEPENDENT_CPU_STATE_AND_GPU_LOGIT_REPLAY_PASSED','counts':{s:v['count'] for s,v in result.items()},'splits':result,'gpu_replay_sha256':sha(GPU),'collector_receipt_sha256':gpu['collector_receipt_sha256'],'gpu_runtime':gpu['runtime'],'cpu_scope':'state/logit bit anchors, residuals, continuity, source and independently recomputed FP64 lens/accounting','gpu_scope':'saved full BF16 logits to same-runtime CUDA FP32 logsoftmax/gather/sum, exact scores, attention parameters/positions audit','limitation':'No causal identification; saved attention arguments, not underlying kernel identity.','script_sha256':sha(Path(__file__))}
 with OUT.open('x') as f:json.dump(evidence,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('COMBINED_COVERAGE_PASS',sha(OUT),flush=True)
if __name__=='__main__':main()
