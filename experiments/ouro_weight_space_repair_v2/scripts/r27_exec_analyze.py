"""Full saved-vector integrity before any label evaluation/effect summary."""
from r27_exec_common import ROOT,LOCK,read,sha,save,gate
from r27_exec_protocol import ordered_instances,execution_schema,label_metrics,functional_summary,check_resources

def main():
 import numpy as np,hashlib
 spec=gate();c=read('results/r27_EXEC_COMPLETE.json');sup=read('logs/r27_EXEC_SUPERVISOR.json')
 ledgers=spec['budget']['inherited_ledger_sha256'];assert len(ledgers)==15
 for p,h in ledgers.items():assert sha(p)==h
 inherited=sum(read(p)['used_seconds'] for p in ledgers);assert inherited==18874.135196831077
 book=read('logs/r27_EXEC_gpu_budget.json');check_resources(sup,book,[read(p)['used_seconds'] for p in ledgers])
 assert c['protocol_sha256']==sha(LOCK) and c['raw_sha256']==sha('results/r27_EXEC_RAW.json') and c['vectors_sha256']==sha('results/r27_EXEC_VECTORS.npz') and c['input_audit_sha256']==sha('results/r27_EXEC_INPUT_AUDIT.json')
 assert read('results/r27_EXEC_INPUT_AUDIT.json')==read('results/r27_EXEC_INPUT_PREFLIGHT.json')
 assert len(c['all_parameter_hashes_before'])==269 and c['all_parameter_hashes_before']==c['all_parameter_hashes_after']==read('results/r26_COMPLETE.json')['all_parameter_hashes_after']
 raw=read('results/r27_EXEC_RAW.json');assert c['cleanup']==raw['cleanup']
 if c['status']=='invalid':save('results/r27_EXEC_ANALYSIS.json',dict(status='invalid_no_functional_interpretation',reasons=raw['invalid_reasons']));return
 phaseB=c['status']=='complete_phaseB';assert c['status'] in ['complete_phaseB','complete_phaseA_not_admitted']
 count=execution_schema(raw,ordered_instances(read('configs/r27_CPU_STRUCTURE_MANIFEST.json')),phaseB,spec['projection_width']);assert c['forwards']==count and c['baselines']==8 and c['patches']==(24 if phaseB else 0)
 references={}
 def scan(x):
  if isinstance(x,dict):
   if set(x)=={'key','dtype','shape','raw_sha256'}:
    key=x['key']
    if key in references:assert references[key]==x
    references[key]=x
   else:
    for v in x.values():scan(v)
  elif isinstance(x,list):
   for v in x:scan(v)
 scan(raw)
 with np.load(ROOT/'results/r27_EXEC_VECTORS.npz',allow_pickle=False) as z:
  assert set(z.files)==set(references)
  for key,r in references.items():
   a=z[key];assert list(a.shape)==r['shape'] and a.dtype==np.dtype('uint16' if r['dtype']=='torch.bfloat16' else 'float32') and r['dtype'] in ['torch.bfloat16','torch.float32']
   b=a.tobytes();assert hashlib.sha256(b).hexdigest()==r['raw_sha256'] and hashlib.sha256((r['dtype']+str(tuple(r['shape']))).encode()+b).hexdigest()==key
   decoded=(a.astype(np.uint32)<<16).view(np.float32) if r['dtype']=='torch.bfloat16' else a;assert np.isfinite(decoded).all()
  # Only after all schema/finite/identity/source/cleanup/params/accounting checks pass.
  for r in raw['baselines']+raw['patches']:
   recomputed=label_metrics(z[r['fullV']['key']],r['label_ids'],r['expected_label']);assert recomputed==r['metrics']
  summary=functional_summary(raw['baselines'],raw['patches']);assert summary['phaseA_pass']==phaseB==raw['phaseA_pass']
 out=dict(status=c['status'],protocol_sha256=sha(LOCK),raw_sha256=c['raw_sha256'],vectors_sha256=c['vectors_sha256'],label_free_integrity='passed_all_before_label_metrics',forwards=count,summary=summary,actual_seconds=sup['gpu_worker_seconds']);save('results/r27_EXEC_ANALYSIS.json',out)
if __name__=='__main__':main()
