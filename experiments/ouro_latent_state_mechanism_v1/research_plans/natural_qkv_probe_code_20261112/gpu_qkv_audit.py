"""Post-collection independent CUDA replay, under the project global lock."""
import os,fcntl,json,hashlib
from pathlib import Path
import torch
import full_qkv,full_input
from source_input import ROOT,sha
OUT=ROOT/'research_plans/NATURAL_QKV_INDEPENDENT_CUDA_REPLAY_20261112.json'
def main():
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run=full_qkv.OUT;rp=run/'receipt.json';r=json.loads(rp.read_text());full_qkv.require(r['status']=='FULL_QKV_PENDING_INDEPENDENT_COVERAGE' and r['counts']=={'train':1119,'validation':299,'test':1172},'receipt')
  sources=r['source_hashes'];codes=r['code_hashes'];pr=ROOT/'research_plans/natural_qkv_probe_20261112';paths={**full_qkv.initial_sources(),'terms':full_qkv.TERMS,'review':full_qkv.REVIEW,'l18_coverage':full_qkv.qkv_runner.LCOV,'probe_receipt':pr/'receipt.json','probe_pending':pr/'probe_pending.json'};mods=(full_qkv.qkv_observer,full_qkv.qkv_runner,full_qkv.replay_qkv,full_qkv.natural_l18_hook,full_qkv.natural_sources,full_qkv.technical,full_qkv.full_input,full_qkv.state_capture,full_qkv.source_input);cp={'full':Path(full_qkv.__file__),'probe':Path(__import__('probe_qkv').__file__),**{m.__name__:Path(m.__file__) for m in mods}};receipt_hash=sha(rp);pending=json.loads((pr/'probe_pending.json').read_text());runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()};full_qkv.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported() and all(runtime[k]==pending['runtime'][k] for k in runtime),'runtime differs from probe');full_qkv.require({k:sha(v) for k,v in paths.items()}==sources and {k:sha(v) for k,v in cp.items()}==codes,'source');ledger=[]
  for split,n in [('train',1119),('validation',299),('test',1172)]:
   qs=full_input.coverage(split);keys={split+'_'+hashlib.sha256(q['id'].encode()).hexdigest() for q in qs};full_qkv.require(len(keys)==n and keys=={p.stem for p in run.glob(split+'_*.json')}=={p.stem for p in run.glob(split+'_*.safetensors')},'cohort')
   for q in qs:
    ident=q['id'];full_qkv.require(full_qkv.check(split,ident,sources,codes),'replay');key=split+'_'+hashlib.sha256(ident.encode()).hexdigest();ledger.append({'split':split,'id':ident,'metadata_sha256':sha(run/(key+'.json')),'tensor_sha256':sha(run/(key+'.safetensors'))})
    if len(ledger)%100==0:print('INDEPENDENT_CUDA_REPLAY',len(ledger),flush=True)
  full_qkv.require(len(ledger)==2590 and sha(rp)==receipt_hash and {k:sha(v) for k,v in paths.items()}==sources and {k:sha(v) for k,v in cp.items()}==codes,'ledger');result={'status':'GPU_QKV_LOGIT_REPLAY_PASSED_NOT_FULL_STATE_COVERAGE','count':len(ledger),'questions':ledger,'collector_receipt_sha256':sha(rp),'code_sha256':sha(Path(__file__)),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'runtime_matches_probe':True,'scope':'separate CUDA process reuses collector check implementation; saved logits/endpoint replay only and trusts collector for in-forward interventions'}
  with OUT.open('x') as f:json.dump(result,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  print('QKV_INDEPENDENT_CUDA_REPLAY_PASS',sha(OUT),flush=True)
 finally:os.close(lock)
if __name__=='__main__':main()
