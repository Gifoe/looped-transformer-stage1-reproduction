"""Technical native-repeat diagnosis only. No labels, fitting, or interventions."""
import argparse,os,json,time,inspect,contextlib,datetime
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='/root/autodl-tmp/ouro-k1-k2-mechanism');p.add_argument('--model',default='/root/autodl-tmp/ouro-e4-e5/model');p.add_argument('--backend',choices=['original','deterministic_math'],required=True);p.add_argument('--repeats',type=int,default=4);a=p.parse_args();assert 2<=a.repeats<=4
 # Must precede torch import and any CUDA context creation, including helper imports.
 if a.backend=='deterministic_math':os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 import torch,transformers,fcntl
 import C1_COLLECT_v2 as c
 r=Path(a.root);out=r/('numeric_fp32_native_repeat_'+a.backend+'_v1');out.mkdir(exist_ok=False)
 freeze=c.read(r/'numeric_fp32/SOURCE_AND_SAMPLE_FREEZE.json');fixture=r/'numeric_fp32/GOLDFREE_INPUTS.jsonl';assert c.sha(fixture)==freeze['fixture_sha256']
 with fixture.open() as f:q=json.loads(next(f))
 assert q['id']==freeze['ids'][0] and not any(k in q for k in ['gold','gold_index','label','historical_scores'])
 assert c.sha(r/'C1_COLLECT_v2.py')==freeze['producer_source_sha256'] and c.sha(r/'branch_interfaces_v2.py')==freeze['operator_source_sha256']
 identity={'source_sha256':c.sha(__file__),'producer_source_sha256':freeze['producer_source_sha256'],'operator_source_sha256':freeze['operator_source_sha256'],'sample_freeze_sha256':c.sha(r/'numeric_fp32/SOURCE_AND_SAMPLE_FREEZE.json'),'fixture_sha256':c.sha(fixture),'id':q['id'],'input_ids_sha256':q['input_ids_sha256'],'backend_configuration':a.backend,'repeats':a.repeats,'scope':'One fixed first native-only selected ID; all native N, no P/U, no gold. Source frozen before this process model load.','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 c.write(out/'SOURCE_AND_PERMISSION_FREEZE.json',identity)
 lock=open('/root/autodl-tmp/ouro-e6a/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 transformers.utils.import_utils._torchvision_available=False
 from transformers import AutoModelForCausalLM
 torch.set_grad_enabled(False);torch.set_num_threads(1);torch.manual_seed(2021);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if a.backend=='deterministic_math':torch.use_deterministic_algorithms(True)
 assert torch.__version__=='2.7.1+cu128' and transformers.__version__=='4.55.0' and 'RTX 5090' in torch.cuda.get_device_name()
 for name,sha in [('model.safetensors',c.MODEL_SHA),('modeling_ouro.py',c.MODEL_SOURCE),('config.json',c.CONFIG_SHA),('configuration_ouro.py',c.CONFIGURATION_SHA)]:assert c.sha(Path(a.model)/name)==sha
 t=time.perf_counter();model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,trust_remote_code=True,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;model.model.total_ut_steps=2
 assert c.sha(inspect.getfile(model.__class__))==c.MODEL_SOURCE and all(v.dtype==torch.float32 for v in model.parameters())
 versions=[(n,id(v),v._version) for n,v in model.named_parameters()]
 identity.update(model_load_seconds=time.perf_counter()-t,torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),checkpoint_sha256=c.MODEL_SHA,loaded_model_source_sha256=c.MODEL_SOURCE,CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,matmul_precision=torch.get_float32_matmul_precision(),SDPA_enabled_flags={'flash':torch.backends.cuda.flash_sdp_enabled(),'math':torch.backends.cuda.math_sdp_enabled(),'memory_efficient':torch.backends.cuda.mem_efficient_sdp_enabled(),'cudnn':torch.backends.cuda.cudnn_sdp_enabled()})
 c.write(out/'IDENTITY.json',identity);saved=[];rows=[]
 with torch.inference_mode():
  for repeat in range(a.repeats):
   if a.backend=='deterministic_math':
    from torch.nn.attention import sdpa_kernel,SDPBackend
    ctx=sdpa_kernel(SDPBackend.MATH)
   else:ctx=contextlib.nullcontext()
   with ctx,torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    row,z,k1,qr,sites,branch=c.forward(model,q,K1_head=True)
   events=[{'operator':e.key,'count':e.count} for e in prof.key_averages() if any(s in e.key.lower() for s in ['scaled_dot_product','flash_attention','efficient_attention','cudnn_attention'])];assert events
   tensors={'Q9_K1_Q':qr['K1_Q'].cpu(),'K1_logits':k1.cpu(),'K2_logits':z.cpu()};diff={}
   if saved:
    for name,value in tensors.items():
     base=saved[0][name];diff[name]={'bitwise_equal':c.bits(value,base),'non_equal_elements':int((value!=base).sum()),'max_abs_difference':float((value.double()-base.double()).abs().max()),'first_sha256':c.thash(base),'current_sha256':c.thash(value)}
   archive=c.save(out/f'repeat_{repeat}.safetensors',tensors,r);record={'repeat':repeat,'actual_counts':row['actual_counts'],'native_K1':c.score(k1,q['targets']),'native_K2':row,'actual_SDPA_profiler_events':events,'tensor_hashes':{k:c.thash(v) for k,v in tensors.items()},'comparison_to_repeat0':diff,**archive};c.write(out/f'repeat_{repeat}.json',record);saved.append(tensors);rows.append(record)
   assert versions==[(n,id(v),v._version) for n,v in model.named_parameters()]
   print('NATIVE_REPEAT_COMPLETE',a.backend,repeat,json.dumps(diff),flush=True)
 allpairs=[]
 for i in range(len(saved)):
  for j in range(i+1,len(saved)):
   allpairs.append({'i':i,'j':j,'tensors':{k:{'bitwise_equal':c.bits(saved[i][k],saved[j][k]),'non_equal_elements':int((saved[i][k]!=saved[j][k]).sum()),'max_abs_difference':float((saved[i][k].double()-saved[j][k].double()).abs().max())} for k in saved[i]}})
 c.write(out/'COMPLETE.json',{'status':'COMPLETE','source_identity':identity,'repeats':a.repeats,'all_pair_comparisons':allpairs,'weights_modified':False,'scope':'Technical native determinism diagnosis; cannot attribute a dispatch cause solely from this observation; no scientific contrast evaluated.'});print('NATIVE_REPEAT_DIAGNOSTIC_COMPLETE',a.backend,flush=True)

if __name__=='__main__':main()
