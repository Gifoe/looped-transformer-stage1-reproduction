"""True B prefix stopping, fixed ZL gate and matched output-only/K4 timing."""
import argparse,json,time,hashlib,fcntl,importlib.util
from pathlib import Path
import numpy as np
from E6B_TRAINING_FREE_CONTROLLER import early_exit_prefix,predict
from E6B_NATURAL_FEATURE_EXTRACTION_v3 import features
class StopAtPrefix(Exception):pass
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--native',required=True);p.add_argument('--model',required=True);p.add_argument('--frozen-spec',required=True);p.add_argument('--out',required=True);a=p.parse_args();frozen=json.loads(Path(a.frozen_spec).read_text());threshold=frozen['thresholds_unlabeled_train']['0.5']
 import torch,transformers
 transformers.utils.import_utils._torchvision_available=False
 from transformers import AutoModelForCausalLM
 torch.set_grad_enabled(False);torch.manual_seed(2021);torch.backends.cuda.matmul.allow_tf32=False;lock=open('/root/autodl-tmp/ouro-e6a/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(Path(a.model)/'model.safetensors')=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af';rows=[json.loads(x) for x in Path(a.inputs).read_text().splitlines()];assert all('gold' not in q for q in rows)
 out=Path(a.out);out.mkdir(exist_ok=True,parents=True);model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;model.model.total_ut_steps=4;versions=[(n,id(v),v._version) for n,v in model.named_parameters()];result=[]
 def th(x):return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
 for i,q in enumerate(rows):
  key=q['split']+'_'+hashlib.sha256(q['id'].encode()).hexdigest();native=json.loads((Path(a.native)/(key+'.json')).read_text());assert native['input_ids_sha256']==q['input_ids_sha256'];ids=torch.tensor([q['ids']],device='cuda');R=q['R'];saved_latent=[next(f['values'] for f in native['features'] if f['family']=='latent' and f['depth']==d) for d in range(1,5)]
  order=['native4','output_only','output_latent'] if i%2==0 else ['output_latent','output_only','native4']
  for name in order:
   handles=[];active={};states={};prefix=[];latent_prefix=[];all_a=[];all_x=[];endpoint={};calls=[0]
   def first(m,args,kwargs):calls[0]+=1
   handles.append(model.model.layers[0].register_forward_pre_hook(first,with_kwargs=True))
   if name=='output_latent':
    for j in range(9,16):
     layer=model.model.layers[j]
     def pre(m,args,kwargs,j=j):active[j]=kwargs['current_ut'];states[(active[j],j,'x')]=args[0][:,R,:].detach().clone()
     def norm(m,args,y,j=j):states[(active[j],j,'a')]=y[:,R,:].detach().clone()
     handles.extend([layer.register_forward_pre_hook(pre,with_kwargs=True),layer.input_layernorm_2.register_forward_hook(norm)])
   def callback(m,args,h):
    z=model.lm_head(h)[0,R].detach().clone();score=z.float().log_softmax(-1)[q['targets']].cpu().tolist();prefix.append(score);d=len(prefix);assert th(z)==native['native'][d-1]['full_logits_sha256'],(q['id'],name,d,'full-vocab prefix')
    if name=='output_latent':
     aa=np.stack([states[(d-1,j,'a')].float().cpu().numpy().ravel() for j in range(9,16)]);xx=np.mean([states[(d-1,j,'x')].float().cpu().numpy().ravel() for j in range(9,16)],axis=0);all_a.append(aa);all_x.append(xx);v=features(aa,xx,all_a[-2] if d>=2 else None,all_x[-2] if d>=2 else None,all_a[-3] if d>=3 else None);assert v==saved_latent[d-1],(q['id'],d,'feature prefix');latent_prefix.append(v)
    else:latent_prefix.append([0.]*16)
    stop=d==4 or (d in [2,3] and early_exit_prefix(prefix,latent_prefix,threshold,name=='output_latent',.9))
    if stop:endpoint.update(depth=d,prediction=int(np.argmax(score)),scores=score,full_logits_sha256=th(z));raise StopAtPrefix()
   if name!='native4':handles.append(model.model.norm.register_forward_hook(callback))
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
   try:
    if name=='native4':
     z=model(ids,use_cache=False,exit_at_step=3).logits[0,R].detach().clone();sc=z.float().log_softmax(-1)[q['targets']].cpu().tolist();endpoint={'depth':4,'prediction':int(np.argmax(sc)),'scores':sc,'full_logits_sha256':th(z)}
    else:
     try:model(ids,use_cache=False,exit_at_step=3)
     except StopAtPrefix:pass
   finally:
    for handle in handles:handle.remove()
   torch.cuda.synchronize();seconds=time.perf_counter()-start;d=endpoint['depth'];assert calls[0]==d and endpoint['full_logits_sha256']==native['native'][d-1]['full_logits_sha256']
   if name!='native4':
    replay=predict([n['scores'] for n in native['native']],saved_latent,threshold,'early_exit',name=='output_latent',.9);assert replay['selected_depth']==d and replay['prediction']==endpoint['prediction'],(q['id'],name,'frozen replay')
   result.append({'id':q['id'],'split':q['split'],'method':name,'seconds':seconds,'actual_executed_loops':d,'first_layer_calls':calls[0],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'prefix_only':True,'scores_match_cached_replay':True,**endpoint})
  print('E6B_ONLINE_VALIDATED',i+1,len(rows),q['id'],flush=True)
 assert versions==[(n,id(v),v._version) for n,v in model.named_parameters()];(out/'rows.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in result));(out/'COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','n':len(rows),'source_sha256':sha(__file__),'frozen_spec_sha256':sha(a.frozen_spec),'inputs_sha256':sha(a.inputs),'model_sha256':sha(Path(a.model)/'model.safetensors'),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'policies':'K4; fixed ZL p=.9 q=.5 prefix gate with/without latent; no new selection','scope':'Actual terminating model forward. All observed prefix full-vocabulary hashes, latent vectors and chosen predictions match saved native replay. Alternating method order; cold first example retained separately.'},indent=2))
if __name__=='__main__':main()
