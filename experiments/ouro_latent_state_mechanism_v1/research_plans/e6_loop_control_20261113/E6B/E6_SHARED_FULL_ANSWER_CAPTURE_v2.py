"""Full-answer native four-loop scoring and fixed-candidate0 prefix R capture."""
import argparse,json,hashlib,time,os,fcntl
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(o,allow_nan=False));os.replace(q,p)
def cosine(a,b):
 n=np.linalg.norm(a)*np.linalg.norm(b);return float(np.dot(a.ravel(),b.ravel())/n) if n>0 else 0.
def features(a,x,old=None,oldx=None,older=None):
 a=a.astype(np.float64);x=x.astype(np.float64);norm=np.linalg.norm(a,axis=1);mean=a.mean(0)
 v=[float(norm.mean()),float(norm.std()),float(norm.max()),float(norm.max()/max(norm.sum(),1e-30)),float(np.linalg.norm(mean)),float(np.linalg.norm(mean)/max(norm.mean(),1e-30)),cosine(mean,x),float(np.linalg.norm(x))]
 if old is None:return v+[0.]*8
 d=a-old.astype(np.float64);dn=np.linalg.norm(d,axis=1);dm=d.mean(0);dx=x-oldx.astype(np.float64)
 return v+[float(dn.mean()),float(dn.std()),float(dn.max()),float(dn.mean()/max(np.linalg.norm(old.astype(np.float64),axis=1).mean(),1e-30)),float(np.mean([cosine(u,w) for u,w in zip(a,old)])),cosine(dm,dx),cosine(dm,(old-older).mean(0)) if older is not None else 0.,float(np.linalg.norm(dx)/max(np.linalg.norm(oldx),1e-30))]
FEATURE_NAMES=['a_norm_mean','a_norm_std','a_norm_max','concentration','a_mean_norm','coordination','a_residual_cosine','residual_norm','delta_norm_mean','delta_norm_std','delta_norm_max','relative_delta','adjacent_a_cosine','delta_residual_cosine','successive_delta_cosine','relative_residual_delta']

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);p.add_argument('--technical',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--loopcd-source',default='/root/autodl-tmp/ouro-e6a/repo/experiments/looped_stage1_trajectory_audit_seed2021_v1/extensions/arc_loopcd_ouro14_full_20261026/scripts/methods.py');p.add_argument('--loopcd-protocol',default='/root/autodl-tmp/ouro-e6a/repo/experiments/looped_stage1_trajectory_audit_seed2021_v1/extensions/arc_loopcd_ouro14_full_20261026/configs/protocol.json');args=p.parse_args()
 import torch,transformers
 transformers.utils.import_utils._torchvision_available=False
 from transformers import AutoModelForCausalLM
 torch.set_grad_enabled(False);torch.manual_seed(2021);torch.backends.cuda.matmul.allow_tf32=False
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True);lock=open('/root/autodl-tmp/ouro-e6a/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(Path(args.model)/'model.safetensors')=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
 data=[json.loads(x) for x in Path(args.inputs).read_text().splitlines()];assert all('gold' not in x for x in data)
 data=sorted(data,key=lambda x:(x['split'],x['id']))
 if args.limit:data=data[:args.limit]
 identity={'model_sha256':sha(Path(args.model)/'model.safetensors'),'source_sha256':sha(Path(args.model)/'modeling_ouro.py'),'config_sha256':sha(Path(args.model)/'config.json'),'configuration_sha256':sha(Path(args.model)/'configuration_ouro.py'),'collector_sha256':sha(__file__),'inputs_sha256':sha(args.inputs),'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),'dtype':'bfloat16','attention':'sdpa','use_cache':False,'score_protocol':'full_answer','primary_normalization':'answer_character_count','feature_prefix':'fixed_candidate0_at_context_R','replica_baselines':{'reference_depth':1,'adaptive_cap':1.,'fixed_strength':.5,'rule':'fullvocab native K1/K4 contrast at every answer token; character-normalized sum primary'},'features':FEATURE_NAMES,'target_layers':list(range(10,17)),'control_layers':list(range(2,9)),'defined_delta_depths':[2,3,4],'defined_successive_delta_depths':[3,4]}
 identity['loopcd_source_sha256']=sha(args.loopcd_source);identity['loopcd_protocol_sha256']=sha(args.loopcd_protocol)
 write(out/'IDENTITY.json',identity)
 model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False
 import inspect
 assert sha(inspect.getfile(model.__class__))==identity['source_sha256']
 assert sha(inspect.getfile(model.config.__class__))==identity['configuration_sha256']
 identity['loaded_source_verified']=True;write(out/'IDENTITY.json',identity)
 versions=[(n,id(v),v._version) for n,v in model.named_parameters()];start=time.time();latencies=[];actual_body=0;actual_head=0;actual_technical=0
 def th(x):return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
 for i,q in enumerate(data):
  key=q['split']+'_'+hashlib.sha256(q['id'].encode()).hexdigest();file=out/(key+'.json')
  if file.exists():assert json.loads(file.read_text())['identity']==identity;continue
  assert hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()==q['input_ids_sha256'];R=q['R'];assert R==len(q['ids'])-1
  assert 3<=len(q['answer_targets'])<=5 and len(q['answer_char_lengths'])==len(q['answer_targets'])
  states={};active={};handles=[];events=[]
  native=[{'scores':[],'raw_sum':[],'mean_token':[],'full_prediction_logits_sha256':[]} for _ in range(4)];actual_inputs=[]
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
  production_seconds=0.;replica={name:{'scores':[],'raw_sum':[],'mean_token':[],'strengths':[]} for name in ['loopcd_adaptive_cap1','loopcd_fixed_0.5','loopcd_cap0_identity']}
  for candidate,target in enumerate(q['answer_targets']):
   torch.cuda.synchronize();candidate_started=time.perf_counter()
   assert target and q['answer_char_lengths'][candidate]>0
   seq=q['ids']+target[:-1];actual_inputs.append(hashlib.sha256(json.dumps(seq).encode()).hexdigest());ids=torch.tensor([seq],device='cuda')
   if candidate==0:
    for layerindex in sorted(set(range(1,8))|set(range(9,16))|{17}):
     layer=model.model.layers[layerindex]
     def pre(m,a,k,li=layerindex):
      u=k['current_ut'];assert k['use_cache'] is False;events.append((u,li));active[li]=u;states[(u,li,'x')]=a[0][:,R,:].detach().clone()
     def norm(m,a,y,li=layerindex):states[(active[li],li,'a')]=y[:,R,:].detach().clone()
     def post(m,a,k,y,li=layerindex):states[(active[li],li,'y')]=y[:,R,:].detach().clone()
     handles.extend([layer.register_forward_pre_hook(pre,with_kwargs=True),layer.input_layernorm_2.register_forward_hook(norm),layer.register_forward_hook(post,with_kwargs=True)])
   model.model.total_ut_steps=4
   try:_,hidden,gates=model.model(ids,use_cache=False,past_key_values=None)
   finally:
    for h in handles:h.remove()
    handles=[]
   actual_body+=1
   assert len(hidden)==4
   if candidate==0:
    assert events==[(u,j) for u in range(4) for j in sorted(set(range(1,8))|set(range(9,16))|{17})]
    assert len(states)==4*15*3
   index=torch.tensor(target,device='cuda');zs=[];early=None
   for depth,h in enumerate(hidden,1):
    actual_head+=1;z=model.lm_head(h)[0,R:R+len(target)].detach().clone();assert z.shape[0]==len(target) and z.dtype==torch.bfloat16 and bool(torch.isfinite(z).all())
    lp=z.float().log_softmax(-1).gather(-1,index[:,None]).squeeze(-1);value=float(lp.sum().item());zs.append(th(z));native[depth-1]['raw_sum'].append(value);native[depth-1]['mean_token'].append(value/len(target));native[depth-1]['scores'].append(value/q['answer_char_lengths'][candidate]);native[depth-1]['full_prediction_logits_sha256'].append(th(z))
    if depth==1:early=z.clone()
    if depth==4:
     top=z.float().softmax(-1).topk(2).values;strength=1-(top[:,0]-top[:,1])
     for name,amount in [('loopcd_adaptive_cap1',strength),('loopcd_fixed_0.5',torch.full_like(strength,.5)),('loopcd_cap0_identity',torch.zeros_like(strength))]:
      adjusted=z.float()+amount[:,None]*(z.float()-early.float());ls=adjusted.log_softmax(-1).gather(-1,index[:,None]).squeeze(-1);val=float(ls.sum().item());replica[name]['raw_sum'].append(val);replica[name]['mean_token'].append(val/len(target));replica[name]['scores'].append(val/q['answer_char_lengths'][candidate]);replica[name]['strengths'].append(amount.cpu().tolist())
   torch.cuda.synchronize();production_seconds+=time.perf_counter()-candidate_started
   if args.technical:
    for depth in range(1,5):
     model.model.total_ut_steps=depth;actual_technical+=1;z=model(ids,labels=None,use_cache=False,exit_at_step=depth-1).logits[0,R:R+len(target)].clone();assert th(z)==zs[depth-1],('full-answer full-vocab prefix identity',q['id'],candidate,depth)
   del hidden,gates,z
  torch.cuda.synchronize();technical_total_seconds=time.perf_counter()-t;seconds=production_seconds;latencies.append(seconds)
  for n in native:
   n['prediction']=int(np.argmax(n['scores']));n['top_tie']=n['scores'].count(max(n['scores']))>1
   n['sum_prediction']=int(np.argmax(n['raw_sum']));n['token_mean_prediction']=int(np.argmax(n['mean_token']))
  assert replica['loopcd_cap0_identity']['scores']==native[3]['scores']
  for name,value in replica.items():value['prediction']=int(np.argmax(value['scores']));value['top_tie']=value['scores'].count(max(value['scores']))>1
  feature_started=time.perf_counter();feature_rows=[];arrays={}
  for family,inds in [('latent',list(range(9,16))),('unrelated',list(range(1,8)))]:
   a=np.stack([np.stack([states[(u,j,'a')].float().cpu().numpy().ravel() for j in inds]) for u in range(4)])
   x=np.stack([np.mean([states[(u,j,'x')].float().cpu().numpy().ravel() for j in inds],axis=0) for u in range(4)])
   arrays[family]=a;arrays[family+'_x']=x
   for u in range(4):feature_rows.append({'family':family,'depth':u+1,'values':features(a[u],x[u],a[u-1] if u else None,x[u-1] if u else None,a[u-2] if u>=2 else None)})
  arrays['l18_x']=np.stack([states[(u,17,'x')].float().cpu().numpy().ravel() for u in range(4)]);arrays['l18_y']=np.stack([states[(u,17,'y')].float().cpu().numpy().ravel() for u in range(4)])
  # NumPy float32 preserves the exact BF16 values; no full-token activations retained.
  np.savez_compressed(out/(key+'.npz'),**arrays)
  row={'id':q['id'],'split':q['split'],'input_ids_sha256':q['input_ids_sha256'],'answer_targets':q['answer_targets'],'answer_char_lengths':q['answer_char_lengths'],'candidate_actual_input_ids_sha256':actual_inputs,'feature_prefix_candidate':0,'R':R,'identity':identity,'native':native,'replica_baselines':replica,'features':feature_rows,'state_sha256':sha(out/(key+'.npz')),'wall_seconds':seconds,'timed_including_technical_seconds':technical_total_seconds,'feature_cpu_and_storage_seconds':time.perf_counter()-feature_started,'state_storage_bytes':(out/(key+'.npz')).stat().st_size,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'technical_prefix_and_unhooked_identity':args.technical}
  write(file,row);del states;print('E6_NATIVE_CAPTURE',i+1,len(data),q['split'],q['id'],round(seconds,3),flush=True)
 assert versions==[(n,id(v),v._version) for n,v in model.named_parameters()]
 write(out/'COMPLETE.json',{'status':'COMPLETE','n':len(data),'native_model_body_forwards':actual_body,'native_head_calls':actual_head,'technical_unhooked_native_forwards':actual_technical,'logical_retained_native_body_forwards':sum(len(q['answer_targets']) for q in data),'wall_seconds':time.time()-start,'measured_seconds':sum(latencies),'identity':identity})
if __name__=='__main__':main()
