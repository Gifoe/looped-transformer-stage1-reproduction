"""Read-only native four-loop R capture; no labels, interventions or weight updates."""
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
 identity={'model_sha256':sha(Path(args.model)/'model.safetensors'),'source_sha256':sha(Path(args.model)/'modeling_ouro.py'),'config_sha256':sha(Path(args.model)/'config.json'),'configuration_sha256':sha(Path(args.model)/'configuration_ouro.py'),'collector_sha256':sha(__file__),'inputs_sha256':sha(args.inputs),'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),'dtype':'bfloat16','attention':'sdpa','use_cache':False,'replica_baselines':{'reference_depth':1,'adaptive_cap':1.,'fixed_strength':.5,'rule':'final.float()+strength*(final.float()-early.float()); adaptive strength=1-fullvocab probability top1top2 gap'},'features':FEATURE_NAMES,'target_layers':list(range(10,17)),'control_layers':list(range(2,9)),'defined_delta_depths':[2,3,4],'defined_successive_delta_depths':[3,4]}
 identity['loopcd_source_sha256']=sha(args.loopcd_source);identity['loopcd_protocol_sha256']=sha(args.loopcd_protocol)
 write(out/'IDENTITY.json',identity)
 model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False
 import inspect
 assert sha(inspect.getfile(model.__class__))==identity['source_sha256']
 assert sha(inspect.getfile(model.config.__class__))==identity['configuration_sha256']
 identity['loaded_source_verified']=True;write(out/'IDENTITY.json',identity)
 versions=[(n,id(v),v._version) for n,v in model.named_parameters()];start=time.time();latencies=[]
 def th(x):return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
 for i,q in enumerate(data):
  key=q['split']+'_'+hashlib.sha256(q['id'].encode()).hexdigest();file=out/(key+'.json')
  if file.exists():assert json.loads(file.read_text())['identity']==identity;continue
  assert hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()==q['input_ids_sha256'];R=q['R'];assert R==len(q['ids'])-1
  ids=torch.tensor([q['ids']],device='cuda');states={};active={};handles=[]
  for layerindex in sorted(set(range(1,8))|set(range(9,16))|{17}):
   layer=model.model.layers[layerindex]
   def pre(m,a,k,li=layerindex):
    u=k['current_ut'];assert k['use_cache'] is False;active[li]=u;states[(u,li,'x')]=a[0][:,R,:].detach().clone()
   def norm(m,a,y,li=layerindex):states[(active[li],li,'a')]=y[:,R,:].detach().clone()
   def post(m,a,k,y,li=layerindex):states[(active[li],li,'y')]=y[:,R,:].detach().clone()
   handles.extend([layer.register_forward_pre_hook(pre,with_kwargs=True),layer.input_layernorm_2.register_forward_hook(norm),layer.register_forward_hook(post,with_kwargs=True)])
  model.model.total_ut_steps=4;torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
  try:_,hidden,gates=model.model(ids,use_cache=False,past_key_values=None)
  finally:
   for h in handles:h.remove()
  assert len(hidden)==4 and len(states)==4*15*3
  native=[];early=None;replica={}
  for depth,h in enumerate(hidden,1):
   z=model.lm_head(h)[0,R].detach().clone();lp=z.float().log_softmax(-1);v=z[q['targets']];s=lp[q['targets']]
   native.append({'scores':s.cpu().tolist(),'candidate_logits':v.float().cpu().tolist(),'prediction':int(v.argmax()),'top_tie':int((v==v.max()).sum())>1,'full_logits_sha256':th(z)})
   if depth==1:early=z.clone()
   if depth==4:
    conf=z.float().softmax(-1).topk(2).values;strength=1-(conf[0]-conf[1])
    for name,amount in [('loopcd_adaptive_cap1',strength),('loopcd_fixed_0.5',.5),('loopcd_cap0_identity',0.)]:
     adjusted=z.float()+amount*(z.float()-early.float());rs=adjusted.log_softmax(-1)[q['targets']];replica[name]={'scores':rs.cpu().tolist(),'prediction':int(rs.argmax()),'top_tie':bool((rs==rs.max()).sum()>1),'strength':float(amount),'reference_depth':1,'uses_full_vocabulary_confidence':True}
    assert replica['loopcd_cap0_identity']['scores']==native[3]['scores']
  torch.cuda.synchronize();seconds=time.perf_counter()-t;latencies.append(seconds)
  if args.technical:
   for d in range(1,5):
    model.model.total_ut_steps=d;z=model(ids,labels=None,use_cache=False,exit_at_step=d-1).logits[0,R].clone();assert th(z)==native[d-1]['full_logits_sha256'],('prefix identity',q['id'],d)
  feature_started=time.perf_counter();feature_rows=[];arrays={}
  for family,inds in [('latent',list(range(9,16))),('unrelated',list(range(1,8)))]:
   a=np.stack([np.stack([states[(u,j,'a')].float().cpu().numpy().ravel() for j in inds]) for u in range(4)])
   x=np.stack([np.mean([states[(u,j,'x')].float().cpu().numpy().ravel() for j in inds],axis=0) for u in range(4)])
   arrays[family]=a;arrays[family+'_x']=x
   for u in range(4):feature_rows.append({'family':family,'depth':u+1,'values':features(a[u],x[u],a[u-1] if u else None,x[u-1] if u else None,a[u-2] if u>=2 else None)})
  arrays['l18_x']=np.stack([states[(u,17,'x')].float().cpu().numpy().ravel() for u in range(4)]);arrays['l18_y']=np.stack([states[(u,17,'y')].float().cpu().numpy().ravel() for u in range(4)])
  # NumPy float32 preserves the exact BF16 values; no full-token activations retained.
  np.savez_compressed(out/(key+'.npz'),**arrays)
  row={'id':q['id'],'split':q['split'],'input_ids_sha256':q['input_ids_sha256'],'targets':q['targets'],'R':R,'identity':identity,'native':native,'replica_baselines':replica,'features':feature_rows,'state_sha256':sha(out/(key+'.npz')),'wall_seconds':seconds,'feature_cpu_and_storage_seconds':time.perf_counter()-feature_started,'state_storage_bytes':(out/(key+'.npz')).stat().st_size,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'technical_prefix_and_unhooked_identity':args.technical}
  write(file,row);del hidden,gates,states,z;print('E6_NATIVE_CAPTURE',i+1,len(data),q['split'],q['id'],round(seconds,3),flush=True)
 assert versions==[(n,id(v),v._version) for n,v in model.named_parameters()]
 write(out/'COMPLETE.json',{'status':'COMPLETE','n':len(data),'native_model_body_forwards':len(latencies),'native_head_calls':4*len(latencies),'technical_unhooked_native_forwards':4*len(latencies) if args.technical else 0,'wall_seconds':time.time()-start,'measured_seconds':sum(latencies),'identity':identity})
if __name__=='__main__':main()
