"""Three Train full-model FP32 E1 sensitivity. No optimizer/model writes."""
import os,json,gzip,pathlib,hashlib,fcntl,time,resource
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
import fp32_observer as obs
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=pathlib.Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E1';P=B/'E1/full_model_fp32_three_train';CAT=B/'results/compact_input_catalog.jsonl.gz';SELECT=B/'orders/E1_DISCOVERY64_FROZEN.json';AUDIT=E/'discovery64_cuda_audit.json';OLD=ROOT/'research_plans/natural_qkv_eight_corner_20261112';MODEL=ROOT/'model/model.safetensors';ORDER=E/'ORDER_FULLMODEL_FP32_THREE_TRAIN.md';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def scores(z,targets):return [float(z.float().log_softmax(-1).gather(-1,torch.tensor([[t]],device=z.device)).sum()) for t in targets]
def main():
 assert sha(MODEL)=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af' and sha(CAT)=='a38218bfe0c9940f326663b8ecc0447f875a18c07f0091813ddfef4a3d12342d' and sha(SELECT)=='6faad196db958744780ee6477af8839c693798b8b24d2a01a2d450c856c720e1';sel=json.loads(SELECT.read_text());ids=sorted(x['id'] for group in sel['groups'].values() for x in group)[:3];assert ids==['ACTAAP_2010_7_7','AKDE&ED_2008_4_41','AKDE&ED_2008_8_32'];groups={x['id']:g for g,v in sel['groups'].items() for x in v};assert [groups[i] for i in ids]==['damage','stable_correct','stable_wrong'];audit=json.loads(AUDIT.read_text());assert audit['count']==64 and audit['status']=='DISCOVERY64_INDEPENDENT_CUDA_SCORE_AND_SOURCE_AUDIT_PASSED';links={x['id']:x for x in audit['questions']}
 with gzip.open(CAT,'rt') as f:cat={(r['split'],r['id']):r for r in map(json.loads,f)}
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);model=None;oldsteps=None
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not P.exists();P.mkdir();sources=[MODEL,CAT,SELECT,AUDIT,ORDER,pathlib.Path(__file__).resolve(),pathlib.Path(obs.__file__).resolve(),ROOT/'model/modeling_ouro.py',ROOT/'model/config.json'];before={str(p):sha(p) for p in sources};save(P/'STARTED.json',{'status':'STARTED','ids':ids,'source_sha256':before});torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32;torch.cuda.reset_peak_memory_stats();start=time.monotonic();model=AutoModelForCausalLM.from_pretrained(MODEL.parent,local_files_only=True,trust_remote_code=True,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;oldsteps=model.model.total_ut_steps;params=dict(model.named_parameters());validated=0
  with safe_open(str(MODEL),framework='pt',device='cpu') as f:
   keys=set(f.keys());assert keys==set(params),(len(keys),len(params),sorted(keys-set(params))[:5],sorted(set(params)-keys)[:5]);
   for k in sorted(keys):
    original=f.get_tensor(k);assert original.dtype==torch.bfloat16 and params[k].dtype==torch.float32 and torch.equal(original.float(),params[k].detach().cpu());validated+=1
  assert validated==269;versions=[(n,p.data_ptr(),p._version) for n,p in list(model.named_parameters())+list(model.named_buffers())];metadata={'status':'STARTED_MODEL_VERIFIED','sample_groups':{i:groups[i] for i in ids},'sample_limit':'No natural rescue cases; cannot establish rescue-group precision stability.','checkpoint_tensors_verified_exact_bf16_to_fp32':validated,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'matmul_precision':torch.get_float32_matmul_precision(),'flash_sdp_enabled':torch.backends.cuda.flash_sdp_enabled(),'math_sdp_enabled':torch.backends.cuda.math_sdp_enabled(),'mem_efficient_sdp_enabled':torch.backends.cuda.mem_efficient_sdp_enabled(),'scope':'Full FP32 model weights/activations freshly computed; no weight update; E1 corners are local interventions on natural K1 receiver.'};save(P/'RUNTIME.json',metadata)
  def run(r,depth,patch=None,profile=False):
   model.model.total_ut_steps=depth;ids_tensor=torch.tensor([r['input_ids']],device='cuda');roots=[]
   def root(m,args,kw):
    x=args[0] if args else kw.get('input_ids');assert torch.is_tensor(x) and obs.bits(x,ids_tensor) and kw.get('use_cache') is False and kw.get('past_key_values') is None;roots.append(1)
   h=model.register_forward_pre_hook(root,with_kwargs=True)
   try:
    with obs.observe(model,depth-1,r['R'],patch=patch,profile=profile) as rec:z=model(ids_tensor,labels=None,use_cache=False,exit_at_step=depth-1).logits[:,r['R'],:].detach()
   finally:h.remove()
   assert len(roots)==1 and z.dtype==torch.float32 and z.shape==(1,49152) and torch.isfinite(z).all();return z,rec
  def capture(r,profile=False):
   c=[];z=[]
   for depth in (1,2):
    model.model.total_ut_steps=depth;ids_tensor=torch.tensor([r['input_ids']],device='cuda');plain=model(ids_tensor,labels=None,use_cache=False,exit_at_step=depth-1).logits[:,r['R'],:].detach();v,rec=run(r,depth,profile=profile and depth==1);assert obs.bits(v,plain) and scores(v,r['targets'])==scores(plain,r['targets']);q,k,val=rec['q'],rec['k'],rec['v'];assert not rec['tail'] and rec['kwargs'].get('attn_mask') is None and rec['kwargs'].get('dropout_p')==0 and rec['kwargs'].get('is_causal') is True and rec['kwargs'].get('enable_gqa') is True and float(rec['kwargs']['scale'])==0.08838834764831845;c.append(rec);z.append(v)
   return z,c
  def corners(c,R,q):
   out={}
   for k in (1,2):
    for v in (1,2):out[(k,v)]=F.scaled_dot_product_attention(c[q-1]['q'],c[k-1]['k'],c[v-1]['v'],*c[q-1]['tail'],**c[q-1]['kwargs'])[...,R,:].detach().clone()
   return out
  completed=[]
  with torch.inference_mode():
   for n,ident in enumerate(ids):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.monotonic();r=cat[('train',ident)];key=hashlib.sha256(ident.encode()).hexdigest();bfmeta=E/'train'/(key+'.json');bftensor=E/'train'/(key+'.safetensors');link=links[ident];assert sha(bfmeta)==link['metadata_sha256'] and sha(bftensor)==link['tensor_sha256'];bm=json.loads(bfmeta.read_text());oldmeta=OLD/'train'/(key+'.json');assert sha(oldmeta)==link['old_metadata_sha256'];old=json.loads(oldmeta.read_text());assert bm['targets']==old['targets']==r['targets'] and bm['input_ids_sha256']==old['input_ids_sha256']==r['input_ids_sha256'];z,c=capture(r,profile=n==0);backend=c[0].get('profiler_ops',[]);assert n!=0 or backend;corners_by_q={q:corners(c,r['R'],q) for q in (1,2)};assert obs.bits(corners_by_q[1][(1,1)],c[0]['native_sdpa_R']);tensors={'native1_logits':z[0].cpu().contiguous(),'native2_logits':z[1].cpu().contiguous()};entry={'id':ident,'input_ids_sha256':r['input_ids_sha256'],'R':r['R'],'targets':r['targets'],'gold':r['gold_index'],'source_bf16_e1_metadata_sha256':sha(bfmeta),'source_bf16_eight_metadata_sha256':sha(oldmeta),'native':{},'corners':{},'controls':{},'actual_sdpa':{'qkv_shapes':[list(c[0][j].shape) for j in ('q','k','v')],'kwargs':c[0]['kwargs'],'profiler_ops':backend}}
    for u in (0,1):entry['native'][str(u+1)]={'scores':scores(z[u],r['targets']),'candidate_logits':[float(z[u][0,t]) for t in r['targets']],'logits_sha256':thash(z[u]),'baseline_bf16_scores':r['native_scores'][u]}
    for q in (1,2):
     C=corners_by_q[q];entry['corners'][str(q)]={};cz={}
     for k in (1,2):
      for v in (1,2):
       tag=f'G{q}{k}{v}';res,rec=run(r,1,patch=C[(k,v)]);assert obs.bits(rec['native_sdpa_R'],c[0]['native_sdpa_R'])
       if tag=='G111':assert obs.bits(res,z[0])
       cz[(k,v)]=res;tensors[tag+'_logits']=res.cpu().contiguous();tensors[tag+'_sdpa_R']=C[(k,v)].cpu().contiguous();entry['corners'][str(q)][tag]={'scores':scores(res,r['targets']),'candidate_logits':[float(res[0,t]) for t in r['targets']],'bf16_scores':old['scores'][tag],'logits_sha256':thash(res)}
     donor=bm['controls']['conditions'][str(q)]['donor']['id'];dr=cat[('train',donor)];assert dr['input_ids']!=r['input_ids'];_,dc=capture(dr);D=corners(dc,dr['R'],q);donor_add=(D[(2,1)].double()+D[(1,2)].double()-D[(1,1)].double()).float();donor_delta=D[(2,2)].double()-donor_add.double();s11,s12,s21,s22=[C[x] for x in ((1,1),(1,2),(2,1),(2,2))];add_exact=s21.double()+s12.double()-s11.double();add=add_exact.float();delta=s22.double()-add.double();dn=delta.norm();on=donor_delta.norm();an=add.double().norm();tn=s22.double().norm();normal=(s22.clone() if tn==0 else None) if an==0 else (add.double()*(tn/an)).float();unrelated=(s22.clone() if dn==0 else None) if on==0 else (s22.double()-donor_delta*(dn/on)).float();restored=(add.double()+delta).float();assert obs.bits(restored,s22);ctrl={'ID':s22,'ADD':add,'ADD_NORM':normal,'UNRELATED':unrelated,'restore':restored};entry['controls'][str(q)]={'donor_id':donor,'donor_input_ids_sha256':dr['input_ids_sha256'],'donor_source':'fresh complete FP32 model forwards','delta_norm':float(dn),'donor_delta_norm':float(on),'conditions':{}}
     for name,value in ctrl.items():
      if value is None:entry['controls'][str(q)]['conditions'][name]={'status':'UNDEFINED_ZERO_DENOMINATOR'};continue
      res,rec=run(r,1,patch=value)
      if name in ('ID','restore'):assert obs.bits(res,cz[(2,2)])
      bf=bm['controls']['conditions'][str(q)].get(name);tensors[f'q{q}_{name}_logits']=res.cpu().contiguous();tensors[f'q{q}_{name}_sdpa_R']=value.cpu().contiguous();entry['controls'][str(q)]['conditions'][name]={'status':'OK','scores':scores(res,r['targets']),'candidate_logits':[float(res[0,t]) for t in r['targets']],'bf16_scores':bf.get('scores') if bf else None,'logits_sha256':thash(res)}
     tensors[f'q{q}_add_cast_residual_fp64']=(add.double()-add_exact).cpu().contiguous()
    torch.cuda.synchronize();entry['wall_seconds_cuda_synchronized_excludes_save']=time.monotonic()-t0;entry['peak_allocated_bytes']=torch.cuda.max_memory_allocated();entry['peak_reserved_bytes']=torch.cuda.max_memory_reserved();tp=P/(key+'.safetensors');save_file(tensors,str(tp));entry['tensor_sha256']=sha(tp);mp=P/(key+'.json');save(mp,entry);completed.append({'id':ident,'metadata_sha256':sha(mp),'tensor_sha256':sha(tp)});print('FULL_FP32_CASE',n+1,ident,entry['wall_seconds_cuda_synchronized_excludes_save'],flush=True)
  assert {str(p):sha(p) for p in sources}==before and [(n,p.data_ptr(),p._version) for n,p in list(model.named_parameters())+list(model.named_buffers())]==versions;save(P/'PENDING.json',{'status':'THREE_TRAIN_FULL_FP32_PENDING_INDEPENDENT_AUDIT','ids':ids,'cases':completed,'source_sha256':before,'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'cpu_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
 finally:
  if model is not None and oldsteps is not None:model.model.total_ut_steps=oldsteps
  os.close(lock)
if __name__=='__main__':main()
