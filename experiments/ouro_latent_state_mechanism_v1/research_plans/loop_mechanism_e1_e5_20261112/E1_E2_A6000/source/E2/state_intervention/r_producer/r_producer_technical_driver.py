"""Fixed three-case coarse R producer technical gate; no scientific effects selected."""
import os,json,gzip,pathlib,hashlib,fcntl,time
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import torch
from safetensors import safe_open
from safetensors.torch import load_file,save_file
from transformers import AutoModelForCausalLM
import e2_r_producer as engine
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=pathlib.Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E2';BASE=E/'state_intervention';OUT=BASE/'r_producer/technical_three';PR=BASE/'technical_three';M=BASE/'mediation/discovery64';BR=ROOT/'research_plans/natural_branch_full_run_20261111';CAT=B/'results/compact_input_catalog.jsonl.gz';ROWS=E/'E2_latent_transition_rows.jsonl.gz';ORDER=BASE/'r_producer/ORDER_R_PRODUCER_COARSE_v1.md';FROZEN=BASE/'ASTRA_E2_TECHNICAL_THREE.json';MODEL=ROOT/'model/model.safetensors';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def score(z,targets):return [float(z.float().log_softmax(-1).gather(-1,torch.tensor([[t]],device=z.device)).sum()) for t in targets]
def main():
 assert sha(M/'CUDA_REPLAY.json')=='dec38117c2141807b58208dc1d7cb3e27d6d96a02491c2612f6a8d60634f4632' and sha(M/'CONTROL_MATH_AUDIT.json')=='090f37017a57ffc05bbddef7a23ad00f7144d76dcb68d36d68bbdc4f87cf9a82';assert sha(FROZEN)=='a0ccf57717fa7c7846ee3bbe1d853fbbd9da707c7e693e3c33a87afe6464b3dc' and sha(MODEL)=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af' and sha(CAT)=='a38218bfe0c9940f326663b8ecc0447f875a18c07f0091813ddfef4a3d12342d' and sha(ROWS)=='5d749790c56ab7bf2de3a9a64609e0a5b383fe3b7c5b2ec0cc5d5129586daac4'
 assert sha(PR/'CUDA_REPLAY_STRICT.json')=='242116c7d56d66f17c943fe920dfb8917ee8b1ddf8e25e611f07a53714bcc85e';mprior=json.loads((M/'PENDING.json').read_text())['source_sha256'];assert all(str(p) in mprior and sha(p)==mprior[str(p)] for p in [MODEL,CAT,ROOT/'model/modeling_ouro.py',ROOT/'model/config.json']);strict_data=json.loads((PR/'CUDA_REPLAY_STRICT.json').read_text());assert strict_data['pending_sha256']==sha(PR/'PENDING_INDEPENDENT_AUDIT.json');order=json.loads(FROZEN.read_text());assert {(q['split'],q['id']) for q in order['questions']}=={('test','ACTAAP_2013_7_16'),('test','Mercury_405804'),('test','MSA_2012_8_30')};prior=json.loads((PR/'PENDING_INDEPENDENT_AUDIT.json').read_text());links={(x['split'],x['id']):x for x in prior['cases']}
 with gzip.open(CAT,'rt') as f:cat={(r['split'],r['id']):r for r in map(json.loads,f)}
 with gzip.open(ROWS,'rt') as f:rows={(r['split'],r['id']):r for r in map(json.loads,f)}
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);model=None;oldsteps=None
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not OUT.exists();OUT.mkdir();sources=[MODEL,CAT,ROWS,ORDER,FROZEN,M/'CUDA_REPLAY.json',M/'CONTROL_MATH_AUDIT.json',PR/'CUDA_REPLAY_STRICT.json',pathlib.Path(__file__).resolve(),pathlib.Path(engine.__file__).resolve(),pathlib.Path(__file__).with_name('r_producer_cuda_replay.py').resolve(),ROOT/'model/modeling_ouro.py',ROOT/'model/config.json'];before={str(p):sha(p) for p in sources};save(OUT/'STARTED.json',{'status':'STARTED','source_sha256':before});start=time.monotonic();assert torch.__version__==prior['torch'] and torch.version.cuda==prior['cuda'] and torch.cuda.get_device_name()==prior['gpu'];model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;oldsteps=model.model.total_ut_steps;model.model.total_ut_steps=2;params=[(n,p.data_ptr(),p._version) for n,p in list(model.named_parameters())+list(model.named_buffers())];complete=[]
  with torch.inference_mode():
   for q in order['questions']:
    r=cat[(q['split'],q['id'])];rr=rows[(q['split'],q['id'])];ident=r['id'];key=hashlib.sha256((r['split']+'\0'+ident).encode()).hexdigest();oldp=PR/(key+'.safetensors');oldm=PR/(key+'.json');assert sha(oldp)==links[(r['split'],ident)]['tensor_sha256'] and sha(oldm)==links[(r['split'],ident)]['metadata_sha256'];old=load_file(str(oldp));branch=BR/(r['split']+'_'+hashlib.sha256(ident.encode()).hexdigest()+'.safetensors');assert sha(branch)==rr['branch_tensor_sha256'];ids=torch.tensor([r['input_ids']],device='cuda');R=r['R'];assert R==len(r['input_ids'])-1;ih=hashlib.sha256(json.dumps(r['input_ids']).encode()).hexdigest();saved={};meta={'id':ident,'split':r['split'],'R':R,'targets':r['targets'],'input_ids_sha256':ih,'branch_sha256':sha(branch),'prior_metadata_sha256':sha(oldm),'prior_tensor_sha256':sha(oldp),'cuts':{}}
    def run(cut,prod=None,target=None,expect_cut=None,expect_target=None):
     seen=[]
     def root(m,args,kw):
      inp=args[0] if args else kw.get('input_ids');assert torch.is_tensor(inp) and engine.bits(inp,ids) and kw.get('use_cache') is False and kw.get('past_key_values') is None;seen.append(1)
     h=model.register_forward_pre_hook(root,with_kwargs=True)
     try:
      with engine.producer_transfer(model.model.layers,cut_index=cut,R=R,expected_id=ident,current_id=ident,input_sha=ih,expected_input_sha=ih,producer_patch=prod,target_patch=target,expected_cut_full=expect_cut,expected_target_full=expect_target) as rec:z=model(ids,labels=None,use_cache=False,exit_at_step=1).logits[:,R,:].detach()
     finally:h.remove()
     assert len(seen)==1;return z,rec
    plain=model(ids,labels=None,use_cache=False,exit_at_step=1).logits[:,R,:].detach();assert engine.bits(plain.cpu(),old['native2_logits'])
    with safe_open(str(branch),framework='pt',device='cpu') as f:
     for cut in (0,8,16,17):
      donor=f.get_tensor(f'u0_layer{cut:02d}_x').cuda();expected_k2=f.get_tensor(f'u1_layer{cut:02d}_x').cuda();z0,clean=run(cut);assert engine.bits(z0,plain) and engine.bits(clean['cut_before_full'][...,R,:],expected_k2);clean_x=clean['target_before_full'];z1,perturbed=run(cut,prod=donor,expect_cut=clean['cut_before_full']);xpert=perturbed['target_before_full'];mask=torch.arange(xpert.shape[-2],device=xpert.device)!=R;assert engine.bits(xpert[...,mask,:],clean_x[...,mask,:]);target_clean=clean_x[...,R,:];target_pert=xpert[...,R,:];zr,restore=run(cut,prod=donor,target=target_clean,expect_cut=clean['cut_before_full'],expect_target=xpert);zt,transfer=run(cut,target=target_pert,expect_cut=clean['cut_before_full'],expect_target=clean_x);assert engine.bits(zr,plain) and engine.bits(zt,z1)
      if cut==17:assert engine.bits(z1.cpu(),old['rollback_full_logits']) and engine.bits(target_pert.cpu(),old['rollback_full_injected_R'])
      cmeta={'cut':cut,'donor_R_sha256':hashlib.sha256(donor.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),'conditions':{}}
      for name,z,rec in [('clean',z0,clean),('producer',z1,perturbed),('restore_target',zr,restore),('target_only',zt,transfer)]:
       prefix=f'cut{cut}_{name}';saved[prefix+'_logits']=z.cpu().contiguous();saved[prefix+'_cut_before_full']=rec['cut_before_full'].cpu().contiguous();saved[prefix+'_cut_after_R']=rec['cut_after_R'].cpu().contiguous();saved[prefix+'_target_before_full']=rec['target_before_full'].cpu().contiguous();saved[prefix+'_target_after_R']=rec['target_after_R'].cpu().contiguous();saved[prefix+'_a_R']=rec['a_R'].cpu().contiguous();cmeta['conditions'][name]={'candidate_full_vocab_logprob':score(z,r['targets']),'target_nonR_matches_clean':engine.bits(rec['target_before_full'][...,mask,:],clean_x[...,mask,:])}
      meta['cuts'][str(cut)]=cmeta
    out=OUT/(key+'.safetensors');save_file(saved,str(out));meta['tensor_sha256']=sha(out);mp=OUT/(key+'.json');save(mp,meta);complete.append({'id':ident,'split':r['split'],'metadata_sha256':sha(mp),'tensor_sha256':sha(out)});print('R_PRODUCER_TECH_CASE',ident,flush=True)
  assert {str(p):sha(p) for p in sources}==before and [(n,p.data_ptr(),p._version) for n,p in list(model.named_parameters())+list(model.named_buffers())]==params;save(OUT/'PENDING.json',{'status':'R_PRODUCER_THREE_PENDING_INDEPENDENT_REPLAY','cases':complete,'source_sha256':before,'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()})
 finally:
  if model is not None and oldsteps is not None:model.model.total_ut_steps=oldsteps
  os.close(lock)
if __name__=='__main__':main()
