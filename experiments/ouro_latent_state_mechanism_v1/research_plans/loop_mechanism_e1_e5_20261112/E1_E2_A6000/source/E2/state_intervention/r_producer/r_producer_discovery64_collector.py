"""Frozen 64 R-producer coarse cuts. Only three upstream forwards per item."""
import os,json,gzip,pathlib,hashlib,fcntl,time,uuid
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import torch
from safetensors import safe_open
from safetensors.torch import load_file,save_file
from transformers import AutoModelForCausalLM
import e2_r_producer as engine
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=pathlib.Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E2';BASE=E/'state_intervention';OUT=BASE/'r_producer/discovery64';TECH=BASE/'r_producer/technical_three';PR=BASE/'discovery64';BR=ROOT/'research_plans/natural_branch_full_run_20261111';CAT=B/'results/compact_input_catalog.jsonl.gz';ROWS=E/'E2_latent_transition_rows.jsonl.gz';ORDER=BASE/'r_producer/ORDER_R_PRODUCER_COARSE_v1.md';FROZEN=B/'orders/E1_DISCOVERY64_FROZEN.json';MODEL=ROOT/'model/model.safetensors';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dh(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def score(z,targets):return [float(z.float().log_softmax(-1).gather(-1,torch.tensor([[t]],device=z.device)).sum()) for t in targets]
def save(p,x):
 with p.open('x') as f:json.dump(x,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 assert sha(MODEL)=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af' and sha(CAT)=='a38218bfe0c9940f326663b8ecc0447f875a18c07f0091813ddfef4a3d12342d' and sha(ROWS)=='5d749790c56ab7bf2de3a9a64609e0a5b383fe3b7c5b2ec0cc5d5129586daac4' and sha(FROZEN)=='6faad196db958744780ee6477af8839c693798b8b24d2a01a2d450c856c720e1';treplay=TECH/'CUDA_REPLAY.json';assert sha(treplay)=='a863c7d5609a25e8e646c5471478c03184129a070fce7e36a3ee4fea9f381ff4';tr=json.loads(treplay.read_text());assert tr['status']=='R_PRODUCER_THREE_INDEPENDENT_REPLAY_PASSED' and len(tr['cases'])==3 and tr['pending_sha256']==sha(TECH/'PENDING.json');tpending=json.loads((TECH/'PENDING.json').read_text());oldsrc=tpending['source_sha256'];shared=[MODEL,CAT,ROWS,ORDER,pathlib.Path(engine.__file__).resolve(),ROOT/'model/modeling_ouro.py',ROOT/'model/config.json'];assert all(str(p) in oldsrc and sha(p)==oldsrc[str(p)] for p in shared)
 strict=PR/'CUDA_REPLAY_64_STRICT.json';assert sha(strict)=='94297e44f3bc87a6bf40737d2be604bb5aee5e837934a72a33e2e5ea90f3bde8';assert json.loads(strict.read_text())['pending_sha256']==sha(PR/'PENDING_INDEPENDENT_AUDIT.json');prior=json.loads((PR/'PENDING_INDEPENDENT_AUDIT.json').read_text());links={(x['split'],x['id']):x for x in prior['cases']};selection=json.loads(FROZEN.read_text());assert selection['count']==64 and set(selection['groups'])=={'rescue','damage','stable_correct','stable_wrong'} and all(len(v)==16 for v in selection['groups'].values());ids=[q['id'] for g in sorted(selection['groups']) for q in selection['groups'][g]];assert len(set(ids))==64 and {('train',i) for i in ids}==set(links)
 with gzip.open(CAT,'rt') as f:cat={(r['split'],r['id']):r for r in map(json.loads,f)}
 with gzip.open(ROWS,'rt') as f:rows={(r['split'],r['id']):r for r in map(json.loads,f)}
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);model=None;oldsteps=None
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=True);assert not (OUT/'PENDING.json').exists();sources=[MODEL,CAT,ROWS,ORDER,FROZEN,treplay,TECH/'PENDING.json',strict,PR/'PENDING_INDEPENDENT_AUDIT.json',pathlib.Path(__file__).resolve(),pathlib.Path(engine.__file__).resolve(),pathlib.Path(__file__).with_name('r_producer_discovery64_cuda_replay.py').resolve(),ROOT/'model/modeling_ouro.py',ROOT/'model/config.json'];before={str(p):sha(p) for p in sources};start_receipt={'status':'STARTED','source_sha256':before};sp=OUT/'STARTED.json'
  if sp.exists():assert json.loads(sp.read_text())==start_receipt
  else:save(sp,start_receipt)
  start=time.monotonic();assert torch.__version__==prior['torch'] and torch.version.cuda==prior['cuda'] and torch.cuda.get_device_name()==prior['gpu'];model=AutoModelForCausalLM.from_pretrained(MODEL.parent,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;oldsteps=model.model.total_ut_steps;model.model.total_ut_steps=2;params=[(n,p.data_ptr(),p._version) for n,p in list(model.named_parameters())+list(model.named_buffers())];complete=[]
  with torch.inference_mode():
   for ident in ids:
    r=cat[('train',ident)];rr=rows[('train',ident)];key=hashlib.sha256(('train\0'+ident).encode()).hexdigest();out=OUT/(key+'.safetensors');outj=OUT/(key+'.json')
    if out.exists() or outj.exists():
     assert out.exists() and outj.exists();prev=json.loads(outj.read_text());assert prev['id']==ident and prev['targets']==r['targets'] and prev['R']==r['R'] and prev['input_ids_sha256']==r['input_ids_sha256'] and prev['tensor_sha256']==sha(out) and prev['source_sha256']==before;complete.append({'id':ident,'split':'train','metadata_sha256':sha(outj),'tensor_sha256':sha(out)});continue
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.monotonic();oldp=PR/(key+'.safetensors');oldm=PR/(key+'.json');assert sha(oldp)==links[('train',ident)]['tensor_sha256'] and sha(oldm)==links[('train',ident)]['metadata_sha256'];old=load_file(str(oldp));branch=BR/('train_'+hashlib.sha256(ident.encode()).hexdigest()+'.safetensors');assert sha(branch)==rr['branch_tensor_sha256'];ids_tensor=torch.tensor([r['input_ids']],device='cuda');R=r['R'];assert R==len(r['input_ids'])-1;ih=hashlib.sha256(json.dumps(r['input_ids']).encode()).hexdigest();assert ih==r['input_ids_sha256'];target_clean=old['native2_input_full'].cuda();mask=torch.arange(target_clean.shape[-2],device='cuda')!=R;saved={'native2_logits':old['native2_logits'],'target17_logits':old['rollback_full_logits'],'target17_a_R':old['rollback_full_a_R'],'target17_x_R':old['rollback_full_injected_R']};meta={'id':ident,'split':'train','R':R,'targets':r['targets'],'input_ids_sha256':ih,'branch_sha256':sha(branch),'prior_metadata_sha256':sha(oldm),'prior_tensor_sha256':sha(oldp),'cuts':{}}
    with safe_open(str(branch),framework='pt',device='cpu') as f:
     for cut in (0,8,16):
      donor=f.get_tensor(f'u0_layer{cut:02d}_x').cuda();k2=f.get_tensor(f'u1_layer{cut:02d}_x').cuda();seen=[]
      def root(m,args,kw):
       inp=args[0] if args else kw.get('input_ids');assert torch.is_tensor(inp) and engine.bits(inp,ids_tensor) and kw.get('use_cache') is False and kw.get('past_key_values') is None;seen.append(1)
      h=model.register_forward_pre_hook(root,with_kwargs=True)
      try:
       with engine.producer_transfer(model.model.layers,cut_index=cut,R=R,expected_id=ident,current_id=ident,input_sha=ih,expected_input_sha=ih,producer_patch=donor) as rec:z=model(ids_tensor,labels=None,use_cache=False,exit_at_step=1).logits[:,R,:].detach()
      finally:h.remove()
      assert len(seen)==1 and engine.bits(rec['cut_before_full'][...,R,:],k2) and engine.bits(rec['target_before_full'][...,mask,:],target_clean[...,mask,:]);prefix=f'cut{cut}';saved[prefix+'_logits']=z.cpu().contiguous();saved[prefix+'_cut_original_R']=rec['cut_before_full'][...,R,:].cpu().contiguous();saved[prefix+'_cut_after_R']=rec['cut_after_R'].cpu().contiguous();saved[prefix+'_target_R']=rec['target_after_R'].cpu().contiguous();saved[prefix+'_a_R']=rec['a_R'].cpu().contiguous();meta['cuts'][str(cut)]={'candidate_full_vocab_logprob':score(z,r['targets']),'cut_input_full_sha256':dh(rec['cut_before_full']),'target_input_full_sha256':dh(rec['target_before_full']),'target_nonR_matches_clean':True,'R_only':rec['cut_outside_R_unchanged'],'injection_delta_l2':float((donor.double()-k2.double()).norm())}
    torch.cuda.synchronize();meta['wall_seconds_cuda_synchronized_excludes_save']=time.monotonic()-t0;meta['peak_allocated_bytes']=torch.cuda.max_memory_allocated();meta['peak_reserved_bytes']=torch.cuda.max_memory_reserved();meta['source_sha256']=before;tmp=OUT/(key+'.'+uuid.uuid4().hex+'.partial')
    try:save_file(saved,str(tmp));os.link(tmp,out)
    finally:
     if tmp.exists():tmp.unlink()
    meta['tensor_sha256']=sha(out);save(outj,meta);complete.append({'id':ident,'split':'train','metadata_sha256':sha(outj),'tensor_sha256':sha(out)});print('R_PRODUCER_64_CASE',len(complete),ident,meta['wall_seconds_cuda_synchronized_excludes_save'],flush=True)
  assert {str(p):sha(p) for p in sources}==before and [(n,p.data_ptr(),p._version) for n,p in list(model.named_parameters())+list(model.named_buffers())]==params;save(OUT/'PENDING.json',{'status':'R_PRODUCER_DISCOVERY64_PENDING_INDEPENDENT_REPLAY','cases':complete,'source_sha256':before,'elapsed_seconds':time.monotonic()-start,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'scope':'three upstream cut forwards; L18 target anchor reused; full cut and target inputs hash only, not stored'})
 finally:
  if model is not None and oldsteps is not None:model.model.total_ut_steps=oldsteps
  os.close(lock)
if __name__=='__main__':main()
