"""E2.4 three-case technical gate; requires exclusive GPU lock."""
import os,json,gzip,pathlib,hashlib,fcntl,time,uuid
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import torch
from safetensors.torch import load_file,save_file
from safetensors import safe_open
from transformers import AutoModelForCausalLM
import e2_double_hook as dh
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
import e2_state_intervention as single
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=pathlib.Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E2';T=E/'state_intervention/discovery64';OUT=E/'state_intervention/mediation/discovery64';CAT=B/'results/compact_input_catalog.jsonl.gz';ST=E/'E2_fixed_probe_states.safetensors';MODEL=ROOT/'model/model.safetensors';ORDER=E/'state_intervention/mediation/ORDER_E2_4_FULLR_ATTENTION_v1.md';FROZEN=B/'orders/E1_DISCOVERY64_FROZEN.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def score(z,targets):return [float(z.float().log_softmax(-1).gather(-1,torch.tensor([[t]],device=z.device)).sum()) for t in targets]
def save(p,x):
 with p.open('x') as f:json.dump(x,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 assert sha(MODEL)=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af' and sha(CAT)=='a38218bfe0c9940f326663b8ecc0447f875a18c07f0091813ddfef4a3d12342d' and sha(ST)=='8ccae86df9786e88fa63219d10c9ba101ae96f49793da7f63ae01449e49e62e5';strict=T/'CUDA_REPLAY_64_STRICT.json';assert sha(strict)=='94297e44f3bc87a6bf40737d2be604bb5aee5e837934a72a33e2e5ea90f3bde8';strict_data=json.loads(strict.read_text());assert strict_data['pending_sha256']==sha(T/'PENDING_INDEPENDENT_AUDIT.json');assert json.loads((T/'CONTROL_MATH_AUDIT.json').read_text())['status']=='E2_DISCOVERY64_INDEPENDENT_CONTROL_MATH_PASSED';assert sha(FROZEN)=='6faad196db958744780ee6477af8839c693798b8b24d2a01a2d450c856c720e1';prior_start=json.loads((T/'STARTED.json').read_text())['source_sha256'];shared=[MODEL,CAT,ST,ROOT/'model/modeling_ouro.py',ROOT/'model/config.json',pathlib.Path(single.__file__).resolve()];assert all(str(p) in prior_start and sha(p)==prior_start[str(p)] for p in shared);prior_pending=json.loads((T/'PENDING_INDEPENDENT_AUDIT.json').read_text());prior_cases={(x['split'],x['id']):x for x in prior_pending['cases']};assert len(prior_cases)==64;selection=json.loads(FROZEN.read_text());assert selection['count']==64 and set(selection['groups'])=={'rescue','damage','stable_correct','stable_wrong'} and all(len(v)==16 for v in selection['groups'].values());order={'questions':[{'split':'train','id':x['id']} for group in sorted(selection['groups']) for x in selection['groups'][group]]};assert len({q['id'] for q in order['questions']})==64;tech=E/'state_intervention/mediation/technical_three';tech_replay=tech/'CUDA_REPLAY.json';assert sha(tech_replay)=='9afcc605f8547d55b2fc7ca1325be113d0eb703f1922d50ff837d1327575d955';tech_pending=json.loads((tech/'PENDING.json').read_text());tech_source=tech_pending['source_sha256'];assert all(str(p) in tech_source and sha(p)==tech_source[str(p)] for p in [pathlib.Path(dh.__file__).resolve(),pathlib.Path(single.__file__).resolve(),MODEL,CAT,ST,ORDER,ROOT/'model/modeling_ouro.py',ROOT/'model/config.json']);tech_result=json.loads(tech_replay.read_text());assert tech_result['status']=='E2_4_THREE_INDEPENDENT_DOUBLEHOOK_CUDA_REPLAY_PASSED' and tech_result['source_pending_sha256']==sha(tech/'PENDING.json') and len(tech_result['cases'])==3 and all(set(c['conditions'])=={'C00','C11','C10','C01','full_norm','full_unrelated'} and all(v=='BITWISE_FULL_LOGITS_AND_CANDIDATES_MATCH' for v in c['conditions'].values()) for c in tech_result['cases'])
 with gzip.open(CAT,'rt') as f:catalog={(r['split'],r['id']):r for r in map(json.loads,f)}
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);model=None;oldsteps=None
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=True);assert not (OUT/'PENDING.json').exists();source=[MODEL,CAT,ST,ORDER,FROZEN,strict,T/'PENDING_INDEPENDENT_AUDIT.json',T/'CONTROL_MATH_AUDIT.json',tech_replay,tech/'PENDING.json',pathlib.Path(__file__).resolve(),pathlib.Path(dh.__file__).resolve(),pathlib.Path(single.__file__).resolve(),pathlib.Path(__file__).with_name('e2_4_cuda_replay.py').resolve(),pathlib.Path(__file__).with_name('e2_4_discovery64_cuda_replay.py').resolve(),pathlib.Path(__file__).with_name('e2_4_discovery64_control_audit.py').resolve(),ROOT/'model/modeling_ouro.py',ROOT/'model/config.json'];before={str(p):sha(p) for p in source};start_receipt={'status':'STARTED','source_sha256':before};start_path=OUT/'STARTED.json';
  if start_path.exists():assert json.loads(start_path.read_text())==start_receipt
  else:save(start_path,start_receipt)
  start=time.monotonic();assert torch.__version__==prior_pending['torch'] and torch.version.cuda==prior_pending['cuda'] and torch.cuda.get_device_name()==prior_pending['gpu'];model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;layer=model.model.layers[17];oldsteps=model.model.total_ut_steps;params=[(n,p.data_ptr(),p._version) for n,p in list(model.named_parameters())+list(model.named_buffers())];completed=[]
  with torch.inference_mode(),safe_open(str(ST),framework='pt',device='cpu') as st:
   for case in order['questions']:
    r=catalog[(case['split'],case['id'])];ident=r['id'];key=hashlib.sha256((r['split']+'\0'+ident).encode()).hexdigest();out=OUT/(key+'.safetensors');outj=OUT/(key+'.json');
    if out.exists() or outj.exists():
     assert out.exists() and outj.exists();prev=json.loads(outj.read_text());assert prev['id']==ident and prev['split']==r['split'] and prev['R']==r['R'] and prev['targets']==r['targets'] and prev['input_ids_sha256']==r['input_ids_sha256'] and prev['tensor_sha256']==sha(out) and prev['source_sha256']==before;completed.append({'id':ident,'split':r['split'],'metadata_sha256':sha(outj),'tensor_sha256':sha(out)});continue
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();case_start=time.monotonic();mp=T/(key+'.json');tp=T/(key+'.safetensors');meta0=json.loads(mp.read_text());link=prior_cases[(r['split'],ident)];assert sha(mp)==link['metadata_sha256'] and sha(tp)==link['tensor_sha256']==meta0['tensor_sha256'];old=load_file(str(tp));assert meta0['id']==ident and meta0['R']==r['R'] and meta0['targets']==r['targets'];R=r['R'];ids=torch.tensor([r['input_ids']],device='cuda');ih=hashlib.sha256(json.dumps(r['input_ids']).encode()).hexdigest();full=old['native2_input_full'].cuda();x2=old['identity_injected_R'].cuda();x1=old['rollback_full_injected_R'].cuda();a2=old['identity_a_R'].cuda();ap=old['rollback_full_a_R'].cuda();assert dh.bits(x2,full[...,R,:]) and dh.bits(x1,old['native1_input_full'][...,R,:].cuda());saved={};meta={'id':ident,'split':r['split'],'R':R,'targets':r['targets'],'input_ids_sha256':ih,'source_metadata_sha256':sha(mp),'source_tensor_sha256':sha(tp),'conditions':{}}
    def common_call(ctx):
     model.model.total_ut_steps=2;seen=[]
     def root(m,args,kw):
      inp=args[0] if args else kw.get('input_ids');assert torch.is_tensor(inp) and dh.bits(inp,ids) and kw.get('use_cache') is False and kw.get('past_key_values') is None;seen.append(1)
     h=model.register_forward_pre_hook(root,with_kwargs=True)
     try:
      with ctx as rec:y=model(ids,labels=None,use_cache=False,exit_at_step=1)
     finally:h.remove()
     assert len(seen)==1;return y.logits[:,R,:].detach(),rec
    kwargs={'physical_index':17,'ut':1,'R':R,'expected_full':full,'expected_id':case['id'],'current_id':ident,'expected_input_ids_sha':ih,'input_ids_sha':ih}
    for name,x,a,expected_a,anchor in [('C00',x2,a2,a2,'identity'),('C11',x1,ap,ap,'rollback_full'),('C10',x1,a2,ap,None),('C01',x2,ap,a2,None)]:
     z,rec=common_call(dh.double_intervene(layer,**kwargs,x_patch=x,a_patch=a));assert dh.bits(rec['a_receiver_R'],expected_a) and rec['x_outside_R_unchanged'] and rec['a_outside_R_unchanged']
     if anchor:assert dh.bits(z.cpu(),old[anchor+'_logits']) and dh.bits(rec['a_injected_R'].cpu(),old[anchor+'_a_R']) and dh.bits(rec['x_injected_R'].cpu(),old[anchor+'_injected_R'])
     saved[name+'_logits']=z.cpu().contiguous()
     for k in ('x_receiver_R','x_injected_R','a_receiver_R','a_injected_R','y_R','z_R'):saved[name+'_'+k]=rec[k].cpu().contiguous()
     meta['conditions'][name]={'status':'OK','candidate_full_vocab_logprob':score(z,r['targets']),'full_logits_sha256':thash(z),'a_receiver_sha256':thash(rec['a_receiver_R']),'x_outside_R_unchanged':True,'a_outside_R_unchanged':True,'diagonal_anchor':anchor}
    donor=meta0['donor']['id'];donorkey='train_'+hashlib.sha256(donor.encode()).hexdigest();dother=st.get_tensor(donorkey+'_u1_l18_input').double()-st.get_tensor(donorkey+'_u0_l18_input').double();assert donor!=ident or r['split']!='train';assert catalog[('train',donor)]['input_ids']!=r['input_ids'];a=x1.double();b=x2.double();delta=b-a;an=a.norm();bn=b.norm();on=dother.norm();dn=delta.norm();meta['full_control_donor']={'id':donor,'split':'train','inherited_frozen_rule':meta0['donor']['selection'],'source_state_sha256':sha(ST),'donor_delta_l2':float(on),'full_delta_l2':float(dn),'x1_l2':float(an),'x2_l2':float(bn)};controls={'full_norm':None if an==0 and bn!=0 else (b if an==0 else a*(bn/an)),'full_unrelated':None if on==0 and dn!=0 else (b if on==0 else b-dother.cuda()*(dn/on.cuda()))}
    for name,exact in controls.items():
     if exact is None:meta['conditions'][name]={'status':'UNDEFINED'};continue
     patch=exact.to(torch.bfloat16);ctx=single.intervene(layer,physical_index=17,ut=1,R=R,expected_id=ident,current_id=ident,expected_shape=tuple(full.shape),expected_input_ids_sha=ih,input_ids_sha=ih,patch=patch,expected_full=full);z,rec=common_call(ctx);assert rec['outside_R_unchanged'];saved[name+'_logits']=z.cpu().contiguous();saved[name+'_injected_R']=patch.cpu().contiguous();saved[name+'_a_R']=rec['a_R'].cpu().contiguous();saved[name+'_cast_residual_fp64']=(patch.double()-exact).cpu().contiguous();meta['conditions'][name]={'status':'OK','candidate_full_vocab_logprob':score(z,r['targets']),'full_logits_sha256':thash(z),'outside_R_unchanged':True}
    torch.cuda.synchronize();meta['wall_seconds_cuda_synchronized_excludes_save']=time.monotonic()-case_start;meta['peak_allocated_bytes']=torch.cuda.max_memory_allocated();meta['peak_reserved_bytes']=torch.cuda.max_memory_reserved();meta['source_sha256']=before;tmp=OUT/(key+'.'+uuid.uuid4().hex+'.partial');
    try:save_file(saved,str(tmp));os.link(tmp,out)
    finally:
     if tmp.exists():tmp.unlink()
    meta['tensor_sha256']=sha(out);save(outj,meta);completed.append({'id':ident,'split':r['split'],'metadata_sha256':sha(outj),'tensor_sha256':sha(out)});print('E2_4_CASE',ident,flush=True)
  assert [(n,p.data_ptr(),p._version) for n,p in list(model.named_parameters())+list(model.named_buffers())]==params and {str(p):sha(p) for p in source}==before;save(OUT/'PENDING.json',{'status':'DISCOVERY64_PENDING_INDEPENDENT_BARE_DOUBLEHOOK_REPLAY','cases':completed,'source_sha256':before,'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda})
 finally:
  if model is not None and oldsteps is not None:model.model.total_ut_steps=oldsteps
  os.close(lock)
if __name__=='__main__':main()
