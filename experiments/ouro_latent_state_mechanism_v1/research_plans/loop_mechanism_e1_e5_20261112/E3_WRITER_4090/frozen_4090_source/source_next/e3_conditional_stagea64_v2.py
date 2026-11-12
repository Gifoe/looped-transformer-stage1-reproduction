"""Conditional Stage A fixed technical source and actual L9 KV endpoint gate."""
import os,json,hashlib,fcntl,time,pathlib,uuid,torch,transformers
from safetensors.torch import save_file,load_file
from transformers import AutoModelForCausalLM
from e3_qkey_producer_observer import producer_observer,bits
from capture_branch_reference import capture_real_branches,reference_deltas
from e3_conditional_l9_observer import observe_l9
import e3_conditional_factor_question_v2 as factor, unrelated_reference_ledger as ledger
from writer4090_model_digest import model_digest
ROOT=pathlib.Path('/root/autodl-tmp/ouro-e1-e3-offload-20261113');OLD=pathlib.Path('/root/autodl-tmp/ouro-e4-e5');Q=ROOT/'results';MODEL=OLD/'model';INPUT=OLD/'inputs/runner_inputs.jsonl';PREFIX=OLD/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/natural_prefix_identity_20261111/reference_prefix_ids.json';TECH=Q/'technical_reference_three';TRAIN=Q/'reference_train64';L=Q/'frozen_unrelated_4090.json';OUT=Q/'conditional_stage_a_train64_v2';TCH=Q/'conditional_stage_a_technical_v2';ORDER=ROOT/'source_next/ORDER_E3_FIRST_READ_VS_LATER_WRITES_CONDITIONAL_v1.md';HOLD=Q/'holdout_analysis/summary.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def loadrec(base,c):
 stem=hashlib.sha256(c['id'].encode()).hexdigest();mp=base/(stem+'.json');tp=base/(stem+'.safetensors');assert sha(mp)==c['metadata_sha256'] and sha(tp)==c['tensor_sha256'];return json.loads(mp.read_text()),load_file(str(tp))
def same(a,b):
 if torch.is_tensor(a) or torch.is_tensor(b):return torch.is_tensor(a) and torch.is_tensor(b) and bits(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return isinstance(b,type(a)) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
def main():
 assert sha(ORDER)=='94469c7fe667e7bf46bf5bb03117436d8163a08987144f0331d34c5a20b9bb00' and json.loads(HOLD.read_text())['joint_support'] is True;hs=Q/'holdout_analysis/SOURCE.json';hsrc=json.loads(hs.read_text());assert hsrc['status']=='4090_HOLDOUT_ANALYSIS' and all(sha(pathlib.Path(k))==v for k,v in hsrc['source_sha256'].items());catalog_audit=ROOT/'inputs/INPUT_CATALOG_AUDIT.json';assert sha(catalog_audit)=='270c97913ccc161bed73e72a6de655f149160a3e01f9a9bf8780c34efa7f8a39';frozen=json.loads(L.read_text());tr=json.loads((TRAIN/'RECEIPT.json').read_text());te=json.loads((TECH/'RECEIPT.json').read_text());assert frozen['status']=='4090_UNRELATED_SITE_LEDGER_FROZEN_BEFORE_TREATMENTS' and frozen['train_reference_receipt_sha256']==sha(TRAIN/'RECEIPT.json') and all(sha(pathlib.Path(k))==v for k,v in tr['source_sha256'].items()) and all(sha(pathlib.Path(k))==v for k,v in te['source_sha256'].items());pool=[]
 for c in tr['cases']:
  m,t=loadrec(TRAIN,c);pool.append({'id':m['id'],'input_ids_sha256':m['input_ids_sha256'],'deltas':{(j,k):t[f'layer{j:02d}_{k}_delta'] for j in range(8,16) for k in ('a','b')}})
 tp=json.loads((TCH/'PENDING.json').read_text());br=json.loads((TCH/'CUDA_REPLAY.json').read_text());er=json.loads((TCH/'ENDPOINT_BARE.json').read_text());ma=json.loads((TCH/'INDEPENDENT_CPU_MATH_AUDIT.json').read_text());assert br['status']=='4090_CONDITIONAL_THREE_BARE_PASSED' and er['status']=='4090_CONDITIONAL_THREE_INDEPENDENT_KV_ENDPOINTS_PASSED' and ma['status']=='4090_CONDITIONAL_THREE_CPU_MATH_PASSED' and br['pending_sha256']==er['pending_sha256']==ma['pending_sha256']==sha(TCH/'PENDING.json') and ma['cuda_replay_sha256']==sha(TCH/'CUDA_REPLAY.json') and ma['endpoint_bare_sha256']==sha(TCH/'ENDPOINT_BARE.json');ids=[c['id'] for c in tr['cases']];assert len(ids)==len(set(ids))==64 and ids==frozen['pool_ids'];rows=[json.loads(x) for x in INPUT.read_text().splitlines()];qs={q['id']:q for q in rows if q['split']=='train' and q['id'] in ids};prefix=json.loads(PREFIX.read_text())['input_ids'];files=[pathlib.Path(__file__).resolve(),ORDER,HOLD,hs,catalog_audit,TCH/'PENDING.json',TCH/'CUDA_REPLAY.json',TCH/'ENDPOINT_BARE.json',TCH/'INDEPENDENT_CPU_MATH_AUDIT.json',L,TRAIN/'RECEIPT.json',TECH/'RECEIPT.json',INPUT,PREFIX,MODEL/'model.safetensors',MODEL/'config.json',MODEL/'modeling_ouro.py']+sorted((ROOT/'source').glob('*.py'))+sorted((ROOT/'source_next').glob('*.py'));sources={str(p):sha(p) for p in files};OUT.mkdir(exist_ok=True);assert not (OUT/'PENDING.json').exists() and not list(OUT.glob('*.partial'));started={'status':'STARTED','source_sha256':sources};sp=OUT/'STARTED.json';assert json.loads(sp.read_text())==started if sp.exists() else True;(None if sp.exists() else write(sp,started));lock=os.open(OLD/'gpu.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);m=None;old=None
 try:
  fcntl.flock(lock,fcntl.LOCK_EX);transformers.utils.import_utils._torchvision_available=False;m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();m.config.use_cache=False;m.model.config.use_cache=False;old=m.model.total_ut_steps;m.model.total_ut_steps=2;ver=lambda:[(n,id(x),x.data_ptr(),x._version) for n,x in list(m.named_parameters())+list(m.named_buffers())];v0=ver();mh=model_digest(m);prior=json.loads((Q/'treatment_technical_three/PENDING.json').read_text());assert mh==prior['model_state_sha256'];guard=lambda:None if ver()==v0 else (_ for _ in ()).throw(ValueError('model mutation'));cases=[]
  for c in tr['cases']:
   q=qs[c['id']];prior_meta,prior_t=loadrec(TRAIN,c);stem=hashlib.sha256(q['id'].encode()).hexdigest();mp=OUT/(stem+'.json');tp=OUT/(stem+'.safetensors');assert mp.exists()==tp.exists();
   if mp.exists():
    mm=json.loads(mp.read_text());tt=load_file(str(tp));expected={f'{bg}_{verb}_{ff}_{kind}' for ff in ('E','L') for bg,verb in (('P','plus'),('N','minus')) for kind in ('main','norm','unrelated')};assert mm['id']==q['id'] and mm['R']==q['R'] and mm['targets']==q['targets'] and mm['input_ids_sha256']==q['input_ids_sha256'] and mm['tensor_sha256']==sha(tp) and set(mm['conditions'])==expected and torch.equal(tt['N_logits'],prior_t['N_logits']) and torch.equal(tt['P_logits'],prior_t['P_logits']);cases.append({'id':q['id'],'metadata_sha256':sha(mp),'tensor_sha256':sha(tp)});continue
   assert q['id']==prior_meta['id'] and q['input_ids_sha256']==prior_meta['input_ids_sha256']==hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest() and q['R']==prior_meta['R']==len(q['ids'])-1 and q['targets']==prior_meta['targets'] and q['ids'][:len(prefix)]==prefix;x=torch.tensor([q['ids']],device='cuda');R=q['R'];P=len(prefix);t0=time.monotonic();torch.cuda.reset_peak_memory_stats()
   def run(hidden=None,clean=None,dk=None,dv=None):
    full=[];a=[]
    def pre(mod,args,kw):
     if kw.get('current_ut')==1:full.append((args[0] if args else kw['hidden_states']).detach().clone())
    def norm(mod,args,out):a.append(out[:,R,:].detach().clone())
    h1=m.model.layers[17].register_forward_pre_hook(pre,with_kwargs=True);h2=m.model.layers[17].input_layernorm_2.register_forward_hook(norm)
    try:
     with producer_observer(m,8,R,P,hidden_donor=hidden,expected_clean_cut=clean) as rec,observe_l9(m,R,P,donor_k=dk,donor_v=dv) as l9,capture_real_branches(m,R,expected_length=len(q['ids'])) as branch:z=m(x,labels=None,use_cache=False,exit_at_step=1).logits[:,R,:]
    finally:h1.remove();h2.remove()
    assert len(full)==1 and len(a)==2;rec['target_full']=full[0];return z,rec,l9,branch
   with torch.inference_mode():
    N,n,n9,na=run();Pv,p,p9,pa=run(n['loop1_cut_hidden'],n['receiver_cut_hidden']);assert bits(N,prior_t['N_logits'].cuda()) and bits(Pv,prior_t['P_logits'].cuda()) and bits(n9['input_R'],p9['input_R']) and bits(n9['q'][...,R,:],p9['q'][...,R,:]) and same(n9['position_arguments'],p9['position_arguments']) and same(n9['sdpa_args'],p9['sdpa_args']) and same(n9['sdpa_kwargs'],p9['sdpa_kwargs']);mask=torch.ones(n9['k'].shape[-2],device='cuda',dtype=torch.bool);mask[P:R]=False;assert all(bits(n9[k][...,mask,:],p9[k][...,mask,:]) for k in ('k','v'));delta=reference_deltas(na,pa);assert all(torch.equal(na[(j,k)].cpu(),prior_t[f'layer{j:02d}_{k}_N']) and torch.equal(pa[(j,k)].cpu(),prior_t[f'layer{j:02d}_{k}_P']) and torch.equal(delta[(j,k)].cpu(),prior_t[f'layer{j:02d}_{k}_delta']) for j in range(8,16) for k in ('a','b'));ref={'hidden1':n['loop1_cut_hidden'],'clean_cut':n['receiver_cut_hidden'],'clean_k':n['k'],'backgrounds':{'N':{'target_full':n['target_full'],**{j:n[j] for j in ('q','k','v')}},'P':{'target_full':p['target_full'],**{j:p[j] for j in ('q','k','v')}}},'deltas':delta};recipient={'id':q['id'],'input_ids_sha256':prior_meta['input_ids_sha256'],'deltas':{(j,k):prior_t[f'layer{j:02d}_{k}_delta'] for j in range(8,16) for k in ('a','b')}};mapping,u=ledger.unrelated_directions(recipient,pool);assert mapping==frozen['mapping'][q['id']]['sites'];conditions,saved=factor.run_treatments(m,ident=q['id'],ids=x,R=R,prefix_len=P,targets=q['targets'],ref=ref,unrelated=u,mapping=mapping,guard=guard);assert len(conditions)==12

   torch.cuda.synchronize();stem=hashlib.sha256(q['id'].encode()).hexdigest();saved.update({'N_logits':N.cpu().contiguous(),'P_logits':Pv.cpu().contiguous(),'N_l9_a_R':n9['a_R'].cpu().contiguous(),'P_l9_a_R':p9['a_R'].cpu().contiguous(),'N_l9_post_mlp_R':n9['post_mlp_R'].cpu().contiguous(),'P_l9_post_mlp_R':p9['post_mlp_R'].cpu().contiguous()});tmp=OUT/(stem+'.'+uuid.uuid4().hex+'.partial');save_file(saved,str(tmp));back=load_file(str(tmp));assert set(back)==set(saved) and all(torch.equal(back[k],v) for k,v in saved.items());os.link(tmp,tp);tmp.unlink();meta={'id':q['id'],'R':R,'targets':q['targets'],'input_ids_sha256':q['input_ids_sha256'],'conditions':conditions,'tensor_sha256':sha(tp),'l9_qR_and_PR_KV_equal':True,'elapsed_seconds_cuda_synchronized_excludes_save':time.monotonic()-t0,'peak_allocated_bytes':torch.cuda.max_memory_allocated()};mp=OUT/(stem+'.json');write(mp,meta);cases.append({'id':q['id'],'metadata_sha256':sha(mp),'tensor_sha256':sha(tp)});print('CONDITIONAL_STAGEA64',q['id'],flush=True)
  assert ver()==v0 and model_digest(m)==mh and all(sha(pathlib.Path(k))==v for k,v in sources.items());write(OUT/'PENDING.json',{'status':'4090_CONDITIONAL_64_PENDING_INDEPENDENT_BARE','cases':cases,'source_sha256':sources,'model_state_sha256':mh,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda})
 finally:
  if m is not None and old is not None:m.model.total_ut_steps=old
  os.close(lock)
if __name__=='__main__':main()
