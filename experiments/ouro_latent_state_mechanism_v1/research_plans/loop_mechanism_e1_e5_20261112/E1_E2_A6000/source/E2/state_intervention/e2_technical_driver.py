"""E2.3 technical runner: requires Astra-frozen order and exclusive GPU lock."""
import os,json,gzip,hashlib,pathlib,time,fcntl,resource
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import torch,numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
import e2_state_intervention as primitive
from e2_state_intervention import intervene,conditions,bits,raw_probe_span,project
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=pathlib.Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E2';CAT=B/'results/compact_input_catalog.jsonl.gz';ST=E/'E2_fixed_probe_states.safetensors';ROWS=E/'E2_latent_transition_rows.jsonl.gz';CHECK=ROOT/'model/model.safetensors';OUT=E/'state_intervention/technical_three'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def key(r,u):return r['split']+'_'+hashlib.sha256(r['id'].encode()).hexdigest()+f'_u{u}_l18_input'
def save_json(p,obj):
 with p.open('x') as f:json.dump(obj,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def main(order_path):
 assert sha(CHECK)=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af';assert sha(CAT)=='a38218bfe0c9940f326663b8ecc0447f875a18c07f0091813ddfef4a3d12342d';assert sha(ST)=='8ccae86df9786e88fa63219d10c9ba101ae96f49793da7f63ae01449e49e62e5';assert sha(ROWS)=='5d749790c56ab7bf2de3a9a64609e0a5b383fe3b7c5b2ec0cc5d5129586daac4'
 order=json.loads(order_path.read_text());assert order['status']=='ASTRA_FROZEN_E2_TECHNICAL_THREE' and len(order['questions'])==3 and len({(q['split'],q['id']) for q in order['questions']})==3
 with gzip.open(CAT,'rt') as f:cat={(r['split'],r['id']):r for r in map(json.loads,f)}
 with gzip.open(ROWS,'rt') as f:rows=[r for r in map(json.loads,f)]
 pool=sorted((r for r in rows if r['split']=='train' and not r['demo']),key=lambda x:x['id']);assert len(pool)==1094
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);model=None;oldsteps=None
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not OUT.exists();OUT.mkdir();source_files=[CAT,ST,ROWS,CHECK,pathlib.Path(__file__).resolve(),pathlib.Path(primitive.__file__).resolve(),ROOT/'model/modeling_ouro.py',ROOT/'model/config.json',E/'probe_fit_attempt3/selected_models.npz',E/'probe_fit_attempt3/train_val_selection.json',E/'state_intervention/technical_train_probe_span.npz',order_path];source_hashes={str(p):sha(p) for p in source_files};save_json(OUT/'STARTED.json',{'order_sha256':sha(order_path),'source_sha256':source_hashes,'status':'STARTED'})
  assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;layer=model.model.layers[17];oldsteps=model.model.total_ut_steps;params=[(n,p.data_ptr(),p._version) for n,p in list(model.named_parameters())+list(model.named_buffers())];summ=[]
  span_path=E/'state_intervention/technical_train_probe_span.npz';assert sha(span_path)=='76302d62517710db248b58e209657943d87a563137a1a9d5b609b738525dc657';span=np.load(span_path,allow_pickle=False)['basis'];assert span.shape==(2048,4);probe_path=E/'probe_fit_attempt3/selected_models.npz';assert sha(probe_path)=='204a3788301c7aa826a198b9e40ce495ef81284b6cd41a4758dfd942f8650890';probe=np.load(probe_path,allow_pickle=False);pw=probe['l18_input_raw_trainK1_real_w'];std=probe['l18_input_raw_trainK1_real_std'];classes=[330,340,389,414,422];rng=np.random.default_rng(20261112);random_basis=np.linalg.qr(rng.standard_normal((2048,4)),mode='reduced')[0]
  with safe_open(str(ST),framework='pt',device='cpu') as store,torch.inference_mode():
   for q in order['questions']:
    r=cat[(q['split'],q['id'])];ids=torch.tensor([r['input_ids']],device='cuda');R=r['R'];idhash=hashlib.sha256(json.dumps(r['input_ids']).encode()).hexdigest();meta={'id':r['id'],'split':r['split'],'R':R,'input_ids_sha256':idhash,'targets':r['targets'],'conditions':{}};saved={};original={};logits={};full_inputs={}
    def run(ut,patch=None,expected_full=None):
     model.model.total_ut_steps=ut+1;seen=[]
     def root(m,args,kw):
      data=args[0] if args else kw.get('input_ids');assert torch.is_tensor(data) and bits(data,ids) and kw.get('use_cache') is False and kw.get('past_key_values') is None;seen.append(1)
     hook=model.register_forward_pre_hook(root,with_kwargs=True)
     try:
      with intervene(layer,physical_index=17,ut=ut,R=R,expected_id=q['id'],current_id=r['id'],expected_shape=(1,len(r['input_ids']),2048),expected_input_ids_sha=idhash,input_ids_sha=idhash,patch=patch,expected_full=expected_full) as rec:y=model(ids,labels=None,use_cache=False,exit_at_step=ut)
     finally:hook.remove()
     assert len(seen)==1;return y.logits[:,R,:].detach(),rec
    for ut in (0,1):
     model.model.total_ut_steps=ut+1;plain=model(ids,labels=None,use_cache=False,exit_at_step=ut).logits[:,R,:].detach();z,rec=run(ut);assert bits(plain,z);prior=[float(plain.float().log_softmax(-1).gather(-1,torch.tensor([[t]],device=plain.device)).sum()) for t in r['targets']];assert prior==r['native_scores'][ut];ref=store.get_tensor(key(r,ut));assert bits(rec['original_R'].cpu(),ref);original[ut]=rec['original_R'];full_inputs[ut]=rec['original_full'];logits[ut]=z;saved[f'native{ut+1}_input_full']=rec['original_full'].cpu().contiguous();saved[f'native{ut+1}_logits']=z.cpu().contiguous()
    start=int(hashlib.sha256(('E2.3-unrelated-v1\0'+r['split']+'\0'+r['id']).encode()).hexdigest(),16)%len(pool);donor=None
    for i in range(len(pool)):
     x=pool[(start+i)%len(pool)]
     if x['id']==r['id'] and r['split']=='train':continue
     if cat[('train',x['id'])]['input_ids']==r['input_ids']:continue
     other=store.get_tensor(key(x,1)).double()-store.get_tensor(key(x,0)).double()
     if torch.linalg.vector_norm(other)>0:donor=x;break
    assert donor is not None;cs=conditions(original[0],original[1],other.cuda(),projection=span,random_basis=random_basis);indices=[classes.index(t) for t in r['targets']];offered,ospec,otol=raw_probe_span(pw[:,indices],std);dv=cs['delta_fp64'].cpu().numpy().ravel();offered_delta=project(dv,offered);global_delta=project(dv,span);missing=global_delta-offered_delta;assert np.linalg.norm(offered-span@(span.T@offered))<1e-8;cs['rollback_offered']=(original[1].double()-torch.as_tensor(offered_delta,device=ids.device).reshape_as(original[1])).to(torch.bfloat16);cs['exact_fp64']['rollback_offered']=original[1].double()-torch.as_tensor(offered_delta,device=ids.device).reshape_as(original[1]);meta['offered_span']={'rank':offered.shape[1],'singular_values':ospec.tolist(),'tolerance':otol,'global_minus_offered_dimensions':span.shape[1]-offered.shape[1],'offered_delta_l2':float(np.linalg.norm(offered_delta)),'global_delta_l2':float(np.linalg.norm(global_delta)),'global_minus_offered_delta_l2':float(np.linalg.norm(missing)),'subset_residual_l2':float(np.linalg.norm(offered-span@(span.T@offered))),'orthogonality_error':float(np.linalg.norm(offered.T@offered-np.eye(offered.shape[1])))};meta['donor']={'id':donor['id'],'split':'train','selection':'sha256 E2.3-unrelated-v1, then sorted train-nondemo cyclic first nonzero, no gold','donor_state_sha256':sha(ST)}
    for name in ('identity','rollback','rollback_complement','rollback_full','rollback_offered','norm_matched','unrelated','random_rank4','restore','forward_add'):
     patch=cs[name];ut=0 if name=='forward_add' else 1
     if patch is None:meta['conditions'][name]={'status':'UNDEFINED'};continue
     z,rec=run(ut,patch,full_inputs[ut])
     if name in ('identity','restore'):assert bits(z,logits[1])
     assert rec['outside_R_unchanged'];tag=name; saved[tag+'_logits']=z.cpu().contiguous()
     for k in ('original_R','injected_R','Q_R','A_R_fp64','sdpa_R','sdpa_R_fp64','a_R','y_R','z_R'):
      saved[tag+'_'+k]=rec[k].cpu().contiguous()
     zz=z.float();logp=torch.log_softmax(zz,dim=-1);meta['conditions'][name]={'status':'OK','candidate_logits':[float(zz[0,t]) for t in r['targets']],'candidate_full_vocab_logprob':[float(logp.gather(-1,torch.tensor([[t]],device=z.device)).sum()) for t in r['targets']],'full_logits_sha256':digest(z),'outside_R_unchanged':True,'sdpa_kwargs':{k:({'shape':list(v.shape),'dtype':str(v.dtype),'sha256':digest(v)} if torch.is_tensor(v) else v) for k,v in rec['sdpa_kwargs'].items()},'sdpa_tail':[({'shape':list(v.shape),'dtype':str(v.dtype),'sha256':digest(v)} if torch.is_tensor(v) else v) for v in rec['sdpa_tail']],'sdpa_shapes':[list(v.shape) for v in rec['QKV']]}
    for n,exact in cs['exact_fp64'].items():saved[n+'_cast_residual_fp64']=(cs[n].double()-exact).cpu().contiguous()
    saved['offered_basis_fp64']=torch.from_numpy(offered.copy());saved['random_basis_fp64']=torch.from_numpy(random_basis.copy());saved['delta_fp64']=cs['delta_fp64'].cpu().contiguous();saved['component_fp64']=cs['component_fp64'].cpu().contiguous();meta['rollback_cast_error']=cs['rollback_cast_error'];tp=OUT/(hashlib.sha256((r['split']+'\0'+r['id']).encode()).hexdigest()+'.safetensors');save_file(saved,str(tp));meta['tensor_sha256']=sha(tp);mp=tp.with_suffix('.json');save_json(mp,meta);summ.append({'id':r['id'],'split':r['split'],'metadata_sha256':sha(mp),'tensor_sha256':sha(tp)})
  assert [(n,p.data_ptr(),p._version) for n,p in list(model.named_parameters())+list(model.named_buffers())]==params;assert {str(p):sha(p) for p in source_files}==source_hashes;model.model.total_ut_steps=oldsteps;save_json(OUT/'PENDING_INDEPENDENT_AUDIT.json',{'status':'TECHNICAL_PENDING_INDEPENDENT_AUDIT','cases':summ,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'order_sha256':sha(order_path),'scope':'L18 state only at R; changes Q, local K/V and residual; not pure-query or E3 path proof'})
 finally:
  if model is not None and oldsteps is not None:model.model.total_ut_steps=oldsteps
  os.close(lock)
if __name__=='__main__':
 import sys
 assert len(sys.argv)==2;main(pathlib.Path(sys.argv[1]))
