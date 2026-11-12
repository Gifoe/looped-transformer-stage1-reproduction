"""Independent bare-hook full model replay of fixed Q→R interval treatment technical cases."""
import os,json,pathlib,hashlib,fcntl,torch,transformers
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
import sys
ROOT=pathlib.Path('/root/autodl-tmp/ouro-e1-e3-offload-20261113');OLD=pathlib.Path('/root/autodl-tmp/ouro-e4-e5');sys.path.insert(0,str(ROOT/'source'))
import writer4090_bare_hook,writer4090_fresh_reference,writer4090_model_digest
from writer4090_bare_hook import bare,bits
from writer4090_model_digest import model_digest
from writer4090_fresh_reference import score
if not __debug__:raise RuntimeError('optimized Python forbidden')
Q=ROOT/'results';P=Q/'treatment_validation299';MODEL=OLD/'model/model.safetensors';INPUT=OLD/'inputs/runner_inputs.jsonl';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 pp=P/'PENDING.json';pending=json.loads(pp.read_text());assert pending['status']=='4090_ATTENTION_VALIDATION299_PENDING_BARE' and [x['id'] for x in pending['cases']]==[v['id'] for v in json.loads((ROOT/'inputs/WRITER_HOLDOUT_IDS_1030_299.json').read_text())['validation_299']] and len({x['id'] for x in pending['cases']})==299;sources=pending['source_sha256'];assert all(sources[str(pathlib.Path(m.__file__).resolve())]==sha(pathlib.Path(m.__file__).resolve()) for m in (writer4090_bare_hook,writer4090_fresh_reference,writer4090_model_digest)) and all(sha(pathlib.Path(k))==v for k,v in sources.items()) and torch.__version__==pending['torch'] and torch.version.cuda==pending['cuda'] and torch.cuda.get_device_name()==pending['gpu'];rows=[json.loads(x) for x in INPUT.read_text().splitlines()];byid={q['id']:q for q in rows if q['split']=='validation'};lock=os.open(OLD/'gpu.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);model=None;old=None
 try:
  fcntl.flock(lock,fcntl.LOCK_EX);transformers.utils.import_utils._torchvision_available=False;model=AutoModelForCausalLM.from_pretrained(MODEL.parent,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;old=model.model.total_ut_steps;model.model.total_ut_steps=2;mh=model_digest(model);assert mh==pending['model_state_sha256'];cases=[]
  with torch.inference_mode():
   for i,case in enumerate(pending['cases']):
    stem=hashlib.sha256(case['id'].encode()).hexdigest();mp=P/(stem+'.json');tp=P/(stem+'.safetensors');assert sha(mp)==case['metadata_sha256'] and sha(tp)==case['tensor_sha256'];m=json.loads(mp.read_text());saved=load_file(str(tp));e=byid[case['id']];assert m['id']==case['id'] and m['R']==e['R'] and m['targets']==e['targets'] and m['input_ids_sha256']==hashlib.sha256(json.dumps(e['ids']).encode()).hexdigest();R=m['R'];prefix=1839;ids=torch.tensor([e['ids']],device='cuda');assert R==ids.shape[1]-1
    def forward():return model(ids,labels=None,use_cache=False,exit_at_step=1).logits[:,R,:]
    with bare(model,8,R,prefix) as n:N=forward()
    with bare(model,8,R,prefix,hidden=n['donor_hidden']) as p:Pv=forward()
    with bare(model,8,R,prefix,hidden=n['donor_hidden'],key=n['k']) as k:PK=forward()
    assert all(bits(z,saved[name+'_logits'].cuda()) and all(bits(rec[k+'_R'],saved[name+'_'+k+'_R'].cuda()) for k in ('x','a','sdpa')) for name,z,rec in [('N',N,n),('P',Pv,p),('PK',PK,k)]);results={};expected={f'{bg}_{verb}_{branch}_{kind}' for branch in ('a',) for bg,verb,kinds in (('P','plus',('main','norm','unrelated','main_plus_key')),('N','minus',('main','norm','unrelated'))) for kind in kinds};assert set(m['conditions'])==expected
    for name,entry in m['conditions'].items():
     if entry['status']!='OK':assert entry['status'] in ('UNDEFINED_ZERO_PROPOSAL','UNDEFINED_NO_NONZERO_DONOR');results[name]=entry['status'];continue
     branch='a' if '_a_' in name else 'b';isP=name.startswith('P_');key=name.endswith('_plus_key');events={j:[] for j in range(8,16)};active=[];handles=[];seen=[]
     def pre(j):
      def fn(mod,args,kw):
       u=kw['current_ut'];assert u==len(events[j]);events[j].append(u);active.append((j,u))
      return fn
     def post(j):
      def fn(mod,args,kw,out):assert active.pop()==(j,kw['current_ut'])
      return fn
     def patch(j):
      def fn(mod,args,out):
       if active[-1][1]!=1:return
       before=saved[f'{name}_{j:02d}_before'].cuda();after=saved[f'{name}_{j:02d}_after'].cuda();assert bits(out[:,R,:],before);y=out.clone();y[:,R,:]=after;mask=torch.arange(y.shape[1],device=y.device)!=R;assert bits(out[:,mask,:],y[:,mask,:]);seen.append(j);return y
      return fn
     for j in range(8,16):
      layer=model.model.layers[j];site=layer.input_layernorm_2 if branch=='a' else layer.post_attention_layernorm_2;handles.extend([layer.register_forward_pre_hook(pre(j),with_kwargs=True),site.register_forward_hook(patch(j)),layer.register_forward_hook(post(j),with_kwargs=True)])
     try:
      with bare(model,8,R,prefix,hidden=n['donor_hidden'] if isP else None,key=n['k'] if key else None) as rec:z=forward()
     finally:
      for h in handles:h.remove()
     assert all(events[j]==[0,1] for j in events) and seen==list(range(8,16)) and not active and bits(z,saved[name+'_logits'].cuda()) and bits(rec['x_R'],saved[name+'_target_x_R'].cuda()) and bits(rec['sdpa_R'],saved[name+'_sdpa_R'].cuda()) and bits(rec['a_R'],saved[name+'_target_a_R'].cuda()) and score(z,m['targets'])==entry['scores'];results[name]='BITWISE_FULL_LOGITS_TARGET_AND_SCORE_MATCH'
    assert set(results)==expected;cases.append({'id':case['id'],'conditions':results});print('Q_TO_R_BARE_REPLAY_CASE',i,case['id'],flush=True)
  assert model_digest(model)==mh and all(sha(pathlib.Path(k))==v for k,v in sources.items());out=P/'CUDA_REPLAY.json';assert not out.exists();out.write_text(json.dumps({'status':'4090_ATTENTION_VALIDATION299_BARE_PASSED','pending_sha256':sha(pp),'script_sha256':sha(pathlib.Path(__file__)),'cases':cases,'scope':'Independent bare Q hook plus saved actual branch injection replay, checks full logits/target/score; same-site input source and candidate scorer reused; bare forward hook is independently implemented.'},indent=2)+'\n');print('4090_ATTENTION_VALIDATION299_BARE_PASS',sha(out),flush=True)
 finally:
  if model is not None and old is not None:model.model.total_ut_steps=old
  os.close(lock)
if __name__=='__main__':main()
