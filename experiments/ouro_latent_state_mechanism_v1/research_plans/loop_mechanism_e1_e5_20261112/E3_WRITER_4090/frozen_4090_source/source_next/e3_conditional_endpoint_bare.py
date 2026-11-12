"""Independent L9 bare KV endpoint replay; full model and R-only SDPA."""
import os,json,pathlib,hashlib,fcntl,torch,transformers
import torch.nn.functional as F
from contextlib import contextmanager
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from writer4090_bare_hook import bare,bits
from writer4090_model_digest import model_digest
R=pathlib.Path('/root/autodl-tmp/ouro-e1-e3-offload-20261113');O=pathlib.Path('/root/autodl-tmp/ouro-e4-e5');P=R/'results/conditional_stage_a_technical';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
@contextmanager
def l9(m,r,prefix,dk=None,dv=None):
 l=m.model.layers[8];old=F.scaled_dot_product_attention;active=[];attn=[];state=[];v={}
 def pre(mod,args,kw):u=kw['current_ut'];assert u==len(state);state.append(u);active.append(u)
 def post(mod,args,kw,out):assert active.pop()==kw['current_ut']
 def ap(mod,args,kw):attn.append(1)
 def az(mod,args,kw,out):assert attn.pop()==1
 def a(mod,args,out):
  if active and active[-1]==1:v['a']=out[:,r,:].clone()
 def f(*args,**kw):
  if not active or active[-1]!=1 or not attn:return old(*args,**kw)
  q,k,val=args[:3];v['k']=k.clone();v['v']=val.clone();base=old(*args,**kw)
  if dk is None and dv is None:return base
  kk=k.clone();vv=val.clone()
  if dk is not None:kk[...,prefix:r,:]=dk[...,prefix:r,:]
  if dv is not None:vv[...,prefix:r,:]=dv[...,prefix:r,:]
  alt=old(q,kk,vv,*args[3:],**kw);out=base.clone();out[...,r,:]=alt[...,r,:];mask=torch.arange(q.shape[-2],device=q.device)!=r;assert bits(out[...,mask,:],base[...,mask,:]);return out
 hs=[l.register_forward_pre_hook(pre,with_kwargs=True),l.register_forward_hook(post,with_kwargs=True),l.self_attn.register_forward_pre_hook(ap,with_kwargs=True),l.self_attn.register_forward_hook(az,with_kwargs=True),l.input_layernorm_2.register_forward_hook(a)]
 try:F.scaled_dot_product_attention=f;yield v;assert state==[0,1] and not active and not attn and 'a' in v
 finally:
  F.scaled_dot_product_attention=old
  for h in hs:h.remove()
def main():
 pending=json.loads((P/'PENDING.json').read_text());assert pending['status']=='4090_CONDITIONAL_THREE_PENDING_INDEPENDENT_BARE' and [x['id'] for x in pending['cases']]==['ACTAAP_2013_7_16','Mercury_405804','MSA_2012_8_30'] and all(sha(pathlib.Path(k))==v for k,v in pending['source_sha256'].items());rows=[json.loads(x) for x in (O/'inputs/runner_inputs.jsonl').read_text().splitlines()];qs={x['id']:x for x in rows if x['split']=='test'};lock=os.open(O/'gpu.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);m=None;oldsteps=None
 try:
  fcntl.flock(lock,fcntl.LOCK_EX);transformers.utils.import_utils._torchvision_available=False;m=AutoModelForCausalLM.from_pretrained(O/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();m.config.use_cache=False;m.model.config.use_cache=False;oldsteps=m.model.total_ut_steps;m.model.total_ut_steps=2;mh=model_digest(m);assert mh==pending['model_state_sha256'];done=[]
  with torch.inference_mode():
   for c in pending['cases']:
    q=qs[c['id']];stem=hashlib.sha256(c['id'].encode()).hexdigest();mp=P/(stem+'.json');tp=P/(stem+'.safetensors');assert sha(mp)==c['metadata_sha256'] and sha(tp)==c['tensor_sha256'];saved=load_file(str(tp));x=torch.tensor([q['ids']],device='cuda');r=q['R'];prefix=1839
    def fw():return m(x,labels=None,use_cache=False,exit_at_step=1).logits[:,r,:]
    with bare(m,8,r,prefix) as n,l9(m,r,prefix) as n9:N=fw()
    with bare(m,8,r,prefix,hidden=n['donor_hidden']) as p,l9(m,r,prefix) as p9:Pv=fw()
    with bare(m,8,r,prefix,hidden=n['donor_hidden']) as z,l9(m,r,prefix,dk=n9['k'],dv=n9['v']) as ze:zp=fw()
    with bare(m,8,r,prefix) as y,l9(m,r,prefix,dk=p9['k'],dv=p9['v']) as ye:zn=fw()
    assert bits(N,saved['N_logits'].cuda()) and bits(Pv,saved['P_logits'].cuda()) and bits(ze['a'],n9['a']) and bits(ye['a'],p9['a']) and bits(zp,saved['endpoint_P_logits'].cuda()) and bits(zn,saved['endpoint_N_logits'].cuda()) and bits(zp,saved['P_plus_E_main_logits'].cuda()) and bits(zn,saved['N_minus_E_main_logits'].cuda());done.append({'id':q['id'],'status':'BITWISE_BOTH_ENDPOINTS_AND_FULL_SUFFIX'});print('CONDITIONAL_ENDPOINT_BARE',q['id'],flush=True)
  assert model_digest(m)==mh and all(sha(pathlib.Path(k))==v for k,v in pending['source_sha256'].items());out=P/'ENDPOINT_BARE.json';assert not out.exists();out.write_text(json.dumps({'status':'4090_CONDITIONAL_THREE_INDEPENDENT_KV_ENDPOINTS_PASSED','pending_sha256':sha(P/'PENDING.json'),'script_sha256':sha(pathlib.Path(__file__)),'cases':done,'scope':'Separate full model with independently implemented L9 KV row hook and bare Q splice; checks both endpoint a and complete logits.'},indent=2)+'\n');print('ENDPOINT_BARE_PASS',sha(out),flush=True)
 finally:
  if m is not None and oldsteps is not None:m.model.total_ut_steps=oldsteps
  os.close(lock)
if __name__=='__main__':main()
