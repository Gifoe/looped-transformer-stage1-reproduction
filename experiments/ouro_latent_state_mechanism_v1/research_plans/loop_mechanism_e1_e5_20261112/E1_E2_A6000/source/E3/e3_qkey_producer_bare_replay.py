"""Independent bare-hook model-forward replay of E3 Q-key producer technical cases."""
import os,json,hashlib,fcntl,torch
from pathlib import Path
from contextlib import contextmanager
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
import recovery_adapter,probe_eight,technical
import torch.nn.functional as F
ROOT=Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';OUT=B/'E3/qkey_producer_technical';SEL=['ACTAAP_2013_7_16','Mercury_405804','MSA_2012_8_30']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.device==b.device and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def need(x,s):
 if not x:raise ValueError(s)
@contextmanager
def bare(model,cut,R,P,hidden=None,key=None):
 layers=model.model.layers;h=[];active=[];attn=[];seq=[];r={'calls':0};old=F.scaled_dot_product_attention
 def cut_hook(m,args,kw):
  u=kw.get('current_ut'); x=args[0] if args else kw['hidden_states']
  if u==0:r['donor_hidden']=x.detach().clone()
  if u==1:
   r['cut_input']=x.detach().clone()
   if hidden is not None:
    z=x.clone();z[:,P:R,:]=hidden[:,P:R,:];need(bits(z[:,:P,:],x[:,:P,:]) and bits(z[:,R:,:],x[:,R:,:]),'hidden outside Q')
    if args:return (z,)+args[1:],kw
    n=dict(kw);n['hidden_states']=z;return args,n
 def pre(m,args,kw):
  u=kw.get('current_ut');need(u==len(seq) and not active,'target event');seq.append(u);active.append(u)
  if u==1:r['x_R']=(args[0] if args else kw['hidden_states'])[:,R,:].detach().clone()
 def post(m,args,kw,out):need(len(active)==1 and not attn,'target close');active.pop()
 def ap(m,args,kw):need(len(active)==1 and not attn and kw.get('current_ut')==active[0],'attn event');attn.append(1)
 def az(m,args,kw,out):need(len(attn)==1,'attn close');attn.pop()
 def ar(m,args,out):
  if active and active[0]==1:r['a_R']=out[:,R,:].detach().clone()
 def sdpa(*args,**kw):
  if not active or active[0]!=1:return old(*args,**kw)
  need(len(attn)==1 and r['calls']==0 and len(args)>=3,'sdpa event');r['calls']+=1;q,k,v=args[:3];r['k']=k.detach().clone();base=old(*args,**kw)
  if key is not None:
   need(key.shape==k.shape and key.dtype==k.dtype and key.device==k.device,'key match');km=k.clone();km[...,P:R,:]=key[...,P:R,:];need(bits(km[...,:P,:],k[...,:P,:]) and bits(km[...,R:,:],k[...,R:,:]),'key outside Q');alt=old(q,km,v,*args[3:],**kw);z=base.clone();z[...,R,:]=alt[...,R,:];mask=torch.ones(z.shape[-2],device=z.device,dtype=torch.bool);mask[R]=False;need(bits(z[...,mask,:],base[...,mask,:]),'output outside R');base=z
  r['sdpa_R']=base[...,R,:].detach().clone();return base
 try:
  h=[layers[cut].register_forward_pre_hook(cut_hook,with_kwargs=True),layers[17].register_forward_pre_hook(pre,with_kwargs=True),layers[17].register_forward_hook(post,with_kwargs=True),layers[17].self_attn.register_forward_pre_hook(ap,with_kwargs=True),layers[17].self_attn.register_forward_hook(az,with_kwargs=True),layers[17].input_layernorm_2.register_forward_hook(ar)];F.scaled_dot_product_attention=sdpa;yield r;need(seq==[0,1] and r['calls']==1 and not active and not attn and 'a_R' in r,'incomplete')
 finally:
  F.scaled_dot_product_attention=old
  for v in h:v.remove()
def main():
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX);p=OUT/'pending.json';old=OUT/'replay.json';pend=json.loads(p.read_text());rr=json.loads(old.read_text());need(rr['status']=='E3_QKEY_PRODUCER_THREE_CASE_CUDA_SCORE_REPLAY_PASSED' and rr['pending_sha256']==sha(p) and all(sha(Path(k))==v for k,v in pend['source_hashes'].items()),'source proof');need(torch.cuda.is_available() and torch.__version__==pend['runtime']['torch'] and torch.version.cuda==pend['runtime']['cuda'] and torch.cuda.get_device_name()==pend['runtime']['gpu'],'runtime');recovery_adapter.initialize()
  with torch.no_grad():m=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  m.config.use_cache=False;m.model.config.use_cache=False;m.model.total_ut_steps=2;mh=probe_eight.model_hash(m);need(mh==pend['model_state_sha256'],'model hash');q=[]
  for i,ident in enumerate(SEL):
   mp=OUT/(str(i)+'.json');tp=mp.with_suffix('.safetensors');need(sha(mp)==pend['questions'][i]['metadata_sha256'] and sha(tp)==pend['questions'][i]['tensor_sha256'],'technical tensor source');meta=json.loads(mp.read_text());saved=load_file(str(tp));source,_,_=recovery_adapter.load('test',ident);e=source['natural'];need(meta['id']==ident and meta['R']==e['R'] and meta['targets']==e['targets'],'identity');R=e['R'];P=1839;x=torch.tensor([e['ids']],device='cuda')
   def fw():return m(x,labels=None,use_cache=False,exit_at_step=1).logits[0,R,:].unsqueeze(0)
   for cut in (0,8,16,17):
    with torch.inference_mode(),bare(m,cut,R,P) as c:z0=fw()
    with torch.inference_mode(),bare(m,cut,R,P,hidden=c['donor_hidden']) as d:z1=fw()
    for name,h,k,z0_ref,rec in (('A0K0',None,None,z0,c),('A1K1',c['donor_hidden'],None,z1,d),('A0K1',None,d['k'],None,None),('A1K0',c['donor_hidden'],c['k'],None,None)):
     if z0_ref is None:
      with torch.inference_mode(),bare(m,cut,R,P,hidden=h,key=k) as r:z=fw()
     else:r=rec;z=z0_ref
     tag=f'cut{cut}_{name}';need(bits(r['cut_input'],c['cut_input']) and bits(z,saved[tag+'_logits'].cuda()) and bits(r['sdpa_R'],saved[tag+'_sdpa_R'].cuda()) and bits(r['a_R'],saved[tag+'_a_R'].cuda()) and bits(r['x_R'],saved[tag+'_receiver_x_R'].cuda()) and technical.score(z,e['targets'])==meta['conditions'][str(cut)][name]['scores'],'bare full parity')
   q.append({'id':ident,'metadata_sha256':sha(mp),'tensor_sha256':sha(tp),'full_forward_conditions':16});print('BARE_CASE_PASSED',ident,flush=True)
  need(probe_eight.model_hash(m)==mh and all(sha(Path(k))==v for k,v in pend['source_hashes'].items()),'model/source drift');probe_eight.write_json(OUT/'bare_replay.json',{'status':'E3_QKEY_PRODUCER_THREE_CASE_INDEPENDENT_BARE_FORWARD_REPLAY_PASSED','pending_sha256':sha(p),'score_replay_sha256':sha(old),'bare_script_sha256':sha(Path(__file__)),'model_state_sha256':mh,'questions':q,'scope':'Independent bare hooks/full model forward; source adapter and score function reused.'});print('BARE_REPLAY_OK',sha(OUT/'bare_replay.json'),flush=True)
 finally:os.close(lock)
if __name__=='__main__':main()
