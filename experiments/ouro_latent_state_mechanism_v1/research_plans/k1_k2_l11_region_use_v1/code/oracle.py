import pathlib,json,torch,numpy as np,datetime,hashlib
from guard import verify,sha
R=pathlib.Path('/root/ouro-region-use-sol-20261117');RAW=pathlib.Path('/root/autodl-tmp/ouro-region-use-sol-20261117/raw');torch.set_num_threads(4);fs=verify();tech=json.loads((R/'TECHNICAL_GPU.json').read_text());assert sha(RAW/'technical_heads.pt')==tech['raw_heads_sha256'] and sha(RAW/'technical_witnesses.pt')==tech['witness_sha256'];raw=torch.load(RAW/'technical_heads.pt',map_location='cpu',weights_only=False);wits=torch.load(RAW/'technical_witnesses.pt',map_location='cpu',weights_only=False);nv=0;maxlp=0.;maxratio=0.;maxrecon=0.;sites=0
for c in tech['cases']:
 for name,z in c['conditions'].items():
  for d in [1,2]:
   x=raw[c['id']][name][d];s=z['scores'][str(d)];assert hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==s['full49152_sha256'];v=x.flatten().double().numpy();a=v[[330,389,340,422]];assert a.tolist()==s['candidate_logits'];ties=np.flatnonzero(a==a.max()).tolist();assert ties==s['ties'] and ties[0]==s['prediction'];m=v.max();err=float(np.abs(a-m-np.log(np.exp(v-m).sum())-s['candidate_logprobs']).max());assert err<2e-5;maxlp=max(maxlp,err);nv+=1
for name,case in wits.items():
 q=case['q']
 for (u,l),w in case['events'].items():
  assert (u,l)==(1,10);Q=w['Q_H'].double().numpy();K=w['K_full'].double().numpy();V=w['V_full'].double().numpy();a,b=w['record_spans'][0];c,d=w['record_spans'][1];h=w['H_start'];T=K.shape[-2];dim=K.shape[-1];future=np.arange(T)[None,:]>np.arange(h,T)[:,None];outputs={}
  for kind,ck,cv in [('base',0,0),('K',1,0),('V',0,1),('joint',1,1)]:
   k=K.copy();v=V.copy()
   if ck:k[:,:,a:b]=K[:,:,c:d];k[:,:,c:d]=K[:,:,a:b]
   if cv:v[:,:,a:b]=V[:,:,c:d];v[:,:,c:d]=V[:,:,a:b]
   s=np.matmul(Q,np.swapaxes(k,-1,-2))*w['scale'];s=np.where(future[None,None],-np.inf,s);s-=s.max(-1,keepdims=True);p=np.exp(s);p/=p.sum(-1,keepdims=True);outputs[kind]=np.matmul(p,v)
  delta=w['paired_delta_FP64'].numpy();scale=np.maximum(1,np.abs(V).max(axis=(-2,-1)));tau=128*np.finfo(np.float64).eps*(np.arange(h,T)[None,None,:,None]+dim+2)*scale[:,:,None,None]
  for kind in ['K','V','joint']:
   err=np.abs(outputs[kind]-outputs['base']-(0 if kind=='joint' else delta));assert (err<=tau).all();maxratio=max(maxratio,float((err/tau).max()))
  before=w['native_SDPA_H'];after=w['edited_SDPA_H'];mask=w['region_mask'];opened=[t for bit,span in zip([1,2,4],[q['Q_span'],q['O_span'],[q['R'],q['R']+1]]) if mask&bit for t in range(*span)];row=np.array([t in opened for t in range(h,T)]);assert torch.equal(after[:,:,~row].contiguous().view(torch.uint8),before[:,:,~row].contiguous().view(torch.uint8))
  if w['kind']=='semantic':assert torch.equal((before.double()+w['paired_delta_FP64']*torch.tensor(row)[None,None,:,None]).to(before.dtype).contiguous().view(torch.uint8),after.contiguous().view(torch.uint8))
  rel=np.linalg.norm(before.double().numpy()-outputs['base'],axis=-1)/np.maximum(np.linalg.norm(outputs['base'],axis=-1),1e-12);assert rel.max()<=.01;maxrecon=max(maxrecon,float(rel.max()));e=case['record']['audit']['events'][0];proj=(w['actual_projected_H'].double()-w['native_base_projected_H'].double())[0].norm(dim=-1).numpy();assert np.allclose(proj,e['projected_actual_delta_norms'],atol=1e-10,rtol=1e-11)
  if w['kind']=='random':assert e['dose']['actual']==e['actual_delta_norms'];assert all(t==0 for t,ok in zip(e['dose']['target'],row) if not ok)
  sites+=1
assert nv==72 and sites==24
z={'status':'PASS','freeze_sha256':fs,'body_forwards':36,'saved_full49152_endpoints':nv,'witness_sites':sites,'independent_NumPy_fullKV_oracle_max_error_over_tau':maxratio,'native_reconstruction_per_head_row_relative_max':maxrecon,'full_logprob_max_error':maxlp,'mask0_native_exact':True,'mask7_parent_exact_both_conditions':True,'partial_open_immediate_rows_match_full':True,'closed_rows_bitwise_unchanged':True,'Q_labelprefix_actual_states_KV_all24layers_bothloops':'PASS technicalpairedlabels; allsemdevelopment/freshpairs auditedduringexecution','executor':'Sol CPU self-audit, not independent neural replication','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'TECHNICAL_GATE.json').write_text(json.dumps(z,indent=2));print('REGION_TECHNICAL_GATE_PASS',sites,nv,flush=True)
