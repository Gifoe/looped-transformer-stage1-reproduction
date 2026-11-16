"""Independent full-KV softmax oracle for actual three single-event cases."""
import pathlib,json,datetime,numpy as np,torch
R=pathlib.Path('/root/ouro-binding-interface-sol-20261117');RAW=pathlib.Path('/root/autodl-tmp/ouro-binding-interface-sol-20261117/raw');torch.set_num_threads(4);data=torch.load(RAW/'technical_witnesses.pt',map_location='cpu',weights_only=False);tests=[];maxratio=0.;maxrecon=0.
for name,case in data.items():
 assert len(case['events'])==1
 for (u,l),w in case['events'].items():
  assert [u+1,l+1]==[2,case['layer']];Q=w['Q_H'].double().numpy();K=w['K_full'].double().numpy();V=w['V_full'].double().numpy();a,b=w['record_spans'][0];c,d=w['record_spans'][1];h=w['H_start'];T=K.shape[-2];dim=K.shape[-1];future=np.arange(T)[None,:]>np.arange(h,T)[:,None];outputs={}
  for kind,ck,cv in [('base',0,0),('K',1,0),('V',0,1),('joint',1,1)]:
   k=K.copy();v=V.copy()
   if ck:k[:,:,a:b]=K[:,:,c:d];k[:,:,c:d]=K[:,:,a:b]
   if cv:v[:,:,a:b]=V[:,:,c:d];v[:,:,c:d]=V[:,:,a:b]
   s=np.matmul(Q,np.swapaxes(k,-1,-2))*w['scale'];s=np.where(future[None,None],-np.inf,s);s-=s.max(-1,keepdims=True);p=np.exp(s);p/=p.sum(-1,keepdims=True);assert np.isfinite(p).all();outputs[kind]=np.matmul(p,v)
  delta=w['paired_delta_FP64'].numpy();scale=np.maximum(1,np.abs(V).max(axis=(-2,-1)));tau=128*np.finfo(np.float64).eps*(np.arange(h,T)[None,None,:,None]+dim+2)*scale[:,:,None,None]
  for kind in ['K','V','joint']:
   error=np.abs(outputs[kind]-outputs['base']-(0 if kind=='joint' else delta));assert np.all(error<=tau);maxratio=max(maxratio,float(np.max(error/tau)))
  before=w['native_SDPA_H'];after=w['edited_SDPA_H'];rel=np.linalg.norm(before.double().numpy()-outputs['base'],axis=-1)/np.maximum(np.linalg.norm(outputs['base'],axis=-1),1e-12);assert np.max(rel)<=.01;maxrecon=max(maxrecon,float(rel.max()))
  if w['kind']=='self':assert torch.equal(before.view(torch.uint8),after.view(torch.uint8))
  elif w['kind']=='semantic':assert torch.equal((before.double()+w['paired_delta_FP64']).to(before.dtype).contiguous().view(torch.uint8),after.contiguous().view(torch.uint8))
  expected=(w['actual_projected_H'].double()-w['native_base_projected_H'].double())[0].norm(dim=-1).numpy();assert np.allclose(expected,case['audit']['events'][0]['projected_actual_delta_norms'],atol=1e-10,rtol=1e-11);tests.append({'case':name,'layer':l+1,'native_reconstruction_relative_max':float(rel.max())})
z={'status':'PASS','source':'IndependentNumPyfullcausalKVpermutation vs inheritedparent idealpairedDelta; exactsingleeventselector+nativeanchors+selfcheckedGPU','witness_events':len(tests),'maximum_error_div_tau':maxratio,'maximum_native_reconstruction_relative':maxrecon,'tests':tests,'body_forwards':12,'executor':'Sol self-audit, notindependentagentGPUreplication','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'TECHNICAL_GATE.json').write_text(json.dumps(z,indent=2));print('SINGLE_EVENT_TECHNICAL_GATE_PASS',len(tests),maxratio,maxrecon)
