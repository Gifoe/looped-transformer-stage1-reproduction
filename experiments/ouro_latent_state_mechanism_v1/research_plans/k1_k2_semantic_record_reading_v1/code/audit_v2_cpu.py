"""Independent NumPy full-KV permutation oracle; no paired production function imported."""
import pathlib,json,numpy as np,torch,hashlib,datetime
R=pathlib.Path('/root/ouro-semantic-read-sol-20261117');RAW=pathlib.Path('/root/autodl-tmp/ouro-semantic-read-sol-20261117/raw');torch.set_num_threads(4);gpu=json.loads((R/'V2_TECHNICAL_GPU.json').read_text());records={u['id']:u for u in gpu['records']};tests=[];maximum_ratio=0.;native_rel_max=0.;fullheads=0;writechecks=0;projectionchecks=0;maxlp=0.
def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def bits(a,b):return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
for pth in sorted(RAW.glob('v2_technical_*.pt')):
 id=pth.stem[len('v2_technical_'):];data=torch.load(pth,map_location='cpu',weights_only=False);r=records[id]
 for condition,byround in data.items():
  for loop,z in byround.items():
   fullheads+=1;s=r['conditions'][condition]['scores'][str(loop)];x=z.flatten().double().numpy();off=x[[330,389,340,422]];assert off.tolist()==s['candidate_logits'];assert digest(z)==s['full49152_sha256'];norm=np.log(np.exp(x-x.max()).sum())+x.max();lp=off-norm;error=np.max(np.abs(lp-np.array(s['candidate_logprobs'])));maxlp=max(maxlp,float(error));assert error<2e-5;ties=np.flatnonzero(off==off.max()).tolist();assert ties==s['ties'] and ties[0]==s['prediction']
for pth in sorted(RAW.glob('v2_witness_*.pt')):
 pack=torch.load(pth,map_location='cpu',weights_only=False);q=pack['q'];id=q['id'];arm=pack['arm'];r=records[id]
 for kind,events in pack['witnesses'].items():
  assert len(events)==8
  for (u,layer),w in events.items():
   qh=w['Q_H'].double().numpy();k=w['K_full'].double().numpy();v=w['V_full'].double().numpy();a,b=w['record_spans'][0];c,d=w['record_spans'][1];h=w['H_start'];T=k.shape[-2];D=k.shape[-1];H=qh.shape[-2];assert (u+1)==arm and layer in range(8,16);mask=np.arange(T)[None,:]>np.arange(h,T)[:,None];outputs={};sumerrs=[]
   for name,changeK,changeV in [('native',False,False),('K',True,False),('V',False,True),('joint',True,True)]:
    kk=k.copy();vv=v.copy()
    if changeK:kk[:,:,a:b]=k[:,:,c:d];kk[:,:,c:d]=k[:,:,a:b]
    if changeV:vv[:,:,a:b]=v[:,:,c:d];vv[:,:,c:d]=v[:,:,a:b]
    logits=np.matmul(qh,np.swapaxes(kk,-1,-2))*w['scale'];logits=np.where(mask[None,None],-np.inf,logits);logits-=logits.max(-1,keepdims=True);p=np.exp(logits);p/=p.sum(-1,keepdims=True);assert np.isfinite(p).all();sume=np.abs(p.sum(-1)-1);bound=128*np.finfo(np.float64).eps*(np.arange(h,T)+2);assert np.all(sume<=bound[None,None]);sumerrs.append(float(sume.max()));outputs[name]=np.matmul(p,vv)
   s=np.maximum(1,np.abs(v).max(axis=(-2,-1)));visible=np.arange(h,T)+1;tau=128*np.finfo(np.float64).eps*(visible[None,None,:,None]+D+1)*s[:,:,None,None];delta=w['paired_delta_FP64'].numpy();deltas={'K':outputs['K']-outputs['native'],'V':outputs['V']-outputs['native'],'joint':outputs['joint']-outputs['native']};errors={name:float(np.max(np.abs(val-(0 if name=='joint' else delta)))) for name,val in deltas.items()}
   for name,val in deltas.items():
    err=np.abs(val-(0 if name=='joint' else delta));ratio=float(np.max(err/tau));maximum_ratio=max(maximum_ratio,ratio);assert np.all(err<=tau),(id,kind,layer,name,float(err.max()),float(tau.max()))
   before=w['native_SDPA_H'];after=w['edited_SDPA_H'];base=before.double().numpy();nativeerr=base-outputs['native'];rel=np.linalg.norm(nativeerr,axis=-1)/np.maximum(np.linalg.norm(outputs['native'],axis=-1),1e-12);native_rel_max=max(native_rel_max,float(rel.max()));assert np.max(rel)<=.01,(id,kind,layer,float(rel.max()))
   if kind in ['self','joint']:assert bits(before,after)
   elif kind in ['semantic','V']:assert bits((before.double()+w['paired_delta_FP64']).to(before.dtype),after)
   writechecks+=1;actual=(after.double()-before.double()).permute(0,2,1,3).reshape(H,-1).norm(dim=-1).numpy();e=next(e for e in r['conditions'][kind]['audit']['events'] if e['layer']==layer+1);assert np.allclose(actual,e['actual_delta_norms'],atol=1e-11,rtol=1e-11)
   pp=w['actual_projected_H'];bp=w['native_base_projected_H'];projected=(pp.double()-bp.double())[0].norm(dim=-1).numpy();assert np.allclose(projected,e['projected_actual_delta_norms'],atol=1e-10,rtol=1e-11)
   if kind in ['self','joint']:assert bits(pp,bp)
   projectionchecks+=1;tests.append({'unit':id,'arm':arm,'kind':kind,'layer':layer+1,'head_dim':D,'T':T,'H':H,'oracle_max_abs_errors':errors,'maximum_tau':float(tau.max()),'native_reconstruction_max_relative':float(rel.max()),'softmax_sum_maxerr':max(sumerrs),'actual_write_formula_verified':True,'projected_actual_norm_verified':True})
for r in gpu['records']:
 for kind,z in r['conditions'].items():
  assert z['audit']['actual_original_SDPA_invocations']==48
  if kind!='native':assert z['audit']['prefix_identity_sites']==194
  if kind not in ['native','noop']:assert len(z['audit']['events'])==8
out={'status':'PASS','full49152_endpoints':fullheads,'maximum_fullvocab_logprob_error':maxlp,'witness_worlds':4,'witness_events':len(tests),'write_formula_checks':writechecks,'projection_norm_checks':projectionchecks,'maximum_oracle_error_div_frozen_scale_tau':maximum_ratio,'maximum_native_reconstruction_relative_perheadrow':native_rel_max,'full_oracle_details':tests,'actualSDPAcalls_andprefixcoverage_all32worlds':True,'scope':'CPU NumPy fullKV causalattention oracle vsGPU pairedDelta plusexactactualBF16write. Constructedself/joint/KVidentities arenot scientificproof. No newsemanticdataset inference.','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'V2_CPU_ORACLE_AUDIT.json').write_text(json.dumps(out,indent=2));(R/'V2_TECHNICAL_GATE.json').write_text(json.dumps({'status':'PASS','GPU_receipt':'V2_TECHNICAL_GPU.json','CPU_receipt':'V2_CPU_ORACLE_AUDIT.json','V1_still_failed':True,'forwards':224,'independent_agent_reproduction':False,'executor':'Sol GPUimplementationandCPUoracle; Astra designed/reviewed scope'},indent=2));print('V2_CPU_ORACLE_PASS',fullheads,len(tests),maximum_ratio,native_rel_max)
