"""B2 freeze candidate only. No loader/cohorts/CLI. Independent replay required."""
import torch
from b2_forward import forward,_same,_expect_ref,_nonr_equal,_tensor,th,corner
from b2_controls import control
from branch_interfaces_v2 import UndefinedControl
from ab1_metrics import score
SITES=range(9,16)
def receipt(br):
 return {k:v for k,v in br.items() if not torch.is_tensor(v) and k!='technical_only_full_qkv'}
def pack(name,z,bc,sites,br,trace,targets,out):
 out[name+'_logits']=_tensor(z)
 for k,v in br.items():
  if torch.is_tensor(v):out[name+'_branch_'+k]=_tensor(v)
 for i,r in sites.items():
  for k,v in r.items():
   if torch.is_tensor(v):out[f'{name}_site_{i:02d}_{k}']=_tensor(v)
 return {'status':'OK','full_logits_sha256':th(z),'scores':score(z,targets),'branch':receipt(br),'nonR_trace':{str(k):v for k,v in trace.items()},'L1_kwargs':bc['L1_kwargs'],'L9_kwargs':bc['L9_kwargs']}
def checkmeta(q,m):
 assert m['input_ids_sha256']==q['input_ids_sha256'] and m['targets']==q['targets'] and m['R']==q['R'] and m['prefix_len']==q['prefix_len']
def delta(ab):return {i:ab[f'H11_L{i+1:02d}_a_R'].double()-ab[f'H01_L{i+1:02d}_a_R'].double() for i in SITES}
@torch.inference_mode()
def reference(model,q,abmeta,abtensors):
 checkmeta(q,abmeta);device=next(model.parameters()).device
 E=abtensors['source_E'].to(device);B=abtensors['source_B'].to(device);U=corner(E,B,q['prefix_len'],q['R'],'E','B')
 # Reanchors are technical provenance forwards, not extra scientific endpoints.
 n,nb,ns,nl,nt=forward(model,q);u,ub,us,ul,ut=forward(model,q,boundary=U)
 _expect_ref(n,nb,ns,nl,abtensors,'H11');_expect_ref(u,ub,us,ul,abtensors,'H01')
 for name,br in [('H11',nl),('H01',ul)]:
  _nonr_equal(br,abmeta['conditions'][name]['L18_non_R_hashes'])
  assert _same(br['final_norm_R'],abtensors[name+'_finalnorm_R'])
 d=delta(abtensors);z,bc,sites,br,tr=forward(model,q,delta=d)
 assert _same(bc['E'],E) and _same(bc['B'],B) and _same(bc['Q9_actual'],abtensors['native_Q9_full'])
 _nonr_equal(br,nl);assert tr==nt
 out={f'd_{i:02d}':_tensor(v) for i,v in d.items()}
 for i in SITES:
  out[f'N_mlp_{i:02d}']=_tensor(ns[i]['mlp_before']);out[f'U_mlp_{i:02d}']=_tensor(us[i]['mlp_before'])
  out[f'd_mlp_{i:02d}']=out[f'N_mlp_{i:02d}'].double()-out[f'U_mlp_{i:02d}'].double()
 item=pack('T',z,bc,sites,br,tr,q['targets'],out)
 return {'status':'B2_REFERENCE_CANDIDATE','id':q['id'],'split':q['split'],'input_ids_sha256':q['input_ids_sha256'],'source_identity':abmeta['source_identity'],'N_receipt':receipt(nl),'N_trace':{str(k):v for k,v in nt.items()},'U_receipt':receipt(ul),'U_trace':{str(k):v for k,v in ut.items()},'conditions':{'T':item},'scope':'boundary H11-minus-H01 fixed upstream counterfactual; not natural K2-minus-K1'},out
@torch.inference_mode()
def treatments(model,q,abmeta,abtensors,tmeta,ttensors,donor_ab,donor_t):
 checkmeta(q,abmeta);assert tmeta['id']==q['id'] and tmeta['input_ids_sha256']==q['input_ids_sha256'] and tmeta['source_identity']==abmeta['source_identity']
 # Caller must freeze sorted cyclic within-original-split donor mapping before any T outcomes.
 assert donor_ab['id']!=q['id'] and donor_ab['split']==q['split'] and donor_t['id']==donor_ab['id'] and donor_t['split']==q['split']
 dab=donor_ab['tensors'];dt=donor_t['tensors'];assert donor_ab['source_identity']==donor_t['source_identity']==abmeta['source_identity']
 d=delta(abtensors);device=next(model.parameters()).device
 U=corner(abtensors['source_E'].to(device),abtensors['source_B'].to(device),q['prefix_len'],q['R'],'E','B')
 nx=abtensors['H11_L18_x_R'];na=abtensors['H11_L18_a_R']
 cfg={'TX':{'delta':d,'x_branch':nx},'TA':{'delta':d,'a_branch':na},'TXA':{'delta':d,'x_branch':nx,'a_branch':na},'U_plus':{'boundary':U,'delta':d,'sign':1},'upstream_norm':{'delta':d,'norm_preserve':True}}
 unavailable={};out={};stats={}
 try:
  dd=delta(dab);ud={}
  for i in SITES:
   requested=d[i].norm();dn=dd[i].norm()
   if float(requested)==0:ud[i]=torch.zeros_like(d[i])
   elif float(dn)==0:raise UndefinedControl('zero upstream donor direction')
   else:ud[i]=requested*dd[i]/dn
  cfg['upstream_unrelated']={'delta':ud}
  cfg['U_plus_unrelated']={'boundary':U,'delta':ud,'sign':1}
  for i,v in ud.items():out[f'upstream_unrelated_control_requested_delta_{i:02d}']=v
 except UndefinedControl as e:
  unavailable['upstream_unrelated']=str(e);unavailable['U_plus_unrelated']=str(e)
 for branch,n in [('X','x'),('A','a')]:
  key='x_branch' if branch=='X' else 'a_branch';N=abtensors['H11_L18_'+n+'_R'];T=ttensors['T_branch_'+n+'_actual_R']
  donor_delta=dab['H11_L18_'+n+'_R'].double()-dt['T_branch_'+n+'_actual_R'].double()
  for direction,base,target,sign in [('T',T,N,1),('N',N,T,-1)]:
   for kind in ('norm','unrelated'):
    name=direction+'_'+branch+'_'+kind
    try:
     value,ct,st=control(base,target,donor_delta,kind,sign);cfg[name]={**({'delta':d} if direction=='T' else {}),key:value};stats[name]=st
     out.update({name+'_'+k:_tensor(v) for k,v in ct.items()})
    except UndefinedControl as e:unavailable[name]=str(e)
 cfg['U_plus_norm_preserve']={'boundary':U,'delta':d,'sign':1,'norm_preserve':True}
 md={i:ttensors[f'd_mlp_{i:02d}'] for i in SITES}
 cfg.update({'M':{'mlp_delta':md},'U_M_plus':{'boundary':U,'mlp_delta':md,'mlp_sign':1},'M_norm_preserve':{'mlp_delta':md,'mlp_norm_preserve':True},'U_M_plus_norm_preserve':{'boundary':U,'mlp_delta':md,'mlp_sign':1,'mlp_norm_preserve':True}})
 try:
  mud={}
  for i in SITES:
   v=dt[f'd_mlp_{i:02d}'];requested=md[i].norm();dn=v.norm()
   if float(requested)==0:mud[i]=torch.zeros_like(md[i])
   elif float(dn)==0:raise UndefinedControl('zero MLP donor direction')
   else:mud[i]=requested*v/dn
  cfg['M_unrelated']={'mlp_delta':mud};cfg['U_M_plus_unrelated']={'boundary':U,'mlp_delta':mud,'mlp_sign':1}
  for i,v in mud.items():out[f'M_unrelated_requested_delta_{i:02d}']=_tensor(v)
 except UndefinedControl as e:
  unavailable['M_unrelated']=str(e);unavailable['U_M_plus_unrelated']=str(e)
 conditions={k:{'status':'UNAVAILABLE','reason':v,'forward_attempted':False} for k,v in unavailable.items()}
 for name,c in cfg.items():
  try:z,bc,sites,br,tr=forward(model,q,**c)
  except UndefinedControl as e:
   if not hasattr(e,'witness'):raise
   w=e.witness;wm={k:v for k,v in w.items() if not torch.is_tensor(v)};wm['tensors']={}
   for k,v in w.items():
    if torch.is_tensor(v):
     key=name+'_UNDEFINED_'+k;out[key]=_tensor(v);wm['tensors'][k]={'tensor_key':key,'shape':list(v.shape),'dtype':str(v.dtype),'sha256':th(v)}
   conditions[name]={'status':'UNAVAILABLE','reason':str(e),'forward_attempted':True,'UNDEFINED_WITNESS':wm};continue
  assert _same(bc['E'],abtensors['source_E']) and _same(bc['B'],abtensors['source_B'])
  _nonr_equal(br,tmeta['U_receipt'] if name.startswith('U_plus') or name.startswith('U_M_plus') else tmeta['N_receipt']);assert {str(k):v for k,v in tr.items()}=={str(k):v for k,v in (tmeta['U_trace'] if name.startswith('U_plus') or name.startswith('U_M_plus') else tmeta['N_trace']).items()}
  if name in ('TX','TA','TXA') or name.startswith('T_'):
   for k in ('x','a','Q','K','V','O'):assert _same(br[k+'_actual_R'],ttensors['T_branch_'+k+'_actual_R'])
   for i in SITES:
    for k in ('before','after','exact_before_cast','cast_residual'):assert _same(sites[i][k],ttensors[f'T_site_{i:02d}_{k}'])
  if name=='TXA':assert _same(z,abtensors['H11_logits']),'dual branch structural identity failed'
  conditions[name]=pack(name,z,bc,sites,br,tr,q['targets'],out);conditions[name]['forward_attempted']=True
 assert len(conditions)==22
 return {'status':'B2_TREATMENTS_CANDIDATE','id':q['id'],'split':q['split'],'input_ids_sha256':q['input_ids_sha256'],'donor_id':donor_ab['id'],'source_identity':abmeta['source_identity'],'conditions':conditions,'work_counts':{'body_forward_attempts':sum(c['forward_attempted'] for c in conditions.values()),'head_completions':sum(c['status']=='OK' for c in conditions.values()),'reference_complete_forwards':3},'control_stats':stats,'scope':'FINAL25: N/U/T plus22 treatments; attention upstream6, L18branch12, MLPsecondary6 separately Bonferroni; undefined retained'},out
