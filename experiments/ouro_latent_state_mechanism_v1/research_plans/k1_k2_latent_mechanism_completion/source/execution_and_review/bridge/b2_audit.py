"""Independent CPU saved-tensor B2 math audit. Not model replay."""
import argparse,pathlib,json,hashlib,sys,torch,numpy as np

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def same(a,b):assert a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
def checked(root,path,digest):
 p=(root/path).resolve();assert p.is_relative_to(root.resolve()) and sha(p)==digest;return p,json.loads(p.read_text())
def score(z,s,targets):
 z=z.reshape(-1);assert z.numel()==49152 and z.dtype==torch.bfloat16 and torch.isfinite(z).all();v=z[targets];ties=torch.nonzero(v==v.max()).reshape(-1).tolist();lse=float(z.double().logsumexp(0));bound=32*torch.finfo(torch.float32).eps*(abs(lse)+float(z.abs().max())+1);assert [float(x) for x in v]==s['candidate_logits'] and ties==s['ties'] and ties[0]==s['prediction'] and abs(lse-s['logsumexp'])<=bound and all(abs(float(x)-lse-y)<=bound for x,y in zip(v,s['scores']))
def high_norm(x):
 v=x.double().numpy().astype(np.longdouble);return np.sqrt(np.sum(v*v,dtype=np.longdouble))
def site_math(t,name,i,delta=None,sign=-1,norm=False,mlp=False):
 prefix=f'{name}_site_{i:02d}_'+('mlp_' if mlp else '');before=t[prefix+'before'];raw=before.double();scale=t[prefix+'normalization_scale_fp64'];error=0.;envelope=0.
 if delta is not None:raw=raw+sign*delta
 if norm and delta is not None:
  n0=high_norm(before);n1=high_norm(raw);assert n1!=0 or n0==0;expected=n0/n1 if n1!=0 else np.longdouble(1)
  actual=np.longdouble(float(scale));bound=np.longdouble(32*max(2048,before.numel())*np.finfo(np.float64).eps)*(1+abs(expected));err=abs(actual-expected);assert err<=bound,('norm scale reduction envelope',float(err),float(bound));error=float(err);envelope=float(bound);exact=raw*scale
 else:
  same(scale,torch.ones((),dtype=torch.float64));exact=raw
 after=exact.to(before.dtype)
 for key,value in dict(after=after,exact_before_cast=exact,cast_residual=after.double()-exact,effective_delta=after.double()-before.double()).items():same(value,t[prefix+key])
 return dict(scale_error=error,scale_envelope=envelope)
def branch_control(base,target,direction,kind,sign):
 b=base.double();scale=torch.ones((),dtype=torch.float64)
 if kind=='norm':
  assert b.norm()!=0 or target.double().norm()==0
  if b.norm()!=0:scale=target.double().norm()/b.norm()
  exact=b*scale
 else:
  dose=(target.double()-b).norm();dn=direction.norm();assert dose==0 or dn!=0;scale=scale*0 if dose==0 else dose/dn;exact=b+sign*scale*direction
 after=exact.to(base.dtype);return dict(control_before=base,control_exact_before_cast=exact,control_scale_fp64=scale,control_after=after,control_cast_residual=after.double()-exact)
def audit_case(m,t,ab,refm,ref,dab=None,donor=None,reference_only=False):
 targets=m['targets']
 delta={i:ab[f'H11_L{i+1:02d}_a_R'].double()-ab[f'H01_L{i+1:02d}_a_R'].double() for i in range(9,16)};md={i:ref[f'N_mlp_{i:02d}'].double()-ref[f'U_mlp_{i:02d}'].double() for i in range(9,16)}
 for i in range(9,16):same(delta[i],ref[f'd_{i:02d}']);same(md[i],ref[f'd_mlp_{i:02d}'])
 counts={};scale_checks=[];assert set(m['conditions'])==({'T'} if reference_only else set(['TX','TA','TXA','U_plus','upstream_norm','upstream_unrelated','U_plus_norm_preserve','U_plus_unrelated','M','U_M_plus','M_norm_preserve','M_unrelated','U_M_plus_norm_preserve','U_M_plus_unrelated']+[f'{n}_{b}_{k}' for n in ('T','N') for b in ('X','A') for k in ('norm','unrelated')]))
 for name,info in m['conditions'].items():
  if info['status']!='OK':
   assert info['status']=='UNAVAILABLE' and info['reason'];proved=False
   if name.startswith(('T_','N_')):
    label,branch,kind=name.split('_');key=branch.lower();N=ab[f'H11_L18_{key}_R'];T=ref[f'T_branch_{key}_actual_R'];base=T if label=='T' else N;target=N if label=='T' else T
    if kind=='norm':proved=bool(base.double().norm()==0 and target.double().norm()!=0)
    else:
     direction=dab[f'H11_L18_{key}_R'].double()-donor[f'T_branch_{key}_actual_R'].double();proved=bool((target.double()-base.double()).norm()!=0 and direction.norm()==0)
   elif name in ('upstream_unrelated','U_plus_unrelated'):
    proved=any(bool(delta[i].norm()!=0 and (dab[f'H11_L{i+1:02d}_a_R'].double()-dab[f'H01_L{i+1:02d}_a_R'].double()).norm()==0) for i in range(9,16))
   elif name in ('M_unrelated','U_M_plus_unrelated'):
    proved=any(bool(md[i].norm()!=0 and donor[f'd_mlp_{i:02d}'].norm()==0) for i in range(9,16))
   elif name in ('upstream_norm','U_plus_norm_preserve','M_norm_preserve','U_M_plus_norm_preserve'):
    witness=info.get('UNDEFINED_WITNESS');assert witness is not None;site=int(witness['site']);assert site in range(9,16) and witness['norm_preserve'] is True;assert witness['component']==('mlp' if name.startswith(('M','U_M_plus')) else 'attention');assert info['forward_attempted'] is True
    for wk,ws in witness['tensors'].items():
     value=t[ws['tensor_key']];assert ws['tensor_key']==name+'_UNDEFINED_'+wk and list(value.shape)==ws['shape'] and str(value.dtype)==ws['dtype'] and hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()==ws['sha256']
    sign=1 if name.startswith(('U_plus','U_M_plus')) else -1;assert witness['sign']==sign
    before=t[name+'_UNDEFINED_before'];proposal=t[name+'_UNDEFINED_proposal_before_norm'];reference=t[name+'_UNDEFINED_reference_delta'];same(reference,md[site] if name.startswith(('M','U_M_plus')) else delta[site]);same(before.double()+sign*reference,proposal);proved=bool(proposal.norm()==0 and before.double().norm()!=0)
   assert proved,('UNAVAILABLE not independently proven',name);counts[name]='UNAVAILABLE_MATH_PROVEN';continue
  z=t[name+'_logits'];score(z,info['scores'],targets);assert hashlib.sha256(z.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()==info['full_logits_sha256']
  mlp=name.startswith(('M','U_M_plus'));att=name in ('T','TX','TA','TXA','upstream_norm','upstream_unrelated') or name.startswith(('T_','U_plus'));sign=1 if name.startswith(('U_plus','U_M_plus')) else -1
  for i in range(9,16):
   da=delta[i] if att else None;db=md[i] if mlp else None
   if name in ('upstream_unrelated','U_plus_unrelated'):
    dd=dab[f'H11_L{i+1:02d}_a_R'].double()-dab[f'H01_L{i+1:02d}_a_R'].double();assert delta[i].norm()==0 or dd.norm()!=0;da=torch.zeros_like(dd) if delta[i].norm()==0 else delta[i].norm()*dd/dd.norm();same(da,t[f'upstream_unrelated_control_requested_delta_{i:02d}'])
   if name in ('M_unrelated','U_M_plus_unrelated'):
    dd=donor[f'd_mlp_{i:02d}'];assert md[i].norm()==0 or dd.norm()!=0;db=torch.zeros_like(dd) if md[i].norm()==0 else md[i].norm()*dd/dd.norm();same(db,t[f'M_unrelated_requested_delta_{i:02d}'])
   scale_checks.append(dict(condition=name,site=i,component='attention',**site_math(t,name,i,da,sign,name in ('upstream_norm','U_plus_norm_preserve'))));scale_checks.append(dict(condition=name,site=i,component='mlp',**site_math(t,name,i,db,sign,'norm_preserve' in name,True)))
  bp=name+'_branch_';same(t[bp+'y_branch_R'],t[bp+'x_branch_used_R']+t[bp+'a_branch_used_R']);same(t[bp+'MLP_skip_shared_alias_R'],t[bp+'y_branch_R']);same(t[bp+'after_mlp_R'],t[bp+'MLP_skip_shared_alias_R']+t[bp+'b_actual_R'])
  if name.startswith(('T_','N_')):
   label,branch,kind=name.split('_');key=branch.lower();N=ab[f'H11_L18_{key}_R'];T=ref[f'T_branch_{key}_actual_R'];direction=dab[f'H11_L18_{key}_R'].double()-donor[f'T_branch_{key}_actual_R'].double();receipt=branch_control(T if label=='T' else N,N if label=='T' else T,direction,kind,1 if label=='T' else -1)
   for key,value in receipt.items():same(value,t[name+'_'+key])
  if name=='TXA':same(z,ab['native_K2_logits'])
  assert {str(k):v for k,v in info['nonR_trace'].items()}=={str(k):v for k,v in refm['U_trace' if name.startswith(('U_plus','U_M_plus')) else 'N_trace'].items()};
  if name in ('TX','TXA'):same(t[bp+'x_branch_used_R'],ab['H11_L18_x_R'])
  if name in ('TA','TXA'):same(t[bp+'a_branch_used_R'],ab['H11_L18_a_R'])
  if name in ('TX','TA','TXA') or name.startswith('T_'):
   for key in ('x','a'):same(t[bp+key+'_actual_R'],ref['T_branch_'+key+'_actual_R'])
  if name.startswith(('T_','N_')):same(t[bp+('x' if name.split('_')[1]=='X' else 'a')+'_branch_used_R'],t[name+'_control_after'])
  counts[name]='PASS'
 return dict(id=m['id'],conditions=counts,scale_checks=scale_checks,full_defined_coverage=all(v=='PASS' for v in counts.values()),work_counts=m.get('work_counts',{}))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--stage',required=True);ap.add_argument('--kind',choices=['references','treatments'],default='treatments');a=ap.parse_args();assert np.finfo(np.longdouble).nmant>52,'Extended-precision independent reduction required';root=pathlib.Path(a.root).resolve();sys.path[:0]=[str(root/'code'),str(root/'storage')];from ab1_prefix_codec import load
 d=root/'b2_data'/a.kind/a.stage;pp=d/'PENDING.json';p=json.loads(pp.read_text());assert p['status']=='COMPLETE' and p['n']==len(p['cases']);records=[];freeze=root/'B2_FULL_SOURCE_FREEZE.json';fz=json.loads(freeze.read_text());assert sha(freeze)==p['source_freeze_sha256'];assert all(sha(root/k)==v for k,v in fz['sources'].items())
 for c in p['cases']:
  mp=d/c['metadata_file'];assert sha(mp)==c['metadata_sha256'];m=json.loads(mp.read_text());t=load(m,d);abp,abm=checked(root,m['AB_metadata_path'],m['AB_metadata_sha256']);ab=load(abm,abp.parent);assert m['source_identity']==abm['source_identity'] and all(m[k]==abm[k] for k in ('id','split','R','prefix_len','targets','input_ids_sha256'));targets=m['targets'];score(ab['native_K1_logits'],abm['identity']['K1_scores'],targets);score(ab['native_K2_logits'],abm['identity']['N_scores'],targets);score(ab['H01_logits'],abm['conditions']['H01']['scores'],targets)
  if a.kind=='references':ref=t;refm=m;donor=None;dab=None
  else:
   rp,refm=checked(root,m['reference_metadata_path'],m['reference_metadata_sha256']);ref=load(refm,rp.parent);dp,dm=checked(root,m['donor_AB_metadata_path'],m['donor_AB_metadata_sha256']);dab=load(dm,dp.parent);drp,drm=checked(root,m['donor_reference_metadata_path'],m['donor_reference_metadata_sha256']);donor=load(drm,drp.parent);assert dm['id']==drm['id']==m['donor_id'] and dm['id']!=m['id'] and dm['split']==m['split'] and dm['source_identity']==m['source_identity']
  records.append(audit_case(m,t,ab,refm,ref,dab,donor,a.kind=='references'))
 assert sha(freeze)==p['source_freeze_sha256'] and all(sha(root/k)==v for k,v in fz['sources'].items());out=dict(status='PASS' if all(all(v in ('PASS','UNAVAILABLE_MATH_PROVEN') for v in r['conditions'].values()) for r in records) else 'INCOMPLETE_UNAVAILABLE_PROOF',longdouble_nmant=int(np.finfo(np.longdouble).nmant),pending_sha256=sha(pp),script_sha256=sha(__file__),records=records,scope='CPU saved-tensor arithmetic audit only; not independent model replay. UNAVAILABLE zero predicates independently proven; retained undefined is not full-defined scientific coverage.');op=d/'INDEPENDENT_COVERAGE_AUDIT.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print(out['status'])
if __name__=='__main__':main()
