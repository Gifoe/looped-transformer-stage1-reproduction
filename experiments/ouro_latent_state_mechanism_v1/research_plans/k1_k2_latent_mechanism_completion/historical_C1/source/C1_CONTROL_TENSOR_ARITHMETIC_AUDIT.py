"""Independent all-case branch/control/cast audit, no model or GPU work.

FP64 scale arithmetic is compared to independent extended-precision norms with
an a priori rounding envelope. Actual stored proposals must cast bitwise exactly.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from C1_AUDIT_TENSOR_IO import load_checked
from C1_CAUSAL_ANALYSIS import ENDPOINTS,sha,read,write,lines

def digest(t):return hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def eq(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
def high(t):return t.double().numpy().astype(np.longdouble)
def norm(v):return np.sqrt(np.sum(v*v,dtype=np.longdouble))
def close(actual,expected):
 v=actual.double().numpy().astype(np.longdouble);expected=np.asarray(expected,dtype=np.longdouble)
 # Conservative bound for sums/products/division of <=2048 FP64 entries.
 bound=np.longdouble(32*max(2048,v.size,expected.size)*np.finfo(np.float64).eps)*(1+np.max(np.abs(expected)))
 err=np.max(np.abs(v-expected));assert err<=bound,('FP64 arithmetic',float(err),float(bound))
 return float(err),float(bound)
FIELDS={'x_R':'x_actual_R','a_R':'a_actual_R','O_R':'O_actual_R','Q_R':'Q_actual_R','K_R':'K_actual_R','V_R':'V_actual_R',
        'y_R':'y_branch_R','y_before_R':'y_actual_before_branch_R','b_R':'b_actual_R',
        'after_mlp_R':'after_mlp_R','final_norm_R':'final_norm_R','MLP_input_R':'MLP_input_R','MLP_skip_R':'MLP_skip_shared_alias_R'}
NONR=['non_R_x_sha256','non_R_Q_sha256','non_R_K_sha256','non_R_V_sha256','non_R_O_sha256','non_R_after_mlp_sha256','non_R_final_norm_sha256']

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--split',choices=['train','validation'],required=True);p.add_argument('--technical-only',action='store_true');a=p.parse_args()
 assert not torch.cuda.is_initialized();torch.set_num_threads(1)
 assert np.finfo(np.longdouble).nmant>52,'Extended-precision independent reduction is required'
 start=time.time();r=a.root;inputs=list(lines(r/'inputs'/(a.split+'.jsonl')));byid={q['id']:q for q in inputs}
 assert len(inputs)==(1094 if a.split=='train' else 299)
 if a.technical_only:
  assert a.split=='train';technical=set(read(r/'inputs/BATCH_IDS.json')['technical2']);inputs=[q for q in inputs if q['id'] in technical];assert len(inputs)==2
 donors=read(r/'inputs'/(a.split+'_donor_map.json'));out=[];undefined=[];max_error=0.;ncontrols=0;nsites=0
 def ref(ident):
  path=r/'references'/a.split/(hashlib.sha256(ident.encode()).hexdigest()+'.json');m=read(path);v,storage=load_checked(path.with_suffix('.safetensors'),m)
  return m,v
 for q in inputs:
  ident=q['id'];meta,t=ref(ident);dm,dt=ref(donors[ident]);case={'id':ident,'donor_id':donors[ident],'control_checks':{},'site_checks':{}}
  for i in range(9,16):
   assert eq(t[f'd_{i:02d}'],t[f'N_a_{i:02d}'].double()-t[f'P_a_{i:02d}'].double())
   before=t[f'U_site_{i:02d}_before'];exact=t[f'U_site_{i:02d}_exact_before_cast'];after=t[f'U_site_{i:02d}_after']
   assert eq(exact,before.double()-t[f'd_{i:02d}']) and eq(after,exact.to(torch.bfloat16))
   assert eq(t[f'U_site_{i:02d}_cast_residual'],after.double()-exact)
   assert float(t[f'U_site_{i:02d}_normalization_scale_fp64'])==1
  for background in ['N','P','U']:
   for field,receipt_field in FIELDS.items():assert digest(t[background+'_'+field])==meta[background+'_receipt'][receipt_field]['sha256']
   assert eq(t[background+'_y_R'],t[background+'_x_R']+t[background+'_a_R'])
   assert eq(t[background+'_after_mlp_R'],t[background+'_y_R']+t[background+'_b_R'])
  for name in ENDPOINTS:
   if name in ['N','P','U']:continue
   path=r/'treatments'/a.split/hashlib.sha256(ident.encode()).hexdigest()/(name+'.json');c=read(path)
   if c['status']=='UNAVAILABLE':
    undefined.append({'id':ident,'condition':name,'reason':c['reason']});continue
   v,storage=load_checked(path.with_suffix('.safetensors'),c);br=c['actual_L18_receipt']
   background=c['actual_background_reference'];nonr_background='P' if name=='P_plus_L' else 'N'
   for key in NONR:assert br[key]==meta[nonr_background+'_receipt'][key],(ident,name,key)
   if background:
    for field in ['x_R','a_R','O_R','Q_R','K_R','V_R']:assert digest(t[background+'_'+field])==br[FIELDS[field]]['sha256']
    x=v['x_branch_R'];att=v['a_branch_R']
   else:
    x=v['x_R'];att=v['a_R']
   assert eq(v['y_R'],x+att) and eq(v['MLP_input_R'],v['y_R']) and eq(v['MLP_skip_R'],v['y_R'])
   assert eq(v['after_mlp_R'],v['y_R']+v['b_R'])
   assert digest(v['y_R'])==br['y_branch_R']['sha256']
   assert digest(v['after_mlp_R'])==br['after_mlp_R']['sha256']
   assert digest(v['final_norm_R'])==br['final_norm_R']['sha256']
   if name in ['UX','UA','UXA']:
    assert eq(x,t['N_x_R'] if name in ['UX','UXA'] else t['U_x_R'])
    assert eq(att,t['N_a_R'] if name in ['UA','UXA'] else t['U_a_R'])
   ctrl=c.get('control_metadata')
   if ctrl:
    parts=name.split('_');direction,branch,kind=parts[:3];key='x_R' if branch=='X' else 'a_R'
    b=t[direction+'_'+key];other='N' if direction=='U' else 'U';target=t[other+'_'+key]
    assert eq(v['control_before'],b)
    bv,tv=high(b),high(target);dose=norm(tv-bv);bn=norm(bv);tn=norm(tv)
    if kind=='norm':
     assert bn!=0 or tn==0
     scale=tn/bn if bn!=0 else np.longdouble(1);expected=bv*scale
    else:
     dv=high(dt['N_'+key])-high(dt['U_'+key]);dn=norm(dv)
     assert dose==0 or dn!=0
     scale=dose/dn if dose!=0 else np.longdouble(0)
     expected=bv+(1 if direction=='U' else -1)*scale*dv
    err,bound=close(v['control_exact_before_cast'],expected);max_error=max(max_error,err)
    close(v['control_scale_fp64'],scale)
    assert eq(v['control_after'],v['control_exact_before_cast'].to(torch.bfloat16))
    assert eq(v['control_cast_residual'],v['control_after'].double()-v['control_exact_before_cast'])
    assert eq(v['control_after'],x if branch=='X' else att)
    assert eq(att if branch=='X' else x,t[direction+('_a_R' if branch=='X' else '_x_R')])
    for field,value in [('requested_delta_norm',dose),('base_norm',bn),('target_norm',tn),('after_norm',norm(high(v['control_after']))),('actual_after_cast_delta_norm',norm(high(v['control_after'])-bv))]:
     close(torch.tensor(ctrl[field],dtype=torch.float64),value)
    assert ctrl['zero_requested_dose']==bool(dose==0)
    case['control_checks'][name]={'status':'PASS','max_extended_precision_error':err,'a_priori_FP64_envelope':bound};ncontrols+=1
   sites=c['upstream_site_receipts']
   if c['upstream_site_shared_reference']=='U_site_':
    for i in range(9,16):
     for field in ['before','after','exact_before_cast','cast_residual','normalization_scale_fp64']:
      assert digest(t[f'U_site_{i:02d}_{field}'])==sites[str(i)][field]['sha256']
     assert digest(t[f'd_{i:02d}'])==sites[str(i)]['reference_delta']['sha256']
     assert digest(t[f'U_site_{i:02d}_after'].double()-t[f'U_site_{i:02d}_before'].double())==sites[str(i)]['effective_delta']['sha256']
   elif name in ['P_plus_L','U_upstream_norm','U_upstream_unrelated']:
    for i in range(9,16):
     before=v[f'site_{i:02d}_before'];actual=v[f'site_{i:02d}_exact_before_cast'];after=v[f'site_{i:02d}_after']
     delta=t[f'd_{i:02d}'];bv=high(before)
     if name=='P_plus_L':expected=bv+high(delta);scale=np.longdouble(1)
     elif name=='U_upstream_norm':
      proposal=bv-high(delta);pn=norm(proposal);bn=norm(bv);assert pn!=0 or bn==0
      scale=bn/pn if pn!=0 else np.longdouble(1);expected=proposal*scale
     else:
      targetdose=norm(high(delta));dv=high(dt[f'd_{i:02d}']);dn=norm(dv);assert targetdose==0 or dn!=0
      direction=dv*(targetdose/dn) if targetdose!=0 else np.zeros_like(dv)
      requested=v[f'control_requested_delta_{i:02d}'];close(requested,direction)
      expected=bv-high(requested);scale=np.longdouble(1)
     err,bound=close(actual,expected);close(v[f'site_{i:02d}_normalization_scale_fp64'],scale);max_error=max(max_error,err)
     assert eq(after,actual.to(torch.bfloat16)) and eq(v[f'site_{i:02d}_cast_residual'],after.double()-actual)
     for field in ['before','after','exact_before_cast','cast_residual','normalization_scale_fp64']:assert digest(v[f'site_{i:02d}_{field}'])==sites[str(i)][field]['sha256']
     requested=v[f'control_requested_delta_{i:02d}'] if name=='U_upstream_unrelated' else delta
     assert digest(requested)==sites[str(i)]['reference_delta']['sha256']
     assert digest(after.double()-before.double())==sites[str(i)]['effective_delta']['sha256']
     nsites+=1
   elif background=='N':
    for i in range(9,16):
     for field in ['before','after']:assert digest(t[f'N_a_{i:02d}'])==sites[str(i)][field]['sha256']
   case['site_checks'][name]='PASS'
  out.append(case)
 assert not torch.cuda.is_initialized()
 write(r/'analysis'/(a.split.upper()+('_TECHNICAL' if a.technical_only else '')+'_ALL_CONTROL_TENSOR_AUDIT.json'),{'status':'PASS','split':a.split,'technical_only':a.technical_only,'n_questions':len(inputs),
   'control_vectors_checked':ncontrols,'independent_upstream_control_sites':nsites,'max_extended_precision_error':max_error,
   'undefined_controls':undefined,'cases':out,'source_sha256':sha(__file__),'tensor_reader_sha256':sha(Path(__file__).with_name('C1_AUDIT_TENSOR_IO.py')),
   'CPU_seconds':time.time()-start,'longdouble_mantissa_bits':int(np.finfo(np.longdouble).nmant),
   'scope':'All stored branch sums/MLP skip arithmetic, shared actual background vectors, non-R bridges, control direction/dose/scale/cast and upstream injections; FP64 against independent extended-precision norms within a priori rounding envelope. Not a new model forward or independently recomputed full SDPA for every question.'})
 print('C1_ALL_CONTROL_TENSOR_AUDIT_PASS',a.split,len(inputs),ncontrols,nsites,flush=True)
if __name__=='__main__':main()
