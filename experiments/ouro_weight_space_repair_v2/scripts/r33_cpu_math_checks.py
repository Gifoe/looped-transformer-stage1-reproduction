"""Independent NumPy/Torch common-gain tests; no real head or outcome input."""
import numpy as np,torch,json,hashlib,inspect
from pathlib import Path
from r33_math import numpy_readouts,torch_readouts,np_signed_min,np_center,np_checked_norm,torch_center,torch_checked_norm,NumericDomainError,np_logp
from r33_common import pack_diag
ROOT=Path(__file__).resolve().parents[1]
def rejected(f):
 try:f()
 except NumericDomainError:return
 raise AssertionError('Invalid numerical domain accepted')
def torch_args(a):return [torch.from_numpy(x) for x in a]
def main():
 rng=np.random.default_rng(2021);maxout=0.;maxdiag=0.;checks=[];active=0
 for _ in range(30):
  q1=rng.normal(size=37).astype('float32');q4=rng.normal(size=37).astype('float32');n1=rng.normal(size=(3,37)).astype('float32');n4=rng.normal(size=(3,37)).astype('float32');perm=rng.permutation(37);args=[q1,q4,n1,n4,perm];a,d=numpy_readouts(*args);active+=int(d['main_norm2']>0);b,e=torch_readouts(*torch_args(args));e=pack_diag(e)
  for k in a:maxout=max(maxout,float(np.max(np.abs(a[k]-b[k].numpy()))))
  for k,v in d.items():
   if isinstance(v,str):assert v==e[k]
   elif isinstance(v,(bool,np.bool_)):assert bool(v)==bool(e[k])
   else:maxdiag=max(maxdiag,float(abs(v-e[k])))
  for tag in ['G','null','consensus']:
   if not d[tag+'_unmatched']:assert d[tag+'_norm_error']<=1e-10
  if not d['scale_unmatched']:assert d['scale_norm_error']<=1e-10
  fixed=np.float32(q4+np.float32(.5)*(q4-q1));zero,zd=numpy_readouts(q4,q4,n1,n4,perm);zt,ztg=torch_readouts(*torch_args([q4,q4,n1,n4,perm]));assert zero['primary'].tobytes()==q4.tobytes() and zt['primary'].numpy().tobytes()==q4.tobytes() and zd['h_exact_zero']
  for k in ['G_matched','vocabulary_null','reference_consensus_matched','matched_scale']:assert zero[k].tobytes()==q4.tobytes()
  neutral,nd=numpy_readouts(q1,q4,n4,n4,perm);nt,_=torch_readouts(*torch_args([q1,q4,n4,n4,perm]));assert neutral['primary'].tobytes()==fixed.tobytes() and nt['primary'].numpy().tobytes()==fixed.tobytes()
  for k in ['G_matched','vocabulary_null','reference_consensus_matched','matched_scale']:assert neutral[k].tobytes()==fixed.tobytes()
  order=np.array([2,0,1]);other,od=numpy_readouts(q1,q4,n1[order],n4[order],perm);assert np.max(np.abs(a['primary']-other['primary']))<=1e-6
  for j,oldj in enumerate(order):assert np.max(np.abs(other['LOO_N'+str(j)]-a['LOO_N'+str(oldj)]))<=1e-6
 assert maxout<=1e-6 and maxdiag<=1e-10;checks+=['independent_numpy_torch','zero_main_and_controls_raw_exact_Fixed','three_LOO_and_reference_permutation','all_matched_norm_errors']
 for _ in range(50):
  G=rng.normal(size=37);E=rng.normal(size=(3,37));v=np.concatenate([G[None],E]);c=np_signed_min(v);res=G-c;assert np.all(np.where(G>=0,(res>=0)&(res<=G),(res>=G)&(res<=0)));assert np.array_equal(c,np_signed_min(np.concatenate([v,E[1:2]])))
  for j in range(3):assert np.all(np.abs(np_signed_min(np.concatenate([G[None],np.delete(E,j,axis=0)])))>=np.abs(c))
 checks+=['local_gain_interval_theorem','duplicate_existing_reference_idempotence','LOO_removal_relaxes_clamp']
 native=np.array([.2,.3,.5]);fixed=np.array([.4,.29,.31]);rn=np.array([.1,.8,.1]);rf=np.array([.2,.65,.15]);G=np.log(fixed)-np.log(native);E=np.tile(np.log(rf)-np.log(rn),(3,1));c=np_signed_min(np.concatenate([G[None],E]));h=np_center(native,c);z=np.log(fixed)-h;new=np.exp(z-z.max());new/=new.sum();assert G[1]<0 and new[1]>.3 and abs(new[1]-.37037037037037035)<1e-12;checks.append('renormalization_reverses_probability_gain_sign_counterexample')
 q4=np.array([0.,1.,2.,3.],dtype='float32');q1=np.array([-2.,1.,0.,5.],dtype='float32');n4=np.tile(q4,(3,1));n1=np.tile(q1,(3,1));perm=np.arange(4);base,bd=numpy_readouts(q1,q4,n1,n4,perm);assert bd['main_norm2']>0;shift,_=numpy_readouts(q1+8,q4+16,n1+4,n4+12,perm);assert np.max(np.abs(np_logp(base['primary'])-np_logp(shift['primary'])))<1e-6
 big=np.float32(2**27);assert not np.array_equal((q4+big)-big,q4);checks.append('exact_representable_gauge_and_large_FP32_boundary')
 q4=np.zeros(4,dtype='float32');q1=np.array([-4.,-2.,2.,4.],dtype='float32');n4=np.zeros((3,4),dtype='float32');n1=np.array([[-2.,0.,2.,4.],[-4.,0.,1.,2.],[-1.,-2.,1.,4.]],dtype='float32')
 def inspect_gain(a,b,c,d):
  f=np.float32(b+np.float32(.5)*(b-a));nf=np.float32(d+np.float32(.5)*(d-c));G=np_logp(f)-np_logp(b);E=np_logp(nf)-np_logp(d);cor=np_signed_min(np.concatenate([G[None],E]));p=np.exp(b.astype('float64')-b.max());p/=p.sum();return G,cor,np_center(p,cor)
 G0,c0,h0=inspect_gain(q1,q4,n1,n4);G1,c1,h1=inspect_gain(q1+8,q4+16,n1+4,n4+12);assert np.any(c0!=0) and np.any(c0!=G0) and np.array_equal(G0,G1) and np.array_equal(c0,c1) and np.array_equal(h0,h1)
 a,_=numpy_readouts(q1,q4,n1,n4,perm);b,_=numpy_readouts(q1+8,q4+16,n1+4,n4+12,perm);gauge_bound=8*np.finfo(np.float32).eps*max(1.,float(np.abs(a['primary']).max()),float(np.abs(b['primary']).max()));assert np.max(np.abs(np_logp(a['primary'])-np_logp(b['primary'])))<=gauge_bound;checks.append('partial_nonzero_common_correction_gauge_case')
 before={k:v.tobytes() for k,v in a.items()};numpy_readouts(q1[::-1].copy(),q4[::-1].copy(),n1[:,::-1].copy(),n4[:,::-1].copy(),perm);after,_=numpy_readouts(q1,q4,n1,n4,perm);assert before=={k:v.tobytes() for k,v in after.items()};checks.append('same_prefix_stateless_across_unrelated_call')
 q1=np.array([0.,-2.,-4.,-6.],dtype='float32');q4=np.array([0.,1.,2.,3.],dtype='float32');args=[q1,q4,np.tile(q1,(3,1)),np.tile(q4,(3,1)),perm];same,_=numpy_readouts(*args);delta=np.max(np.abs(np_logp(same['primary'])-np_logp(q4)));bound=8*np.finfo(np.float32).eps*max(1.,float(np.max(np.abs(same['primary']))),float(np.max(np.abs(q4))));assert delta<=bound;checks.append('identical_references_native_distribution_with_declared_rounding_bound')
 # A shared tail coordinate has consensus gain but the generic mean is exactly zero.
 ng=np.array([[2.,0.,-2.,0.],[-2.,2.,0.,0.],[0.,-2.,2.,0.]],dtype='float32');n4=np.full((3,4),-10.,dtype='float32')
 for j,i in enumerate([2,0,1]):n4[j,i]=0.
 n1=n4-2*ng;q4=np.zeros(4,dtype='float32');q1=np.array([0.,0.,0.,-2.],dtype='float32');_,d=numpy_readouts(q1,q4,n1,n4,perm);assert d['main_norm2']>0 and d['G_unmatched'];checks.append('deliberate_generic_unmatched_without_epsilon')
 q4=np.array([128.,0.,64.,32.],dtype='float32');g=np.float32(2**-17);q1=q4-2*g;_,d=numpy_readouts(q1,q4,np.tile(q1,(3,1)),np.tile(q4,(3,1)),perm);assert d['main_norm2']>0 and d['scale_unmatched'];checks.append('literal_rounding_zero_guidance_nonzero_main_scale_unmatched')
 q4=np.array([0.,-1000.],dtype='float32');args=[q4,q4,np.tile(q4,(3,1)),np.tile(q4,(3,1)),np.arange(2)];rejected(lambda:numpy_readouts(*args));rejected(lambda:torch_readouts(*torch_args(args)));p=np.array([1.,1e-320]);h=np_center(p,np.array([0.,1e-3]));rejected(lambda:np_checked_norm(p,h));rejected(lambda:torch_checked_norm(torch.from_numpy(p),torch_center(torch.from_numpy(p),torch.tensor([0.,1e-3],dtype=torch.float64))));checks+=['softmax_underflow_zero_rejected','positive_p_weighted_nonzero_h_norm_underflow_rejected']
 bad=[x.copy() for x in args];bad[0][0]=np.nan;rejected(lambda:numpy_readouts(*bad));checks.append('nonfinite_rejected')
 signature=list(inspect.signature(numpy_readouts).parameters);assert signature==['q1','q4','n1','n4','permutation'];packed=pack_diag(dict(tensor=torch.tensor(1.),boolean=True,integer=3,text='actual_gain'));assert packed==dict(tensor=1.,boolean=True,integer=3,text='actual_gain');checks+=['estimator_has_no_target_or_candidate_menu_input','mixed_type_diagnostic_packing']
 result=dict(status='passed_CPU_synthetic_only',checks=checks,FP32_output_max_error=maxout,active_random_cases=active,FP64_stat_max_error=maxdiag,renormalization_counterexample=dict(original_second_gain=float(G[1]) if len(G)==3 else None,new_second_probability=float(new[1])),model_loaded=False,new_head_run=False,GPU_worker_seconds=0,torch=torch.__version__,numpy=np.__version__,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/r33_math.py','scripts/r33_common.py','scripts/r33_cpu_math_checks.py']})
 with (ROOT/'results/r33_CPU_MATH_CHECKS.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
