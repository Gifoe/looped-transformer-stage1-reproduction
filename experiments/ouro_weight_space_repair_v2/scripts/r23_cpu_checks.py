"""CPU-only explicit one-hot versus O(V) and estimator invariance tests."""
import numpy as np
from r23_math import numpy_evolution,torch_evolution,torch_logits

def explicit(z):
 z=np.asarray(z,dtype=np.float64);p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True);weights=np.zeros_like(p[3]);v=z.shape[-1]
 for k in range(3):
  d=z[k]-z[3];d=d-d.mean(-1,keepdims=True)
  for t in range(z.shape[1]):
   if (d[t]*d[t]).sum()<=64*np.finfo(float).eps*np.max(np.abs(d[t]))**2:continue
   for j in range(v):
    gradient=p[k,t]-np.eye(v)[j];g2=gradient@gradient
    if g2<=64*np.finfo(float).eps*max(1.,np.max(p[k,t]**2)):continue
    cosine=d[t]@gradient/np.sqrt((d[t]@d[t])*g2);weights[t,j]+=max(cosine,0.)**2
 total=weights.sum(-1,keepdims=True);latent=np.divide(weights,np.where(total==0,1,total));latent=np.where(total==0,p[3],latent)
 return latent-p[3],latent

def main():
 import torch
 torch.set_num_threads(2);rng=np.random.default_rng(2021);q=rng.normal(size=(4,3,13)).astype(np.float32);refs=rng.normal(size=(3,4,3,13)).astype(np.float32);perm=rng.permutation(13)
 u,d=numpy_evolution(q);eu,ep=explicit(q);assert np.max(np.abs(u-eu))<1e-10 and np.max(np.abs(d['latent']-ep))<1e-10
 tq=torch.from_numpy(q);tr=torch.from_numpy(refs);tu,td=torch_evolution(tq);assert np.max(np.abs(u-tu.numpy()))<1e-10
 values,diag,parts=torch_logits(tq,tr,perm)
 assert np.max(np.abs(parts[0].numpy()-u))<1e-10
 for key in ['shuffle_P_norm2','probability_P_norm2']:assert torch.max(torch.abs(diag[key]-diag['c_P_norm2']))<1e-10
 shifted=q.astype(np.float64)+np.array([2.,-1.,3.,5.])[:,None,None];su,sd=numpy_evolution(shifted);assert np.max(np.abs(su-u))<1e-10
 same=np.repeat(q[3:4],4,axis=0);zero,zd=numpy_evolution(same);assert np.array_equal(zero,np.zeros_like(zero))
 identity=torch_logits(tq,tq.unsqueeze(0).expand(3,-1,-1,-1),perm);assert torch.equal(identity[0]['primary'],identity[0]['M3']) and bool(identity[1]['c_zero'].all())
 duplicate=torch_logits(tq,tr[:1].expand(3,-1,-1,-1),perm);assert bool(torch.isfinite(duplicate[0]['primary']).all())
 reordered=torch_logits(tq,tr[[2,0,1]],perm);assert torch.max(torch.abs(values['primary']-reordered[0]['primary']))<=2e-5 and torch.max(torch.abs(parts[2]-reordered[2][2]))<=1e-10
 extq=torch.cat([tq,tq[:,:1]],dim=1);extr=torch.cat([tr,tr[:,:,:1]],dim=2);ex=torch_logits(extq,extr,perm);assert torch.equal(values['primary'],ex[0]['primary'][:3])
 bad=q.copy();bad[0,0,0]=np.nan
 try:numpy_evolution(bad)
 except ValueError:pass
 else:raise AssertionError('Nonfinite accepted')
 assert np.max(np.abs(d['latent'].sum(-1)-1))<1e-12
 saturated=np.full((4,1,5),-1000.,dtype=np.float32);saturated[:,:,0]=1000.;saturated[0,0,1]=-900.;sat,ds=numpy_evolution(saturated);assert np.isfinite(sat).all()
 try:torch_evolution(torch.from_numpy(bad))
 except ValueError:pass
 else:raise AssertionError('Torch nonfinite accepted')
 assert torch.max(torch.abs(torch_evolution(tq.double()+torch.tensor([2.,-1.,3.,5.])[:,None,None])[0]-tu))<1e-10
 import inspect
 assert all('target' not in inspect.signature(f).parameters for f in [numpy_evolution,torch_evolution,torch_logits])
 return dict(status='passed_CPU_synthetic',explicit_onehot_vs_linear_maxerror=float(np.max(np.abs(u-eu))),NumPy_Torch_maxerror=float(np.max(np.abs(u-tu.numpy()))),gauge=True,identical_depth_zero=True,identical_Q_references_exact_M3=True,duplicate_reference=True,reference_permutation=True,candidate_extension_exact=True,normmatch=True,fullV_normalization=True,nonfinite_rejected=True,no_target_arguments=True,arithmetic='c=mean_j(uQ-uNj), E=mean_j(pNj4-pQ4), FP64 paired differences; no epsilon')
if __name__=='__main__':
 import json
 print(json.dumps(main()))
