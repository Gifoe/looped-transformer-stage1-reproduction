"""Gold-free R22 fixed trailing-step features. No target/model access."""
import numpy as np

def numpy_feature(states):
 a=[np.asarray(x,dtype=np.float64) for x in states]
 if len(a)!=6 or any(x.ndim!=3 or x.shape[0]!=1 or x.shape!=a[0].shape or x.shape[1]==0 or x.shape[2]==0 for x in a):raise ValueError('Invalid complete six-state shape')
 if not all(np.isfinite(x).all() for x in a):raise ValueError('Nonfinite state')
 e45=np.mean((a[4]-a[3])**2,axis=-1)[0];e56=np.mean((a[5]-a[4])**2,axis=-1)[0];den=e45+e56;r=np.full(den.shape,.5);np.divide(e45,den,out=r,where=den!=0)
 if not all(np.isfinite(x).all() for x in [e45,e56,r]):raise ValueError('Nonfinite energy')
 return dict(V=float(r.mean()),C_energy=float(-e56.mean()),degenerate=int((den==0).sum()),tokens=len(r))

def torch_feature(states):
 import torch
 if len(states)!=6 or any(x.dtype!=torch.bfloat16 or x.device.type!='cpu' or x.ndim!=3 or x.shape[0]!=1 or x.shape!=states[0].shape or x.shape[1]==0 or x.shape[2]==0 for x in states):raise ValueError('Expected six CPU BF16 states')
 if not all(bool(torch.isfinite(x).all()) for x in states):raise ValueError('Nonfinite state')
 h4,h5,h6=[x.double() for x in states[3:6]];e45=(h5-h4).square().mean(-1)[0];e56=(h6-h5).square().mean(-1)[0];den=e45+e56;r=torch.where(den==0,torch.full_like(den,.5),e45/torch.where(den==0,torch.ones_like(den),den))
 if not all(bool(torch.isfinite(x).all()) for x in [e45,e56,r]):raise ValueError('Nonfinite energy')
 return dict(V=float(r.mean()),C_energy=float(-e56.mean()),degenerate=int((den==0).sum()),tokens=len(r))
