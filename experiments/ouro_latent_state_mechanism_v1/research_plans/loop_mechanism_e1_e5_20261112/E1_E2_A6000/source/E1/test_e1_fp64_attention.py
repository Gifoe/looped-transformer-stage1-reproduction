import numpy as np
from e1_fp64_attention import attention_row,decompose
rng=np.random.default_rng(309);q=rng.normal(size=(1,4,5,3));k=rng.normal(size=(1,2,5,3));v=rng.normal(size=(1,2,5,3));R=3
A,o,valid=attention_row(q,k,v,R,is_causal=True,scale=.37,enable_gqa=True)
assert valid.all() and np.all(A[:,:,R+1:]==0) and np.all(A[:,:,:R+1]>0)
# An incorrectly cropped one-query causal call uses row0 and must be rejected.
try:attention_row(q[:,:,R:R+1,:],k,v,0,is_causal=True,scale=.37,enable_gqa=True)
except ValueError:pass
else:raise AssertionError('incorrect local row0 causal mask accepted')
k2=k+rng.normal(size=k.shape);v2=v+rng.normal(size=v.shape);d=decompose(q,k,v,k2,v2,R,is_causal=True,scale=.37,enable_gqa=True)
assert np.max(np.abs(d['accounting_residual']))<1e-12
mask=np.zeros((1,1,5,5),dtype=bool);mask[:,:,R,:2]=True
A,_,valid=attention_row(q,k,v,R,mask=mask,enable_gqa=True);assert valid.all() and np.all(A[:,:,2:]==0)
mask[:]=False;A,o,valid=attention_row(q,k,v,R,mask=mask,enable_gqa=True);assert not valid.any() and np.all(A==0) and np.all(o==0)
import torch
import torch.nn.functional as F
qt=torch.tensor(q);kt=torch.tensor(np.repeat(k,2,axis=1));vt=torch.tensor(np.repeat(v,2,axis=1))
_,ours,_=attention_row(q,k,v,R,is_causal=True,scale=.37,enable_gqa=True)
ref=F.scaled_dot_product_attention(qt,kt,vt,is_causal=True,scale=.37,dropout_p=0).numpy()[:,:,R,:]
assert np.allclose(ours,ref,atol=1e-12,rtol=1e-12)
floatmask=np.zeros((1,1,5,5),dtype=np.float64);floatmask[:,:,:,1]=.4;floatmask[:,:,:,4]=-np.inf
_,ours,_=attention_row(q,k,v,R,mask=floatmask,scale=.37,enable_gqa=True)
ref=F.scaled_dot_product_attention(qt,kt,vt,attn_mask=torch.tensor(floatmask),scale=.37,dropout_p=0).numpy()[:,:,R,:];assert np.allclose(ours,ref,atol=1e-12,rtol=1e-12)
boolmask=np.ones((1,1,5,5),dtype=bool);boolmask[:,:,:,2]=False
_,ours,_=attention_row(q,k,v,R,mask=boolmask,scale=.37,enable_gqa=True)
ref=F.scaled_dot_product_attention(qt,kt,vt,attn_mask=torch.tensor(boolmask),scale=.37,dropout_p=0).numpy()[:,:,R,:];assert np.allclose(ours,ref,atol=1e-12,rtol=1e-12)
try:attention_row(q,k,v,R,mask=np.ones((1,1,5,5),dtype=int),enable_gqa=True)
except ValueError:pass
else:raise AssertionError('non-boolean/non-float mask accepted')
print('CPU_ACTUAL_GLOBAL_R_GQA_AND_DECOMPOSITION_PASS')
