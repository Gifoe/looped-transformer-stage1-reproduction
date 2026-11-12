"""Partition fixed-query KV interaction by absolute value-token position."""
import numpy as np

def split_value_interaction(A1,A2,V1,V2,R,prefix_length=1839):
 A1=np.asarray(A1,dtype=np.float64);A2=np.asarray(A2,dtype=np.float64);V1=np.asarray(V1,dtype=np.float64);V2=np.asarray(V2,dtype=np.float64)
 if A1.shape!=A2.shape or A1.ndim!=3 or V1.shape!=V2.shape or V1.ndim!=4 or V1.shape[:3]!=A1.shape or not all(np.isfinite(x).all() for x in (A1,A2,V1,V2)):raise ValueError('shape/finite')
 L=A1.shape[-1]
 if (A1<0).any() or (A2<0).any() or not np.allclose(A1.sum(-1),1.,atol=1e-12,rtol=1e-12) or not np.allclose(A2.sum(-1),1.,atol=1e-12,rtol=1e-12):raise ValueError('attention probabilities')
 if type(R)!=int or type(prefix_length)!=int or not(0<prefix_length<R<L):raise ValueError('bad P/Q/R boundary')
 if np.any(A1[...,R+1:]!=0) or np.any(A2[...,R+1:]!=0):raise ValueError('causal keys after R')
 labels=np.empty(L,dtype=np.int8);labels[:prefix_length]=0;labels[prefix_length:]=1;labels[R]=2
 if (labels==0).sum()!=prefix_length or (labels==2).sum()!=1 or (labels==1).sum()!=L-prefix_length-1:raise ValueError('partition')
 term=(A2-A1)[...,None]*(V2-V1);out={k:term[:,:,labels==v,:].sum(axis=2) for k,v in [('P',0),('Q',1),('R',2)]};total=term.sum(axis=2)
 if not np.allclose(out['P']+out['Q']+out['R'],total,atol=1e-10,rtol=1e-10):raise ValueError('partition accounting')
 return out,total
