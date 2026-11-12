"""L9 R-row K_Q/V_Q intervention primitive; no hooks or model execution."""
import torch
import torch.nn.functional as F

def l9_qkv_row(q,k,v,*,R,prefix,donor_k=None,donor_v=None,mask=None,scale=None,is_causal=False,enable_gqa=False):
 if q.ndim!=4 or k.ndim!=4 or v.ndim!=4 or q.shape[0]!=1 or k.shape[0]!=1 or v.shape!=k.shape:raise ValueError('qkv shape')
 if q.shape[-2]!=k.shape[-2] or R!=q.shape[-2]-1 or not 0<prefix<R:raise ValueError('absolute R/cache boundary')
 if k.dtype!=q.dtype or v.dtype!=q.dtype or not all(torch.isfinite(x).all() for x in (q,k,v)):raise ValueError('dtype or finite')
 for d,x in ((donor_k,k),(donor_v,v)):
  if d is not None and (d.shape!=x.shape or d.dtype!=x.dtype or d.device!=x.device or not torch.isfinite(d).all()):raise ValueError('donor')
 km=k.clone();vm=v.clone()
 if donor_k is not None:km[...,prefix:R,:]=donor_k[...,prefix:R,:]
 if donor_v is not None:vm[...,prefix:R,:]=donor_v[...,prefix:R,:]
 if not torch.equal(km[...,:prefix,:],k[...,:prefix,:]) or not torch.equal(km[...,R:,:],k[...,R:,:]) or not torch.equal(vm[...,:prefix,:],v[...,:prefix,:]) or not torch.equal(vm[...,R:,:],v[...,R:,:]):raise ValueError('outside Q')
 kwargs={'attn_mask':mask,'dropout_p':0.0,'is_causal':is_causal,'scale':scale,'enable_gqa':enable_gqa};base=F.scaled_dot_product_attention(q,k,v,**kwargs);mixed=F.scaled_dot_product_attention(q,km,vm,**kwargs);out=base.clone();out[...,R,:]=mixed[...,R,:];outside=torch.arange(q.shape[-2],device=q.device)!=R
 if not torch.equal(out[...,outside,:],base[...,outside,:]):raise ValueError('outside R output')
 return out,{'base_R':base[...,R,:].detach().clone(),'mixed_R':mixed[...,R,:].detach().clone()}
