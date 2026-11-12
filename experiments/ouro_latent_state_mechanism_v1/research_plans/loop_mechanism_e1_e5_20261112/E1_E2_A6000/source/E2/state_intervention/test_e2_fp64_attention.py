import math,torch
import torch.nn.functional as F
from e2_state_intervention import fp64_attention
torch.manual_seed(7);q=torch.randn(1,4,5,3,dtype=torch.float64);k=torch.randn(1,2,5,3,dtype=torch.float64);v=torch.randn(1,2,5,3,dtype=torch.float64)
for R in (0,2,4):
 A,O=fp64_attention(q,k,v,R,(),{'dropout_p':0.,'is_causal':True,'scale':.37,'enable_gqa':True});ref=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=True,scale=.37,enable_gqa=True);assert torch.allclose(O.squeeze(-2),ref[...,R,:],atol=1e-12,rtol=1e-12);assert torch.allclose(A.sum(-1),torch.ones_like(A.sum(-1)))
mask=torch.tensor([[[[1,1,0,1,0]]]],dtype=torch.bool);A,O=fp64_attention(q,k,v,2,(),{'attn_mask':mask,'dropout_p':0.,'enable_gqa':True});ref=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,enable_gqa=True);assert torch.allclose(O.squeeze(-2),ref[...,2,:],atol=1e-12,rtol=1e-12)
print('PASS FP64 SDPA reconstruction causal three R positions, GQA, custom scale and bool mask')
