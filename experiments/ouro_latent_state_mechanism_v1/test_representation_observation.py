import torch
from representation_observation import pair_stats
x=torch.tensor([[1.,0.],[0.,0.]],dtype=torch.bfloat16);y=x.clone();z=torch.tensor([[0.,1.],[0.,0.]],dtype=torch.bfloat16)
a=pair_stats(x,y);assert a['tokens']==2 and a['mean_l2_change']==0 and a['mean_cosine']==1 and a['cosine_valid_tokens']==1
b=pair_stats(x,z);assert b['mean_cosine']==0 and abs(b['mean_l2_change']-2**0.5/2)<1e-6
c=pair_stats(torch.zeros((1,2)),torch.zeros((1,2)));assert c['mean_cosine'] is None and c['cosine_valid_tokens']==0
print('PASS identity, rotation, zero norm and denominator')
