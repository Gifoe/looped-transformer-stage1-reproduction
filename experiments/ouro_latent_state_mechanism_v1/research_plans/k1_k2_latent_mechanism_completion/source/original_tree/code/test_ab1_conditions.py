import torch
from ab1_conditions import build_boundary_endpoints,EDGES
E=torch.arange(48,dtype=torch.float32).reshape(1,8,6).to(torch.bfloat16);B=(E.float()+torch.arange(8).reshape(1,8,1)).to(torch.bfloat16);p=3;r=7;out=build_boundary_endpoints(E,B,p,r,split="train",ident="question")
assert len(out)==12
assert EDGES['Q_R_B']==('H11','H01','Q') and EDGES['R_Q_B']==('H11','H10','R')
assert EDGES['Q_R_E']==('H00','H10','Q') and EDGES['R_Q_E']==('H00','H01','R')
for name,(receiver,donor,region) in EDGES.items():
 a=out[receiver]['input'];b=out[donor]['input'];mask=torch.ones(8,dtype=torch.bool);mask[p:r] = False if region=='Q' else True
 if region=='R':mask[r]=False
 assert torch.equal(a[:,mask],b[:,mask]);assert set(out[name+'_norm']) >= {'status','input','receipt','receiver','donor'}
 for kind in ('norm','random'):
  v=out[name+'_'+kind];assert v['status']=='OK' and torch.equal(v['input'][:,mask],a[:,mask]);assert v['receiver']==receiver and v['donor']==donor
print('PASS 12 boundary endpoints, 4 receiver-specific paired edge controls, no labels, outside-region invariance')
