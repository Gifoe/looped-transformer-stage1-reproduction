import numpy as np
from test_technical import obs,m
from decompose import decompose
W=m.lm_head.weight.detach().numpy();gamma=m.model.norm.w.detach().numpy();v=decompose(obs['states'],W,gamma,1e-5,[1,2],0)
assert len(v)==48
for r in v:
 a=np.array(r['attention']['delta']);b=np.array(r['mlp']['delta']);assert r['attention']['identity_max_error']<1e-12 and r['mlp']['identity_max_error']<1e-12 and len(a)==len(b)==2
print('BRANCH_LENS_DECOMPOSITION_CPU_PASS; no real model')
