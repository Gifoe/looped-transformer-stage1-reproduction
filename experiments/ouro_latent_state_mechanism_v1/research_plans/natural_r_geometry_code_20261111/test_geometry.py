import tempfile,torch,numpy as np
from pathlib import Path
from safetensors.torch import save_file
from safetensors import safe_open
from geometry import compute,candidate_basis
# BF16 R is the second saved position, not a position ordinal or token ID.
h1=torch.tensor([[9.,9.,9.,9.],[1.,2.,3.,4.],[8.,8.,8.,8.]],dtype=torch.bfloat16);h2=h1.clone();h2[1]=torch.tensor([2.,1.,3.,4.],dtype=torch.bfloat16)
W=np.array([[0.,0.,0.,0.],[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],dtype=np.float64);gamma=np.array([.5,1.,1.5,2.]);targets=[4,1,3];gold=1;wrong=0;eps=1e-6
with tempfile.TemporaryDirectory() as d:
 f=Path(d)/'t.safetensors';save_file({'l1':h1,'l2':h2,'post1':h1.clone(),'logits1':torch.tensor([[7.,8.,9.,10.,11.]],dtype=torch.bfloat16),'logits2':torch.tensor([[12.,13.,14.,15.,16.]],dtype=torch.bfloat16)},str(f))
 with safe_open(str(f),framework='pt',device='cpu') as s:
  positions=[3,7,12];actual_R=7;idx=positions.index(actual_R);a=s.get_slice('l1')[idx:idx+1,:].float().numpy()[0];b=s.get_slice('l2')[idx:idx+1,:].float().numpy()[0]
  assert np.array_equal(a,[1,2,3,4]) and np.array_equal(b,[2,1,3,4])
  raw=[s.get_slice('logits1')[0,j:j+1].float().item() for j in targets];assert raw==[11.,8.,10.]
  sv,tol,rank,basis=candidate_basis(W,targets);r,u,z=compute(a,b,W,targets,gamma,eps,gold,wrong,basis,rank,sv,tol)
  manual=[v/np.sqrt(np.mean(v.astype(np.float64)**2)+eps)*gamma for v in (a,b)];assert np.allclose(u,manual,rtol=0,atol=0)
  assert np.allclose(z[0],W[targets]@manual[0],rtol=0,atol=0)
  expected=(z[1][gold]-z[1][wrong])-(z[0][gold]-z[0][wrong]);assert r['proxy_fixed_gap_delta']==expected and expected>0
  assert abs(r['answer_visible_norm']**2+r['answer_orthogonal_norm']**2-r['delta_u_norm']**2)<1e-12
  saved=s.get_slice('post1')[idx:idx+1,:].float().numpy()[0].astype(np.float64);assert max(abs(u[0]-saved))>0 # calibrator reports actual difference, not assumed zero
  r0,_,_=compute(a,a,W,targets,gamma,eps,gold,wrong,basis,rank,sv,tol);assert r0['zero_delta'] and r0['answer_visible_square_fraction'] is None
print('GEOMETRY_R_TARGET_NORM_SIGN_TOY_PASS; not real data')
