"""CPU float64 layer geometry and candidate lens; all reductions explicit."""
import numpy as np

def candidate_basis(W,targets):
 diffs=W[targets[1:]]-W[targets[0]];_,sv,vh=np.linalg.svd(diffs,full_matrices=False);tol=max(diffs.shape)*np.finfo(np.float64).eps*(sv[0] if len(sv) else 0.);rank=int((sv>tol).sum());return sv,tol,rank,vh[:rank]

def compute(h1,h2,W,targets,gamma,eps,gold,wrong,basis,rank,sv,tol):
 a=np.asarray(h1,dtype=np.float64);b=np.asarray(h2,dtype=np.float64);delta=b-a;n1=float(np.linalg.norm(a));n2=float(np.linalg.norm(b));dn=float(np.linalg.norm(delta));alpha=float(np.dot(delta,a)/(n1*n1)) if n1 else None
 u=[v/np.sqrt(np.mean(v*v)+eps)*gamma for v in (a,b)];z=[W[targets]@v for v in u];dz=z[1]-z[0];common=float(dz.mean());du=u[1]-u[0];visible=basis.T@(basis@du) if rank else np.zeros_like(du);vn=float(np.linalg.norm(visible));un=float(np.linalg.norm(du))
 out={'h1_norm':n1,'h2_norm':n2,'delta_l2':dn,'relative_delta':dn/n1 if n1 else None,'cosine':float(np.dot(a,b)/(n1*n2)) if n1 and n2 else None,'norm_ratio':n2/n1 if n1 else None,'radial_alpha':alpha,'orthogonal_delta_l2':float(np.linalg.norm(delta-alpha*a)) if alpha is not None else None,'zero_h1':n1==0,'zero_h2':n2==0,'zero_delta':dn==0,'proxy_z1':z[0].tolist(),'proxy_z2':z[1].tolist(),'proxy_delta':dz.tolist(),'proxy_common_delta':common,'proxy_centered_delta':(dz-common).tolist(),'proxy_fixed_gap_delta':float((z[1][gold]-z[1][wrong])-(z[0][gold]-z[0][wrong])),'proxy_all_wrong_gap_delta':float(np.mean([(z[1][gold]-z[1][j])-(z[0][gold]-z[0][j]) for j in range(len(targets)) if j!=gold])),'answer_subspace_rank':rank,'answer_singular_values':sv.tolist(),'answer_svd_tolerance':float(tol),'delta_u_norm':un,'answer_visible_norm':vn,'answer_orthogonal_norm':float(np.linalg.norm(du-visible)),'answer_visible_square_fraction':(vn/un)**2 if un else None}
 return out,u,z
