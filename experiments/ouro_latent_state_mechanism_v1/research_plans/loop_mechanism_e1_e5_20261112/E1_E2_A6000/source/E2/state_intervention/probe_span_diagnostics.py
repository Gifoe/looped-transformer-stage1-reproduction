import pathlib,json,numpy as np,hashlib
from e2_state_intervention import raw_probe_span
E=pathlib.Path(__file__).resolve().parent.parent;src=E/'probe_fit_attempt3/selected_models.npz';z=np.load(src,allow_pickle=False);out={}
for name in sorted(k[:-2] for k in z.files if k.endswith('_w')):
 w=z[name+'_w'];std=z[name+'_std'];d=(w[:-1,1:]-w[:-1,:1])/std[:,None];U,s,t=raw_probe_span(w,std);out[name]={'rank':U.shape[1],'singular_values':s.tolist(),'tolerance':float(t),'orthogonality_error_fro':float(np.linalg.norm(U.T@U-np.eye(U.shape[1]))),'reconstruction_residual_fro':float(np.linalg.norm(d-U@(U.T@d))),'warning':'Raw input-coordinate linear directions valid only for raw variant; unit_l2 is nonlinear in original inputs.' if '_unit_l2_' in name else None}
p=pathlib.Path(__file__).resolve().parent/'all_probe_span_diagnostics.json';assert not p.exists();p.write_text(json.dumps({'source_npz_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'models':out},indent=2,sort_keys=True)+'\n');print('16 model spectra receipt',hashlib.sha256(p.read_bytes()).hexdigest())
