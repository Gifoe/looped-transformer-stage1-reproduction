"""Independent CPU checkpoint/lens and state-availability audit of technical probes."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
R=Path('/root/autodl-tmp/ouro-e7');M=Path('/root/autodl-tmp/ouro-e4-e5/model');torch.set_num_threads(1)
config=json.loads((M/'config.json').read_text());eps=config['rms_norm_eps'];rows=[]
with safe_open(str(M/'model.safetensors'),framework='pt',device='cpu') as f:
 norm=f.get_tensor('model.norm.weight').float()
 for p in sorted((R/'technical2').glob('train_*.json')):
  q=json.loads(p.read_text());state=p.with_suffix('.npz');assert hashlib.sha256(state.read_bytes()).hexdigest()==q['states_sha256'];a=np.load(state);w=torch.stack([f.get_slice('lm_head.weight')[t:t+1].squeeze(0).float() for t in q['targets']]);errors={}
  for family in ['candidate','unrelated']:
   x=torch.from_numpy(a[family+'_x']);y=torch.from_numpy(a[family+'_y']);inc=torch.from_numpy(a[family+'_a']);assert x.shape==y.shape==inc.shape==(4,7,2048)
   assert torch.equal((x.bfloat16()+inc.bfloat16()).float(),y)
   # Independent PyTorch implementation; production lens uses NumPy FP32.
   def reference(z):return torch.nn.functional.linear((z*torch.rsqrt(z.square().mean(-1,keepdim=True)+eps))*norm,w)
   b=reference(y)-reference(x);b-=b.mean(-1,keepdim=True);saved=torch.from_numpy(a[family+'_b']);err=float((b-saved).abs().max());assert torch.allclose(b,saved,rtol=2e-5,atol=2e-4),(q['id'],family,err)
   assert q['features'][family][0] is None and all(len(v)==16 and np.isfinite(v).all() for v in q['features'][family][1:]);errors[family]=err
  rows.append({'id':q['id'],'checkpoint_row_mapping_verified':True,'per_layer_states_shape':[4,7,2048],'actual_BF16_add_identity':True,'independent_torch_vs_numpy_FP32_max_abs_error':errors,'loop1_unavailable':True,'all_remaining_features_finite':True})
assert len(rows)==2
(R/'H2_LENS_INDEPENDENT_QA.json').write_text(json.dumps({'status':'PASS','n':2,'scope':'R-only checkpoint candidate-row and FP32 operator check, not causal contribution or full-dataset accuracy','epsilon':eps,'torch':torch.__version__,'numpy':np.__version__,'tolerances':{'rtol':2e-5,'atol':2e-4},'rows':rows,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'GPU_forwards':0},indent=2))
print('H2_INDEPENDENT_LENS_QA_PASS',rows,flush=True)
