"""Explicit gated AB1 model loader. Import does not initialize CUDA or download.

No caller is allowed to invoke this draft until server authorization, resource
check, scientific/source freeze and real technical gate plan are in place.
"""
import hashlib,inspect,pathlib
EXPECTED={'model.safetensors':'58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af','config.json':'ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3','modeling_ouro.py':'c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413','configuration_ouro.py':'950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb'}
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify_source(model_dir):
 p=pathlib.Path(model_dir).resolve();actual={name:sha(p/name) for name in EXPECTED};assert actual==EXPECTED,'checkpoint/config/source mismatch';return actual
def load_model(model_dir,*,authorized=False,device=None):
 if not authorized or device is None:raise PermissionError('explicit launch authorization and selected device required')
 initial=verify_source(model_dir)
 import torch,transformers
 from transformers import AutoModelForCausalLM
 if not str(device).startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('selected CUDA device unavailable')
 model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval();model.config.use_cache=False;model.model.config.use_cache=False;model.model.total_ut_steps=2
 actual_model=pathlib.Path(inspect.getsourcefile(type(model))).resolve();actual_body=pathlib.Path(inspect.getsourcefile(type(model.model))).resolve();actual_config=pathlib.Path(inspect.getsourcefile(type(model.config))).resolve()
 assert sha(actual_model)==sha(actual_body)==EXPECTED['modeling_ouro.py'] and sha(actual_config)==EXPECTED['configuration_ouro.py'],'actually imported source mismatch'
 assert all(p.dtype==torch.bfloat16 for p in model.parameters()) and verify_source(model_dir)==initial
 return model,{'checkpoint_files_sha256':initial,'actual_model_source_sha256':sha(actual_model),'actual_body_source_sha256':sha(actual_body),'actual_config_source_sha256':sha(actual_config),'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(device),'dtype':'BF16','attn_implementation':'sdpa','cache':False,'total_ut_steps':2}
