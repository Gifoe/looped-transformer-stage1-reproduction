"""Experiment-local configuration, frozen inference and independent6h budget checks."""
from pathlib import Path
import hashlib,json,os,random,time

ROOT=Path(__file__).resolve().parents[1]
REVISION='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
def save(path,value):
 p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8');tmp.replace(p)
def configure():
 dirs={'TMPDIR':'tmp','HF_HOME':'cache/hf','XDG_CACHE_HOME':'cache/xdg','PYTHONPYCACHEPREFIX':'cache/pycache','TORCH_HOME':'cache/torch','TORCHINDUCTOR_CACHE_DIR':'cache/torchinductor','TRITON_CACHE_DIR':'cache/triton','CUDA_CACHE_PATH':'cache/cuda'}
 for key,rel in dirs.items():
  (ROOT/rel).mkdir(parents=True,exist_ok=True);os.environ[key]=str(ROOT/rel)
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
def check_budget(reserve=0):
 deadline=os.environ.get('GPU_DEADLINE_UNIX')
 if deadline and time.time()+reserve>=float(deadline):raise RuntimeError('Independent6 GPU-hour budget reserve reached')
def load_model():
 configure()
 import numpy as np
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 random.seed(2021);np.random.seed(2021);torch.manual_seed(2021);torch.set_num_threads(4);torch.set_grad_enabled(False)
 torch.backends.cuda.matmul.allow_tf32=False
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
 spec=json.loads((ROOT/'models/Ouro-1.4B/snapshot_metadata.json').read_text());assert spec['sha']==REVISION
 verified=json.loads((ROOT/'logs/MODEL_SERVER_VERIFIED.json').read_text());assert verified['revision']==REVISION and verified['status']=='passed'
 path=ROOT/'models/Ouro-1.4B'
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
 model,info=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
 assert all(not info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
 model=model.to('cuda').eval().requires_grad_(False)
 model.config.use_cache=False
 model.model.config.use_cache=False
 assert not any(p.requires_grad for p in model.parameters())
 assert model.config.num_hidden_layers==24 and model.config.hidden_size==2048 and model.model.total_ut_steps==4
 return tok,model
def encode_pair(tok,context,continuation):
 from types import SimpleNamespace
 from lm_eval.api.model import TemplateLM
 coder=SimpleNamespace(tok_encode=lambda text,**kw:tok.encode(text,add_special_tokens=False))
 prefix,answer=TemplateLM._encode_pair(coder,context,continuation)
 assert prefix and answer and len(prefix)+len(answer)<=32768
 return prefix,answer
def outcome(scores,lengths,gold=None):
 import numpy as np
 a=np.asarray(scores,dtype=float);normalized=a/np.maximum(np.asarray(lengths,dtype=float),1)
 pred=int(normalized.argmax());r=dict(sum_scores=a.tolist(),normalized_scores=normalized.tolist(),prediction=pred)
 if gold is not None:r.update(gold=int(gold),correct=pred==gold,margin=float(normalized[gold]-max(np.delete(normalized,gold))))
 return r
