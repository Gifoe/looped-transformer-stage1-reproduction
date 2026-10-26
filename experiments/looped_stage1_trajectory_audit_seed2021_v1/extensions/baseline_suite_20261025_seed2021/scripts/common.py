"""Shared paths, immutable protocol, cumulative GPU budget and readouts."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]
CFG=json.loads((ROOT/'configs/protocol.json').read_text(encoding='utf-8'))
PROTOCOL_SHA=hashlib.sha256((ROOT/'configs/protocol.json').read_bytes()).hexdigest()
_code_files=[Path(__file__),ROOT/'scripts/methods.py',BASE/'scripts/qwen_recipe.py',Path(sys.argv[0]).resolve()]
IMPLEMENTATION_SHA=hashlib.sha256(''.join(str(path.name)+hashlib.sha256(path.read_bytes()).hexdigest() for path in _code_files if path.is_file()).encode()).hexdigest()
os.environ.setdefault('HF_HOME',str(ROOT/'cache/huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
sys.path.insert(0,str(BASE/'scripts'))

def read(path):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def save(path,value):
    path=ROOT/path;path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    for attempt in range(40):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==39:raise
            time.sleep(.05)

def append(path,value):
    value=dict(value,protocol_sha256=PROTOCOL_SHA,implementation_sha256=IMPLEMENTATION_SHA)
    path=ROOT/path;path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')

def existing(path,key='id'):
    path=ROOT/path
    if not path.exists():return {}
    rows=read(path)
    assert all(row['protocol_sha256']==PROTOCOL_SHA for row in rows),'Protocol changed after predictions'
    assert all(row['implementation_sha256']==IMPLEMENTATION_SHA for row in rows),'Implementation changed; investigate before resume'
    result={row[key]:row for row in rows};assert len(result)==len(rows)
    return result

class Budget:
    def __init__(self,tag):
        self.tag=tag;self.started=time.monotonic()
        path=ROOT/'logs/gpu_budget.json'
        if path.exists():self.previous=json.loads(path.read_text())
        else:
            prior=json.loads((BASE/'logs/gpu_budget.json').read_text())['used_seconds']
            self.previous=dict(stage1_used_seconds=prior,used_seconds=prior,runs=[])
        self.check()
        save('logs/active_gpu.json',dict(pid=os.getpid(),tag=tag,started_unix=time.time(),protocol_sha256=PROTOCOL_SHA))
    def check(self,reserve=30):
        if self.previous['used_seconds']+time.monotonic()-self.started+reserve>=CFG['hard_total_gpu_hours_including_stage1']*3600:raise RuntimeError('Cumulative 50 GPU-hour limit')
    def close(self,state='finished'):
        seconds=time.monotonic()-self.started
        self.previous['used_seconds']+=seconds
        self.previous['runs'].append(dict(tag=self.tag,seconds=seconds,state=state,pid=os.getpid()))
        self.previous['extension_used_seconds']=self.previous['used_seconds']-self.previous['stage1_used_seconds']
        save('logs/gpu_budget.json',self.previous)
        save('logs/active_gpu.json',dict(pid=os.getpid(),tag=self.tag,state=state,finished_unix=time.time()))
    def __enter__(self):return self
    def __exit__(self,kind,value,tb):self.close('failed' if kind else 'finished')

def setup_torch():
    import numpy as np
    import torch
    random.seed(2021);np.random.seed(2021);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021)
    torch.set_grad_enabled(False);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    return torch

def load(name):
    torch=setup_torch()
    from transformers import AutoModelForCausalLM,AutoTokenizer
    directory=BASE/'models'/name
    tok=AutoTokenizer.from_pretrained(directory,local_files_only=True)
    model,loading=AutoModelForCausalLM.from_pretrained(directory,local_files_only=True,trust_remote_code=name.startswith('Ouro'),torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
    assert not loading['missing_keys'] and not loading.get('mismatched_keys'),loading
    model=model.to('cuda').eval()
    assert len(model.model.layers)==model.config.num_hidden_layers
    if name.startswith('Ouro'):assert model.model.total_ut_steps==4
    for parameter in model.parameters():assert not parameter.is_quantized
    save('results/model_load_'+name+'.json',dict(model=name,loading_info=loading,config=model.config.to_dict(),dtype=str(next(model.parameters()).dtype),revision=json.loads((directory/'snapshot_metadata.json').read_text())['sha']))
    return tok,model

def encode_pair(tok,context,continuation):
    from types import SimpleNamespace
    from lm_eval.api.model import TemplateLM
    coder=SimpleNamespace(tok_encode=lambda text,**kw:tok.encode(text,add_special_tokens=False))
    prefix,answer=TemplateLM._encode_pair(coder,context,continuation)
    assert prefix and answer
    assert len(prefix)+len(answer)<=CFG['max_context_tokens'],'Context cap; never truncate silently'
    return prefix,answer

def vocab_stats(z):
    import torch
    lp=z.float().log_softmax(-1);p=lp.exp();top=p.topk(2).values
    sorted_z=z.float().topk(2).values
    return dict(confidence=float(top[0]),entropy=float(-(p*lp).sum()),probability_margin=float(top[0]-top[1]),logit_margin=float(sorted_z[0]-sorted_z[1])),lp

def candidate_result(sums,lengths,token_counts,gold,metric,proper_probability=True):
    import numpy as np
    raw=np.asarray(sums,dtype=float);normalized=raw/np.maximum(lengths,1)
    score=raw if metric=='acc' else normalized
    log_weights=score-(float(np.log(np.exp(score-score.max()).sum()))+score.max())
    probabilities=np.exp(log_weights)
    prediction=int(score.argmax())
    return dict(prediction=prediction,correct=prediction==gold,raw_prediction=int(raw.argmax()),normalized_prediction=int(normalized.argmax()),
                raw_correct=int(raw.argmax())==gold,normalized_correct=int(normalized.argmax())==gold,
                sum_scores=raw.tolist(),normalized_scores=normalized.tolist(),scores=score.tolist(),choice_weights=probabilities.tolist(),choice_log_weights=log_weights.tolist(),
                gold_choice_log_weight=float(log_weights[gold]),gold_nll_per_token=float(-raw[gold]/token_counts[gold]) if proper_probability else None,
                primary_metric=metric,score_semantics='conditional log likelihood' if proper_probability else 'official DoLa contrastive score; post-softmax then relative-top mask')

@contextmanager
def raw_states(model,selection):
    states=[]
    handle=model.model.norm.register_forward_pre_hook(lambda module,args:states.append(args[0][:,selection,:].clone()))
    try:yield states
    finally:handle.remove()

def hidden_change(current,previous):
    import torch.nn.functional as F
    a=current.float();b=previous.float()
    return dict(hidden_relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-8)),hidden_absolute_l2=float((a-b).norm()),hidden_cosine=float(F.cosine_similarity(a,b,dim=0)))
