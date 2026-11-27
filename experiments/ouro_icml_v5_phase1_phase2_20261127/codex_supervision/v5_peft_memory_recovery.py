"""Recover memory-heavy V5 arms without changing their effective batch, data or optimizer."""
import ast
import gc
import inspect
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

import v5_matched_peft as matched
from v4_common import sha256_file


def backward_micro(archive,model,tokenizer,mb,weights,denom,key,depth,net,micro):
    value=0.
    with torch.enable_grad():
        for k in range(0,len(mb),micro):
            sub=mb[k:k+micro]
            raw=matched.batch_logits(archive,model,tokenizer,sub,depth,net)
            loss=sum(-F.log_softmax(x,dim=-1)[row[key]]*w/denom
                     for x,row,w in zip(raw,sub,weights[k:k+micro],strict=True))
            loss.backward()
            value+=float(loss.detach())
    return value


def logical_backward(archive,model,tokenizer,mb,weights,denom,key,depth,net,opt,micro,method,n):
    if method in ("M5b_IA3_All","M6b_LoRA_AllQV"):
        micro=min(2,micro)
    while True:
        opt.zero_grad(set_to_none=True)
        try:
            result=backward_micro(archive,model,tokenizer,mb,weights,denom,key,depth,net,micro)
        except torch.OutOfMemoryError:
            if micro==1:
                raise
        else:
            return result,micro
        opt.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        replacement=max(1,micro//2)
        print("V5_GPU_OOM_RETRY",method,n+1,micro,replacement,flush=True)
        micro=replacement


def install():
    source=inspect.getsource(matched.train)
    node=ast.parse(source).body[0]
    lines=source.splitlines(keepends=True)
    zero=next(i for i,line in enumerate(lines) if "opt.zero_grad(set_to_none=True)" in line)
    grad=next(i for i,line in enumerate(lines) if "grad=torch.nn.utils.clip_grad_norm_" in line)
    assert len(node.body)>5 and zero<grad
    lines[zero:grad]=["        loss_n,last_micro=logical_backward(archive,model,tokenizer,mb,weights,denom,key,(2,3,4)[n%3],net,opt,micro,method,n)\n"]
    modified="".join(lines)
    if modified==source:
        raise ValueError("Pinned 1C trainer structure changed")
    scope=dict(vars(matched),logical_backward=logical_backward)
    exec(compile(modified,str(Path(matched.__file__).with_name("v5_matched_peft_memory_recovery.py")), "exec"),scope)
    changed=scope["train"]
    fingerprint=sha256_file(Path(__file__))

    def receipt(*args,**kwargs):
        net,doc=changed(*args,**kwargs)
        doc.update(memory_recovery_sha256=fingerprint,
                   logical_batch_unchanged=4,gradient_accumulation="projection-heavy arms use at most two probes per call; retries halve batch")
        return net,doc
    matched.train=receipt


if __name__=="__main__":
    sys.argv[0]=str(Path(matched.__file__))
    install()
    root=Path(sys.argv[1])
    src=sys.argv[2]
    seed=int(sys.argv[3])
    matched.torch.set_num_threads(4)
    matched.run(root,src,seed,sys.argv[4:] or None)

