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


def single_option_forward(packed, model, depth, net, col):
    from types import SimpleNamespace
    from advantages import forward
    inputs, groups, positions, tokens=packed.packed
    selected=groups==col
    option=SimpleNamespace(packed=(inputs[col:col+1],groups[selected]*0,positions[selected],tokens[selected]))
    is_hook=isinstance(net,matched.parts.HookAdapterBase)
    from contextlib import nullcontext
    with (net.attach(model) if is_hook else nullcontext()):
        output,_=forward(model,option,depth,None if is_hook else net)
    return -F.cross_entropy(output,tokens[selected],reduction="sum") * (16./packed.metadata[0]["characters"][col])


def one_option_at_a_time(archive,model,tokenizer,mb,weights,denom,key,depth,net):
    from advantages import choice_batch
    total=0.
    for row,weight in zip(mb,weights,strict=True):
        request=matched.make_candidate_row(row["id"],row["question"],row["options"],0)
        packed=choice_batch(archive,tokenizer,[request])
        columns=len(row["options"])
        target=row[key]
        with torch.no_grad():
            scores=torch.stack([single_option_forward(packed,model,depth,net,i) for i in range(columns)])
            value=-scores.log_softmax(0)[target]
            coeff=scores.softmax(0)
            coeff[target]-=1.
            coeff*=weight/denom
            total+=float(value)*weight/denom
        with torch.enable_grad():
            for i in range(columns):
                score=single_option_forward(packed,model,depth,net,i)
                (score*coeff[i]).backward()
    return total



def logical_backward(archive,model,tokenizer,mb,weights,denom,key,depth,net,opt,micro,method,n):
    if method in ("M5b_IA3_All","M6b_LoRA_AllQV"):
        micro=min(2,micro)
    while True:
        opt.zero_grad(set_to_none=True)
        try:
            result=backward_micro(archive,model,tokenizer,mb,weights,denom,key,depth,net,micro)
        except torch.OutOfMemoryError:
            pass
        else:
            return result,micro
        opt.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        if micro==1:
            print("V5_GPU_OOM_EXACT_OPTION_GRADIENT",method,n+1,flush=True)
            return one_option_at_a_time(archive,model,tokenizer,mb,weights,denom,key,depth,net),1
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
                   logical_batch_unchanged=4,gradient_accumulation="projection-heavy arms use at most two probes per call; retries halve batch; at one probe use exact choice-wise softmax chain rule")
        return net,doc
    matched.train=receipt


if __name__=="__main__":
    sys.argv[0]=str(Path(matched.__file__))
    install()
    root=Path(sys.argv[1])
    src=sys.argv[2]
    seed=int(sys.argv[3])
    matched.torch.set_num_threads(4)
    reserved=['M0_InLoop_Rank1', 'M2_Learned_Constant', 'M4_PostLoop_LMHead_Rank1', 'M1a_CAA', 'M1b_RepE']
    if src=="S2" and seed==26112712 and not sys.argv[4:]:
        matched.run(root,src,seed,[x for x in matched.METHODS+matched.STATIC if x not in reserved])
        work=root/"track_b_baselines"
        import time
        while any(not (work/f"S2.{method}.s{seed}.json").is_file() for method in reserved):
            print("V5_S2_RESERVED_OLD_WAITING",flush=True)
            time.sleep(40)
    else:
        matched.run(root,src,seed,sys.argv[4:] or None)

