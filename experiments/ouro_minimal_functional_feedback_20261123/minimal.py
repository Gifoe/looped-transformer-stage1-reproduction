"""Single low-rank state update injected into the actual Ouro recurrence."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

WIDTH=2048
DEPTHS=(1,2,3,4)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def dump(path,x):
    p=Path(path); tmp=p.with_suffix('.partial'); tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); tmp.replace(p)


def runtime(archive,previous,checkpoint):
    sys.path.insert(0,str(Path(previous).resolve()))
    from arc_features import archive as import_arc
    old=import_arc(Path(archive)); model,tok,info=old.load_runtime(Path(checkpoint)); model.requires_grad_(False)
    return old,model,tok,info


def rms(x):
    return x.square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt()


class Minimal(nn.Module):
    def __init__(self,kind,rank,bound=.45,basis=None,fixed=False):
        super().__init__()
        self.kind,self.rank,self.bound,self.fixed=kind,rank,bound,fixed
        if kind=='constant':
            self.vector=nn.Parameter(torch.zeros(WIDTH))
            return
        self.encode=nn.Linear(WIDTH,rank,bias=False)
        self.out=nn.Linear(rank,WIDTH,bias=False)
        nn.init.zeros_(self.out.weight)
        if kind=='history':
            self.history=nn.Linear(WIDTH,rank,bias=False)
        if basis is not None:
            with torch.no_grad():
                self.out.weight.copy_(basis[:,:rank])
                self.encode.weight.mul_(.02)
        if fixed:
            self.out.weight.requires_grad_(False)

    def delta(self,h,previous=None):
        scale=rms(h)
        if self.kind=='constant':
            x=scale*self.vector.expand_as(h)
        elif self.kind in ('linear','plain'):
            x=self.out(self.encode(h))
        else:
            value=self.encode(h/scale)
            if self.kind=='history' and previous is not None:
                d=h-previous
                value=value+self.history(d/rms(d))
            x=scale*self.out(F.silu(value))
        return x if self.bound is None else x*(self.bound*scale/rms(x)).clamp_max(1.)


def load_net(path):
    z=torch.load(path,map_location='cpu',weights_only=True)
    net=Minimal(**z['arch'])
    net.load_state_dict(z['state'])
    return net.cuda().eval()


def save_net(path,net,info):
    path=Path(path); p=path.with_suffix('.partial')
    torch.save(dict(arch=dict(kind=net.kind,rank=net.rank,bound=net.bound,fixed=net.fixed),
                    state={k:v.detach().cpu() for k,v in net.state_dict().items()},meta=info),p)
    p.replace(path)


def forward(model,packed,depth,net=None,dose=1.,leaf=False):
    """Write only at eligible causal predictor positions, then run the frozen suffix."""
    inp,group,pos,_=packed
    core=model.model
    core.total_ut_steps=depth
    previous=None; anchor=None; counter=0

    def edit(tensor):
        nonlocal anchor
        anchor=tensor[group,pos].float()
        if leaf:
            anchor=anchor.detach().requires_grad_(True)
        if net is not None:
            value=anchor+dose*net.delta(anchor,previous)
        elif leaf:
            value=anchor
        else:
            return tensor
        return tensor.index_put((group,pos),value.to(tensor.dtype))

    def before_norm(_,args):
        nonlocal previous,counter
        h=args[0]
        if counter==depth-3 and depth>=3 and net is not None and net.kind=='history':
            previous=h[group,pos].float()
        if counter==depth-2:
            h=edit(h)
        counter+=1
        return (h,*args[1:]) if h is not args[0] else None

    if depth==1:
        handle=core.layers[20].register_forward_hook(lambda _,args,tensor:edit(tensor))
    else:
        handle=core.norm.register_forward_pre_hook(before_norm)
    try:
        _,states,_=core(input_ids=inp,use_cache=False)
    finally:
        handle.remove()
    assert anchor is not None and len(states)==depth
    return states[-1][group,pos],anchor


def label_prompt(row):
    options=[chr(65+i) for i in range(len(row['choices']['label']))]
    inline='\n'.join(f'{a}. {x}' for a,x in zip(options,row['choices']['text'],strict=True))
    text=f"Question: {row['question']}\n{inline}\nAnswer with the option letter.\nAnswer:"
    return text,options,options[row['choices']['label'].index(row['answerKey'])]


def load_jsonl(p):
    with Path(p).open(encoding='utf-8') as f:
        return [json.loads(x) for x in f]
