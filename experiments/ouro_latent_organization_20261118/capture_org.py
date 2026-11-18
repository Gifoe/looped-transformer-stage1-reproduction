"""Capture native residuals of complete prompts at every Loop × Layer coordinate."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from manifest_org import GOALS, TASKS, digest, dump_line, get_encoding, load_lines, write_json
from collect import fast_forward, load_runtime, predictions
from run import depth_scores

ROLES=("source","bridge","mapping","query","readout")
LAYER_COUNT=24
PROJECTION_WIDTH=96


def projection():
    rng=np.random.default_rng(2026111807)
    frame,_=np.linalg.qr(rng.standard_normal((2048,PROJECTION_WIDTH)))
    return frame.astype(np.float32)


def sample_span(seq,cap=12):
    if len(seq)<=cap:
        return list(seq)
    return [seq[j] for j in np.linspace(0,len(seq)-1,cap,dtype=int)]


def positions(g,e):
    r=e["roles"]
    n=e["read"]
    if g["task"]=="relation":
        src=r["payload_0"]+r[f'payload_{g["meta"]["other"]}']
        bridge=r["bind"]
    else:
        src=r.get("signal",list(range(max(0,n//5-2),n//5+1)))
        bridge=r.get("bridge",list(range(max(0,n//3-2),n//3+1)))
    mapping=r.get("map",list(range(n//2,max(n//2+1,3*n//4))))
    query=r.get("query",[max(0,n-3)])
    if g["task"] not in TASKS:
        query=query[-8:]
    return dict(zip(ROLES,[sample_span(sorted(set(p))) for p in (src,bridge,mapping,query,[n])]))


@contextmanager
def native_bank(model,select,edits=None,full=False,modules=False,depth=4,tap_layers=None):
    """Use original method and lifetime-bounded detached tensors."""
    core=model.model
    previous=core.total_ut_steps
    core.total_ut_steps=depth
    observed=[]
    bank={}
    norm=[]
    branch={}
    hooks=[]
    if not full:
        index=torch.tensor(select,device=model.lm_head.weight.device)
    def grab(x):
        return x[0].detach().clone() if full else x[0].index_select(0,index).detach()
    def layer_hook(i):
        def hook(module,args,kw,out):
            event=(kw["current_ut"]+1,i)
            assert event==(len(observed)//24+1,len(observed)%24)
            if edits and event in edits:
                pos,value=edits[event]
                out=out.clone()
                out[0,pos]=value
            observed.append(event)
            if tap_layers is None or i in tap_layers:
                bank[event]=grab(out)
            return out
        return hook
    def norm_hook(module,args,out):
        k=len(norm)+1
        event=(k,24)
        if edits and event in edits:
            pos,value=edits[event]
            out=out.clone()
            out[0,pos]=value
        norm.append(grab(out))
        bank[event]=norm[-1]
        return out
    def branch_hook(i,kind):
        def hook(module,args,out):
            k=len(observed)//24+1
            branch[k,i,kind]=grab(out[0] if isinstance(out,tuple) else out)
        return hook
    try:
        for i,m in enumerate(core.layers):
            hooks.append(m.register_forward_hook(layer_hook(i),with_kwargs=True))
            if modules and i in (5,11,17,23):
                hooks.append(m.self_attn.register_forward_hook(branch_hook(i,"attention")))
                hooks.append(m.mlp.register_forward_hook(branch_hook(i,"mlp")))
        hooks.append(core.norm.register_forward_hook(norm_hook))
        yield bank,norm,branch
        assert len(observed)==24*depth and len(norm)==depth
    finally:
        for h in hooks:
            h.remove()
        core.total_ut_steps=previous


def capture(model,e,select,edits=None,full=False,modules=False,depth=4,tap_layers=None):
    coords=select if full else sorted(set(select+[e["read"]]))
    pos=e["read"] if full else coords.index(e["read"])
    x=torch.tensor([e["ids"]],device=model.lm_head.weight.device)
    with native_bank(model,coords,edits,full,modules,depth,tap_layers) as (bank,norm,branches):
        native=model(x,use_cache=False,exit_at_step=depth-1,logits_to_keep=1).logits
    scores,head=depth_scores(model,[t[pos] for t in norm],e["target"])
    assert torch.equal(native[0,0],head[-1])
    return scores,head,bank,branches,coords


def tensor_features(bank,coords,roles,p):
    t=torch.stack(list(bank.values())).reshape(4,25,len(coords),2048).float()
    mask=torch.zeros((len(roles),len(coords)),device=t.device,dtype=torch.float32)
    index={x:i for i,x in enumerate(coords)}
    for i,seq in enumerate(roles.values()):
        mask[i,[index[x] for x in seq]]=1/len(seq)
    pooled=torch.einsum("rp,klpd->klrd",mask,t)
    norm=pooled.norm(dim=-1)
    stats=torch.full((*pooled.shape[:3],4),float("nan"),device=t.device)
    stats[...,0]=norm/np.sqrt(2048)
    stats[:,1:,:,1]=F.cosine_similarity(pooled[:,1:],pooled[:,:-1],dim=-1)
    stats[1:,:,:,2]=F.cosine_similarity(pooled[1:],pooled[:-1],dim=-1)
    stats[1:,:,:,3]=(pooled[1:]-pooled[:-1]).norm(dim=-1)/norm[:-1].clamp_min(1e-12)
    return (pooled @ p).half().cpu().numpy(),stats.cpu().numpy()


def parity(model,e,roles):
    coord=[p for v in roles.values() for p in v]
    scores,heads,bank,_,positions_used=capture(model,e,coord)
    old=fast_forward(model,e)
    assert scores==old[0] and torch.equal(heads,old[4])
    for k in range(1,5):
        a=capture(model,e,[],depth=k,tap_layers=())
        assert torch.equal(heads[k-1],a[1][-1])
    return True


def run(study,model_path,task,pilot):
    groups=[g for g in load_lines(study/"groups.jsonl") if g["task"]==task]
    if pilot:
        groups=groups[:4]
    planned=[(g,name) for g in groups for name in g["capture_worlds"]]
    folder=study/("pilot" if pilot else "natural")/task
    folder.mkdir(parents=True)
    model,tok,runtime=load_runtime(model_path)
    write_json(folder/"runtime.json",runtime)
    p_np=projection()
    p=torch.from_numpy(p_np).cuda()
    features=np.lib.format.open_memmap(folder/"vectors.npy",mode="w+",dtype=np.float16,
                          shape=(len(planned),4,25,len(ROLES),PROJECTION_WIDTH))
    stats=np.lib.format.open_memmap(folder/"stats.npy",mode="w+",dtype=np.float32,
                          shape=(len(planned),4,25,len(ROLES),4))
    start=time.perf_counter()
    verified=False
    with torch.inference_mode(),(folder/"rows.jsonl").open("x",encoding="utf-8") as f:
        for i,(g,name) in enumerate(planned):
            w=g["worlds"][name]
            e=get_encoding(tok,g,w)
            role=positions(g,e)
            if i==0:
                verified=parity(model,e,role)
            selected=[u for v in role.values() for u in v]
            scores,_,bank,_,index=capture(model,e,selected)
            features[i],stats[i]=tensor_features(bank,index,role,p)
            dump_line(f,{"i":i,"id":g["id"],"split":g["split"],"task":task,
                         "family":g.get("family"),"mode":g.get("mode"),"world":name,
                         "gold":w["gold"],"mid":w.get("mid"),"scores":scores,
                         "pred":predictions(scores),"ids":e["ids"],"target_ids":e["target"],
                         "semantic_roles":role,"prompt_sha256":e["prompt_sha256"]})
            if (i+1)%250==0:
                f.flush()
                print(task,i+1,round(time.perf_counter()-start,1),flush=True)
    features.flush()
    stats.flush()
    torch.cuda.synchronize()
    write_json(folder/"complete.json",{"groups":len(groups),"rows":len(planned),
               "input_sha256":digest(study/"groups.jsonl"),"capture_sha256":digest(Path(__file__)),
               "projection_sha256":hashlib.sha256(p_np.tobytes()).hexdigest(),
               "independent_K1_K4_logits_exact":verified,"gpu_phase_seconds":time.perf_counter()-start,
               "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"roles":ROLES})


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("study",type=Path)
    cli.add_argument("--model",type=Path,required=True)
    cli.add_argument("--task",choices=GOALS,required=True)
    cli.add_argument("--pilot",action="store_true")
    a=cli.parse_args()
    run(a.study,a.model,a.task,a.pilot)
