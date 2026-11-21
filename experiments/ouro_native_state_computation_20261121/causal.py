"""Finite state perturbations and equal-suffix native factorial swaps."""
import argparse
from pathlib import Path

import torch

from native import Forward, Recorder, SEED, flatnorm, load, output, save, setup, sha, tensors, timer


def copy_heads(to, donor, heads):
    a=to.clone()
    a[:,heads]=donor[:,heads]
    return a


def att_output(a,v):
    return a@v


def direction_like(x,seed):
    return torch.randn(x.shape,dtype=x.dtype,device=x.device,
                       generator=torch.Generator(device=x.device).manual_seed(seed))


def unit(a,ref):
    return a*ref.float().norm()/a.float().norm().clamp_min(1e-12)


def projection(y,origin,target):
    delta=(target-origin).float()
    return float(((y-origin).float()*delta).sum()/delta.square().sum().clamp_min(1e-12))


def centered(y,origin,target):
    clean=lambda x:x-x.mean(dim=-1,keepdim=True)
    return projection(clean(y),clean(origin),clean(target))


def vector_stats(y,ref):
    y,ref=y.float(),ref.float()
    inner=(y*ref).sum()
    return dict(ratio=float(y.norm()/ref.norm().clamp_min(1e-12)),
          cosine=float(inner/(y.norm()*ref.norm()).clamp_min(1e-12)),
          rms=float(y.square().mean().sqrt()))


def source_tokens(tok,row):
    if "choices" not in row:
        return []
    return [tok.encode(" "+x,add_special_tokens=False)[0] for x in row["choices"]]


class Edit:
    def __init__(self,f,l,kind,arm,donor,heads,other,seed=SEED):
        self.f,self.l,self.kind,self.arm=f,l,kind,arm
        self.d,self.heads,self.other,self.seed=donor,heads,other,seed
        self.delta_norm=0.

    def resid(self,k,l,x):
        return x

    def attn(self,key,a,v):
        if key!=(0,self.l) or self.kind!="attention":
            return a,v
        arm=self.arm
        other=arm=="other_heads"
        heads=self.other if other else self.heads
        pa=copy_heads(a,self.d["a"],heads)
        pv=copy_heads(v,self.d["v"],heads)
        if arm=="factor":
            self.delta_norm=float(((pa-a)@v).norm())
            return pa,v
        if arm=="content":
            self.delta_norm=float((a@(pv-v)).norm())
            return a,pv
        if arm=="joint":
            self.delta_norm=float(((pa@pv)-(a@v)).norm())
            return pa,pv
        # Matched controls run at o_proj's actual input via a temporary hook.
        return a,v

    def mlp(self,key,z,r):
        if key!=(0,self.l) or self.kind!="mlp":
            return z
        live=r.data[key]
        act=self.f.layers[self.l].mlp.act_fn
        g,dg=act(live["g_pre"]),act(self.d["g_pre"])
        u,du=live["u"],self.d["u"]
        ref=dg*u-z
        if self.arm=="factor":
            y=dg*u
        elif self.arm=="content":
            y=g*du
        elif self.arm=="joint":
            y=dg*du
        else:
            noise=direction_like(ref,self.seed)
            if self.arm=="other_tokens":
                noise=ref.roll(1,dims=1)
            y=z+unit(noise,ref)
        self.delta_norm=float((y-z).norm())
        return y


def swap_once(f,ids,start,pos,rec,recv,donor,kind,l,arm,c,ci):
    h=c["heads_zero_based"]
    o=c["control_heads_zero_based"]
    anchor=rec.data[recv,l]
    source=rec.data[donor,l]
    edit=Edit(f,l,kind,arm,source,h,o)
    hook=None
    if kind=="attention" and arm in ("random","other_heads","other_tokens"):
        a,v=anchor["a"],anchor["v"]
        ref=(copy_heads(a,source["a"],h)-a)@v
        if arm=="random":
            z=torch.zeros_like(ref)
            z[:,h]=direction_like(ref[:,h],SEED)
        elif arm=="other_heads":
            z=(copy_heads(a,source["a"],o)-a)@v
        else:
            z=ref.roll(1,dims=2)
        z=unit(z,ref).transpose(1,2).flatten(2)
        edit.delta_norm=float(z.norm())
        hook=f.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda _,a:(a[0]+z,))
    try:
        y=f.run(ids,1,start=start,record=Recorder(keys={(0,l)}),edit=edit)[-1]
    finally:
        if hook is not None:
            hook.remove()
    return y,edit.delta_norm


def same_perturbation(f,ids,pos,starts,clean,rec,l,row,tok):
    h=starts[0]
    d=torch.zeros_like(h)
    center=(ids.shape[-1]-1)//2
    d[:,center]=direction_like(h[:,center],SEED)
    amount=.12*float(h[:,center].norm())
    d=unit(d,h.new_tensor(amount))
    rew,_=tensors(tok,[row.get("rewrite",row["prompt"])])
    ds={"fixed":d}
    if rew.shape==ids.shape and not torch.equal(rew,ids):
        other=f.run(rew,1)[-1]
        natural=(other-starts[1]).clone()
        natural[:,center+1:]=0
        if float(natural.norm())>1e-8:
            ds["paired_rewrite"]=unit(natural,d)
    result={}
    for name,delta in ds.items():
        states={}
        vectors=[]
        for k,(base,native) in enumerate(zip(starts,clean,strict=True)):
            traced=Recorder(keys={(0,l)})
            z=f.run(ids,1,start=base+delta,record=traced)[-1]
            diff=z-native
            vectors.append(diff)
            ref=rec.data[k,l]
            new=traced.data[0,l]
            state_score=output(f.model,z,pos)-output(f.model,native,pos)
            states[str(k+1)]=dict(gain=float(diff.norm()/delta.norm().clamp_min(1e-12)),rms=float(base.square().mean().sqrt()),
                token_response=diff.norm(dim=-1)[0].tolist(),vocab_logit_response_rms=float(state_score.square().mean().sqrt()),
                attention_change=float((new["a"]-ref["a"]).square().mean().sqrt()),
                mlp_change=float((new["z"]-ref["z"]).square().mean().sqrt()))
        for k in range(1,len(vectors)):
            states[str(k+1)]["vs_previous_response"]=vector_stats(vectors[k],vectors[k-1])
        result[name]=dict(dose=float(delta.norm()),token=center,response=states)
    return result


def case(f,tok,row,c,pairs):
    ids,pos=tensors(tok,[row["prompt"]])
    al=c["candidate_attention_layer"]-1
    ml=c["candidate_mlp_layer"]-1
    layers={al,ml,c["attention_control_layer"]-1,c["mlp_control_layer"]-1}
    trace=Recorder(keys={(k,l) for k in range(4) for l in layers})
    t=timer()
    clean=f.run(ids,4,record=trace)
    starts=[f.core.embed_tokens(ids),*clean[:-1]]
    tokens=source_tokens(tok,row)
    logits=[output(f.model,x,pos) for x in clean]
    response=same_perturbation(f,ids,pos,starts,clean,trace,al,row,tok)
    tests={}
    for lo,hi in pairs:
        for recv,donor in ((lo,hi),(hi,lo)):
            out={}
            origin=clean[recv]
            target=clean[donor]
            lg0,lg1=logits[recv],logits[donor]
            for kind,l in (("attention",al),("mlp",ml)):
                arms={}
                control=c[f"{kind}_control_layer"]-1
                interventions=[("factor",l,"factor"),("content",l,"content"),("joint",l,"joint"),
                               ("random",l,"random"),("other_tokens",l,"other_tokens"),("other_layer",control,"factor")]
                if kind=="attention":
                    interventions.append(("other_heads",l,"other_heads"))
                for name,layer,arm in interventions:
                    y,dose=swap_once(f,ids,starts[recv],pos,trace,recv,donor,kind,layer,arm,c,None)
                    z=output(f.model,y,pos)
                    arms[name]=dict(readout_progress=projection(z,lg0,lg1),centered_progress=centered(z,lg0,lg1),
                        state_progress=projection(y,origin,target),top_token=int(z.argmax()),dose=dose,
                        rank_tokens=z[0,tokens].tolist(),maximum_state_ratio=float((y.norm(dim=-1)/origin.norm(dim=-1).clamp_min(1e-9)).max()),
                        delta_rms=float((y-origin).square().mean().sqrt()),
                        target_token_effect=float((z-lg0)[0,lg1.argmax()]))
                out[kind]=arms
            tests[f"{recv+1}from{donor+1}"]=out
    return dict(id=row["id"],family=row["family"],response=response,tests=tests,
                native_first=[z[0,tokens].tolist() for z in logits],candidate_first_ids=tokens,
                native_top_full_vocab=[int(z.argmax()) for z in logits],
                native_state_rms=[float(z.square().mean().sqrt()) for z in clean],gpu_seconds=timer()-t)


def predict(root,inputs):
    c=load(root/"CANDIDATES.json")
    dev=load(root/"CAUSAL_dev.json")["rows"]
    d={}
    for arm in ("1from2","2from1"):
        d[arm]={m:sum(row["tests"][arm][m]["factor"]["centered_progress"] for row in dev)/len(dev)
                for m in ("attention","mlp")}
    save(root/"PREDICTIONS.json",dict(training_phase="development, no labels used",
         expected_positive_progress=d,candidates_sha256=sha(root/"CANDIDATES.json"),
         no_data_refit=True,
         items={k:[r["id"] for r in inputs[k]] for k in ("confirm_arc","confirm_external","controlled")}))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("manifest",type=Path)
    p.add_argument("phase")
    p.add_argument("--limit",type=int)
    a=p.parse_args()
    data=load(a.manifest)
    if a.phase=="predict":
        predict(a.out,data)
        return
    c=load(a.out/"CANDIDATES.json")
    if a.phase not in ("pilot","dev"):
        pred=load(a.out/"PREDICTIONS.json")
        assert pred["candidates_sha256"]==sha(a.out/"CANDIDATES.json")
        assert pred["items"][a.phase]==[row["id"] for row in data[a.phase]]
    rows=data[a.phase][:a.limit]
    model,tok=setup(a.model)
    f=Forward(model)
    dest=a.out/f"CAUSAL_{a.phase}.json"
    exists=load(dest)["rows"] if dest.exists() else []
    pairs=[(0,1),(1,2),(2,3)] if a.phase not in ("pilot","dev") else [(0,1)]
    for i,row in enumerate(rows[len(exists):],start=len(exists)):
        exists.append(case(f,tok,row,c,pairs))
        if (i+1)%4==0 or i+1==len(rows):
            save(dest,dict(rows=exists,expected=len(rows),candidate_sha256=sha(a.out/"CANDIDATES.json"),
                        prediction_sha256=sha(a.out/"PREDICTIONS.json") if a.phase not in ("pilot","dev") else None,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated()))
            print(a.phase,i+1,len(rows),flush=True)


if __name__=="__main__":
    main()

