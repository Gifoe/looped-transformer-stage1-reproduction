"""Full answer likelihood after equal-budget component swaps restricted to the prompt."""
import argparse
from pathlib import Path

import torch

from behavior import answer_tokens
from causal import Edit, centered, copy_heads, direction_like, unit
from native import Forward, Recorder, SEED, load, pad, save, setup, sha, tensors, timer


class PromptEdit(Edit):
    def __init__(self,*args,prefix,**kw):
        super().__init__(*args,**kw)
        self.prefix=prefix

    def attn(self,key,a,v):
        if key!=(0,self.l) or self.kind!="attention":
            return a,v
        n=self.prefix
        if self.arm in ("factor","joint"):
            a=a.clone()
            a[:,self.heads,:n,:n]=self.d["a"][:,self.heads,:n,:n]
        if self.arm in ("content","joint"):
            v=v.clone()
            v[:,self.heads,:n]=self.d["v"][:,self.heads,:n]
        return a,v

    def mlp(self,key,z,r):
        if key!=(0,self.l) or self.kind!="mlp":
            return z
        n=self.prefix
        a=r.data[key]
        act=self.f.layers[self.l].mlp.act_fn
        g=act((self.d if self.arm in ("factor","joint") else a)["g_pre"][:,:n])
        u=(self.d if self.arm in ("content","joint") else a)["u"][:,:n]
        result=z.clone()
        result[:,:n]=g*u
        return result


def score_case(f,tok,row,c):
    begins=[answer_tokens(tok,row["prompt"],v) for v in row["choices"]]
    starts={n for a,n in begins}
    assert len(starts)==1
    n=starts.pop()
    ids,ps=pad([x for x,_ in begins])
    layers={k:c[f"candidate_{k}_layer"]-1 for k in ("attention","mlp")}
    r=Recorder(keys={(k,l) for k in (0,1) for l in layers.values()})
    base=f.run(ids,2,record=r)
    corpus=[(i,j-1) for i,(tokens,start) in enumerate(begins) for j in range(start,len(tokens))]
    idx=torch.tensor(corpus,device=ids.device)
    targets=ids[idx[:,0],idx[:,1]+1]
    sizes=[len(tokens)-start for tokens,start in begins]
    def read(x):
        logits=f.model.lm_head(x[idx[:,0],idx[:,1]]).float()
        scores=logits.log_softmax(-1).gather(-1,targets[:,None]).flatten()
        return [v.tolist() for v in scores.split(sizes)]
    result=dict(native=[read(x) for x in base],tests={})
    for recv,donor in ((0,1),(1,0)):
        state=f.core.embed_tokens(ids) if recv==0 else base[0]
        controls={}
        for mode,l in layers.items():
            cfg=c["heads_zero_based"]
            tgt=c["control_heads_zero_based"]
            out={}
            for arm in ("factor","content","joint"):
                edit=PromptEdit(f,l,mode,arm,r.data[donor,l],cfg,tgt,prefix=n)
                out[arm]=read(f.run(ids,1,start=state,record=Recorder(keys={(0,l)}),edit=edit)[-1])
            controls[mode]=out
        result["tests"][f"{recv+1}from{donor+1}"]=controls
    return dict(id=row["id"],**result)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("blind",type=Path)
    p.add_argument("phase")
    p.add_argument("--limit",type=int,default=24)
    a=p.parse_args()
    c=load(a.out/"CANDIDATES.json")
    pth=a.out/"PREDICTIONS.json"
    assert load(pth)["candidates_sha256"]==sha(a.out/"CANDIDATES.json")
    inputs=load(a.blind)[a.phase][:a.limit]
    model,tok=setup(a.model)
    f=Forward(model)
    rows=[]
    start=timer()
    for item in inputs:
        rows.append(score_case(f,tok,item,c))
        if len(rows)%4==0:
            print("complete candidate patch",a.phase,len(rows),len(inputs),flush=True)
    save(a.out/f"FULL_PATCH_{a.phase}.json",dict(rows=rows,prediction_sha256=sha(pth),
          budget="one recipient's remaining loop, patch restricted to original prompt, gold excluded",
          peak_cuda_bytes=torch.cuda.max_memory_allocated(),seconds=timer()-start))


if __name__=="__main__":
    main()

