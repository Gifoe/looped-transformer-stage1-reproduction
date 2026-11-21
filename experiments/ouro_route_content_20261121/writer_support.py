"""Separate support of the K1 writer that forms the K2 position-zero attention mass."""
import argparse
from pathlib import Path

import torch

from native import Forward, load, save, setup, sha, timer
from route_ops import HEADS, LAYER, Patch, capture, lane_plan, norms


class SliceWrite(Patch):
    def __init__(self,lengths,kind,control=False):
        super().__init__(lengths)
        self.kind,self.randomized=kind,control
        self.norms=[]

    def write(self,key,name,x):
        if key!=(0,0) or name!="a_write":
            return x
        z=x.clone()
        for lane,n in enumerate(self.lengths):
            seg={"pos0":slice(0,1),"rest":slice(1,n),"all":slice(None,n)}[self.kind]
            a=x[lane,seg]
            self.norms.append(float(a.norm()))
            z[lane,seg]=a-a.roll(37,dims=-1) if self.randomized else 0.
        return z


def score(f,tok,q,prompt):
    ids,lens,read=lane_plan(tok,prompt,q["choices"])
    y,r=capture(f,ids)
    n=lens[0]
    early=r.data[0,LAYER]["a"][0,list(HEADS),:n,:n]
    late=r.data[1,LAYER]["a"][0,list(HEADS),:n,:n]
    d=late-early
    def mass(a):
        return float(a[:,:,0].mean())
    result={}
    suite=[(name,False) for name in ("pos0","rest","all")]+[(name,True) for name in ("pos0","rest")]
    for k,random in suite:
        edit=SliceWrite(lens,k,random)
        z,tr=capture(f,ids,edit=edit)
        att=tr.data[1,LAYER]["a"][0,list(HEADS),:n,:n]
        diff=float(((late-att)*d).sum()/d.square().sum().clamp_min(1e-12))
        result[k+("_permuted" if random else "")]=dict(
            ll=read(f.model,z[-1]),mass=mass(att),factor_loss=diff,
            norm_ratio=norms(z[-1],y[-1],lens),write_norm=edit.norms[0])
    return dict(native=[read(f.model,h) for h in y],mass=[mass(early),mass(late)],arms=result)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("manifest",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("family")
    a=p.parse_args()
    data=load(a.manifest)["cohorts"][a.family]["rows"][:128]
    model,tok=setup(a.model)
    f=Forward(model)
    rows=load(a.out)["rows"] if a.out.exists() else []
    assert [q["id"] for q in rows]==[q["id"] for q in data[:len(rows)]]
    for i,q in enumerate(data[len(rows):],start=len(rows)):
        t=timer()
        q={k:v for k,v in q.items() if k!="gold"}
        old=q["prompt"]
        alt=old.replace("Question:","Problem:",1)
        a0,a1=[tok.encode(z,add_special_tokens=False) for z in (old,alt)]
        ok=i<48 and len(a0)==len(a1) and a0[0]!=a1[0] and a0[1:]==a1[1:]
        rows.append(dict(id=q["id"],standard=score(f,tok,q,old),
                         first_token_rewrite=score(f,tok,q,alt) if ok else None,seconds=timer()-t))
        if len(rows)%4==0 or len(rows)==len(data):
            save(a.out,dict(expected=len(data),manifest_sha256=sha(a.manifest),rows=rows,
                            peak_bytes=torch.cuda.max_memory_allocated()))
            print(a.family,len(rows),len(data),flush=True)


if __name__=="__main__":
    main()

