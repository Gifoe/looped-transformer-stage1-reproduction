"""Native factor interactions at the state of a naturally successful or unsuccessful K2."""
import argparse
from pathlib import Path

import torch

from native import Forward, Recorder, load, save, setup, sha, timer
from route_ops import HEADS, LAYER, Patch, lane_plan, norms


class MLP(Patch):
    def __init__(self,lengths,f,gate=None,up=None):
        super().__init__(lengths)
        self.gate,self.up,self.f=gate,up,f

    def mlp(self,key,z,trace):
        if key!=(0,0):
            return z
        d=trace.data[key]
        v=z.clone()
        for lane,n in enumerate(self.lengths):
            a=self.gate if self.gate is not None else d["g_pre"]
            b=self.up if self.up is not None else d["u"]
            v[lane,:n]=self.f.layers[0].mlp.act_fn(a[lane,:n])*b[lane,:n]
        return v


def case(f,tok,q):
    start=timer()
    ids,lens,read=lane_plan(tok,q["prompt"],q["choices"])
    n=lens[0]
    old=Recorder(keys={(k,l) for k in (0,1) for l in (0,LAYER)})
    y=f.run(ids,2,record=old)
    drop=f.run(ids,2,edit=Patch(lens,write=1.))
    ans={}
    def norm_scores(h,ref):
        return dict(ll=read(f.model,h),ratio=norms(h,ref,lens))
    for name,head in (("L6_four_heads",HEADS),("L6_all_heads",list(range(16))),("producer_deleted",HEADS)):
        deficit=name=="producer_deleted"
        anchor=drop if deficit else y
        other=old.data[1 if deficit else 0,LAYER]
        a=other["a"][:,list(head),:n,:n]
        v=other["v"][:,list(head),:n,:]
        z={"base":norm_scores(anchor[1],y[1])}
        for tag,aa,vv in (("first",a,None),("second",None,v),("both",a,v)):
            h=f.run(ids,1,start=anchor[0],edit=Patch(lens,a=aa,v=vv,heads=head))[0]
            z[tag]=norm_scores(h,y[1])
        ans[name]=z
    l=old.data[0,0]
    buf={}
    for tag,g,u in (("first",l["g_pre"],None),("second",None,l["u"]),("both",l["g_pre"],l["u"])):
        h=f.run(ids,1,start=y[0],edit=MLP(lens,f,g,u),record=Recorder(keys={(0,0)}))[0]
        buf[tag]=norm_scores(h,y[1])
    buf["base"]=norm_scores(y[1],y[1])
    ans["L1_mlp"]=buf
    return dict(id=q["id"],ll_native=[read(f.model,h) for h in y],factorials=ans,seconds=timer()-start)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("manifest",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("family")
    a=p.parse_args()
    data=load(a.manifest)["cohorts"][a.family]["rows"]
    model,tok=setup(a.model)
    f=Forward(model)
    rows=load(a.out)["rows"] if a.out.exists() else []
    assert [q["id"] for q in rows]==[q["id"] for q in data[:len(rows)]]
    for q in data[len(rows):]:
        rows.append(case(f,tok,{k:v for k,v in q.items() if k!="gold"}))
        if len(rows)%4==0 or len(rows)==len(data):
            save(a.out,dict(expected=len(data),manifest_sha256=sha(a.manifest),rows=rows,
                            peak_bytes=torch.cuda.max_memory_allocated()))
            print(a.family,len(rows),len(data),flush=True)


if __name__=="__main__":
    main()

