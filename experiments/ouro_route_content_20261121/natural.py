"""Native correction, exact rollback and downstream-route rescue on uncensored fresh cohorts."""
import argparse
from pathlib import Path

import torch

from native import Forward, load, save, sha, setup, timer
from route_ops import HEADS, OTHER, LAYER, Patch, capture, dose, lane_plan, norms, prefix_pos0, scramble, sink_mix


def row(f,tok,q):
    t=timer()
    ids,lengths,read=lane_plan(tok,q["prompt"],q["choices"])
    n=lengths[0]
    native,r=capture(f,ids)
    early=r.data[0,LAYER]["a"][:1,list(HEADS),:n,:n]
    late=r.data[1,LAYER]["a"][:1,list(HEADS),:n,:n]
    early_o=r.data[0,LAYER]["a"][:1,list(OTHER),:n,:n]
    late_o=r.data[1,LAYER]["a"][:1,list(OTHER),:n,:n]
    rng,sym=scramble(early,late)
    controls={"rollback":early,"position0":sink_mix(late,early,"position0"),
              "rest":sink_mix(late,early,"rest"),"random_first":rng,"random_second":sym}
    arms={}
    for name,a in controls.items():
        z=f.run(ids,1,start=native[0],edit=Patch(lengths,a=a))[0]
        arms[name]=dict(ll=read(f.model,z),dose=dose(a,late),norm_ratio=norms(z,native[1],lengths))
    scale=min(1.,dose(early,late)/max(dose(early_o,late_o),1e-12))
    z=f.run(ids,1,start=native[0],edit=Patch(lengths,a=late_o+scale*(early_o-late_o),heads=OTHER))[0]
    arms["other_heads"]=dict(ll=read(f.model,z),dose=dose(scale*(early_o-late_o),torch.zeros_like(early_o)),
                             norm_ratio=norms(z,native[1],lengths))
    # Reciprocal intervention at K1 has exactly K1's remaining computation.
    z=f.run(ids,1,edit=Patch(lengths,a=late))[0]
    arms["early_add"]=dict(ll=read(f.model,z),dose=dose(early,late),norm_ratio=norms(z,native[0],lengths))
    drop,rr=capture(f,ids,edit=Patch(lengths,write=1.))
    cut=rr.data[1,LAYER]["a"][:1,list(HEADS),:n,:n]
    alt=rr.data[1,LAYER]["a"][:1,list(OTHER),:n,:n]
    control_alpha=min(1.,dose(cut,late)/max(dose(alt,late_o),1e-12))
    place={
        "restore":(late,HEADS),
        "restore_pos0":(sink_mix(cut,late,"position0"),HEADS),
        "restore_rest":(sink_mix(cut,late,"rest"),HEADS),
        "restore_other":(alt+control_alpha*(late_o-alt),OTHER)
    }
    for name,(a,which) in place.items():
        z=f.run(ids,1,start=drop[0],edit=Patch(lengths,a=a,heads=which))[0]
        arms[name]=dict(ll=read(f.model,z),dose=dose(a,cut if which==HEADS else alt),norm_ratio=norms(z,native[1],lengths))
    arms["drop"]=dict(ll=read(f.model,drop[1]),dose=dose(cut,late),norm_ratio=norms(drop[1],native[1],lengths))
    for name,d,method in (("half_drop",.5,None),("matched_writer",1.,"permute")):
        z=f.run(ids,2,edit=Patch(lengths,write=d,writer_control=method))[1]
        arms[name]=dict(ll=read(f.model,z),norm_ratio=norms(z,native[1],lengths))
    forced=f.run(ids,1,start=native[0],edit=Patch(lengths,a=late))[0]
    parity=float((forced-native[1]).abs().max())
    assert parity<5e-5,(q["id"],parity)
    return dict(id=q["id"],native=[read(f.model,h) for h in native],arms=arms,
                mean_position0=[prefix_pos0(a,n) for a in (early,late,cut)],same_pattern_parity=parity,
                seconds=timer()-t)


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
    out=load(a.out)["rows"] if a.out.exists() else []
    assert [q["id"] for q in out]==[q["id"] for q in data[:len(out)]]
    for q in data[len(out):]:
        out.append(row(f,tok,{k:v for k,v in q.items() if k!="gold"}))
        if len(out)%4==0 or len(out)==len(data):
            save(a.out,dict(expected=len(data),manifest_sha256=sha(a.manifest),rows=out,
                            peak_bytes=torch.cuda.max_memory_allocated()))
            print(a.family,len(out),len(data),flush=True)


if __name__=="__main__":
    main()

