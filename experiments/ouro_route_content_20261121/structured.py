"""Transfer the same natural A under two recipient contents, with full-answer scoring."""
import argparse
from pathlib import Path

import torch

from native import Forward, load, save, setup, sha, tensors, timer
from route_ops import HEADS, OTHER, LAYER, Patch, capture, dose, lane_plan, scramble


def item(f,tok,q):
    begin=timer()
    plans=[lane_plan(tok,x[0],q["choices"]) for x in q["receivers"]]
    assert all(len(set(p[1]))==1 for p in plans)
    n=q["prompt_tokens"]
    assert all(p[1][0]==n for p in plans)
    ids,_=tensors(tok,q["donors"])
    _,donor=capture(f,ids)
    a=donor.data[1,LAYER]["a"]
    route=a[:,list(HEADS),:n,:n].split(1)
    other=a[:,list(OTHER),:n,:n].split(1)
    delta=dose(*route)
    shift=min(1.,delta/max(dose(*other),1e-10))
    m=(other[0]+other[1])/2
    other=(m+shift*(other[0]-m),m+shift*(other[1]-m))
    random=scramble(*route)
    oracle=[]; refs=[]; values=[]
    for ci,(ids,lens,read) in enumerate(plans):
        z,rec=capture(f,ids)
        refs.append(z[0])
        values.append(rec.data[1,LAYER]["v"][:,list(HEADS),:n])
        ids1,lens1,read1=lane_plan(tok,q["receivers"][ci][1],q["choices"])
        z1=f.run(ids1,2)
        oracle.append([[read(f.model,u) for u in z],[read1(f.model,u) for u in z1]])
    def arms(take,heads,joint):
        outcomes=[]
        for c in (0,1):
            prefix=q["anchor"] if joint else c
            ids,lens,read=plans[prefix]
            vv=values[c] if joint else None
            z=[f.run(ids,1,start=refs[prefix],edit=Patch(lens,a=ax,v=vv,heads=heads))[0] for ax in take]
            outcomes.append([read(f.model,u) for u in z])
        return outcomes
    result={
        "natural_receivers":arms(route,HEADS,False),
        "native_values":arms(route,HEADS,True),
        "scramble_receivers":arms(random,HEADS,False),
        "scramble_values":arms(random,HEADS,True),
        "other_heads_receivers":arms(other,OTHER,False)
    }
    maxsum=max(float((x.sum(-1)-1).abs().max()) for x in (*route,*random,*other))
    assert maxsum<1e-5
    return dict(id=q["id"],layout=q["layout"],oracle=oracle,conditions=result,dose=delta,
                scramble_dose=dose(*random),other_dose=dose(*other),max_row_sum_error=maxsum,
                seconds=timer()-begin)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("manifest",type=Path)
    p.add_argument("works",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("phase")
    a=p.parse_args()
    pred=load(a.works)
    assert pred["manifest_sha256"]==sha(a.manifest)
    data=[x for x in load(a.manifest)["rows"] if x["phase"]==a.phase]
    model,tok=setup(a.model)
    f=Forward(model)
    rows=load(a.out)["rows"] if a.out.exists() else []
    assert [q["id"] for q in rows]==[q["id"] for q in data[:len(rows)]]
    for q in data[len(rows):]:
        rows.append(item(f,tok,{k:v for k,v in q.items() if k not in ("targets","donor_targets")}))
        if len(rows)%4==0 or len(rows)==len(data):
            save(a.out,dict(expected=len(data),prediction_sha256=sha(a.works),rows=rows,
                            peak_bytes=torch.cuda.max_memory_allocated()))
            print(a.phase,len(rows),len(data),flush=True)


if __name__=="__main__":
    main()

