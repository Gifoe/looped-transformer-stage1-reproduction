"""Use original recipient rules to test what a frozen transferred factor carries."""
import argparse
from pathlib import Path

import torch

from causal import swap_once
from native import Forward, Recorder, load, output, save, setup, sha, tensors


def run_group(f,tok,group,config):
    rec=group["worlds"][0]
    pool={"source":group["worlds"][1],"rule":group["worlds"][2],"joint":group["worlds"][3],**group["extra"]}
    ids,pos=tensors(tok,[rec["prompt"]])
    al=config["candidate_attention_layer"]-1
    ml=config["candidate_mlp_layer"]-1
    a=Recorder(keys={(1,x) for x in (al,ml)})
    early,actual=f.run(ids,2,record=a)
    indices=rec["candidate_ids"]
    scores=output(f.model,actual,pos)[0,indices].tolist()
    truth={}
    for tag,donor in pool.items():
        donor_ids,donor_pos=tensors(tok,[donor["prompt"]])
        assert ids.shape==donor_ids.shape and donor["candidate_ids"]==indices
        r=Recorder(keys={(1,x) for x in (al,ml)})
        z=f.run(donor_ids,2,record=r)[-1]
        combined=Recorder()
        combined.data={(0,x):a.data[1,x] for x in (al,ml)}|{(1,x):r.data[1,x] for x in (al,ml)}
        arms={}
        for name,l in (("attention",al),("mlp",ml)):
            vv={}
            for q in ("factor","content","joint","random"):
                y,dose=swap_once(f,ids,early,pos,combined,0,1,name,l,q,config,None)
                vv[q]=dict(scores=output(f.model,y,pos)[0,indices].tolist(),dose=dose)
            arms[name]=vv
        truth[tag]=dict(native_donor=output(f.model,z,donor_pos)[0,indices].tolist(),arms=arms)
    return dict(group=group["group"],family=group["family"],native_recipient=scores,donors=truth)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("groups",type=Path)
    a=p.parse_args()
    groups=load(a.groups)
    frozen=a.out/"SEMANTIC_PREDICTIONS.json"
    save(frozen,dict(candidate_sha256=sha(a.out/"CANDIDATES.json"),design_sha256=sha(a.groups),
                targets="predefined from same bridge/different downstream answer, same answer/different bridge, and independently matched worlds",
                data_access="label-bearing world manifest is a separate scorer; run_group reads no gold or bridge"))
    c=load(a.out/"CANDIDATES.json")
    model,tok=setup(a.model)
    f=Forward(model)
    rows=[]
    for group in groups:
        rows.append(run_group(f,tok,group,c))
        if len(rows)%4==0:
            print("semantic group",len(rows),len(groups),flush=True)
    save(a.out/"SEMANTIC_NATIVE.json",dict(rows=rows,prediction_sha256=sha(frozen),
          peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=="__main__":
    main()

