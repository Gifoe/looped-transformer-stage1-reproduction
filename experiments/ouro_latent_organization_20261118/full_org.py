"""Preserve full-token BF16 states for predeclared outcome strata."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze_org import TaskData, performance, ridge, ROLES, TAPS
from capture_org import capture, positions, projection, tensor_features
from manifest_org import GOALS,TASKS,get_encoding,load_lines,write_json,digest
from collect import load_runtime


def choose(data):
    meta=performance(data)
    split=("confirm","transfer") if data.task in TASKS else ("benchmark",)
    available=[r for r in meta if r["split"] in split]
    def order(r):
        return hashlib.sha256(("org-full-token/"+r["id"]).encode()).digest()
    if data.task not in TASKS:
        return sorted(available,key=order)[:1]
    result=[]
    for outcome in ("always_correct","always_wrong","corrected","regressed","transient"):
        for s in split:
            rr=sorted([r for r in available if r["trajectory"]==outcome and r["split"]==s],key=order)
            if rr:
                result.append(rr[0])
    return result


def save_bits(x):
    return x.contiguous().view(torch.int16).cpu().numpy()


def main(study,model_path):
    out=study/"full_token"
    out.mkdir()
    groups={g["id"]:g for g in load_lines(study/"groups.jsonl")}
    frozen=json.loads((study/"analysis_freeze.json").read_text())
    model,tok,runtime=load_runtime(model_path)
    write_json(out/"runtime.json",runtime)
    p=torch.from_numpy(projection()).cuda()
    rank=0
    catalog=[]
    arrays=[]
    start=time.perf_counter()
    with torch.inference_mode():
        for task in GOALS:
            data=TaskData(study,task,groups)
            chosen=choose(data)
            if task in TASKS:
                fit=data.data_mask("fit")
                target=data.labels("intermediate")
                j=ROLES.index(frozen["selected_for_causality"][task]["role"])
                marginal=np.zeros((len(chosen),4,25),dtype=np.float32)
                for k in range(4):
                    for l in range(25):
                        x=data.feature(k,l,j)
                        w=ridge(x[fit],target.y[fit])
                        all_scores=target.signed_margin(w,x)
                        marginal[:,k,l]=all_scores[[r["index"] for r in chosen]]
            else:
                marginal=np.full((len(chosen),4,25),np.nan,np.float32)
                j=ROLES.index("readout")
            for i,r in enumerate(chosen):
                group=groups[r["id"]]
                e=get_encoding(tok,group,group["worlds"]["r"])
                role=positions(group,e)
                scores,head,x,branch,coords=capture(model,e,[],full=True,modules=True)
                old=data.rows[r["index"]]
                assert scores==old["scores"]
                n=len(e["ids"])
                full=torch.stack(list(x.values())).reshape(4,25,n,2048)
                branch_bits=save_bits(torch.stack(list(branch.values())))
                stem=f"{rank:02}_{task}_{r['trajectory']}"
                file=out/f"{stem}.npz"
                np.savez(file,state_bf16_bits=save_bits(full),branch_bf16_bits=branch_bits,
                         branch_coordinates=np.array([":".join(map(str,x)) for x in branch]),
                         input_ids=np.array(e["ids"],np.int32))
                del full,branch_bits
                _,summary=tensor_features(x,list(range(n)),role,p)
                arrays.append((summary,marginal[i],np.array(scores,dtype=np.float32)))
                item={**r,"file":file.name,"sha256":digest(file),"length":n,
                      "annotation_role":ROLES[j],"role_tokens":role,
                      "score_parity":True,"choices":group["options"],"gold":old["gold"],
                      "intermediate":group["worlds"]["r"].get("semantic",{}).get("intermediate")}
                catalog.append(item)
                rank+=1
                print(stem,n,round(time.perf_counter()-start,1),flush=True)
    np.savez_compressed(out/"figure_values.npz",
          geometry=np.stack([a[0] for a in arrays]),mid_margin=np.stack([a[1] for a in arrays]))
    write_json(out/"catalog.json",catalog)
    write_json(out/"complete.json",{"cases":rank,"gpu_phase_seconds":time.perf_counter()-start,
                   "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                   "raw_storage_bytes":sum((out/c["file"]).stat().st_size for c in catalog)})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("study",type=Path)
    p.add_argument("--model",type=Path,required=True)
    a=p.parse_args()
    main(a.study,a.model)
