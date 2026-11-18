"""Recheck the earlier source-to-mapping claim under independent answer slots."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze_org import export, group_ci
from manifest_org import get_encoding, load_lines, digest, dump_line, write_json
from collect import load_runtime, predictions
from intervene import encode_bank, instrument, region, source_region
from run import dose_matched


def evaluate(model,tok,g):
    worlds={n:get_encoding(tok,g,w) for n,w in g["worlds"].items()}
    site=(1,5)
    for r,d,alt in (("r","d","x"),("d","r","y")):
        e=worlds[r]
        src=source_region(g,e)
        mapping=region(e,"map")
        roles={"source":src,"mapping":mapping,"query":region(e,"query_options"),"readout":[e["read"]]}
        initial=encode_bank(model,e,src,site)
        old=initial[1][site]
        new=encode_bank(model,worlds[d],src,site)
        v=new[1][site]
        end={}
        def run(route,edit):
            return instrument(model,e,roles,edit,route,(2,))
        open_state={}
        controls={"none":{}, "semantic":{site:(src,v)}}
        for arm,route in (("open",{}),("cut",{2:[(mapping,src)]})):
            results={key:run(route,edits) for key,edits in controls.items()}
            end[arm]={key:value["scores"] for key,value in results.items()}
            if arm=="open":
                open_state=results
                assert torch.equal(initial[4],results["none"]["head"])
            else:
                restoration={}
                for key,edits in controls.items():
                    value=open_state[key]["norm"][2][mapping]
                    restoration[key]=run(route,{**edits,(2,"norm"):(mapping,value)})
                end["restore"]={key:z["scores"] for key,z in restoration.items()}
        seed=int.from_bytes(hashlib.sha256(f"route/{g['id']}/{r}".encode()).digest()[:8],"big")
        z=torch.randn(old.shape,device="cuda",generator=torch.Generator(device="cuda").manual_seed(seed))
        dose=float((v.float()-old.float()).norm())
        random=dose_matched(old,z,dose)
        self_result=run({},{site:(src,old)})
        assert torch.equal(self_result["head"],initial[4])
        end["random"]={"none":initial[0],"semantic":run({},{site:(src,random)})["scores"]}
        target=g["worlds"][alt]["gold"]
        native=g["worlds"][r]["gold"]
        assert native==g["worlds"][d]["gold"]!=target
        yield {"id":g["id"],"split":g["split"],"receive":r,"donor":d,
               "original":native,"counterfactual":target,"native":initial[0],
               "donor_native":new[0],"site":site,"source_positions":src,
               "mapping_positions":mapping,"arms":end,"self_exact":True,"patch_l2":dose}


def main(root,modeldir):
    out=root/"route_retest"
    out.mkdir()
    groups=[g for g in load_lines(root/"groups.jsonl") if g["task"]=="relation"]
    cohort=[]
    for split,n,candidate_n in (("confirm",40,60),("transfer",20,30)):
        rs=[g for g in groups if g["split"]==split]
        previous=sorted(rs,key=lambda g:hashlib.sha256(("causal-org/"+g["id"]).encode()).digest())[:candidate_n]
        old={g["id"] for g in previous}
        remain=[g for g in rs if g["id"] not in old]
        cohort+=sorted(remain,key=lambda g:hashlib.sha256(("route-retest/"+g["id"]).encode()).digest())[:n]
    write_json(out/"protocol.json",{"groups":len(cohort),"ids":[g["id"] for g in cohort],
        "input_sha256":digest(root/"groups.jsonl"),"operator":"K1 source injection, K2 source→mapping cut and postnorm mapping restoration",
        "selection":"40 confirm + 20 transfer groups disjoint from the decoder intervention"})
    model,tok,rt=load_runtime(modeldir)
    write_json(out/"runtime.json",rt)
    rows=[]
    start=time.perf_counter()
    with torch.inference_mode(),(out/"rows.jsonl").open("x",encoding="utf-8") as file:
        for g in cohort:
            for row in evaluate(model,tok,g):
                rows.append(row)
                dump_line(file,row)
    torch.cuda.synchronize()
    write_json(out/"complete.json",{"groups":len(cohort),"directions":len(rows),
        "gpu_phase_seconds":time.perf_counter()-start,"peak_allocated_bytes":torch.cuda.max_memory_allocated()})
    def get(row,arm,k):
        q=row["arms"][arm]
        c,o=row["counterfactual"],row["original"]
        return sum(sign*(q[key][k][c]-q[key][k][o]) for key,sign in (("semantic",1),("none",-1)))
    stats=[]
    for k in range(4):
        for a,b in (("open",None),("open","cut"),("restore","cut"),("open","random")):
            v=np.array([get(r,a,k)-(get(r,b,k) if b else 0) for r in rows]).reshape(-1,2).mean(1)
            m,lo,hi=group_ci(v)
            stats.append(dict(K=k+1,arm=a,reference=b or "native",groups=len(v),effect=m,lower=lo,upper=hi))
    export(out/"effects.csv",stats)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("study",type=Path)
    p.add_argument("--model",type=Path,required=True)
    a=p.parse_args()
    main(a.study,a.model)
