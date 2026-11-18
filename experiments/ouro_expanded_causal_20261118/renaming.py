"""Fresh cross-name transfer test with exact same payload token IDs."""
import argparse
import csv
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from collect import aligned, get_encoding, load_lines, load_runtime, write_json, dump_line, digest
from factory import group, rng_for
from intervene import (DIRECTIONS, source_region, region, rng, encode_bank,
                       instrument, change, l2, predictions)
from run import dose_matched
from analyze import interval

ALIASES=("Nora","Leo","Lena","Miles","June","Milo","Jill","Rose","Anna","Mark",
         "George","Finn","Zoe","Isaac","Sara","Ruth","Eric","Nina")
SITE=(2,5)


def rename(w, mapping):
    matches=list(re.finditer(r"\b(?:"+"|".join(mapping)+r")\b",w["prompt"]))
    length_delta=[(m.end(),len(mapping[m[0]])-len(m[0])) for m in matches]
    def remap(offset):
        assert not any(m.start()<offset<m.end() for m in matches)
        return offset+sum(d for end,d in length_delta if end<=offset)
    roles={k:[(remap(a),remap(b)) for a,b in spans] for k,spans in w["roles"].items()}
    pattern=r"\b(?:"+"|".join(mapping)+r")\b"
    return {**w,"prompt":re.sub(pattern,lambda m:mapping[m[0]],w["prompt"]),
            "probe":re.sub(pattern,lambda m:mapping[m[0]],w["probe"]),
            "roles":roles,"prefix_chars":remap(w["prefix_chars"])}


def create(tok,existing):
    prior={json.dumps(g["meta"]["signature"]) for g in existing if g["task"]=="relation"}
    for i in range(90):
        def accept(g):
            sig=json.dumps(g["meta"]["signature"])
            if sig in prior or not aligned(tok,g):
                return False
            names=g["meta"]["signature"][0]
            for j in range(25):
                aliases=rng_for(f'{g["id"]}/{j}').sample(ALIASES,len(names))
                mapping=dict(zip(names,aliases))
                new={n:rename(g["worlds"][n],mapping) for n in ("r","d","x","y")}
                for n in new:
                    old=get_encoding(tok,g,g["worlds"][n])
                    e=get_encoding(tok,g,new[n])
                    src=source_region(g,old)
                    if (len(old["ids"])!=len(e["ids"]) or old["roles"]!=e["roles"] or
                        any(old["ids"][t]!=e["ids"][t] for t in src)):
                        break
                else:
                    g["aliases"]=mapping
                    g["renamed_worlds"]=new
                    prior.add(sig)
                    return True
            return False
        yield group("relation","renaming",i,None,None,"context",accept)


def evaluate(model,tok,g):
    src_group=g["worlds"]
    for rx,ds,cf,noise,_ in DIRECTIONS:
        e=get_encoding(tok,g,src_group[rx])
        src=source_region(g,e)
        nz=region(e,"noise")
        points=src+nz
        natural=encode_bank(model,e,points,SITE)
        reference=natural[1][SITE]
        rstate=reference[:len(src)]
        nstate=reference[len(src):]
        d=get_encoding(tok,g,src_group[ds])
        dn=get_encoding(tok,g,g["renamed_worlds"][ds])
        old=encode_bank(model,d,src,SITE)
        new=encode_bank(model,dn,src,SITE)
        rnd=torch.randn(rstate.shape,device="cuda",generator=rng(g["id"]+rx))
        v_old=old[1][SITE]; v_new=new[1][SITE]
        dose=l2(v_old.float()-rstate.float())
        noise_delta=encode_bank(model,get_encoding(tok,g,src_group[noise]),nz,SITE)[1][SITE].float()-nstate.float()
        assert l2(noise_delta)>0
        roles={"source":src,"mapping":region(e,"map"),"query_options":region(e,"query_options"),
               "readout":[e["read"]]}
        base=instrument(model,e,roles)
        assert torch.equal(base["head"],natural[4])
        arm_spec={"same_names":(src,rstate,v_old),
                  "new_names":(src,rstate,v_new),
                  "renamed_matched":(src,rstate,dose_matched(rstate,v_new.float()-rstate.float(),dose)),
                  "random":(src,rstate,dose_matched(rstate,rnd,dose)),
                  "irrelevant":(nz,nstate,dose_matched(nstate,noise_delta,dose)),
                  "self":(src,rstate,rstate)}
        arms={}
        for arm,(pos,original,value) in arm_spec.items():
            result=instrument(model,e,roles,{SITE:(pos,value)})
            if arm=="self":
                assert torch.equal(result["head"],base["head"])
            arms[arm]={"scores":result["scores"],"pred":predictions(result["scores"]),
                       "dose":l2(value.float()-original.float())}
            if arm in ("same_names","renamed_matched"):
                arms[arm]["trace"]=change(base,result,roles)
        mid={}
        pe=get_encoding(tok,g,src_group[rx],True)
        pb=encode_bank(model,pe,src,SITE)
        pr=pb[1][SITE]
        mid["native"]=pb[0]
        mid_dose=l2(v_old.float()-pr.float())
        for arm,vector in (("same_names",v_old),("new_names",dose_matched(pr,v_new.float()-pr.float(),mid_dose)),
                           ("random",dose_matched(pr,rnd,mid_dose))):
            mid[arm]=instrument(model,pe,{"source":src,"readout":[pe["read"]]},
                                {SITE:(src,vector)})["scores"]
        yield {"id":g["id"],"family":g["family"],"receive":rx,"donor":ds,"site":SITE,
               "counterfactual":src_group[cf]["gold"],"original":src_group[rx]["gold"],
               "mid_cf":src_group[ds]["mid"],"mid_original":src_group[rx]["mid"],
               "native":base["scores"],"same_donor":old[0],"renamed_donor":new[0],
               "unchanged_payload_ids":[e["ids"][i] for i in src],
               "payload_positions":src,"arms":arms,"mid":mid}


def summarize(out):
    rows=load_lines(out/"rows.jsonl")
    assert len(rows)==180
    populations={"all":rows}
    for i in range(3):
        populations[f"family_{i}"]=[r for r in rows if r["family"]==i]
    modes=(("same_names","native"),("same_names","random"),
           ("renamed_matched","native"),("renamed_matched","random"),
           ("renamed_matched","irrelevant"),("renamed_matched","same_names"))
    with (out/"effects.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f,lineterminator="\n")
        w.writerow(("partition","endpoint","contrast","K","groups","effect","lower","upper"))
        for label,data in populations.items():
            for k in range(4):
                for a,b in modes:
                    values=[]
                    for r in data:
                        o,c=r["original"],r["counterfactual"]
                        left=r["arms"][a]["scores"][k]
                        right=r["native"][k] if b=="native" else r["arms"][b]["scores"][k]
                        values.append(left[c]-left[o]-right[c]+right[o])
                    val=interval(np.array(values).reshape(-1,2).mean(axis=1),1-.05/3 if k==3 else .95)
                    w.writerow((label,"final",f"{a}-minus-{b}",k+1,len(data)//2,*val))
                for a,b in (("same_names","native"),("new_names","random"),("new_names","same_names")):
                    values=[]
                    for r in data:
                        o,c=r["mid_original"],r["mid_cf"]
                        left,right=r["mid"][a][k],r["mid"][b][k]
                        values.append(left[c]-left[o]-right[c]+right[o])
                    val=interval(np.array(values).reshape(-1,2).mean(axis=1),.95)
                    w.writerow((label,"intermediate",f"{a}-minus-{b}",k+1,len(data)//2,*val))
    def key(r):
        o,c=r["original"],r["counterfactual"]
        a=r["arms"]["renamed_matched"]["scores"][3]
        b=r["native"][3]
        return a[c]-a[o]-b[c]+b[o]
    write_json(out/"case.json",sorted(rows,key=key)[len(rows)//2])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("study",type=Path)
    p.add_argument("--model",type=Path)
    p.add_argument("--summarize-only",action="store_true")
    a=p.parse_args()
    out=a.study/"renaming"
    if a.summarize_only:
        summarize(out)
        return
    out.mkdir()
    model,tok,runtime=load_runtime(a.model)
    write_json(out/"runtime.json",runtime)
    groups=list(create(tok,load_lines(a.study/"inputs.jsonl")))
    with (out/"inputs.jsonl").open("w",encoding="utf-8") as f:
        for g in groups:
            dump_line(f,g)
    write_json(out/"protocol.json",{"input_sha256":digest(out/"inputs.jsonl"),"site":SITE,
               "source_sha256":digest(Path(__file__)),"groups":90,
               "operator":"same source token IDs, rewritten entity names; match random and renamed dose"})
    start=time.perf_counter()
    n=0
    with torch.inference_mode(),(out/"rows.jsonl").open("w",encoding="utf-8") as f:
        for g in groups:
            for r in evaluate(model,tok,g):
                dump_line(f,r)
                n+=1
            f.flush()
            if n%30==0:
                print("Renaming",n,round(time.perf_counter()-start,1),flush=True)
    torch.cuda.synchronize()
    write_json(out/"complete.json",{"groups":len(groups),"rows":n,
        "gpu_phase_seconds":time.perf_counter()-start,"peak_allocated_bytes":torch.cuda.max_memory_allocated()})
    summarize(out)


if __name__=="__main__":
    main()
