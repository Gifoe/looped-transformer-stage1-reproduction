"""Extend the frozen fourth round with new structural worlds and one extra loop."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
EMBED=HERE.parent/"ouro_continuous_embedding_20261120"
CONT=HERE.parent/"ouro_continuous_latent_20261120"
DESIGN=HERE.parent/"ouro_hidden_computation_20261120"
sys.path.insert(0,str(EMBED))
from embedding_run import load_runtime
from dynamics_embedding import EmbeddingOperators, compare, norm, read_json, timed, weight_hash, write_json
from embedding_experiments import causal_compare, forecast, get_ridge, patch_state
from prefix_control import features
sys.path.insert(0,str(DESIGN))
from mechanism_design import encode, group
from study import twohop, role_tokenize


METHODS=("native","chart_anderson","chart","full_loop","branch_flow","prefix_ridge","ridge","persistence")
ARMS=("clean","source","rule","random_source","random_rule","unrelated_source",
      "unrelated_rule","same_y_source","same_y_rule")
LIMITS={"tableau":12,"twohop_open":18,"twohop_options":18}
DTYPES={"float32":torch.float32,"bfloat16":torch.bfloat16}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def key(text):
    return hashlib.sha256(("confirm-fourth/574fa66/"+text).encode()).hexdigest()


def hashes():
    return {str(p.relative_to(HERE.parent)):sha(p) for p in (
        HERE/"confirm.py", EMBED/"dynamics_embedding.py", EMBED/"embedding_experiments.py",
        EMBED/"prefix_control.py", EMBED/"results"/"ridge.npz",
        EMBED/"results"/"prefix_control"/"prefix_ridge.npz")}


def cases(tokenizer, family, index, attempt):
    if family=="tableau":
        config=group("new_structure",index,attempt)
        rows=[]
        for s,m in ((0,0),(1,0),(0,1)):
            w=encode(tokenizer,config,s,m,"open")
            rows.append(dict(w,item=f"{config['id']}:{s}{m}",group=config["id"],family="tableau"))
        semantic=tuple(config["keys"]+config["answers"])
    else:
        worlds=twohop(tokenizer,"new_structure",index,attempt)
        lookup={(w["s"],w["m"],w["r"]):w for w in worlds}
        rows=[lookup[i] for i in ((0,0,0),(1,0,0),(0,1,0))]
        semantic=tuple(tuple(sorted(x)) for x in rows[0]["within"])
        if family=="twohop_options":
            decorated=[]
            for w in rows:
                text=w["text"].removesuffix("\nAnswer:")
                note="\nOptions: "+", ".join(w["choices"])+"."
                ext=role_tokenize(tokenizer,[(None,text+note),("readout","\nAnswer:")],w["choices"],w)
                if ext["ids"][:w["read"]-1]!=w["ids"][:w["read"]-1]:
                    raise ValueError("Context before the choice list shifted")
                ext["roles"]={**w["roles"],**ext["roles"]}
                decorated.append(ext)
            rows=decorated
    z,a,b=rows
    rule="mapping" if family=="tableau" else "bridge"
    if not all(len(x["ids"])==len(z["ids"]) and x["content"]==z["content"] for x in rows):
        raise ValueError("Different alignment or candidate order")
    if z["roles"]["source"]!=a["roles"]["source"] or z["roles"][rule]!=b["roles"][rule]:
        raise ValueError("Different role positions")
    if z["y"] in (a["y"],b["y"]):
        raise ValueError("Semantic target did not change")
    index_noise=z["roles"]["noise"][0]
    for word in (" office"," shed"," hallway"," garden"," forest"," pencil"):
        v=tokenizer.encode(word,add_special_tokens=False)
        if len(v)==1 and v[0] not in z["ids"] and v[0] not in z["content"]:
            break
    ids=z["ids"].copy()
    ids[index_noise]=v[0]
    alternative=dict(z,ids=ids,item=z["item"]+"/same_y",text=tokenizer.decode(ids))
    return {"cohort":family,"worlds":rows+[alternative],"semantic":semantic}


def freeze(model,out):
    out.mkdir(exist_ok=True,parents=True)
    tokenizer=AutoTokenizer.from_pretrained(model,trust_remote_code=True)
    old=read_json(EMBED/"results"/"MANIFEST.json")
    older=read_json(HERE.parent/"ouro_continuum_refinement_20261120"/"results"/"MANIFEST.json")
    arc0=read_json(EMBED/"results"/"ARC_CAUSAL_MANIFEST.json")
    used_arc={w["item"] for k in ("test","dev") for w in old["arc"][k]}
    used_arc.update(w["item"] for z in older["arc"].values() for w in z)
    used_arc.update(g[0]["item"] for g in arc0)
    arc=[]
    seen=set()
    seen_structure=set()
    with (CONT/"results"/"data_manifest.jsonl").open(encoding="utf-8") as fp:
        for w in map(json.loads,fp):
            if w["family"]=="arc":
                if len(w["content"])==4 and w["item"] not in used_arc:
                    arc.append(w)
            else:
                seen.add(tuple(w["ids"]))
                if "within" in w:
                    seen_structure.add(tuple(tuple(sorted(v)) for v in w["within"]))
    for w in old["twohop"]["test"]+[v for g in old["twohop"]["causal"] for v in g]:
        seen_structure.add(tuple(tuple(sorted(v)) for v in w["within"]))
    selected={}
    for j,(family,n) in enumerate(LIMITS.items()):
        records=[]
        for index in range(2000+1000*j,2000+1000*j+500):
            for attempt in range(30):
                try:
                    record=cases(tokenizer,family,index,attempt)
                except ValueError:
                    continue
                if record["semantic"] in seen_structure or any(tuple(w["ids"]) in seen for w in record["worlds"]):
                    continue
                records.append(record)
                seen_structure.add(record["semantic"])
                seen.update(tuple(w["ids"]) for w in record["worlds"])
                break
            if len(records)==n:
                break
        assert len(records)==n
        selected[family]=records
    write_json(out/"FREEZE.json",dict(
        status="FROZEN_BEFORE_MODEL_SCORING",source_commit="bfca32210a0d08d6bc03f070f69e94212ffb88f8",
        hashes=hashes(),methods=METHODS,arms=ARMS,primary={"dtype":"float32","target_loop":2,"groups":selected},
        paired_depth_panel={f:[g["worlds"][0]["group"] for g in sorted(v,key=lambda z:key(z["worlds"][0]["group"]))[:8]]
                            for f,v in selected.items()},
        arc_natural=sorted(arc,key=lambda w:key(w["item"]))[:32]))


class LaneView:
    def __init__(self,op,i,j):
        self.k1=op.k1[i:j]
        self.valid=op.valid[i:j]
        self.lengths=op.lengths[i:j]
        self.positions=op.positions[i:j]
        self.candidates=op.candidates[i:j]
        self.rows=torch.arange(j-i,device=op.ids.device)
        self.op=op

    def scores(self,x):
        with torch.no_grad():
            return self.op.model.lm_head(x[self.rows,self.positions].to(self.op.dtype)).float().gather(-1,self.candidates)


def buckets(data,n):
    for k in range(0,len(data),n):
        yield data[k:k+n]


def prefix_fit():
    p=EMBED/"results"/"prefix_control"/"prefix_ridge.npz"
    with np.load(p) as fp:
        centre,scale,w,shift=(torch.as_tensor(fp[k],device="cuda") for k in ("centre","scale","weight","offset"))
    return lambda x:x.float()+(features(x)-centre)/scale@w+shift


def advance(op,state,name,ridge,prefix):
    if name=="prefix_ridge":
        return prefix(state)
    return forecast(op,state,name,ridge)[0]


def patch(view,record):
    rec,src,rule,noise=record["worlds"]
    wrapper=SimpleNamespace(k1=view.k1[:7])
    edit,spans=patch_state(wrapper,rec,src,rule)
    source=torch.cat((edit,edit[:1].repeat(2,1,1)))
    matches=[]
    for j,idx in enumerate(spans):
        quantity=(edit[j+1,idx].float()-edit[0,idx].float()).norm()
        direction=view.k1[7,idx].float()-edit[0,idx].float()
        native_dose=direction.norm()
        scale=quantity/native_dose.clamp_min(1e-10)
        source[7+j,idx]=(edit[0,idx].float()+scale*direction).to(edit.dtype)
        matches.append(dict(original_dose=float(native_dose),requested=float(quantity),scaling=float(scale)))
    g=rec["y"],src["y"],rule["y"]
    return source,[g[0],g[1],g[2],g[1],g[2],g[1],g[2],g[0],g[0]],matches


def causal(model,groups,depth,dtype,ridge,prefix,destination):
    result=[]
    timing=[]
    for batch in buckets(groups,2):
        lanes=[]
        for r in batch:
            rec,src,rule,same_y=r["worlds"]
            lanes += [rec,src,rule]+[rec]*4+[same_y,rec]
        op=EmbeddingOperators(model,lanes,depth)
        views=[LaneView(op,9*k,9*k+9) for k in range(len(batch))]
        interventions=[patch(v,b) for v,b in zip(views,batch,strict=True)]
        state=torch.cat([q[0] for q in interventions])
        with timed(op,"causal/native",timing):
            actual=op.native_suffix(state)
        assert torch.equal(actual[::9],op.k2[::9])
        predictions={"native":actual}
        for name in METHODS[1:]:
            with timed(op,"causal/"+name,timing):
                predictions[name]=advance(op,state,name,ridge,prefix)
        for k,(view,b,p) in enumerate(zip(views,batch,interventions,strict=True)):
            i=slice(9*k,9*k+9)
            identity=list(range(len(b["worlds"][0]["content"])))
            rows={name:causal_compare(view,output[i],actual[i],p[1],[identity]*9)
                  for name,output in predictions.items()}
            result.append(dict(group=b["worlds"][0]["group"],item=b["worlds"][0]["item"],targets=p[1],
                       true_clean=b["worlds"][0]["y"],source_same_y=p[2],
                       input_dose=norm(view,p[0]-p[0][:1]).tolist(),methods=rows))
    write_json(destination,dict(dtype=dtype,loop=depth,arms=ARMS,rows=result,timing=timing))


def natural(model,groups,depth,dtype,ridge,prefix,destination):
    out=[]
    timing=[]
    for batch in buckets(groups,6):
        op=EmbeddingOperators(model,batch,depth)
        data={}
        for method in METHODS:
            with timed(op,"natural/"+method,timing):
                pred=advance(op,op.k1,method,ridge,prefix)
            data[method]=compare(op,pred,op.k2,op.k1)
        out.append(dict(items=[w["item"] for w in batch],gold=[w["y"] for w in batch],
                        initial_logits=op.scores(op.k1).tolist(),methods=data))
    write_json(destination,dict(dtype=dtype,loop=depth,batches=out,timing=timing))


@torch.no_grad()
def run(model_path,out,dtype,which):
    receipt=read_json(out/"FREEZE.json")
    assert receipt["hashes"]==hashes()
    model,_,runtime=load_runtime(model_path)
    model.requires_grad_(False)
    baseline=weight_hash(model,torch.bfloat16)
    model.to(dtype=DTYPES[dtype])
    ridge,prefix=get_ridge(EMBED/"results"),prefix_fit()
    groups=receipt["primary"]["groups"]
    natural_groups={f:[q["worlds"][0] for q in v] for f,v in groups.items()} | {"arc":receipt["arc_natural"]}
    parity=[]
    for f,data in natural_groups.items():
        for depth in (2,3,4):
            op=EmbeddingOperators(model,data[:1],depth)
            parity.append(dict(cohort=f,loop=depth,**op.parity_receipt()))
            del op
    write_json(out/f"PARITY_{dtype}.json",dict(runtime=runtime,reference_dtype=dtype,rows=parity))
    for depth in (2,3,4):
        for f,g in (groups if which=="causal" else natural_groups).items():
            dest=out/f"{which}_{dtype}_K{depth}_{f}.json"
            if dest.exists():
                continue
            if which=="causal":
                chosen=g if depth==2 else [v for v in g if v["worlds"][0]["group"] in receipt["paired_depth_panel"][f]]
                causal(model,chosen,depth,dtype,ridge,prefix,dest)
            else:
                natural(model,g,depth,dtype,ridge,prefix,dest)
            print("COMPLETED",dest.name,flush=True)
    final=weight_hash(model,torch.bfloat16)
    assert final==baseline
    write_json(out/f"WEIGHTS_{which}_{dtype}.json",dict(before=baseline,after=final,status="PASS"))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("stage",choices=("freeze","natural","causal"))
    p.add_argument("--dtype",choices=tuple(DTYPES),default="float32")
    args=p.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.stage=="freeze":
        freeze(args.model,args.out)
    else:
        run(args.model,args.out,args.dtype,args.stage)
