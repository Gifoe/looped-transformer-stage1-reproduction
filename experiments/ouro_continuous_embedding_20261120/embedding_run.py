"""Run the frozen Ouro embedding study on the inherited token manifest."""
import argparse
import hashlib
import sys
from pathlib import Path

import torch

from dynamics_embedding import (EmbeddingOperators, HERE, REFINE, branch_trial, norm_audit,
    read_json, spectrum, timed, weight_hash, write_json)
sys.path.insert(0,str(HERE.parent/"ouro_cross_task_20261118"))
from run import load_runtime


FAMILIES=("tableau","twohop","arc")
OLD=HERE.parent/"ouro_continuous_latent_20261120"


def key(s):
    return hashlib.sha256(("eb71/embedding/20261120/"+s).encode()).hexdigest()


def freeze(tokenizer):
    import json
    from study import twohop
    with (OLD/"results"/"data_manifest.jsonl").open(encoding="utf-8") as fp:
        rows=list(map(json.loads,fp))
    ref=read_json(REFINE/"results"/"MANIFEST.json")
    earlier={w["group"] for kinds in ref.values() for bundle in kinds.values() for w in bundle}
    groups={}
    result={}
    for w in sorted(rows,key=lambda w:key(w["item"])):
        groups.setdefault(w["group"],[]).append(w)
    for f in FAMILIES:
        choices=[g for g in groups.values() if g[0]["family"]==f and g[0]["group"] not in earlier]
        if f=="arc":
            choices=[g for g in choices if len(g[0]["content"])==4]
            result[f]=dict(dev=[g[0] for g in choices[:4]],test=[g[0] for g in choices[4:12]],
                           causal=[])
            continue
        d=[g for g in choices if g[0]["cohort"]=="development"]
        t=[g for g in choices if g[0]["cohort"]=="new_structure"]
        if f=="twohop":
            t=[]
            signatures=set()
            for j in range(1000,1016):
                for attempt in range(500):
                    try:
                        items=twohop(tokenizer,"new_structure",j,attempt)
                    except ValueError:
                        continue
                    sig=tuple(items[0]["ids"])
                    if sig in signatures:
                        continue
                    signatures.add(sig)
                    t.append(sorted(items,key=lambda w:key(w["item"])))
                    break
                else:
                    raise RuntimeError("Unable to construct aligned held-out two-hop world")
        c=[]
        for g in t[8:16]:
            lookup={tuple(w[k] for k in ("s","m")+(("r",) if f=="twohop" else ())):w for w in g}
            if f=="tableau":
                rec,source,rule=(lookup[k] for k in ((0,0),(1,0),(0,1)))
            else:
                rec,source,rule=(lookup[k] for k in ((0,0,0),(1,0,0),(0,1,0)))
            c.append([rec,source,rule])
        assert len(d)>=4 and len(t)>=12,(f,len(d),len(t))
        result[f]=dict(dev=[g[0] for g in d[:4]],test=[g[0] for g in t[:8]],causal=c)
    return result


def arc_edits(tok,manifest):
    """Counterfactual ARC option permutations with identical token layout."""
    import json
    import re
    prior=read_json(REFINE/"results"/"MANIFEST.json")
    excluded={w["item"] for row in manifest["arc"].values() for w in row}
    excluded.update(w["item"] for row in prior["arc"].values() for w in row)
    with (OLD/"results"/"data_manifest.jsonl").open(encoding="utf-8") as fp:
        worlds=[w for w in map(json.loads,fp) if w["family"]=="arc" and len(w["content"])==4
                and w["item"] not in excluded]
    out=[]
    for w in sorted(worlds,key=lambda w:key("option/"+w["item"])):
        text=tok.decode(w["ids"],clean_up_tokenization_spaces=False)
        if tok.encode(text,add_special_tokens=False)!=w["ids"]:
            continue
        pattern=r"(?m)^([A-D])\) (.+)$"
        options=list(re.finditer(pattern,text))
        if [o.group(1) for o in options]!=list("ABCD"):
            continue
        gold=w["y"]
        record=[]
        for foil in (j for j in range(4) if j!=gold):
            spans={gold,foil}
            parts=[]
            offset=0
            for j,o in enumerate(options):
                a,b=o.span(2)
                parts.extend([text[offset:a],options[gold+foil-j].group(2) if j in spans else text[a:b]])
                offset=b
            new="".join(parts)+text[offset:]
            ids=tok.encode(new,add_special_tokens=False)
            if len(ids)!=len(w["ids"]) or ids[-1]!=w["ids"][-1]:
                continue
            changed=[i for i,(a,b) in enumerate(zip(w["ids"],ids,strict=True)) if a!=b]
            if not changed or not set(changed)<=set(w["roles"]["candidate"]):
                continue
            slot_to_content=list(range(4))
            slot_to_content[gold],slot_to_content[foil]=foil,gold
            record.append((dict(w,item=f"{w['item']}:option{foil}",text=new,ids=ids,y=foil,
                                arc_slot_to_content=slot_to_content),changed))
        if len(record)<2:
            continue
        roles=dict(w["roles"],source=record[0][1],mapping=record[1][1])
        out.append([dict(w,roles=roles,text=text),dict(record[0][0],roles=roles),dict(record[1][0],roles=roles)])
        if len(out)==6:
            break
    return out


def pilot(model,manifest,out):
    bf_hash=weight_hash(model,torch.bfloat16)
    parity=[]
    for f in FAMILIES:
        n=EmbeddingOperators(model,manifest[f]["dev"][:2],2)
        parity.append(dict(family=f,**n.parity_receipt()))
        del n
    model.float()
    assert all(w.dtype==torch.float32 and not w.requires_grad for w in model.parameters())
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.is_autocast_enabled("cuda")
    assert weight_hash(model,torch.bfloat16)==bf_hash
    fp_hash=weight_hash(model,torch.float32)
    for f in FAMILIES:
        op=EmbeddingOperators(model,manifest[f]["dev"][:1],2)
        receipt=op.parity_receipt()
        log=[]
        with timed(op,"BEA_local/"+f,log):
            records=[branch_trial(op,i,(.0625,.125,.25,.5,1.)) for i in (0,1,10,11,22,23,34,35,46,47)]
        with timed(op,"RMSNorm_and_Krylov/"+f,log):
            norms=norm_audit(op)
            spectra={}
            for i in (0,11,46,47):
                for t in (.25,.5,1.):
                    spectra[f"branch{i+1}/{t}"]=spectrum(op,lambda z:z+t*op.branch(i,z),
                                                      op.actual[i],1734+i)
            spectra["norm"]=spectrum(op,op.norm,op.actual[-1],1642)
            spectra["full_loop"]=spectrum(op,op.field,op.k1,1630)
        write_json(out/f"pilot_{f}.json",dict(item=manifest[f]["dev"][0]["item"],parity=receipt,
                  cases=records,norm=norms,spectra=spectra,timing=log))
    fp_after=weight_hash(model,torch.float32)
    assert fp_hash==fp_after and weight_hash(model,torch.bfloat16)==bf_hash
    write_json(out/"NATIVE_PARITY.json",dict(status="PASS",bf16=parity,
          fp32_hash_before=fp_hash,fp32_hash_after=fp_after,weight_values_bf16_sha256=bf_hash,tf32=False))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("stage",choices=("pilot","operators","reconstruct","holdout","causal"))
    a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model,tokenizer,runtime=load_runtime(a.model)
    model.requires_grad_(False)
    if a.stage=="pilot":
        manifest=freeze(tokenizer)
        write_json(a.out/"MANIFEST.json",manifest)
        write_json(a.out/"RUNTIME.json",runtime)
        pilot(model,manifest,a.out)
    else:
        assert read_json(a.out/"NATIVE_PARITY.json")["status"]=="PASS"
        manifest=read_json(a.out/"MANIFEST.json")
        if a.stage=="causal":
            extra=a.out/"ARC_CAUSAL_MANIFEST.json"
            if not extra.exists():
                write_json(extra,arc_edits(tokenizer,manifest))
            manifest["arc"]["causal"]=read_json(extra)
        from embedding_experiments import execute
        execute(model,manifest,a.out,a.stage)


if __name__=="__main__":
    main()
