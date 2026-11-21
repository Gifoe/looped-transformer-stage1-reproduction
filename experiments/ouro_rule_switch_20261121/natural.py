"""Test if same-input V rollback causally reinstates naturally repaired wrong integers."""
import argparse
from collections import Counter
from pathlib import Path
import random

import torch

from causal import align, interval
from channels import NativeKV
from native import Forward, load, output, pad, save, setup, sha
from pathway import Partial
from rule import MAX_NEW, text_results


def pick(full):
    rng=random.Random(261121306)
    pool=[(g,i,o) for g in full if g["split"]=="confirmation" for i in (0,1) for o in (0,1)]
    fix=[]
    clean=[]
    for g,i,o in pool:
        y=g["target"][i][o]
        a,b=(x[2*i+o]["number"] for x in g["native"])
        if a is not None and a!=y and b==y:
            fix.append((g,i,o))
        if a==b==y:
            clean.append((g,i,o))
    sorted_fix=sorted(fix,key=lambda z:(z[0]["id"],z[1],z[2]))
    select=rng.sample(sorted_fix,min(len(fix),72))
    reserved={x[0]["id"] for x in select}
    rng.shuffle(clean)
    free=list(clean)
    matched=[]
    for g,i,o in select:
        idx=next((j for j,(h,_,rule) in enumerate(free) if h["id"] not in reserved and h["band"]==g["band"] and rule==o),None)
        if idx is not None:
            h,u,v=free.pop(idx)
            reserved.add(h["id"])
            matched.append((h,u,v))
    return [("repair",g,i,o) for g,i,o in select]+[("control",g,i,o) for g,i,o in matched],len(fix)


def greedy(f,tok,cases,seeds,mode,site):
    tokens=[[] for _ in cases]
    active=list(range(len(tokens)))
    for _ in range(MAX_NEW):
        seq=[cases[i]["rec"][cases[i]["rule"]]+tokens[i] for i in active]
        ids,last=pad(seq)
        p=[torch.tensor(cases[i]["positions"],device=ids.device) for i in active]
        record=NativeKV(p)
        h1=f.run(ids,1,record=record)[0]
        if mode=="native":
            state=f.run(ids,1,start=h1)[0]
        else:
            r="values_noise" if mode.endswith("_sham") else "values"
            edit=Partial(record,p,[seeds[i] for i in active],r,site)
            state=f.run(ids,1,start=h1,edit=edit)[0]
        z=output(f.model,state,last).argmax(-1).tolist()
        remain=[]
        for j,v in zip(active,z,strict=True):
            tokens[j].append(v)
            if v!=tok.eos_token_id and "\n" not in tok.decode(tokens[j],skip_special_tokens=True).strip():
                remain.append(j)
        active=remain
        if not active:
            break
    return text_results(tok,tokens)


def summarize(rows):
    out={}
    for key in ("repair","control"):
        case=[r for r in rows if r["cohort"]==key]
        seen=[r for r in case if r["aligned"]]
        arms={}
        for m in seen[0]["outputs"] if seen else []:
            z=Counter()
            for r in seen:
                y=r["outputs"][m]["number"]
                target=r["target"]
                why="original" if y==target else ("unparsed" if y is None else ("exact_K1" if y==r["wrong"] else "other"))
                z[why]+=1
            arms[m]=dict(z)
        pair={}
        for m in ("all","window"):
            if m not in arms:
                continue
            values={}
            for r in case:
                ident=r["id"]
                values.setdefault(ident,0)
                if r["aligned"] and r["wrong"] is not None:
                    actual,sham=[r["outputs"][n]["number"]==r["wrong"] for n in (m,m+"_sham")]
                    values[ident]+=int(actual)-int(sham)
            pair[m]=interval(list(values.values()))
        out[key]=dict(n=len(case),aligned=len(seen),n_stems=len({q["id"] for q in case}),arms=arms,matched_excess=pair)
    p=out["repair"]
    c=out["control"]
    fx=p["matched_excess"].get("all",{})
    def damage(x):
        if not x["aligned"]:
            return None
        return (x["arms"]["all_sham"].get("original",0)-x["arms"]["all"].get("original",0))/x["aligned"]
    a,b=damage(p),damage(c)
    out["gate"]=bool(p["n_stems"]>=12 and fx and fx["total"]>=5 and fx["interval"][0]>0
                      and a is not None and b is not None and a>b)
    out["accuracy_damage_excess"]=[a,b]
    return out


@torch.inference_mode()
def run(model,root):
    assert load(root/"NATIVE_SUMMARY.json")["confirmation"]["gate"]
    raw=root/"NATIVE.json"
    pack,available=pick(load(raw)["rows"])
    sites=dict(native=list(range(24)),all=list(range(24)),all_sham=list(range(24)))
    sel=root/"SELECTION.json"
    if sel.exists() and (window:=load(sel)["window"]):
        sites.update(window=window,window_sham=window)
    path=root/"NATURAL_RAW.json"
    result=load(path)["rows"] if path.exists() else []
    assert [(x["cohort"],x["id"],x["orientation"],x["rule"]) for x in result]==[(s,g["id"],i,o) for s,g,i,o in pack[:len(result)]]
    if pack:
        m,tok=setup(model)
        f=Forward(m)
    for offset in range(len(result),len(pack),16):
        chunk=pack[offset:offset+16]
        notes=[]
        cases=[]
        for s,g,i,o in chunk:
            c=align(tok,g,i)
            line=dict(cohort=s,id=g["id"],orientation=i,rule=o,target=g["target"][i][o],aligned=c is not None,
                      wrong=g["native"][0][i*2+o]["number"] if s=="repair" else None)
            notes.append(line)
            if c is not None:
                c["rule"]=o
                cases.append(c)
        if cases:
            rngseeds=[261121321+offset+k for k in range(len(cases))]
            table={m:greedy(f,tok,cases,rngseeds,m,where) for m,where in sites.items()}
            loc=0
            for note in notes:
                if not note["aligned"]:
                    continue
                note["outputs"]={m:z[loc] for m,z in table.items()}
                assert note["outputs"]["native"]["tokens"]==cases[loc]["native"][1][note["rule"]]["tokens"]
                loc+=1
        result.extend(notes)
        save(path,dict(native_sha256=sha(raw),rows=result))
        print("natural rollback",len(result),len(pack),flush=True)
    if not path.exists():
        save(path,dict(native_sha256=sha(raw),rows=result))
    summary=summarize(result)
    summary.update(eligible_before_cap=available)
    save(root/"NATURAL_SUMMARY.json",summary)
    print(summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.model,a.out)

