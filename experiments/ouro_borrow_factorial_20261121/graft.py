"""Recompose independently acquired first-loop operand fields."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib

import torch

from borrow import ci
from native import Forward, load, output, pad, save, setup, sha
from rule import MAX_NEW, PREFIX, prompt, text_results
from state import rotate_about

MODES=("native","a","b","ab","sham","wrong_b")


def encode(tok,pair):
    a,b=pair
    t=prompt(a,b,1)
    z=tok(t,add_special_tokens=False,return_offsets_mapping=True)
    start_a=len(PREFIX+"A = ")
    start_b=len(PREFIX+f"A = {a}, B = ")
    start_rule=len(PREFIX+f"A = {a}, B = {b}, ")
    fields=[[i for i,(s,e) in enumerate(z["offset_mapping"]) if a<=s<e<=b]
            for a,b in ((start_a,start_b),(start_b,start_rule))]
    return z["input_ids"],fields


def key(x,y):
    return hashlib.sha256((x+"::"+y+"::261121910").encode()).hexdigest()


def choose_wrong(groups,g,k):
    wrong=[]
    want=g["pairs"][2 if not k else 1][1]
    donors=2 if not k else 1
    held_a=g["pairs"][1 if not k else 2][0]
    for s in groups:
        if s["split"]!=g["split"] or s["group"]==g["group"]:
            continue
        pair=s["pairs"][donors]
        if pair[1]!=want and 100<=pair[1]-held_a<=999:
            wrong.append((pair[1]//100!=want//100,key(g["id"],s["id"]),s["id"],pair))
    return min(wrong)[2:] if wrong else None


def condition(tok,groups,g,reverse):
    # Receiver, A donor and B donor are controlled separately.
    i,a,b=(3,2,1) if reverse else (0,1,2)
    wrong=choose_wrong(groups,g,reverse)
    if wrong is None:
        return None
    wid,w=wrong
    recv=g["pairs"][i]
    first=g["pairs"][a]
    second=g["pairs"][b]
    assert first[1]==recv[1] and second[0]==recv[0]
    vals=[encode(tok,p) for p in (recv,first,second,w)]
    token,fields=vals[0]
    a_field,b_field=fields
    if any(len(ids)!=len(token) or group!=fields for ids,group in vals):
        return None
    if not a_field or not b_field or set(a_field).intersection(b_field) or max(b_field)>=len(token)-1:
        return None
    changed_a={j for j,(p,q) in enumerate(zip(token,vals[1][0],strict=True)) if p!=q}
    changed_b={j for j,(p,q) in enumerate(zip(token,vals[2][0],strict=True)) if p!=q}
    assert changed_a and changed_a.issubset(a_field) and changed_b and changed_b.issubset(b_field)
    ans=g["answers"]
    expected=[ans[i],ans[a],ans[b],ans[0 if reverse else 3]]
    return dict(id=g["id"],split=g["split"],reverse=reverse,ids=[x[0] for x in vals],
                a=a_field,b=b_field,expected=expected,native=g["pred"][1][i],
                donors=[a,b],receiver=i,wrong_id=wid,wrong_b=w[1],wrong_expected=w[1]-first[0])


def capture(f,cases):
    ids,p=pad([x for c in cases for x in c["ids"]])
    h1=f.run(ids,1)[0]
    table=[]
    for i,c in enumerate(cases):
        base=4*i
        one=h1[base+1,c["a"]]
        two=h1[base+2,c["b"]]
        wrong=h1[base+3,c["b"]]
        table.append((one.contiguous(),two.contiguous(),wrong.contiguous()))
    return table


def seeded(cases,src):
    out=[]
    for index,c in enumerate(cases):
        s=261121925+sum((i+1)*ord(a) for i,a in enumerate(c["id"]))+int(c["reverse"])
        g=torch.Generator(device=src[index][0].device).manual_seed(s)
        out.append(tuple(torch.randn(x.shape,generator=g,device=x.device) for x in src[index][:2]))
    return out


def decoded(f,tok,cases,source):
    order=[(mode,i) for mode in MODES for i in range(len(cases))]
    tokens=[[] for _ in order]
    active=list(range(len(order)))
    draws=seeded(cases,source)
    for pos in range(MAX_NEW):
        batch,last=pad([cases[order[j][1]]["ids"][0]+tokens[j] for j in active])
        h=f.run(batch,1)[0]
        altered=h.clone()
        if pos==0:
            assert torch.equal(f.run(batch,1,start=h)[0],f.run(batch,2)[1])
        for r,j in enumerate(active):
            mode,i=order[j]
            a,b=cases[i]["a"],cases[i]["b"]
            one,two,other=source[i]
            if mode in ("a","ab","wrong_b"):
                altered[r,a]=one
            if mode in ("b","ab","wrong_b"):
                altered[r,b]=other if mode=="wrong_b" else two
            if mode=="sham":
                altered[r,a]=rotate_about(h[r,a],one,draws[i][0])
                altered[r,b]=rotate_about(h[r,b],two,draws[i][1])
        index=torch.arange(len(h),device=h.device)
        assert torch.equal(altered[index,last],h[index,last])
        top=output(f.model,f.run(batch,1,start=altered)[0],last).argmax(-1).tolist()
        more=[]
        for j,v in zip(active,top,strict=True):
            tokens[j].append(v)
            if v!=tok.eos_token_id and "\n" not in tok.decode(tokens[j],skip_special_tokens=True).strip():
                more.append(j)
        active=more
        if not active:
            break
    joined=text_results(tok,tokens)
    end=len(cases)
    return {mode:joined[i*end:(i+1)*end] for i,mode in enumerate(MODES)}


def stats(rows):
    out={}
    for s in ("discovery","confirmation"):
        after={}
        subset=[r for r in rows if r["split"]==s]
        for name in ("forward","reverse"):
            counters=Counter()
            strata=[[0,0] for _ in range(3)]
            effect={q:{r["group"]:0 for r in subset} for q in ("sham","wrong_b")}
            for item in subset:
                box=item[name]
                counters["n"]+=1
                choice=item["amplitude_arm"]
                strata[choice][1]+=1
                if not box["aligned"]:
                    continue
                counters["aligned"]+=1
                c=box["case"]
                d=box["pred"]
                wanted=c["expected"]
                ok=[d[a]["number"]==y for a,y in zip(MODES[:4],wanted,strict=True)]
                for a,b in zip(MODES[:4],ok,strict=True):
                    counters[a+"_exact"]+=b
                exact=all(ok)
                strata[choice][0]+=exact
                counters["quartet"]+=exact
                counters["wrong_target_actual"]+=d["wrong_b"]["number"]==c["wrong_expected"]
                for arm,bin in effect.items():
                    hit=all(ok[:3]) and d[arm]["number"]==wanted[3]
                    counters[arm+"_quartet"]+=hit
                    bin[item["group"]]+=int(exact)-int(hit)
            intervals={q:ci(list(vals.values())) for q,vals in effect.items()}
            passed=bool(counters["quartet"]>=25 and min(z[0] for z in strata)>=8 and
                        all(g["interval"] and g["interval"][0]>0 for g in intervals.values()))
            after[name]=dict(counts=dict(counters),by_amplitude=strata,comparison=intervals,gate=passed)
        out[s]=after
    return out


@torch.inference_mode()
def run(model,root):
    assert load(root/"NATIVE_SUMMARY.json")["graft_gate"]
    groups=load(root/"NATIVE.json")["rows"]
    m,tok=setup(model)
    f=Forward(m)
    path=root/"GRAFT_RAW.json"
    seen=load(path)["rows"] if path.exists() else []
    assert [r["id"] for r in seen]==[g["id"] for g in groups[:len(seen)]]
    for i in range(len(seen),len(groups),8):
        subset=groups[i:i+8]
        stuff=[{k:g[k] for k in ("id","split","group","amplitude_arm")} for g in subset]
        for reverse,name in enumerate(("forward","reverse")):
            candidates=[condition(tok,groups,g,reverse) for g in subset]
            selected=[g for g in candidates if g is not None]
            p=0
            if selected:
                d=decoded(f,tok,selected,capture(f,selected))
            for a,c in zip(stuff,candidates,strict=True):
                local=dict(aligned=c is not None)
                if c is not None:
                    local["case"]={x:y for x,y in c.items() if x!="ids"}
                    local["pred"]={x:y[p] for x,y in d.items()}
                    assert local["pred"]["native"]["tokens"]==c["native"]["tokens"]
                    p+=1
                a[name]=local
        seen.extend(stuff)
        save(path,dict(rows=seen,native_sha256=sha(root/"NATIVE.json")))
        print("factorial graft",len(seen),len(groups),flush=True)
    r=stats(seen)
    save(root/"GRAFT_SUMMARY.json",r)
    print(r,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)

