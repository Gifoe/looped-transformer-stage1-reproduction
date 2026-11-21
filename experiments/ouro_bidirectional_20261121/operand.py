"""Causal recombination of a first-operand H1 prefix with a different second operand."""
import argparse
from pathlib import Path

import numpy as np
import torch

from bidi import unordered
from carry import MAX_NEW, PROMPT, numeric
from native import Forward, load, output, pad, save, setup, sha
from state import prompt_ids, rotate_about


MODES=("native","prefix","prefix_noise","digits","digits_noise")


def alternate_b(a,b):
    return next(i for i in (b+100,b-100) if 100<=i<=999 and 100<=a+i<=999)


def markers(tok,a,b):
    whole=PROMPT+f"{a} + {b} ="
    obj=tok(whole,add_special_tokens=False,return_offsets_mapping=True)
    first=len(PROMPT)
    second=len(PROMPT+f"{a} + ")
    return obj["input_ids"],[i for i,(start,stop) in enumerate(obj["offset_mapping"]) if first<=start<stop<=second]


def make_case(tok,g,k):
    recipient=g["pairs"][k]
    a,b=recipient
    a2,other=g["pairs"][1-k]
    assert b==other and abs(a-a2)==200
    source=(a2,alternate_b(a2,b))
    tokens,prefix=markers(tok,a,b)
    alternate,source_prefix=markers(tok,*source)
    assert source_prefix==prefix
    if len(tokens)!=len(alternate):
        return None
    cap=prompt_ids(tok,(a2,b))
    if len(cap)!=len(tokens):
        return None
    digits=[i for i,(x,y) in enumerate(zip(tokens,cap,strict=True)) if x!=y]
    if not digits or not set(digits).issubset(prefix):
        return None
    kept=[i for i in prefix if i>=digits[0]]
    assert all(cap[i]==alternate[i] for i in kept)
    assert kept[-1]<len(tokens)-1
    donor_answer=sum(source)
    assert donor_answer not in {sum(recipient),a2+b}
    reference=[x[k] for x in g["pred"]]
    return dict(recipient=recipient,donor=source,tokens=tokens,source_tokens=alternate,clean_alternate=cap,
                prefix=kept,digits=digits,answer=sum(recipient),counterfactual=a2+b,
                donor_answer=donor_answer,native=reference)


def minimize(cases):
    return [{k:v for k,v in c.items() if k not in ("tokens","source_tokens","clean_alternate")} for c in cases]


def precompute(f,cases):
    corpus=[c["source_tokens"] for c in cases]+[c["clean_alternate"] for c in cases]
    ids,_=pad(corpus)
    states=f.run(ids,1)[0]
    all_source=[]
    for i,c in enumerate(cases):
        p=c["prefix"]
        observed=states[i,p]
        torch.testing.assert_close(observed,states[i+len(cases),p],rtol=2e-5,atol=2e-5)
        all_source.append({j:x for j,x in zip(p,observed,strict=True)})
    return all_source


def decode_modes(f,tok,cases,source,seeds):
    outputs=[[] for _ in range(len(cases)*len(MODES))]
    modes=[m for m in MODES for _ in cases]
    orig=list(range(len(cases)))*len(MODES)
    active=list(range(len(outputs)))
    max_gap=0.0
    control=[]
    for i,case in enumerate(cases):
        rng=torch.Generator(device="cuda").manual_seed(seeds[i])
        control.append(torch.randn((len(case["prefix"]),f.core.config.hidden_size),device="cuda",generator=rng))
    for t in range(MAX_NEW):
        tokens=[cases[orig[i]]["tokens"]+outputs[i] for i in active]
        ids,last=pad(tokens)
        h1=f.run(ids,1)[0]
        h=h1.clone()
        # Assert native handoff once per batch under identical masks and input shapes.
        if t==0:
            assert torch.equal(f.run(ids,1,start=h1)[0],f.run(ids,2)[1])
        for row,m in enumerate(active):
            mode=modes[m]
            if mode=="native":
                continue
            i=orig[m]
            local=cases[i]
            typ=mode.removesuffix("_noise")
            p=local[typ]
            from_donor=torch.stack([source[i][j] for j in p])
            if mode.endswith("_noise"):
                base=h1[row,p]
                rand=control[i][[local["prefix"].index(j) for j in p]]
                value=rotate_about(base,from_donor,rand)
                error=(value-base).norm(dim=-1)-(from_donor-base).norm(dim=-1)
                max_gap=max(max_gap,float(error.abs().max()))
            else:
                value=from_donor
            h[row,p]=value
        batch=torch.arange(len(h),device=h.device)
        assert torch.equal(h[batch,last],h1[batch,last])
        end=f.run(ids,1,start=h)[0]
        nxt=output(f.model,end,last).argmax(-1).tolist()
        tail=[]
        for i,token in zip(active,nxt,strict=True):
            outputs[i].append(token)
            if token!=tok.eos_token_id and "\n" not in tok.decode(outputs[i],skip_special_tokens=True).strip():
                tail.append(i)
        active=tail
        if not active:
            break
    text=[tok.decode(x,skip_special_tokens=True) for x in outputs]
    z=[dict(text=t,number=numeric(t),tokens=y) for t,y in zip(text,outputs,strict=True)]
    return {mode:z[i*len(cases):(i+1)*len(cases)] for i,mode in enumerate(MODES)},max_gap


def common_effect(v,seed=211123511):
    values=np.asarray(v,dtype=float)
    r=np.random.default_rng(seed)
    idx=r.integers(0,len(values),size=(20000,len(values)))
    return dict(total=int(values.sum()),n=len(values),mean=float(values.mean()),
                interval=np.quantile(values[idx].mean(axis=1),[0.025,0.975]).tolist())


def count_results(bins,raw):
    entries={}
    for split in ("discovery","confirmation"):
        data={m:None for m in sorted(bins) if bins[m]==split}
        data.update({str(x["midpoint"]):x for x in raw if x["split"]==split})
        alignment=[x for x in data.values() if x is not None and x["aligned"]]
        stats={}
        for m in MODES:
            hits={"counterfactual":0,"donor_answer":0,"answer":0,"two_counterfactual":0,
                  "counterfactual_among_native_repairs":0,"native_repairs":0}
            for row in alignment:
                both=True
                for i,c in enumerate(row["cases"]):
                    got=row["modes"][m][i]["number"]
                    for output in ("counterfactual","donor_answer","answer"):
                        hits[output]+=got==c[output]
                    was_repair=c["native"][0]["number"]!=c["answer"] and c["native"][1]["number"]==c["answer"]
                    hits["native_repairs"]+=was_repair
                    hits["counterfactual_among_native_repairs"]+=was_repair and got==c["counterfactual"]
                    both=both and got==c["counterfactual"]
                hits["two_counterfactual"]+=both
            stats[m]=hits
        for family in ("prefix","digits"):
            placebo=family+"_noise"
            differences=[]
            for row in data.values():
                if row is None or not row["aligned"]:
                    differences.append(0)
                    continue
                cf=[c["counterfactual"] for c in row["cases"]]
                switches=[int([q["number"] for q in row["modes"][m]]==cf) for m in (family,placebo)]
                differences.append(switches[0]-switches[1])
            stats[family]["two_switch_excess"]=common_effect(differences)
        p=stats["prefix"]
        entries[split]=dict(n_bins=len(data),aligned_bins=len(alignment),modes=stats,
                            pass_primary=p["two_counterfactual"]>=12 and p["two_switch_excess"]["interval"][0]>0)
    return entries


@torch.inference_mode()
def run(model,root):
    deck=load(root/"DECK.json")
    raw=load(root/"NATIVE.json")
    assert sha(root/"DECK.json")==raw["deck_sha256"]
    assert len(raw["rows"])==len(deck["rows"])
    selected={}
    for row in sorted(raw["rows"],key=lambda x:x["id"]):
        if row["changed_operand"]==0:
            selected.setdefault(row["midpoint"],row)
    order=list(selected.values())
    file=root/"OPERAND_RAW.json"
    old=load(file)["rows"] if file.exists() else []
    assert [r["id"] for r in old]==[r["id"] for r in order[:len(old)]]
    model,tok=setup(model)
    f=Forward(model)
    for i in range(len(old),len(order),4):
        block=order[i:i+4]
        base=[]
        todo=[]
        back=[]
        seeds=[]
        for j,row in enumerate(block):
            assert abs(row["pairs"][0][0]-row["pairs"][1][0])==200
            cs=[make_case(tok,row,k) for k in range(2)]
            valid=all(c is not None for c in cs)
            base.append(dict(id=row["id"],midpoint=row["midpoint"],split=row["split"],aligned=valid))
            if valid:
                base[-1].update(cases=minimize(cs),modes={})
                todo.extend(cs)
                back.extend((j,j))
                seeds.extend(211125000+row["midpoint"]*2+k for k in range(2))
        if todo:
            source=precompute(f,todo)
            outputs,err=decode_modes(f,tok,todo,source,seeds)
            for x,c in zip(outputs["native"],todo,strict=True):
                assert x["tokens"]==c["native"][1]["tokens"]
            for mode,values in outputs.items():
                for j,value in zip(back,values,strict=True):
                    base[j]["modes"].setdefault(mode,[]).append(value)
                    base[j]["max_geometry_abs_error"]=err
        old.extend(base)
        if (i//4)%4==0 or len(old)==len(order):
            save(file,dict(native_sha256=sha(root/"NATIVE.json"),rows=old))
            print("operand",len(old),len(order),flush=True)
    result=count_results(deck["midpoint_splits"],old)
    save(root/"OPERAND_SUMMARY.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)

