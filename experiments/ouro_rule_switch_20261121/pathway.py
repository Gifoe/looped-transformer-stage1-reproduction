"""Frozen layer-window and whole-source value restorations in native K2."""
import argparse
from collections import Counter
from pathlib import Path
import random

import numpy as np
import torch

from causal import align, interval, source_states
from channels import NativeKV, Restore
from native import Forward, load, output, pad, save, setup, sha
from rule import MAX_NEW, prompt, text_results


class Partial(Restore):
    def __init__(self,trace,positions,seeds,mode,selected):
        super().__init__(trace,positions,seeds,mode)
        self.selected=set(selected)

    def place(self,key,kind,x):
        if key[1] in self.selected:
            return super().place(key,kind,x)
        return x


def pick(root,split):
    native={x["id"]:x for x in load(root/"NATIVE.json")["rows"]}
    fix=load(root/"GRAFT.json")
    keys=[]
    for row in sorted(fix["rows"],key=lambda x:x["id"]):
        if row["split"]!=split or not row["aligned"]:
            continue
        for k,c in enumerate(row["cases"]):
            if [x["number"] for x in row["graft"][k]]==c["new"] and [x["number"] for x in row["native"][k]]==c["original"]:
                keys.append((native[row["id"]],row,k))
                break
    if split=="confirmation":
        keys=random.Random(26112105).sample(keys,min(64,len(keys)))
    return keys


def matched_prefix(tok,c,op):
    text=prompt(c["a"],c["b"],op)
    base=c["rec"][op]
    p,q=[tok.encode(text+" "+str(c[k][op])+"\n",add_special_tokens=False) for k in ("original","new")]
    assert p[:len(base)]==q[:len(base)]==base
    for j,(a,b) in enumerate(zip(p,q)):
        if a!=b:
            return p[:j],a,b
    raise ValueError(c)


def mix_inputs(f,cases,donor,ids):
    h1=f.run(ids,1)[0]
    hybrid=h1.clone()
    p=[torch.tensor(c["positions"],device=ids.device) for c in cases]
    for j,c in enumerate(cases):
        hybrid[j,p[j]]=donor[j]
    return h1,hybrid,p


def qstr(f,h,p,yes,no):
    read=output(f.model,h,p)
    i=torch.arange(len(read),device=read.device)
    return read[i,yes]-read[i,no]


def score_windows(f,tok,root,pairs):
    tests=[tuple(range(i,i+4)) for i in range(0,24,4)]
    state=load(root/"SELECTION.json") if (root/"SELECTION.json").exists() else None
    if state:
        return state

    def assess(options):
        records=[]
        for a in range(0,len(pairs),6):
            batch=pairs[a:a+6]
            group=[align(tok,g,k) for g,_,k in batch]
            d=source_states(f,group)
            diag=[matched_prefix(tok,c,o) for c in group for o in (0,1)]
            sequences,yes,no=zip(*diag,strict=True)
            toks,last=pad(sequences)
            yes=torch.tensor(yes,device=toks.device)
            no=torch.tensor(no,device=toks.device)
            expanded=[c for c in group for _ in (0,1)]
            donors=[v for v in d for _ in (0,1)]
            h1,hyb,pos=mix_inputs(f,expanded,donors,toks)
            native_kv=NativeKV(pos)
            graft_kv=NativeKV(pos)
            h2n=f.run(toks,1,start=h1,record=native_kv)[0]
            h2h=f.run(toks,1,start=hyb,record=graft_kv)[0]
            for trace in (native_kv,graft_kv):
                for y in trace.data.values():
                    for i,j in zip(y[::2],y[1::2],strict=True):
                        torch.testing.assert_close(i,j,rtol=2e-5,atol=2e-5)
            base=qstr(f,h2h,last,yes,no)
            native=qstr(f,h2n,last,yes,no)-base
            measured=[]
            for option in options:
                edit=Partial(native_kv,pos,[0]*len(pos),"values",option)
                patch=f.run(toks,1,start=hyb,edit=edit)[0]
                measured.append((qstr(f,patch,last,yes,no)-base).cpu().tolist())
            delta=native.tolist()
            for j,(g,_,k) in enumerate(batch):
                sl=slice(2*j,2*j+2)
                records.append(dict(id=g["id"],orientation=k,native_gain=delta[sl],
                                    restoration_gain=[x[sl] for x in measured]))
        metric=[]
        for i,option in enumerate(options):
            means=np.asarray([p["restoration_gain"][i] for p in records]).mean(axis=0).tolist() if records else [0.0,0.0]
            metric.append(dict(layers=list(option),mean_margin_gain=means,score=min(means)))
        winner=max(metric,key=lambda x:x["score"],default=None)
        return (winner["layers"] if winner and winner["score"]>0 else None),metric,records

    win,score,raw=assess(tests)
    if win:
        layer,layer_scores,layer_raw=assess([(i,) for i in win])
    else:
        layer,layer_scores,layer_raw=None,[],[]
    result=dict(n_discovery=len(pairs),window=win,layer=layer,window_scores=score,layer_scores=layer_scores,
                window_raw=raw,layer_raw=layer_raw,primary_raw_sha256=sha(root/"GRAFT.json"))
    save(root/"SELECTION.json",result)
    print("locked selection",len(pairs),win,layer,flush=True)
    return result


def modes(p):
    limits=list(range(24))
    yield "native",None,False,[]
    yield "graft",None,True,[]
    yield "all_keys","keys",True,limits
    yield "all_v","values",True,limits
    yield "all_v_sham","values_noise",True,limits
    for key in ("window","layer"):
        if p[key]:
            yield key,"values",True,p[key]
            yield key+"_sham","values_noise",True,p[key]


def generate(f,tok,cases,donors,seeds,mode):
    _,restore,swap,layers=mode
    order=[(i,o) for i in range(len(cases)) for o in (0,1)]
    used=list(range(len(order)))
    answers=[[] for _ in order]
    for _ in range(MAX_NEW):
        seq=[cases[i]["rec"][o]+answers[j] for j in used for i,o in [order[j]]]
        ids,last=pad(seq)
        group=[cases[order[j][0]] for j in used]
        inserted=[donors[order[j][0]] for j in used]
        h1,hyb,pos=mix_inputs(f,group,inserted,ids)
        if not swap:
            hyb=h1
        row=torch.arange(len(h1),device=h1.device)
        assert torch.equal(h1[row,last],hyb[row,last])
        if restore:
            trace=NativeKV(pos)
            f.run(ids,1,start=h1,record=trace)
            random=[seeds[order[j][0]] for j in used]
            edit=Partial(trace,pos,random,restore,layers)
            state=f.run(ids,1,start=hyb,edit=edit)[0]
        else:
            state=f.run(ids,1,start=hyb)[0]
        tok_ids=output(f.model,state,last).argmax(-1).tolist()
        remain=[]
        for j,z in zip(used,tok_ids,strict=True):
            answers[j].append(z)
            if z!=tok.eos_token_id and "\n" not in tok.decode(answers[j],skip_special_tokens=True).strip():
                remain.append(j)
        used=remain
        if not used:
            break
    vals=text_results(tok,answers)
    return [vals[i:i+2] for i in range(0,len(vals),2)]


def summarize(rows):
    modes=list(rows[0]["outputs"]) if rows else []
    out={}
    for mode in modes:
        d=Counter()
        for r in rows:
            v=[x["number"] for x in r["outputs"][mode]]
            d["joint_original"]+=v==r["case"]["original"]
            d["joint_counterfactual"]+=v==r["case"]["new"]
            for o in (0,1):
                name=next((a for a in ("original","new","source") if r["case"][a][o]==v[o]),"other")
                d[f"rule{o}_{name}"]+=1
        out[mode]=dict(d)
    paired={}
    for key in ("all_v","window","layer"):
        if key not in modes:
            continue
        sham=key+"_sham"
        effects=[]
        for row in rows:
            answers=row["case"]["original"]
            x,y=(int([t["number"] for t in row["outputs"][m]]==answers) for m in (key,sham))
            effects.append(x-y)
        levels=(.025,.975) if key=="all_v" else (.0125,.9875)
        paired[key]=interval(effects,levels)
    return dict(n=len(rows),modes=out,paired=paired)


@torch.inference_mode()
def run(model,root):
    assert load(root/"GRAFT_SUMMARY.json")["confirmation"]["gate"]
    m,tok=setup(model)
    f=Forward(m)
    discovery=pick(root,"discovery")
    lock=score_windows(f,tok,root,discovery)
    plan=list(modes(lock))
    confirmation=pick(root,"confirmation")
    path=root/"PATH_RAW.json"
    old=load(path)["rows"] if path.exists() else []
    assert [x["id"] for x in old]==[r[0]["id"] for r in confirmation[:len(old)]]
    for i in range(len(old),len(confirmation),6):
        pack=confirmation[i:i+6]
        cases=[align(tok,x,k) for x,_,k in pack]
        donor=source_states(f,cases)
        seeds=[26112110+i+j for j in range(len(cases))]
        y={r[0]:generate(f,tok,cases,donor,seeds,r) for r in plan}
        for j,(g,b,k) in enumerate(pack):
            per={a:v[j] for a,v in y.items()}
            for a in ("native","graft"):
                assert [x["tokens"] for x in per[a]]==[x["tokens"] for x in b[a][k]]
            old.append(dict(id=g["id"],orientation=k,case=b["cases"][k],outputs=per))
        save(path,dict(selection_sha256=sha(root/"SELECTION.json"),primary_raw_sha256=sha(root/"GRAFT.json"),rows=old))
        print("path",len(old),len(confirmation),flush=True)
    report=summarize(old)
    save(root/"PATH_SUMMARY.json",report)
    print(report,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.model,a.out)

