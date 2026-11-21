"""Hold the early operand graft fixed while the recipient rule changes its sign."""
import argparse
from pathlib import Path

import numpy as np
import torch

from native import Forward, load, output, pad, save, setup, sha
from rule import MAX_NEW, PREFIX, prompt, text_results, value
from state import rotate_about

MODES=("native","graft","noise")


def ids(tok,a,b,op):
    return tok.encode(prompt(a,b,op),add_special_tokens=False)


def align(tok,g,i):
    a=g["a"][i]
    alt=g["a"][1-i]
    b=g["b"]
    b2=g["source_b"]
    traces=[prompt(a,b,o) for o in (0,1)]
    info=[tok(t,add_special_tokens=False,return_offsets_mapping=True) for t in traces]
    rec=[q["input_ids"] for q in info]
    alt_rec=[ids(tok,alt,b,o) for o in (0,1)]
    donor=ids(tok,alt,b2,1)
    check=ids(tok,alt,b2,0)
    if len({len(x) for x in rec+alt_rec+[donor,check]})!=1:
        return None
    b_start=len(PREFIX+f"A = {a}, B = ")
    a_start=len(PREFIX+"A = ")
    a_end=a_start+len(str(a))
    ends=[[j for j,(s,t) in enumerate(q["offset_mapping"]) if a_start<=s<t<=b_start] for q in info]
    diffs=[[j for j,(x,y) in enumerate(zip(n,d,strict=True)) if x!=y] for n,d in zip(rec,alt_rec,strict=True)]
    if not diffs[0] or diffs[0]!=diffs[1] or ends[0]!=ends[1] or not set(diffs[0]).issubset(ends[0]):
        return None
    if any(not (a_start<=info[0]["offset_mapping"][j][0]<a_end) for j in diffs[0]):
        return None
    p=list(range(diffs[0][0],ends[0][-1]+1))
    if not all(rec[0][j]==rec[1][j] and donor[j]==check[j]==alt_rec[0][j]==alt_rec[1][j] for j in p):
        return None
    if p[-1]>=len(rec[0])-1:
        return None
    return dict(a=a,alt=alt,b=b,b2=b2,rec=rec,alt_rec=alt_rec,donor=donor,donor_check=check,positions=p,
                new=[value(alt,b,o) for o in (0,1)],original=g["target"][i],
                source=[value(alt,b2,o) for o in (0,1)],native=[n[i*2:i*2+2] for n in g["native"]])


def scenario(c):
    return {k:v for k,v in c.items() if k not in ("rec","alt_rec","donor","donor_check")}


def source_states(f,cases):
    flat=[x for c in cases for x in [c["donor"],c["donor_check"],*c["alt_rec"],*c["rec"]]]
    a,_=pad(flat)
    states=f.run(a,1)[0]
    captured=[]
    for j,c in enumerate(cases):
        views=states[6*j:6*j+6][:,c["positions"]]
        src=views[0]
        for r in views[1:4]:
            torch.testing.assert_close(r,src,rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(views[4],views[5],rtol=2e-5,atol=2e-5)
        captured.append(src)
    return captured


def generate(f,tok,cases,source,seeds,modes=MODES):
    # All clean and counterfactual arms use identical native eager operators.
    orders=[(mode,j,op) for mode in modes for j in range(len(cases)) for op in (0,1)]
    choices=[[] for _ in orders]
    keep=list(range(len(orders)))
    random_rows=[]
    for c,s in zip(cases,seeds,strict=True):
        rng=torch.Generator(device="cuda").manual_seed(s)
        random_rows.append(torch.randn((len(c["positions"]),f.core.config.hidden_size),
                                       device="cuda",generator=rng))
    for step in range(MAX_NEW):
        seq=[cases[orders[j][1]]["rec"][orders[j][2]]+choices[j] for j in keep]
        tokens,last=pad(seq)
        native=f.run(tokens,1)[0]
        h=native.clone()
        if step==0:
            assert torch.equal(f.run(tokens,1,start=native)[0],f.run(tokens,2)[1])
        for k,j in enumerate(keep):
            mode,i,_=orders[j]
            if mode=="native":
                continue
            p=cases[i]["positions"]
            h[k,p]=source[i] if mode=="graft" else rotate_about(native[k,p],source[i],random_rows[i])
        rows=torch.arange(len(h),device=h.device)
        assert torch.equal(h[rows,last],native[rows,last])
        z=output(f.model,f.run(tokens,1,start=h)[0],last).argmax(-1).tolist()
        rem=[]
        for j,v in zip(keep,z,strict=True):
            choices[j].append(v)
            if v!=tok.eos_token_id and "\n" not in tok.decode(choices[j],skip_special_tokens=True).strip():
                rem.append(j)
        keep=rem
        if not keep:
            break
    results=text_results(tok,choices)
    n=2*len(cases)
    return {m:[results[j:j+2] for j in range(i*n,(i+1)*n,2)] for i,m in enumerate(modes)}


def interval(d,levels=(.025,.975)):
    a=np.asarray(d,dtype=float)
    grid=np.random.default_rng(26112102).integers(0,len(a),size=(20000,len(a)))
    bounds=np.quantile(a[grid].mean(axis=1),levels).tolist()
    return dict(n=len(a),total=int(a.sum()),mean=float(a.mean()),interval=bounds)


def summary(stage):
    report={}
    for split in ("discovery","confirmation"):
        sub=[r for r in stage if r["split"]==split]
        out={}
        for mode in MODES:
            counts=dict(alignable=0,orientations=2*len(sub),stems_joint=0,both_orientations=0,
                        per_orientation=[0,0],per_rule=[0,0],original=[0,0],donor_integer=[0,0],
                        fixed_plus=0,fixed_minus=0,repair_matches=[0,0],repair_base=[0,0],
                        bands=[0,0,0],band_n=[0,0,0],carry_joint=0,carry_n=0,borrow_joint=0,borrow_n=0)
            for r in sub:
                if not r["aligned"]:
                    continue
                h=[]
                counts["alignable"]+=1
                for i,c in enumerate(r["cases"]):
                    z=[x["number"] for x in r[mode][i]]
                    want=c["new"]
                    same=z==want
                    h.append(same)
                    counts["per_orientation"][i]+=same
                    counts["bands"][r["band"]]+=same
                    counts["band_n"][r["band"]]+=1
                    counts["carry_n"]+=r["carry_flip"]
                    counts["borrow_n"]+=r["borrow_flip"]
                    counts["carry_joint"]+=r["carry_flip"] and same
                    counts["borrow_joint"]+=r["borrow_flip"] and same
                    displacement=c["alt"]-c["a"]
                    counts["fixed_plus"]+=z==[y+displacement for y in c["original"]]
                    counts["fixed_minus"]+=z==[y-displacement for y in c["original"]]
                    for o in (0,1):
                        counts["per_rule"][o]+=z[o]==want[o]
                        counts["original"][o]+=z[o]==c["original"][o]
                        counts["donor_integer"][o]+=z[o]==c["source"][o]
                        repair=c["native"][0][o]["number"]!=c["original"][o] and c["native"][1][o]["number"]==c["original"][o]
                        counts["repair_matches"][o]+=repair and z[o]==want[o]
                        counts["repair_base"][o]+=repair
                counts["stems_joint"]+=any(h)
                counts["both_orientations"]+=all(h)
            out[mode]=counts
        raw=[]
        for r in sub:
            if not r["aligned"]:
                raw.append(0)
                continue
            paired=[]
            for mode in ("graft","noise"):
                paired.append(sum([x["number"] for x in z]==c["new"]
                                  for z,c in zip(r[mode],r["cases"],strict=True)))
            raw.append(paired[0]-paired[1])
        estimate=interval(raw)
        out["excess_pairs_per_stem"]=estimate
        out["gate"]=out["graft"]["stems_joint"]>=18 and min(out["graft"]["per_orientation"])>=5 and estimate["interval"][0]>0
        report[split]=out
    return report


@torch.inference_mode()
def run(model,root):
    assert load(root/"NATIVE_SUMMARY.json")["confirmation"]["gate"]
    raw=root/"NATIVE.json"
    full=load(raw)["rows"]
    path=root/"GRAFT.json"
    rows=load(path)["rows"] if path.exists() else []
    assert [z["id"] for z in rows]==[z["id"] for z in full[:len(rows)]]
    m,tok=setup(model)
    f=Forward(m)
    for k in range(len(rows),len(full),6):
        src=full[k:k+6]
        cases=[]
        sub=[]
        for g in src:
            a=[align(tok,g,i) for i in (0,1)]
            out={k:g[k] for k in ("id","split","band","carry_flip","borrow_flip")}
            out["aligned"]=all(c is not None for c in a)
            sub.append(out)
            if out["aligned"]:
                cases.extend(a)
                out["cases"]=[scenario(c) for c in a]
        if cases:
            donors=source_states(f,cases)
            same_seeds=[26112209+(k*2+i) for i in range(len(cases))]
            y=generate(f,tok,cases,donors,same_seeds)
            j=0
            for out in sub:
                if out["aligned"]:
                    for mode in MODES:
                        out[mode]=y[mode][j:j+2]
                    for c,got in zip(out["cases"],out["native"],strict=True):
                        assert [x["tokens"] for x in got]==[p["tokens"] for p in c["native"][1]]
                    j+=2
        rows.extend(sub)
        save(path,dict(native_sha256=sha(raw),rows=rows))
        print("graft",len(rows),len(full),flush=True)
    s=summary(rows)
    save(root/"GRAFT_SUMMARY.json",s)
    print(s,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.model,a.out)

