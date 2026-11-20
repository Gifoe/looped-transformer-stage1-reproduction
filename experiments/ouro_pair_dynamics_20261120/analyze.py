"""Group-level inference on fixed four-world contrasts and separate access tiers."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


END=("1/norm","2/norm","3/norm")
LABELS=dict(base="Base flow",base_fine="Fine base",full="Full pair field",self="Self correction",
    order="Order correction",reverse="Reversed bracket",taylor="Discrete Taylor",picard="Picard",
    prefix_anderson="Prefix Anderson",whole_loop="Whole-loop ODE",branch_flow="Branch flow",
    full_half="Full ×0.50",full_quarter="Full ×0.25",persistence="K1 persistence",zero="Additive zero",
    native="Native")


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def write(p,obj):
    p.write_text(json.dumps(obj,indent=2,allow_nan=False,ensure_ascii=False)+"\n",encoding="utf-8")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def contrast(z,target):
    return [x[target]-sum(v for i,v in enumerate(x) if i!=target)/(len(x)-1) for x in z]


def iq(z,target):
    x=contrast(z,target)
    return (x[3]-x[1])-(x[2]-x[0])


def winner(x):
    return int(np.argmax(x)) if x.count(max(x))==1 else None


def stat(v):
    if not len(v):
        return None
    return dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.quantile(v,.95)))


def paired(d):
    """Input contains one average per sampled world group."""
    a=np.asarray(d,dtype=float)
    if not len(a):
        return None
    rng=np.random.default_rng(57460)
    b=a[rng.integers(len(a),size=(4000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),ci=np.quantile(b,[.025,.975]).tolist(),n=len(a))


def measure(y,x):
    if not len(y):
        return None
    return dict(error=stat([abs(p-r) for p,r in zip(x,y,strict=True)]),
          bias=stat([p-r for p,r in zip(x,y,strict=True)]),
          positive_sign_exact=sum(np.sign(p)==np.sign(r) for p,r in zip(x,y,strict=True) if abs(r)>1e-6),
          sign_evaluable=sum(abs(r)>1e-6 for r in y))


def collect_manifest(out):
    m=load(out/"MANIFEST.json")
    return m,{g["group"]:g for phase in m["groups"].values() for groups in phase.values() for g in groups}


def phase_rows(root,phase,lookup):
    pred_path=root/f"PREDICTIONS_{phase}.json"
    p=load(pred_path)
    n=load(root/f"NATIVE_{phase}.json")
    assert n["prediction_sha256"]==sha(pred_path)==load(root/f"PREDICTION_SHA_{phase}.json")["sha256"]
    states={r["group"]:r for r in n["rows"]}
    result=[]
    for rp in p["rows"]:
        k=rp["group"]
        rn=states[k]
        meta=lookup[k]
        target=meta["targets"][-1]
        z=dict(group=k,family=rp["family"],endpoints={})
        assert rp["initial"]["mixed_l2"]==rn["initial"]["mixed_l2"]==0
        for stage in END:
            y=rn["methods"]["native"].get(stage)
            if y is None:
                continue
            true=iq(y["scores"],target)
            effects=dict(zero=0.)
            projected={}
            for name,records in rp["methods"].items():
                if stage not in records:
                    continue
                row=records[stage]
                effects[name]=iq(row["scores"],target)
                a,b=np.array(row["readout_mixed_sketch"]),np.array(y["readout_mixed_sketch"])
                projected[name]=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
            z["endpoints"][stage]=dict(iq=true,pred=effects,score=y["scores"],target=target,
                                       readout_sketch_cos=projected)
        result.append(z)
    return result,p,n


def free_summary(rows):
    out={}
    for k in END:
        d=[v["endpoints"][k] for v in rows if k in v["endpoints"]]
        if not d:
            continue
        names=sorted({m for r in d for m in r["pred"]})
        out[k]={m:(measure([r["iq"] for r in d if m in r["pred"]],[r["pred"][m] for r in d if m in r["pred"]])
                       or {}) for m in names}
        for m in names:
            out[k][m]["missing"]=sum(m not in r["pred"] for r in d)
            out[k][m]["sketch_cos"]=stat([r["readout_sketch_cos"][m] for r in d if m in r["readout_sketch_cos"]])
        out[k]["native"]={"target_correct":sum(winner(r["score"][3])==r["target"] for r in d),"n":len(d),
                         "iq":stat([r["iq"] for r in d])}
    return out


def freeze_method(out):
    _,lookup=collect_manifest(out)
    rows,_,_=phase_rows(out,"development",lookup)
    grid=free_summary(rows)
    family=("full","full_half","full_quarter")
    baselines=("base","self","order","reverse","base_fine","taylor","picard","prefix_anderson","whole_loop","branch_flow","zero")
    best=min(family,key=lambda m:sum(grid[k][m]["error"]["mean"] for k in (END[0],END[2])))
    controls={}
    for k in (END[0],END[2]):
        ctrl=min(baselines,key=lambda m:grid[k][m]["error"]["mean"])
        records=[q["endpoints"][k] for q in rows]
        gap=paired([abs(r["pred"][ctrl]-r["iq"])-abs(r["pred"][best]-r["iq"])
                    for r in records if ctrl in r["pred"] and best in r["pred"]])
        target=max(.05,.1*np.mean([abs(r["iq"]) for r in records]))
        controls[k]=dict(best_control=ctrl,gain=gap,threshold=float(target),pass_gate=bool(
            gap and gap["ci"][0]>0 and gap["mean"]>=target and gap["n"]==len(records)))
    expand=all(v["pass_gate"] for v in controls.values())
    methods=list(dict.fromkeys([best,"full","base","taylor","prefix_anderson"]+
                              [z["best_control"] for z in controls.values() if z["best_control"]!="zero"]))
    write(out/"FREE_METHOD.json",dict(choice=best,gate_pass=expand,phase="confirmation" if expand else "validation",
       methods=methods,controls=controls,family_sizes=[96,96] if expand else [48,48],
       validation_receipt=sha(out/"PREDICTIONS_development.json")))


def local_summary(root,phase,lookup):
    x=root/f"LOCAL_PREDICTIONS_{phase}.json"
    z=load(x)
    native=load(root/f"LOCAL_NATIVE_{phase}.json")
    assert sha(x)==native["prediction_sha256"]==load(root/f"LOCAL_PREDICTION_SHA_{phase}.json")["sha256"]
    obs={(r["group"],r["layer"],r["loop"]):r for r in native["rows"]}
    data=[]
    for p in z["rows"]:
        key=(p["group"],p["layer"],p["loop"])
        g=lookup[key[0]]
        tgt=g["targets"][-1]
        real=obs[key]["methods"]["native"]
        def effect(v,arm):
            baseline="am" if arm=="ba" or arm.startswith("AM_") else "ba"
            return iq(v[arm]["scores"],tgt)-iq(v[baseline]["scores"],tgt)
        true={a:effect(real,a) for a in real if a!="am"}
        predicted={m:{a:effect(a_dict,a) for a in a_dict if a!="am"} for m,a_dict in p["methods"].items()}
        margin={k:contrast(v["scores"],tgt)[3] for k,v in real.items()}
        data.append(dict(group=key[0],layer=key[1],loop=key[2],family=g["family"],pred=predicted,truth=true,margin=margin))
    loops={}
    for k in sorted({d["loop"] for d in data}):
        rows=[r for r in data if r["loop"]==k]
        c={}
        for name in rows[0]["pred"]:
            c[name]=measure([v["truth"]["ba"] for v in rows],[v["pred"][name]["ba"] for v in rows])
        highest=[d["truth"]["BA_comp/1.0"] for d in rows]
        def clustered(fn):
            g=defaultdict(list)
            for q in rows:
                g[q["group"]].append(fn(q))
            return paired([np.mean(d) for d in g.values()])

        gain=clustered(lambda q:abs(q["truth"]["ba"]+q["truth"]["BA_random/1.0"])-
                    abs(q["truth"]["ba"]+q["truth"]["BA_comp/1.0"]))
        plain_gain=clustered(lambda q:abs(q["truth"]["ba"])-abs(q["truth"]["ba"]+q["truth"]["BA_comp/1.0"]))
        donor_gain=clustered(lambda q:abs(q["truth"]["ba"]+q["truth"]["BA_other/1.0"])-
                    abs(q["truth"]["ba"]+q["truth"]["BA_comp/1.0"]))
        dir_rank=clustered(lambda q:abs(q["truth"]["ba"]-q["pred"]["taylor"]["ba"])-
                        abs(q["truth"]["ba"]-q["pred"]["full"]["ba"]))
        dose_effect={}
        for label in ("AM_weak","BA_comp","BA_random","BA_other"):
            for dose in (.25,.5,1.):
                arm=f"{label}/{dose}"
                dose_effect[arm]=measure([q["truth"][arm] for q in rows],[q["pred"]["full"][arm] for q in rows])
        loops[str(k)]=dict(n=len({q["group"] for q in rows}),n_observations=len(rows),order=c,
                          expected_compensation=stat(highest),compensation_vs_random=gain,
                          compensation_vs_ba=plain_gain,compensation_vs_other=donor_gain,
                          full_vs_taylor=dir_rank,dose_prediction=dose_effect)
    return dict(rows=data,loops=loops)


def arc_summary(out,manifest):
    p=load(out/"ARC_PREDICTIONS.json")
    t=load(out/"ARC_NATIVE.json")
    assert t["prediction_sha256"]==sha(out/"ARC_PREDICTIONS.json")
    native={q["item"]:q for q in t["rows"]}
    gold={w["item"]:w["y"] for w in manifest["arc_confirmation"]}
    stats={}
    dv=np.load(out/"ARC_PREDICT_READOUT.npz")
    tv=np.load(out/"ARC_TRUTH_READOUT.npz")
    for k in END:
        s={}
        ref=tv["native_"+k.replace("/","_")]
        for name in p["rows"][0]["methods"]:
            x=[(r["methods"][name][k]["scores"],native[r["item"]]["methods"]["native"][k]["scores"],gold[r["item"]])
               for r in p["rows"]]
            h=dv[name+"_"+k.replace("/","_")]
            s[name]=dict(eq=sum(winner(a)==winner(b) and winner(a)!=None for a,b,_ in x),
                         correct=sum(winner(a)==y for a,_,y in x),native_correct=sum(winner(b)==y for _,b,y in x),
                         n=len(x),readout_error=stat((np.linalg.norm(h-ref,axis=-1)/np.linalg.norm(ref,axis=-1)).tolist()))
        stats[k]=s
    return stats


def bridge_summary(out):
    z=load(out/"BRIDGE.json")
    outb={}
        # The class is an actual cabinet ID; the original arbitrary group label is not fitted.
    for p in ("development","validation","confirmation"):
        rows=[q for q in z["rows"] if q["phase"]==p]
        outb[p]={k:{r:sum(winner(q["loops"][str(k)][r])==q["bridge_label"] for q in rows)
                    for r in ("pre_D","answer_pos")} | {"n":len(rows)} for k in (1,2,3,4)}
    return dict(head=outb,omitted=z["omitted_non_single_token"],
                swaps=[dict(roi=roi,k=2+i,target_results=[sum(winner(r["scores"][i][j])==r["targets"][j] for r in z["swaps"] if r["roi"]==roi)
                            for j in range(3)],n=sum(r["roi"]==roi for r in z["swaps"]))
                    for roi in ("pre_D","readout") for i in range(3)])


def aggregate(out):
    manifest,lookup=collect_manifest(out)
    names=[q for q in ("pilot","development","validation","confirmation") if (out/f"NATIVE_{q}.json").exists()]
    free={}
    raw={}
    timing=[]
    for p in names:
        rows,pred,truth=phase_rows(out,p,lookup)
        timing+=pred["timing"]+truth["timing"]
        raw[p]=rows
        free[p]={"all":free_summary(rows),**{f:free_summary([r for r in rows if r["family"]==f])
                                            for f in ("tableau","twohop")}}
    order={p:local_summary(out,p,lookup) for p in ("development","validation") if (out/f"LOCAL_NATIVE_{p}.json").exists()}
    for p in order:
        for q in (f"LOCAL_NATIVE_{p}",f"LOCAL_PREDICTIONS_{p}"):
            timing+=load(out/f"{q}.json")["timing"]
    for q in ("BRIDGE","ARC_NATIVE","ARC_PREDICTIONS","CLOCKS_AND_BETA","CONTROLS","SHORT"):
        path=out/f"{q}.json"
        if path.is_file():
            timing+=load(path).get("timing",[])
    timings={}
    for x in timing:
        name=x["run"].split("/")[-1]
        timings.setdefault(name,[]).append(x)
    return dict(free=free,order=order,decision=load(out/"FREE_METHOD.json"),
                arc=arc_summary(out,manifest),bridge=bridge_summary(out),
                timed_methods={k:{column:stat([x[column] for x in v]) for column in
                                   ("wall_s","allocated_peak_bytes","attention_calls","mlp_calls","jvp")} for k,v in timings.items()})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    p.add_argument("stage",choices=("freeze","all"))
    a=p.parse_args()
    if a.stage=="freeze":
        freeze_method(a.out)
    else:
        write(a.out/"SUMMARY.json",aggregate(a.out))
