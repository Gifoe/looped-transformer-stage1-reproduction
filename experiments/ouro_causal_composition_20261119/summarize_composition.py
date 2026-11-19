"""Group-wise uncertainty, matched controls and the independent progression gate."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from capture_composition import TASKS, MAIN, digest, load_lines, write_json
from evaluate_composition import export

FRESH=("fresh_same","fresh_structure")
REGIONS=("source","bridge","mapping","query","readout")
CONDS=("natural","patch","random","irrelevant","mismatch")
EFFECTS=("S","M","SM","I")
METRICS=("response_mse","cosine","norm_ratio","mse_source","mse_bridge","mse_mapping",
         "mse_query","mse_readout","mid_error","comp_error","k2_error","k4_error","mid_pred","mid_true",
         "comp_pred","comp_true","k2_pred","k2_true","k3_true","k4_true","k4_pred")


def table(path):
    with path.open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def partitions(rows,fields):
    groups=defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in fields)].append(r)
    return groups


def draws(n):
    return np.random.default_rng(2026111919).integers(n,size=(6000,n),dtype=np.int32)


def boot(x,y=None):
    x=np.asarray(x,dtype=float)
    ids=draws(len(x))
    a=x[ids].mean(1)
    if y is None:
        point=x.mean()
    else:
        y=np.asarray(y,dtype=float)
        b=y[ids].mean(1)
        if y.mean()==0:
            # An exact zero-error comparator cannot be improved.
            return (0.,0.,0.) if x.mean()==0 else (-1.,-1.,-1.)
        point=1-x.mean()/y.mean()
        a=1-np.divide(a,np.maximum(b,1e-20))
    return tuple(float(v) for v in (point,*np.quantile(a,(.025,.975))))


def summary(rows,fields,metrics):
    result=[]
    for tag,group in partitions(rows,fields).items():
        idx=draws(len(group))
        for metric in metrics:
            y=np.array([float(r[metric]) for r in group if r[metric]!=""])
            if not len(y):
                continue
            assert len(y)==len(group)
            lo,hi=np.quantile(y[idx].mean(1),(.025,.975))
            result.append({**dict(zip(fields,tag)),"metric":metric,"groups":len(y),
                           "mean":y.mean(),"lower":lo,"upper":hi})
    return result


def losses(records,arm,metric,method):
    g=defaultdict(list)
    for r in records:
        if r["method"]==method and (r["arm"] in MAIN if arm=="pooled" else r["arm"]==arm):
            g[r["id"]].append(float(r[metric]))
    widths={len(v) for v in g.values()}
    assert widths==({3} if arm=="pooled" else {1})
    return np.array([np.mean(g[k]) for k in sorted(g)])


def gate(records,freeze,task):
    comparisons=[]
    candidate=freeze["candidate"]
    all_methods={r["method"] for r in records}
    references=sorted(m for m in all_methods if not m.endswith("bilinear"))
    names=tuple((arm,metric) for arm,metric in (("pooled","response_mse"),("I","response_mse"),
          ("pooled","mid_error"),("I","comp_error"),("pooled","k4_error")))
    passes={}
    for split in FRESH:
        pool=[r for r in records if r["split"]==split]
        for arm,metric in names:
            pred=losses(pool,arm,metric,candidate)
            all_decisions=[]
            for ref in references:
                base=losses(pool,arm,metric,ref)
                a,lo,hi=boot(pred,base)
                comparisons.append(dict(task=task,split=split,arm=arm,metric=metric,groups=len(base),
                    candidate=candidate,baseline=ref,candidate_error=float(pred.mean()),
                    baseline_error=float(base.mean()),relative_reduction=a,lower=lo,upper=hi))
                success=lo>0
                if arm=="pooled" and metric=="response_mse" and split=="fresh_structure":
                    success=success and a>=.15
                all_decisions.append(success)
            passes[f"{split}/{arm}/{metric}"]=all(all_decisions)
    necessary=[value for k,value in passes.items() if k.startswith("fresh_structure") or
               k=="fresh_same/pooled/response_mse"]
    return comparisons,{"task":task,"candidate":candidate,"tests":passes,"advances":all(necessary)}


def measured_geometry(data,folder):
    meta=load_lines(folder/"groups.jsonl")
    array=np.load(folder/"atlas.npy",mmap_mode="r")
    result,controls=[],[]
    for (split,),members in partitions(meta,("split",)).items():
        ids=np.array([row["i"] for row in members])
        v=array[ids]
        for j,cond in enumerate(CONDS):
            known=np.isfinite(v[:,j,0,0,0,0])
            if not np.any(known):
                continue
            per_group=v[known,j]
            mean=per_group.mean(0)
            # At every event retain mean and SD. Paired CIs follow for prespecified boundary contrasts.
            sd=per_group.std(0)
            for k in range(4):
                for l in range(25):
                    for region in range(5):
                        for b,effect in enumerate(EFFECTS):
                            result.append(dict(task=data,split=split,condition=cond,groups=int(known.sum()),
                                loop=k+1,layer=l,role=REGIONS[region],effect=effect,
                                mean=float(mean[k,l,region,b]),sd=float(sd[k,l,region,b])))
            if cond in ("random","irrelevant","mismatch"):
                for k,l in ((1,23),(1,24),(2,24),(3,24)):
                    for jrole in (2,4):
                        patch=v[known,1,k,l,jrole,3]
                        alt=per_group[:,k,l,jrole,3]
                        a,lo,hi=boot(patch-alt)
                        controls.append(dict(task=data,split=split,reference=cond,loop=k+1,layer=l,
                             role=REGIONS[jrole],groups=len(patch),patch=float(patch.mean()),
                             control=float(alt.mean()),difference=a,lower=lo,upper=hi))
    return result,controls


def effect(score,goal):
    return score[goal[0]]-(score[goal[1]]+score[goal[2]])/2


def outcomes(rows):
    out=[]
    for r in rows:
        goals=[r["gold"][w] for w in ("r","x","y","d")]
        act={a:[effect(s,goals) for s in logits] for a,logits in r["scores"].items()}
        for loop in range(1,5):
            j=loop-1
            for a,arm in enumerate((*MAIN,"I")):
                if arm in MAIN:
                    score=r["scores"][arm][j]
                    delta=act[arm][j]-act["r"][j]
                    label=goals[a+1]
                    before=int(np.argmax(r["scores"]["r"][j]))
                    current=int(np.argmax(score))
                    hits=int(current==label)
                    gain=int(current==label and before!=label)
                    loss=int(current!=label and before==label)
                else:
                    delta=act["SM"][j]-act["S"][j]-act["M"][j]+act["r"][j]
                    score=r["scores"]["SM"][j]
                    hits=gain=loss=0
                out.append({"task":r["task"],"split":r["split"],"difficulty":r["difficulty"],"id":r["id"],
                    "arm":arm,"loop":loop,"contrast_change":delta,"target_hit":hits,
                    "flipped_into_target":gain,"lost_target":loss})
    return out


def choose_examples(inputs,capture,res,case_dir,selected):
    case_dir.mkdir(exist_ok=True)
    known={q["id"]:q for q in load_lines(inputs)}
    cases=[]
    for task in TASKS:
        irows=load_lines(capture/task/"groups.jsonl")
        m=selected[task]["candidate"]
        performance={(r["id"],r["arm"],r["method"]):r for r in table(res/task/"causal.csv")
                        if r["split"].startswith("fresh")}
        unused={r["id"]:r for r in irows if r["split"]=="fresh_structure"}
        def best(case,scoring,eligible=lambda x:True):
            candidates=[r for r in unused.values() if eligible(r)]
            if not candidates:
                return
            data=max(candidates,key=lambda r:(scoring(r),r["id"]))
            del unused[data["id"]]
            models={name:{arm:performance[data["id"],arm,name] for arm in EFFECTS} for name in
                    ("zero",m,"full-mixed-full")}
            cases.append({"selection":case,"selection_stat":float(scoring(data)),"capture":data,
                          "input":known[data["id"]],"models":models})
        def projection_score(r,a,k):
            g=[r["gold"][w] for w in ("r","x","y","d")]
            return effect(r["scores"][a][k],g)
        def interaction(r,k):
            return sum(s*projection_score(r,a,k) for s,a in zip((1,-1,-1,1),("r","S","M","SM")))
        def mixed(r,a="I"):
            return performance[r["id"],a,m]
        best("largest_late_positive_behavioral_interaction",lambda r:interaction(r,3))
        best("largest_late_negative_behavioral_interaction",lambda r:-interaction(r,3))
        best("largest_K2_to_K4_joint_margin_increase",lambda r:float(mixed(r,"SM")["k4_true"])-
             float(mixed(r,"SM")["k2_true"]))
        best("best_model_relative_to_zero_on_interaction",lambda r:float(
             performance[r["id"],"I","zero"]["response_mse"])-float(mixed(r)["response_mse"]))
        best("worst_model_relative_to_zero_on_interaction",lambda r:float(mixed(r)["response_mse"])-
             float(performance[r["id"],"I","zero"]["response_mse"]))
        best("largest_random_control_behavioral_interaction",lambda r:abs(sum(
             s*projection_score(r,a,3) for s,a in zip((1,-1,-1,1),("r","randS","randM","randSM")))),
             lambda r:"randSM" in r["scores"])
    write_json(case_dir/"examples.json",cases)
    return len(cases)


def run(root):
    eval_root=root/"evaluation"
    output=root/"results"
    output.mkdir(exist_ok=True)
    gen=read(root/"data"/"generation.json")
    independent=gen["group_sha256"]
    assert digest(root/"data"/"fresh_groups.jsonl")==independent and gen["overlap"]==0
    gates=[]
    all_compare,natural,causal,atlas,controls,behavior=[],[],[],[],[],[]
    audit={"generation":gen,"capture":{},"fits":{}}
    for task in TASKS:
        f=root/"frozen"/task/"freeze.json"
        freeze=read(f)
        audit["fits"][task]={"sha256":digest(f),"weights_verified":True}
        record=read(eval_root/task/"complete.json")
        assert record["locked_models"]==digest(f) and record["independent_manifest"]==independent
        val=table(eval_root/task/"causal.csv")
        comparisons,decision=gate(val,freeze,task)
        gates.append(decision)
        all_compare+=comparisons
        ckeys=("task","split","method","arm")
        causal+=summary(val,ckeys,METRICS)
        causal+=summary([v for v in val if v["split"]=="fresh_structure"],(*ckeys,"difficulty"),METRICS)
        natural+=summary(table(eval_root/task/"natural.csv"),("task","split","method"),
                        ("state_mse","update_mse","native_K2_top1_agreement","diagnostic_K2_mid_accuracy",
                         "diagnostic_K1_mid_accuracy"))
        for phase in ("development","independent"):
            folder=root/phase/task
            complete=read(folder/"completion.json")
            for name,p in complete["arrays"].items():
                assert (folder/name).stat().st_size==p["size"] and digest(folder/name)==p["sha256"]
            audit["capture"][f"{phase}/{task}"]=complete
            a,b=measured_geometry(task,folder)
            atlas+=a
            controls+=b
            if phase=="independent":
                assert complete["source_manifest_sha256"]==independent
                rows=load_lines(folder/"groups.jsonl")
                assert len(rows)==160 and all("SM" in r["scores"] for r in rows)
                behavior+=outcomes(rows)
    export(output/"baseline_comparisons.csv",all_compare)
    export(output/"natural_summary.csv",natural)
    # Split-level and depth-stratum rows share a common schema.
    export(output/"causal_summary.csv",[{**r,"difficulty":r.get("difficulty","all")} for r in causal])
    export(output/"interaction_atlas.csv",atlas)
    export(output/"interaction_controls.csv",controls)
    export(output/"behavior_by_group.csv",behavior)
    export(output/"behavior_summary.csv",summary(behavior,("task","split","arm","loop"),
           ("contrast_change","target_hit","flipped_into_target","lost_target")))
    count=choose_examples(root/"data"/"fresh_groups.jsonl",root/"independent",eval_root,
                          output/"cases",{r["task"]:r for r in gates})
    write_json(output/"gate.json",{"tasks":gates,"chain_allowed":any(r["advances"] for r in gates)})
    write_json(output/"execution_audit.json",{**audit,"cases":count,"status":"passed"})
    print("decisions",[(r["task"],r["advances"]) for r in gates],"examples",count,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    run(p.parse_args().root)
