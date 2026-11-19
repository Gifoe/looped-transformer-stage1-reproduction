"""Group-level causal contrasts, behavior transitions and incremental prediction tests."""
import argparse
from pathlib import Path
import csv
import json

import numpy as np

from data_loop import TASKS, SITES, rows, jsonfile
from readouts_loop import Data, Probe, SEMANTICS, ci, digest, export, partition, top


def table(p):
    with p.open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def contrasts(p,task,dev):
    traces=rows(p/"arms.jsonl")
    x=np.load(p/"projected_boundary.npy",mmap_mode="r")
    result,transition=[],[]
    for (gid,protocol),members in partition(traces,("id","protocol")).items():
        entries={(m["kind"],m["loop"],m["receiver"],m["arm"]):m for m in members}
        fact=members[0]
        gold=fact["gold"]
        r=entries["native",0,"r","r"]
        native_x=entries["native",0,"x","x"]
        def write(k,h,metric,value):
            result.append(dict(task=task,id=gid,protocol=protocol,split=fact["split"],
                              loop=k,horizon=h,metric=metric,value=float(value)))
        def differences(m,a,b,h):
            sc=m["scores"][h-1]
            return sc[gold[a]]-sc[gold[b]]
        for k in (2,3,4):
            sham,cut,nuisance,rescue,direct=[entries["route",k,"r",a] for a in
                 ("sham","cut","noise_cut","rescue","direct_cut")]
            x_sham=entries["route",k,"x","sham"]
            x_cut=entries["route",k,"x","cut"]
            matched,same,wrong,irrelevant,rand=[entries["interchange",k,"r",a] for a in
                 ("matched","same_Y","mismatched","irrelevant","random")]
            for h in sorted({k,4}):
                a,b,c,d,e,n=(differences(m,"r","x",h) for m in (sham,cut,nuisance,rescue,direct,r))
                v,w=(differences(m,"x","r",h) for m in (x_sham,x_cut))
                for name,val in (("source_cut_cost",a-b),("nuisance_cost",a-c),("source_over_nuisance",c-b),
                    ("same_receiver_rescue",d-b),("sham_native_difference",a-n),("direct_cost",a-e),
                    ("symmetric_native_pair_cost",((a-b)+(v-w))/2),
                    ("source_by_context_delta",(a-b)-(v-w)),
                    ("natural_counterfactual_shift",differences(native_x,"x","r",h)-differences(r,"x","r",h))):
                    write(k,h,name,val)
                base=differences(r,"x","r",h)
                values={name:differences(m,"x","r",h)-base for name,m in
                       (("matched",matched),("same_Y",same),("mismatched",wrong),("irrelevant",irrelevant),("random",rand))}
                for name,val in values.items():
                    write(k,h,"patch_"+name,val)
                    if name!="matched":
                        write(k,h,"matched_over_"+name,values["matched"]-val)
            before=int(top(sham["scores"][k-2]))
            after=int(top(sham["scores"][k-1]))
            is_cut=int(top(cut["scores"][k-1]))
            is_noise=int(top(nuisance["scores"][k-1]))
            is_rescue=int(top(rescue["scores"][k-1]))
            if min(before,after)<0:
                category="ties"
            elif before==gold["r"] and after==gold["r"]:
                category="stay_correct"
            elif before!=gold["r"] and after==gold["r"]:
                category="correction"
            elif before==gold["r"] and after!=gold["r"]:
                category="regression"
            else:
                category="stay_wrong"
            transition.append(dict(task=task,protocol=protocol,split=fact["split"],id=gid,loop=k,
                 category=category,sham_correct=int(after==gold["r"]),cut_correct=int(is_cut==gold["r"]),
                 noise_correct=int(is_noise==gold["r"]),rescued_correct=int(is_rescue==gold["r"])))
    # Test the same frozen natural decoder on the semantic destination of the intervention.
    meta_index={(r["id"],r["protocol"],r["world"]):i for i,r in enumerate(dev.metadata)}
    for protocol in ("choice","open"):
        train=dev.choose(protocol)
        for k in (2,3,4):
            for sem,region in (("S",2),("Y",4)):
                ridge=Probe(dev,train,dev.at["trajectory"][train,k-1,24,region].astype(np.float32),sem)
                for (gid,proto),members in partition(traces,("id","protocol")).items():
                    if proto!=protocol:
                        continue
                    entries={(m["kind"],m["loop"],m["receiver"],m["arm"]):m for m in members}
                    targets=[m for m in members if m["kind"]=="route" and m["loop"]==k and m["receiver"]=="r"]
                    g=dev.groups[gid]
                    choices=g["options"] if sem=="Y" else g["S_options"]
                    expected=g["worlds"]["r"][sem]
                    a=np.stack([x[r["i"],k-1,region] for r in targets])
                    _,margin=ridge.contrast(a,[choices]*len(a),[expected]*len(a))
                    margins=dict(zip((r["arm"] for r in targets),margin))
                    for name,value in (("cut_semantic_"+sem,margins["sham"]-margins["cut"]),
                        ("nuisance_semantic_"+sem,margins["sham"]-margins["noise_cut"]),
                        ("rescue_semantic_"+sem,margins["rescue"]-margins["cut"])):
                        result.append(dict(task=task,id=gid,protocol=proto,split=g["split"],loop=k,
                                           horizon=k,metric=name,value=float(value)))
    return result,transition


def average(rs,key,float_key="value",block=False):
    metrics=[]
    for tag,group in partition(rs,key).items():
        if block:
            values=[np.mean([float(r[float_key]) for r in q]) for q in partition(group,("id",)).values()]
        else:
            values=[float(r[float_key]) for r in group]
        m,lo,hi=ci(values)
        metrics.append({**dict(zip(key,tag)),"groups":len(values),"mean":m,"lower":lo,"upper":hi})
    return metrics


def public_screen(root):
    out=[]
    data=rows(root/"public"/"screen.jsonl")
    for (name,),group in partition(data,("task",)).items():
        for k in range(4):
            correct=[int(top(r["scores"][k])==r["gold"]) for r in group]
            old=[int(top(r["scores"][max(0,k-1)])==r["gold"]) for r in group]
            sample=[str(r["greedy"][k][1])==str(r["gold_number"]) for r in group if "greedy" in r]
            out.append(dict(task=name,loop=k+1,groups=len(group),accuracy=np.mean(correct),
                 corrections=sum(a>b for a,b in zip(correct,old)),regressions=sum(a<b for a,b in zip(correct,old)),
                 greedy_groups=len(sample),greedy_accuracy=float(np.mean(sample)) if sample else ""))
    return out


def make_cases(root,groups,causal):
    all_cases=[]
    for task in TASKS:
        tr=rows(root/"causal"/task/"arms.jsonl")
        available=[r for r in causal if r["task"]==task and r["split"]=="transfer" and r["protocol"]=="open"
                      and r["loop"]==2 and r["horizon"]==4]
        cases=[]
        for name in ("source_over_nuisance","same_receiver_rescue","matched_over_same_Y"):
            candidates=[r for r in available if r["metric"]==name]
            cases.extend((key,pick(candidates,key=lambda z:z["value"])) for key,pick in
                         (("high",max),("low",min)))
        for size,q in cases:
            case=q["id"]
            ids=[r for r in tr if r["id"]==case and r["protocol"]=="open"]
            all_cases.append(dict(rule=f"{q['metric']}/{size}",target=q,input=groups[case],arms=ids))
    return all_cases


def run(root):
    result=root/"results"
    groups={r["id"]:r for r in rows(root/"data"/"groups.jsonl")}
    freeze=json.loads((root/"freeze.json").read_text())
    assert freeze["manifest_sha256"]==digest(root/"data"/"groups.jsonl")
    for name,want in freeze["implementation"].items():
        assert digest(Path(__file__).parent/name)==want
    causal,trans=[],[]
    audit={}
    for task in TASKS:
        q,r=contrasts(root/"causal"/task,task,Data(root/"development"/task,groups))
        causal+=q
        trans+=r
        for stage in ("development","independent","causal"):
            p=root/stage/task
            m=json.loads((p/"complete.json").read_text())
            assert m["manifest_sha256"]==freeze["manifest_sha256"]
            for name,sha in m["files"].items():
                assert digest(p/name)==(sha["sha256"] if isinstance(sha,dict) else sha)
            audit[f"{stage}/{task}"]=m
    raw=table(result/"behavior_worlds.csv")
    bsummary=[]
    for metric in ("correct","tie","corrected","regressed","margin","unconstrained_answer_prefix"):
        bsummary.extend({**r,"metric":metric} for r in average(raw,
            ("task","protocol","split","loop"),metric,block=True))
    tgroups=[]
    for metric in ("sham_correct","cut_correct","noise_correct","rescued_correct"):
        tgroups.extend({**r,"metric":metric} for r in average(trans,("task","protocol","split","loop","category"),metric))
    pred=table(result/"decision_predictions.csv")
    for r in pred:
        y=float(r["next_correct"])
        base=float(r["baseline"])
        aug=float(r["semantic_augmented"])
        r["brier_gain"]=(base-y)**2-(aug-y)**2
        r["state"]=("current_correct" if int(r["current_correct"]) else "current_wrong")
    d=average(pred,("task","protocol","split","loop"),"brier_gain")
    ds=average(pred,("task","protocol","split","loop","state"),"brier_gain")
    c=average(causal,("task","protocol","split","loop","horizon","metric"))
    gates=[]
    for task in TASKS:
        split_results={}
        for s in ("confirm","transfer"):
            outcomes={}
            for e in ("source_over_nuisance","same_receiver_rescue","matched_over_same_Y"):
                r=next(r for r in c if (r["task"],r["protocol"],r["split"],r["loop"],r["horizon"],r["metric"])
                       ==(task,"open",s,2,2,e))
                outcomes[e]=r["lower"]>0
            r=next((r for r in d if (r["task"],r["protocol"],r["split"],r["loop"])==(task,"open",s,"1")),None)
            outcomes["semantic_incremental_Brier"]=r is not None and r["lower"]>0
            split_results[s]=outcomes
        gates.append({"task":task,"tests":split_results,"passes":all(v for s in split_results.values() for v in s.values())})
    for name,data in (("behavior_summary",bsummary),("causal_by_group",causal),("causal_summary",c),
                      ("route_outcomes",trans),("route_outcome_summary",tgroups),
                      ("decision_summary",d),("decision_by_current_state",ds),("public_screen_summary",public_screen(root))):
        export(result/(name+".csv"),data)
    cases=make_cases(root,groups,causal)
    jsonfile(result/"case_studies.json",cases)
    jsonfile(result/"progression.json",{"tasks":gates,"cross_scale_allowed":sum(g["passes"] for g in gates)>=2})
    jsonfile(result/"execution_audit.json",{"status":"passed","groups":len(groups),"raw_checks":audit,"cases":len(cases)})
    print("gates",gates,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    run(p.parse_args().root)
