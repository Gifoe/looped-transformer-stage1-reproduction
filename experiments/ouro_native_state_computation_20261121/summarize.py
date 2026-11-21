"""Paired question/group analysis of the frozen native-computation experiment."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from native import load, save, sha

TRANSITIONS=("1from2","2from1","2from3","3from2","3from4","4from3")
COHORTS=("dev","confirm_arc","confirm_external","controlled")
MODES=("attention","mlp")


def ci(v):
    a=np.asarray(v,dtype=float)
    if not len(a):
        return None
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite effect in metric")
    rng=np.random.default_rng(115821)
    b=a[rng.integers(len(a),size=(3000,len(a)))].mean(1)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),ci=np.quantile(b,[.025,.975]).tolist())


def pick(x):
    m=max(x)
    return x.index(m) if x.count(m)==1 else None


def margin(v,g):
    return v[g]-max(x for j,x in enumerate(v) if j!=g)


def behavior(ob):
    if not ob:
        return {}
    cohorts={}
    for p in ("gen","tf","tf_sum","letter"):
        ks=[[q["k"][str(j)][p] for q in ob] for j in range(1,5)]
        good=[[k==r["gold"] for k,r in zip(pred,ob,strict=True)] for pred in ks]
        cohorts[p]=dict(n=len(ob),correct=[sum(c) for c in good],
            unresolved=[sum(k is None for k in p) for p in ks],
            transitions=[dict(Counter("CC" if a and b else "CW" if a else "WC" if b else "WW"
                         for a,b in zip(x,y,strict=True))) for x,y in zip(good,good[1:])])
    return cohorts


def atlas(rows,c):
    if not rows:
        return {}
    metrics={}
    for k in range(3):
        layer_panels=[]
        for l in range(24):
            rr=[q["measures"][k*24+l] for q in rows]
            parts={}
            for key in MODES:
                a=[q[key] for q in rr]
                parts[key]={m:np.median([q[m] for q in a],axis=0).tolist()
                            for m in ("pre","projected","actual","projection_actual")}
                parts[key]["max_residual"]=float(max(max(x["residual"]) for x in a))
                parts[key]["norm_remainder_median"]=float(np.median([q["post_norm"]["remainder_rms"] for q in a]))
                parts[key]["norm_actual_median"]=float(np.median([q["post_norm"]["actual_rms"] for q in a]))
                parts[key]["post_rms_ratio"]=np.median([np.asarray(x["post_norm"]["single_rms"])/max(x["post_norm"]["actual_rms"],1e-9) for x in a],axis=0).tolist()
            layer_panels.append(parts)
        metrics[f"{k+1}to{k+2}"]=layer_panels
    return dict(metrics=metrics,n=len(rows),tokens=[min(q["tokens"] for q in rows),max(q["tokens"] for q in rows)],
                seconds=sum(q["gpu_seconds"] for q in rows))


def causal(rows, by, obs):
    b={q["id"]:q for q in obs}
    n=len(rows)
    result={}
    factors={}
    eligibility=sum(len(x["candidate_first_ids"])==len(set(x["candidate_first_ids"])) and bool(x["candidate_first_ids"]) for x in rows)
    for t in TRANSITIONS:
        rr=[r for r in rows if t in r["tests"]]
        recv=int(t[0])
        k=min(recv,int(t[-1]))
        if not rr:
            continue
        result[t]={}
        for m in MODES:
            out={}
            for a in rr[0]["tests"][t][m]:
                effects=[]
                ratios=[]
                token_result=[]
                duplicate=0
                for item in rr:
                    state=item["tests"][t][m][a]
                    effects.append(state["centered_progress"])
                    ratios.append(state["maximum_state_ratio"])
                    x=item["candidate_first_ids"]
                    if len(x)!=len(set(x)) or not x:
                        duplicate+=1
                        continue
                    g=by[item["id"]]["gold"]
                    scores=item["native_first"][recv-1]
                    patched=state["rank_tokens"]
                    token_result.append((margin(patched,g)-margin(scores,g),pick(scores)==g,pick(patched)==g))
                out[a]=dict(progress=ci(effects),ood2x=sum(v>2 for v in ratios),dose=ci([q["tests"][t][m][a]["dose"] for q in rr]),
                        first_token_margin=ci([x[0] for x in token_result]),first_token_excluded=duplicate,
                        first_token_correct=sum(x[2] for x in token_result),
                        first_token_rescue=sum(not x[1] and x[2] for x in token_result),
                        first_token_harm=sum(x[1] and not x[2] for x in token_result))
            controls=[a for a in out if a!="factor"]
            d={a:ci([q["tests"][t][m]["factor"]["centered_progress"]-q["tests"][t][m][a]["centered_progress"] for q in rr]) for a in controls}
            st={}
            for prot in ("gen","tf"):
                strata={key:[] for key in ("CC","CW","WC","WW")}
                for q in rr:
                    item=b.get(q["id"])
                    if item is None:
                        continue
                    g=item["gold"]
                    code="".join("C" if item["k"][str(j)][prot]==g else "W" for j in (k,k+1))
                    strata[code].append(q["tests"][t][m]["factor"]["centered_progress"])
                st[prot]={k:ci(v) for k,v in strata.items()}
            result[t][m]=dict(arms=out,contrast=d,stratified_factor=st)
    for name in ("fixed","paired_rewrite"):
        r=[q["response"][name]["response"] for q in rows if name in q["response"]]
        if r:
            factors[name]={"gain":[ci([s[str(k)]["gain"] for s in r]) for k in range(1,5)],
              "ratio":[ci([s[str(k)]["vs_previous_response"]["ratio"] for s in r]) for k in range(2,5)],
              "cosine":[ci([s[str(k)]["vs_previous_response"]["cosine"] for s in r]) for k in range(2,5)],
              "attn_effect":[ci([s[str(k)]["attention_change"] for s in r]) for k in range(1,5)],
              "mlp_effect":[ci([s[str(k)]["mlp_change"] for s in r]) for k in range(1,5)],
              "vocab_response":[ci([s[str(k)]["vocab_logit_response_rms"] for s in r]) for k in range(1,5)]}
    return dict(n=n,predictions=result,response=factors,first_token_distinguishable_n=eligibility,
                 seconds=sum(x["gpu_seconds"] for x in rows))


def complete_answers(r, labels):
    out={}
    for tr in ("1from2","2from1"):
        k=int(tr[0])-1
        groups={}
        for mode in MODES:
            arms={}
            for name in ("factor","content","joint"):
                raw=[]
                for q in r:
                    g=labels[q["id"]]["gold"]
                    a=[float(np.mean(x)) for x in q["native"][k]]
                    b=[float(np.mean(x)) for x in q["tests"][tr][mode][name]]
                    raw.append((margin(b,g)-margin(a,g),pick(a)==g,pick(b)==g))
                arms[name]=dict(margin_effect=ci([a[0] for a in raw]),correct_before=sum(x[1] for x in raw),
                  correct_after=sum(x[2] for x in raw),rescues=sum(not x[1] and x[2] for x in raw),
                  harms=sum(x[1] and not x[2] for x in raw))
            groups[mode]=arms
        out[tr]=groups
    return out


def origin(rows,labels):
    out={}
    for mode in rows[0]["mechanisms"]:
        result={}
        data=[q["mechanisms"][mode] for q in rows]
        proper="route" if mode=="attention" else "gate"
        content="value" if mode=="attention" else "up"
        for arm in data[0]:
            result[arm]=dict(readout_loss=ci([r[arm]["readout_loss"] for r in data]),
                 factor_loss=ci([r[arm]["factor_loss"] for r in data]),
                 boundary_change=ci([r[arm]["hidden_distance"] for r in data]),
                 ood2x=sum(r[arm]["state_norm_max_ratio"]>2 for r in data),
                 components={z:ci([r[arm]["perturb_components"][z] for r in data]) for z in ("q","k","v","a","g_pre","u")})
        c={x:ci([r["drop"]["factor_loss"]-r[x]["factor_loss"] for r in data])
           for x in ("matched_random","shifted_tokens","other_write")}
        restores={x:ci([r["drop"]["readout_loss"]-r[x]["readout_loss"] for r in data])
           for x in ((proper,content,"query","key","mediator_only") if mode=="attention" else (proper,content,"mediator_only"))}
        read=[]
        for q in rows:
            ids=q["candidate_first_ids"]
            if len(ids)!=len(set(ids)) or not ids:
                continue
            gold=labels[q["id"]]["gold"]
            r=q["mechanisms"][mode]
            read.append(dict(damage=margin(r["drop"]["rank_tokens"],gold)-margin(q["native_first"][-1],gold),
                 rescue=margin(r[proper]["rank_tokens"],gold)-margin(r["drop"]["rank_tokens"],gold),
                 other=margin(r[content]["rank_tokens"],gold)-margin(r["drop"]["rank_tokens"],gold),
                 native_hit=pick(q["native_first"][-1])==gold,dropped_hit=pick(r["drop"]["rank_tokens"])==gold,
                 restored_hit=pick(r[proper]["rank_tokens"])==gold))
        out[mode]=dict(arms=result,origin_specificity=c,restore=restores,
             first_token={k:ci([q[k] for q in read]) for k in ("damage","rescue","other")},
             correct=[sum(q[k] for q in read) for k in ("native_hit","dropped_hit","restored_hit")])
    return out


def semantics(rows,g):
    groups={q["group"]:q for q in g}
    out={}
    for f in sorted({r["family"] for r in rows}):
        usable=[r for r in rows if r["family"]==f]
        out[f]={}
        donor_keys=sorted({t for r in usable for t in r["donors"]})
        for dn in donor_keys:
            record={}
            for m in MODES:
                record[m]={}
                for a in ("factor","content","joint","random"):
                    scores=[]
                    for r in usable:
                        if dn not in r["donors"]:
                            continue
                        meta=groups[r["group"]]
                        pool={"source":meta["worlds"][1],"rule":meta["worlds"][2],"joint":meta["worlds"][3],**meta["extra"]}
                        if dn=="same_answer_other_bridge":
                            target=meta["worlds"][1]["gold"]
                        elif dn in ("same_bridge_other_answer","same_bridge_different_source"):
                            target=meta["worlds"][0]["gold"]
                        else:
                            target=pool[dn]["gold"]
                        donor=pool[dn]["gold"]
                        vals=r["donors"][dn]["arms"][m][a]["scores"]
                        control=r["donors"][dn]["arms"][m]["random"]["scores"]
                        native=r["native_recipient"]
                        scores.append((margin(vals,target)-margin(native,target),
                            margin(vals,target)-margin(control,target),pick(vals)==target,pick(vals)==donor,pick(native)==target,
                            pick(control)==target,target!=donor))
                    record[m][a]=dict(n=len(scores),target_margin=ci([q[0] for q in scores]),
                      relative_random_margin=ci([q[1] for q in scores]),target_hits=sum(q[2] for q in scores),
                      donor_hits=sum(q[3] for q in scores),native_hits=sum(q[4] for q in scores),
                      random_hits=sum(q[5] for q in scores),different_target_and_donor=sum(q[6] for q in scores))
            out[f][dn]=record
    return out


def ruler(rows, behavior, atlases, candidates, extra):
    byb={r["id"]:r for r in behavior}
    bya={r["id"]:r for r in atlases}
    X,Y=[],[]
    for q in rows:
        if q["id"] not in byb or q["id"] not in bya:
            continue
        b,a=byb[q["id"]],bya[q["id"]]
        r1,r2=b["k"]["1"],b["k"]["2"]
        scores=sorted([x["mean"] for x in r1["likelihood"]],reverse=True)
        l=candidates["candidate_attention_layer"]-1
        record=a["measures"][l]
        z=[scores[0]-scores[1],record["state_rms"],record["attention"]["entropy"],record["update_rms"]]
        if extra:
            d=q["response"]["fixed"]["response"]["2"]
            z.extend([record["attention"]["pre"][0],d["vs_previous_response"]["cosine"],d["attention_change"]])
        if r1["tf"]==b["gold"]:
            continue
        X.append(z)
        Y.append(r2["tf"]==b["gold"])
    return np.asarray(X),np.asarray(Y,dtype=float)


def logistic(train,y,test):
    mean=train.mean(0)
    sd=train.std(0).clip(1e-6)
    x=np.column_stack((np.ones(len(train)),((train-mean)/sd).clip(-8,8)))
    test=np.column_stack((np.ones(len(test)),((test-mean)/sd).clip(-8,8)))
    w=np.zeros(x.shape[-1])
    sigmoid=lambda x:1/(1+np.exp(-x.clip(-20,20)))
    for _ in range(1600):
        z=sigmoid(x@w)-y
        grad=x.T@z/len(y)
        grad[1:]+=.1*w[1:]
        w-=.1*grad
    return sigmoid(test@w)


def predictor_increment(raw,c):
    dev=raw["dev"]
    out={}
    for phase in ("confirm_arc","confirm_external"):
        z=raw[phase]
        predictions=[]
        for extra in (False,True):
            x,y=ruler(dev["causal"],dev["behavior"],dev["atlas"],c,extra)
            q,t=ruler(z["causal"],z["behavior"],z["atlas"],c,extra)
            if len(set(y))<2 or len(t)<2:
                break
            p=logistic(x,y,q)
            predictions.append(dict(loss=(p-t)**2,n=len(t),positives=int(t.sum()),train_n=len(y)))
        if len(predictions)==2:
            b,p=predictions
            out[phase]=dict(n=b["n"],train_n=b["train_n"],positives=b["positives"],
                       baseline_brier=float(b["loss"].mean()),extra_brier=float(p["loss"].mean()),
                       improvement=ci((b["loss"]-p["loss"]).tolist()))
    return out


def run(root,inputs,structure):
    labels={q["id"]:q for p in COHORTS for q in inputs[p]}
    c=load(root/"CANDIDATES.json")
    data={"candidates":c,"gate":load(root/"GATE.json"),"phases":{}}
    raw={}
    for phase in COHORTS:
        def read(name,inner=None):
            path=root/f"{name}_{phase}.json"
            if not path.exists():
                return []
            r=load(path)
            return r if inner is None else r[inner]
        nt,nb,nc=read("ATLAS"),read("BEHAVIOR","rows"),read("CAUSAL","rows")
        raw[phase]=dict(atlas=nt,behavior=nb,causal=nc)
        q=dict(behavior=behavior(nb),atlas=atlas(nt,c),causal=causal(nc,labels,nb))
        full=read("FULL_PATCH","rows")
        if full:
            q["full_patch"]=complete_answers(full,labels)
        data["phases"][phase]=q
    raw_sem=root/"SEMANTIC_NATIVE.json"
    if raw_sem.exists():
        data["semantics"]=semantics(load(raw_sem)["rows"],structure)
    form=root/"formation"
    if (form/"ORIGIN_LOCK.json").exists():
        data["origin_lock"]=load(form/"ORIGIN_LOCK.json")
        data["formation"]={p.stem.removeprefix("ORIGIN_"):origin(row["rows"],labels)
                  for p in form.glob("ORIGIN_*json") if p.stem not in ("ORIGIN_LOCK","ORIGIN_PREDICTIONS","ORIGIN_SCREEN")
                  if (row:=load(p))["rows"]}
    data["increment"]=predictor_increment(raw,c)
    save(root/"SUMMARY.json",data)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("results",type=Path)
    p.add_argument("manifest",type=Path)
    p.add_argument("structure",type=Path)
    a=p.parse_args()
    run(a.results,load(a.manifest),load(a.structure))

