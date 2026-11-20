"""Reconstruct statistics and seven numbered figures from released GPU records."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def read(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(s) for s in f]


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf-8")


def mean_ci(x, groups):
    by_group = defaultdict(list)
    for value, key in zip(x, groups, strict=True):
        by_group[key].append(value)
    arr = np.asarray([np.mean(items,axis=0) for items in by_group.values()])
    draws = np.random.default_rng(574).integers(len(arr),size=(2500,len(arr)))
    low, up = np.quantile(arr[draws].mean(1),[.025,.975],axis=0)
    value=lambda v: np.asarray(v).tolist()
    return dict(mean=value(arr.mean(0)),low=value(low),high=value(up),n=len(x),groups=len(arr))


def choice(scores, y):
    top=max(scores)
    return scores[y]==top and scores.count(top)==1


def margin(z,y):
    return z[y]-max(v for i,v in enumerate(z) if i!=y)


def acc_rows(rows):
    methods={}
    for row in rows:
        y=row["y"]
        for name, value in row["tracks"].items():
            end=value["scores"][-1]
            methods.setdefault(name,[]).append((bool(choice(end,y)),
                  bool(choice(row["tracks"]["native"]["scores"][0],y)),row["group"],end.count(max(end))>1))
    return {name:dict(n=len(xs),correct=sum(x[0] for x in xs),tied=sum(x[3] for x in xs),
                     rescue=sum(x[0] and not x[1] for x in xs),damage=sum(x[1] and not x[0] for x in xs))
            for name,xs in methods.items()}


def geometry_summary(rows):
    result={}
    for cohort in ("development","new_entities","new_structure","validation"):
        for family in ("tableau","twohop","arc"):
            subset=[r for r in rows if (r["family"],r["cohort"])==(family,cohort)]
            if not subset:
                continue
            keys=sorted({k for r in subset for k in r["comparisons"]})
            con={}
            for k in keys:
                q=[r for r in subset if k in r["comparisons"]]
                con[k]={"time":q[0]["comparisons"][k]["time"],
                        **mean_ci([r["comparisons"][k]["rel"] for r in q],[r["group"] for r in q])}
            end=acc_rows(subset)
            first=sum(choice(r["tracks"]["native"]["scores"][0],r["y"]) for r in subset)
            result[family+"/"+cohort]=dict(items=len(subset), convergence=con, endpoint=end,native_k1=first)
    return result


def semantic_summary(rows):
    bins=defaultdict(list)
    for row in rows:
        bins[row["family"],row["cohort"],row["track"]].append(row)
    output={}
    for (family,cohort,track), rs in bins.items():
        key="/".join((family,cohort,track))
        roles=rs[0]["roles"]
        z=np.asarray([r["labels"] for r in rs])==np.asarray([r["true"] for r in rs])[:,None,:]
        groups=[r["group"] for r in rs]
        item={role:mean_ci(z[:,:,i],groups) for i,role in enumerate(roles)}
        for role in roles:
            starts=[r["onset"][role] for r in rs]
            histogram=Counter("absent" if t is None else str(t) for t in starts)
            item[role]["sustained_onset_counts"]=dict(histogram)
        ordered=None
        if "b" in roles:
            found=[(r["onset"]["b"] is not None and r["onset"]["y"] is not None) for r in rs]
            ordered=mean_ci([ok and r["onset"]["b"]<r["onset"]["y"] for ok,r in zip(found,rs)],groups)
            ordered["both_detected"]=sum(found)
        output[key]=dict(time=rs[0]["time"],roles=item, ordered_b_before_y=ordered,items=len(rs))
    return output


def behaviour_prediction(states, raw):
    records={r["item"]:r for r in raw if r["family"]=="twohop"}
    feats=defaultdict(list)
    for s in states:
        if s["family"]!="twohop" or s["track"]!="native":
            continue
        r=records[s["item"]]
        a=r["tracks"]["native"]
        mid=a["time"].index(.5)
        b_idx=s["roles"].index("b")
        h=s["true_confidence"][s["time"].index(.5)][b_idx]
        feats[s["cohort"]].append((margin(a["scores"][mid],r["y"]),h,
                           int(choice(a["scores"][-1],r["y"])),r["group"]))
    train, test = (feats[k] for k in ("development","new_structure"))
    y=np.array([r[2] for r in train])
    truth=np.array([r[2] for r in test])
    outputs={}
    brier=[]
    for name,width in (("margin",1),("margin_plus_bridge",2)):
        a=np.array([r[:width] for r in train])
        b=np.array([r[:width] for r in test])
        std=a.std(0).clip(1e-4)
        x=np.column_stack((np.ones(len(a)),(a-a.mean(0))/std))
        z=np.column_stack((np.ones(len(b)),(b-a.mean(0))/std))
        opt=minimize(lambda v: np.logaddexp(0,x@v).sum()-y@(x@v)+.5*(v[1:]**2).sum(),np.zeros(width+1))
        p=expit(z@opt.x)
        err=(p-truth)**2
        brier.append(err)
        outputs[name]=dict(brier=float(err.mean()),coef=opt.x.tolist())
    outputs["paired_brier_reduction"]=mean_ci(brier[0]-brier[1],[r[3] for r in test])
    return outputs


def semantic_checks(states, raw):
    worlds={r["item"]:r for r in raw if r["family"]=="twohop" and r["cohort"]=="new_structure"}
    paired=defaultdict(dict)
    for row in states:
        if row["item"] in worlds:
            paired[row["track"]][row["item"]]=row

    def ordered(row):
        b,y=(row["onset"][k] for k in ("b","y"))
        return int(b is not None and y is not None and b < y)

    order={}
    for a,b in (("A8","A16"),("A16","A32"),("B8","B16"),("B16","B32"),
                ("A32","linear"),("B32","linear"),("native","linear"),
                ("A32","norm_linear"),("B32","norm_linear")):
        if a not in paired or b not in paired:
            continue
        x,y=paired[a],paired[b]
        order[f"{a}/{b}"]=mean_ci([ordered(x[i])-ordered(y[i]) for i in worlds],
                                  [r["group"] for r in worlds.values()])

    conditioned={}
    for method in ("native","A32","B32","linear","norm_linear","random"):
        for correct in (False,True):
            chosen=[i for i,r in worlds.items()
                    if choice(r["tracks"][method]["scores"][-1],r["y"])==correct]
            if not chosen:
                continue
            rows=[paired[method][i] for i in chosen]
            index=rows[0]["roles"].index("b")
            conditioned[f"{method}/{int(correct)}"]={
                "time":rows[0]["time"],
                "bridge":mean_ci([[x[index]==r["true"][index] for x in r["labels"]] for r in rows],
                                 [r["group"] for r in rows]),
                "ordered_b_before_y":mean_ci([ordered(r) for r in rows],[r["group"] for r in rows])}
    return dict(paired_order=order, terminal_outcome_conditioned=conditioned)


def causal_summary(rows):
    groups=defaultdict(dict)
    for r in rows:
        groups[r["cohort"],r["group"],r["method"],r["t"]][r["arm"]]=r
    aggregate=defaultdict(list)
    for (cohort,g,method,t), obs in groups.items():
        a=obs["native"]
        b=obs["linear_clean"]
        for arm in ("semantic","same_y_different_b","same_b_different_s","random","unrelated_token",
                    "linear_semantic","semantic_minus_random","same_y_minus_random"):
            edit,control= (obs[arm],b if arm=="linear_semantic" else a) if arm in obs else (
                obs["semantic" if arm=="semantic_minus_random" else "same_y_different_b"],obs["random"])
            target=edit["target"]
            dz=margin(edit["scores"],target)-margin(control["scores"],target)
            hit=int(choice(edit["scores"],target))
            aggregate[cohort,method,t,arm].append((g,dz,hit))
    out={}
    for (cohort,method,t,arm),rs in aggregate.items():
        out["/".join(map(str,(cohort,method,t,arm)))]=dict(
            **mean_ci([r[1] for r in rs],[r[0] for r in rs]), hits=sum(r[2] for r in rs))
    by_event=defaultdict(dict)
    for (cohort,group,method,t), obs in groups.items():
        row=obs["same_y_different_b"]
        target=row["target"]
        by_event[cohort,method,group][t]=margin(row["scores"],target)-margin(obs["random"]["scores"],target)
    for cohort in ("development","new_structure"):
        for method in ("native","A16","B16"):
            cells=[(g,x[.75]-x[.25]) for (co,m,g),x in by_event.items() if (co,m)==(cohort,method)]
            out[f"{cohort}/{method}/late_minus_early_same_y_minus_random"]=mean_ci(
                [x[1] for x in cells],[x[0] for x in cells])
    return out


COLORS={"native":"#242c39","A32":"#206785","B32":"#a94d31","linear":"#7b668d",
        "norm_linear":"#7f875b","random":"#909090","s":"#206785","m":"#7b668d","b":"#008361","y":"#a94d31"}


def figure(out, index, name, fig):
    for extension in ("png","pdf"):
        fig.savefig(out/f"figure_{index}_{name}.{extension}",bbox_inches="tight",dpi=220)
    plt.close(fig)


def charts(root, raw, summary, semantic, causal):
    out=root/"figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42})
    held=[r for r in raw if r["family"]=="twohop" and r["cohort"]=="new_structure"]
    gs=[r["group"] for r in held]

    def curve(ax, label, x, y, c=None, ls="-"):
        ax.plot(x,y["mean"],label=label,color=c,linestyle=ls,linewidth=1.7)
        ax.fill_between(x,y["low"],y["high"],color=c,alpha=.13,linewidth=0)

    fig, ax=plt.subplots(figsize=(7,3.3))
    dist=[[v for v in r["tracks"]["native"]["to_k1"][:-1]]+[r["before_native_norm"]["to_k1"],
                              r["tracks"]["native"]["to_k1"][-1]] for r in held]
    curve(ax,"Native K1 state",np.arange(50),mean_ci(dist,gs),COLORS["native"])
    ax.axvline(48.5,color="#828282",ls=":")
    ax.set(xlabel="Residual update index (49 = loop-end norm)",ylabel="Relative full-state distance from K1",
           title="Native K1 → K2 trajectory on two-hop structural holdout")
    figure(out,1,"native_residuals",fig)

    fig, axes=plt.subplots(1,2,figsize=(9,3.3),constrained_layout=True)
    for ax, family in zip(axes,("A","B")):
        for m,col in zip((2,4,8,16,32),plt.cm.viridis(np.linspace(.1,.85,5))):
            values=[r["tracks"][f"{family}{m}"] for r in held]
            curve(ax,f"m={m}",values[0]["time"],mean_ci([r["to_k1"] for r in values],gs),col)
        ax.set(title="Full-loop Euler" if family=="A" else "Residual branch micro-steps",
               xlabel="Constructed time",ylabel="Relative distance from native K1")
    axes[0].legend(frameon=False,ncols=2)
    figure(out,2,"dynamics_by_step_size",fig)

    fig,ax=plt.subplots(figsize=(7,3.3))
    for idx,key in enumerate(("tableau/new_structure","twohop/new_structure","arc/validation")):
        q=summary["arc/high_resolution" if key=="arc/validation" else key]["convergence"]
        for family in ("A","B"):
            right=[2,4,8,16,32]
            es=[q[f"{family}{i//2}/{family}{i}"]["mean"][-1] for i in right]
            ax.plot(right,es,marker=("o","s","^")[idx],ls="-" if family=="A" else "--",
                    color=("#206785","#a94d31","#6d608b")[idx],label=f"{family} · {key.split('/')[0]}")
    ax.set(xscale="log",yscale="log",xticks=[2,4,8,16,32],xticklabels=["2","4","8","16","32"],
           xlabel="Finer resolution m",ylabel="Relative distance between adjacent endpoints")
    ax.legend(frameon=False,ncols=2)
    figure(out,3,"convergence",fig)

    fig,axes=plt.subplots(1,3,figsize=(11,3.1),sharey=True,constrained_layout=True)
    for ax,method in zip(axes,("native","A32","B32")):
        q=semantic["twohop/new_structure/"+method]
        for role,data in q["roles"].items():
            curve(ax,role.upper(),q["time"],data,COLORS[role])
        ax.axhline(1/3,ls=":",color="#777777")
        ax.set(title=method,xlabel="Native branch fraction or artificial time",ylim=(0,1))
    axes[0].set_ylabel("Frozen semantic readout accuracy")
    axes[0].legend(frameon=False,ncols=2)
    figure(out,4,"semantic_readout",fig)

    fig,axes=plt.subplots(1,2,figsize=(8.8,3.25),constrained_layout=True)
    for m in ("native","A32","B32","linear","norm_linear","random"):
        q=[r["tracks"][m] for r in held]
        curve(axes[0],m,q[0]["time"],mean_ci([r["to_k2"] for r in q],gs),COLORS[m])
        curve(axes[1],m,q[0]["time"],mean_ci([[margin(z,r["y"]) for z in p["scores"]]
                                            for p,r in zip(q,held)],gs),COLORS[m])
    for ax in axes:
        ax.set_xlabel("Native branch fraction or artificial time")
    axes[0].set_ylabel("Distance from native K2")
    axes[1].set_ylabel("Gold − strongest foil logit")
    axes[0].legend(frameon=False,ncols=2,fontsize=8)
    figure(out,5,"computed_vs_interpolation",fig)

    fig,axes=plt.subplots(1,3,figsize=(10.2,3.15),sharey=True,constrained_layout=True)
    times=(.25,.5,.75)
    variables=(("semantic","Same rule, new bridge","#206785"),
               ("same_y_different_b","Same final answer, new bridge","#a94d31"),
               ("random","Dose-matched random","#777777"),("linear_semantic","Interpolation edit","#7b668d"))
    for ax,method in zip(axes,("native","A16","B16")):
        for arm,label,c in variables:
            x=[causal[f"new_structure/{method}/{t}/{arm}"] for t in times]
            curve(ax,label,times,{key:[v[key] for v in x] for key in ("mean","low","high")},c)
        ax.axhline(0,ls=":",color="#777777")
        ax.set(title=method,xlabel="Edit time",xticks=times)
    axes[0].set_ylabel("Target logit margin change")
    axes[2].legend(frameon=False,fontsize=7)
    figure(out,6,"causal_edits",fig)

    fig,axes=plt.subplots(1,2,figsize=(9,3.35),constrained_layout=True)
    gr=["00","01","10","11"]
    for idx,st in enumerate(gr):
        rs=[r for r in raw if r["family"]=="arc" and "A32" in r["tracks"]
            and "".join(str(int(v==r["y"])) for v in r["native_scan"])==st]
        if not rs:
            continue
        groups=[r["group"] for r in rs]
        for ax,track in zip(axes,("A32","B32")):
            v=[r["tracks"][track] for r in rs]
            curve(ax,f"{st} (n={len(rs)})",v[0]["time"],mean_ci(
                [[margin(z,r["y"]) for z in q["scores"]] for q,r in zip(v,rs)],groups),
                ("#242c39","#008361","#c06b36","#206785")[idx])
            ax.set(title=track,xlabel="Constructed time",ylabel="Gold − strongest foil logit")
            ax.yaxis.set_major_locator(MaxNLocator(5))
    axes[1].legend(frameon=False)
    figure(out,7,"arc_transition_strata",fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    args=p.parse_args()
    r=args.root
    assert json.loads((r/"COMPLETE.json").read_text())["status"]=="PASS"
    raw, semantic, causal, pilot, costs = (read(r/f"{name}.jsonl") for name in
                                            ("trajectories","semantic_predictions","causal","pilot","runtime"))
    assert len({x["item"] for x in raw})==len(raw)
    a=[x for x in raw if x["family"]=="arc"]
    assert len(a)==299
    d=geometry_summary(raw)
    panel=[r for r in a if "A32" in r["tracks"]]
    d["arc/high_resolution"]=geometry_summary(panel)["arc/validation"]
    s=semantic_summary(semantic)
    c=causal_summary(causal)
    stats={"geometry":d,"semantic":s,"causal":c,"prediction":behaviour_prediction(semantic,raw),
           "semantic_checks":semantic_checks(semantic,raw),
           "pilot_geometry":geometry_summary(pilot),
           "arc_native_strata":{k:sum("".join(str(int(z==x["y"])) for z in x["native_scan"])==k for x in a)
                                for k in ("00","01","10","11")},
           "arc_scan_vs_main_disagreements":sum(
               x["native_scan"]!=[int(np.argmax(x["tracks"]["native"]["scores"][i])) for i in (0,-1)] for x in a),
           "profile_wall_s":sum(x["wall_s"] for x in costs),"profile_peak_allocated":max(x["peak_allocated"] for x in costs),
           "profile_counters":dict((k,sum(x["counters"][k] for x in costs)) for k in costs[0]["counters"])}
    dump(r/"SUMMARY.json",stats)
    charts(r,raw,d,s,c)
    print("ANALYSIS COMPLETE",len(raw),len(semantic),len(causal),flush=True)


if __name__=="__main__":
    main()
