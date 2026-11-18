"""Generate figures from the frozen benchmark and intervention tables."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(p):
    with p.open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"axes.spines.top":False,
       "axes.spines.right":False,"axes.linewidth":.85,"savefig.facecolor":"white","pdf.fonttype":42})


def save(fig,out,label):
    fig.savefig(out/f"{label}.png",dpi=300,bbox_inches="tight",pad_inches=.1)
    fig.savefig(out/f"{label}.pdf",bbox_inches="tight",pad_inches=.1)
    plt.close(fig)


COLORS=("#0f4d92","#b64342","#42949e","#767676")
TASKS={"relation":"Relation","logic":"Logic","arithmetic":"Arithmetic",
       "prontoqa":"PrOntoQA","arc":"ARC-Challenge","hellaswag":"HellaSwag"}


def trajectory(study,out):
    data=rows(study/"natural_behavior.csv")
    fig,ax=plt.subplots(1,3,figsize=(11.7,3.15),layout="constrained")
    plans=[
       ("benchmarks","correct","worlds",("arc","prontoqa","hellaswag"),"Official benchmark items"),
       ("controlled","correct","worlds",("relation","logic","arithmetic"),"All factorial worlds"),
       ("controlled","joint","groups",("relation","logic","arithmetic"),"All four worlds correct"),
    ]
    for p,(stage,correct,total,tasks,title) in zip(ax,plans):
        for i,t in enumerate(tasks):
            d=[r for r in data if r["study"]=="expanded" and r["stage"]==stage and r["family"]=="all" and r["task"]==t]
            y=[int(r[correct])/int(r[total]) for r in d]
            p.plot(range(1,5),y,"o-",lw=1.55,ms=4,color=COLORS[i],label=TASKS[t])
        p.set(title=title,xticks=range(1,5),xlabel="Recurrence depth K",ylim=(0,1.03))
        p.legend(frameon=False,loc="best",fontsize=8)
    ax[0].set_ylabel("Accuracy")
    save(fig,out,"natural_depth")


def outline(p,d,label,color,est="estimate",xoff=0):
    x=np.array([int(r["K"]) for r in d])+xoff
    v=np.array([float(r[est]) for r in d])
    lo=np.array([float(r["lower"]) for r in d]); hi=np.array([float(r["upper"]) for r in d])
    p.errorbar(x,v,yerr=np.maximum([v-lo,hi-v],0),marker="o",ms=3.6,lw=1.3,capsize=2.0,label=label,color=color)
    p.axhline(0,color="#999999",ls=":",lw=.7)


def contextual(study,out):
    pairs=rows(study/"causal_effects.csv")
    data=[r for r in pairs if r["experiment"]=="A-relation" and r["subset"]=="all"]
    alias=rows(study/"renaming"/"effects.csv")
    fig,ax=plt.subplots(1,3,figsize=(11.8,3.15),layout="constrained")
    for i,(reg,label) in enumerate((("payload","Same-ID box tokens"),("binding","Changed owner names"),("prefix","Whole source prefix"))):
        for j,target in enumerate(("final","intermediate")):
            d=[r for r in data if r["region"]==reg and r["endpoint"]==target and r["contrast"]=="semantic-minus-random"]
            outline(ax[j],d,label,COLORS[i],xoff=(i-1)*.07)
    for i,(contrast,label) in enumerate((("same_names-minus-random","Same names"),
                                        ("renamed_matched-minus-random","Renamed donor"))):
        d=[r for r in alias if r["partition"]=="all" and r["endpoint"]=="final" and r["contrast"]==contrast]
        outline(ax[2],d,label,COLORS[i],est="effect",xoff=(i-.5)*.07)
    for p,title in zip(ax,("Final answer","Intermediate query","Independent renaming cohort")):
        p.set_title(title,loc="left",fontweight="semibold")
        p.set(xticks=range(1,5),xlabel="Readout depth K")
        p.legend(frameon=False,fontsize=7.4)
    ax[0].set_ylabel("Semantic effect vs random (nats)")
    save(fig,out,"same_token_context")


def process(study,out):
    data=rows(study/"causal_effects.csv")
    runs=("explicit_open","k2_both","k3_both","k2k3_both")
    labels=("Open","K2 cut","K3 cut","K2 + K3 cut")
    names=("injected","erase_k1_source","erase_k1_map","erase_k1_suffix")
    names_label=("Intact","Source reset","Mapping reset","Suffix reset")
    fig,ax=plt.subplots(1,3,figsize=(12,3.25),layout="constrained")
    for p,regs,conditions,local,title in ((ax[0],"routes",runs,labels,"Cut source edges"),
            (ax[1],"boundary",names,names_label,"Restore receiver at K1 boundary")):
        for i,(arm,lab) in enumerate(zip(conditions,local)):
            d=[r for r in data if r["experiment"]=="B-relation" and r["region"]==regs and r["contrast"]==arm]
            outline(p,d,lab,COLORS[i],xoff=(i-1.5)*.065)
        p.set_title(title,loc="left",fontweight="semibold")
        p.set(xlabel="Readout depth K",xticks=range(1,5))
        p.legend(frameon=False,fontsize=7.5)
    witnesses=rows(study/"trajectory_witnesses.csv")
    for j,(role,lab) in enumerate((("source","Source"),("mapping","Mapping"),("query_options","Query + options"),("readout","Readout"))):
        d=[r for r in witnesses if r["experiment"]=="B-relation" and r["route"]=="implicit"
           and r["witness"]==f"states/{role}" and r["event"].endswith(":norm")]
        ax[2].plot(range(1,5),[float(r["median"]) for r in d],"o-",ms=3.5,lw=1.4,color=COLORS[j],label=lab)
    ax[2].set(xlabel="Recurrence depth K",xticks=range(1,5),ylabel="Median per-token L2 change")
    ax[2].set_title("Post-loop state differences",loc="left",fontweight="semibold")
    ax[2].legend(frameon=False,fontsize=7.5)
    ax[0].set_ylabel("Counterfactual margin change (nats)")
    save(fig,out,"loop_routes")


def comparison(study,out):
    all_rows=rows(study/"cross_task_causal_matrix.csv")
    selected=[]
    for collection,name,region in (("expanded","main-relation","payload"),
                                   ("expanded","main-logic","payload"),
                                   ("expanded","main-arithmetic","payload"),
                                   ("screened","C-logic","prefix"),
                                   ("screened","C-arithmetic","prefix")):
        s=[r for r in all_rows if (r["dataset"],r["investigation"],r["edited_region"])==(collection,name,region)]
        if s:
            selected.append(s[0])
    fig,ax=plt.subplots(1,2,figsize=(9,3.5),layout="constrained",sharey=True)
    for j,key in enumerate(("final","intermediate")):
        p=ax[j]
        p.axvline(0,color="#999999",ls=":",lw=.7)
        for i,r in enumerate(selected[::-1]):
            y=i
            v=float(r[f"{key}_nats_vs_random"])
            lo,hi=float(r[f"{key}_lower"]),float(r[f"{key}_upper"])
            p.errorbar(v,y,xerr=[[max(v-lo,0)],[max(hi-v,0)]],marker="o",color=COLORS[int(r["dataset"]=="screened")],
                       lw=1.35,ms=4,capsize=2.5)
        p.set_xlabel("Effect at K4 (nats)")
        p.set_title(("Final choices","Intermediate queries")[j],loc="left",fontweight="semibold")
    labels=[f"{TASKS[r['task']]} · {'screened prefix' if r['dataset']=='screened' else 'broad payload'}" for r in selected[::-1]]
    ax[0].set_yticks(range(len(labels)),labels)
    save(fig,out,"cross_task_effects")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("study",type=Path)
    a=p.parse_args()
    out=a.study/"figures"
    out.mkdir(exist_ok=True)
    style()
    trajectory(a.study,out)
    contextual(a.study,out)
    process(a.study,out)
    comparison(a.study,out)
