"""Render the six evidence figures from their indexed numerical artifacts."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_loop import TASKS
from summary_loop import table
from readouts_loop import partition

PALETTE=("#235789","#E57632","#158277","#9A4063","#8693A1")
TITLES=("Relation","Logic","Arithmetic")
SPLIT="transfer"


def lookup(path,fields):
    return {tuple(r[k] for k in fields):r for r in table(path)}


def err(ax,x,rows,color,fmt="o-",field="mean",**kw):
    v,lo,hi=[np.array([float(r[k]) for r in rows]) for k in (field,"lower","upper")]
    ax.errorbar(x,v,yerr=np.maximum(0,np.stack((v-lo,hi-v))),
                color=color,fmt=fmt,lw=1.05,ms=3.7,capsize=2,**kw)


def save(fig,p):
    for ext in (".png",".pdf"):
        fig.savefig(p.with_suffix(ext),dpi=320,bbox_inches="tight",pad_inches=.1)
    plt.close(fig)


def virtual_ticks(ax,n):
    for x in range(n-1,4*n-1,n):
        ax.axvline(x,color="#C7CFD8",lw=.7)
    ax.set_xticks([k*n+(n-1)/2 for k in range(4)],["K1","K2","K3","K4"])


def macro_plot(root,res,out):
    ref=lookup(root/"paper_reference"/"profile.csv",("protocol","loop","layer","metric"))
    lens=lookup(res/"candidate_lens.csv",("task","protocol","split","loop","layer"))
    fig,axs=plt.subplots(1,4,figsize=(13,3.35),layout="constrained")
    for ax,metric,label in zip(axs[:3],("mixing","column_concentration","sink_rate"),
                              ("Row-normalized mixing","Column concentration","Sink-head fraction")):
        for j,(p,name) in enumerate((("teacher_forced","Reference solution included"),
                                     ("question_only","Question only"))):
            rows=[ref[p,str(k),str(l),metric] for k in range(1,5) for l in range(24)]
            vals,lo,hi=[[float(r[key]) for r in rows] for key in ("mean","lower","upper")]
            ax.plot(vals,color=PALETTE[j],lw=1.05,label=name)
            ax.fill_between(range(96),lo,hi,color=PALETTE[j],alpha=.15,lw=0)
        virtual_ticks(ax,24)
        ax.set_title(label)
        ax.grid(axis="y",alpha=.15)
    axs[0].set_ylabel("GSM8K test · 256 paired questions")
    axs[2].legend(frameon=False,fontsize=7)
    for j,(protocol,label) in enumerate((("open","Open"),("choice","With options"))):
        values=[float(lens["relation",protocol,SPLIT,str(k),str(l)]["mean"])
                for k in range(1,5) for l in range(25)]
        axs[3].plot(values,color=PALETTE[j],lw=1.02,label=label)
    virtual_ticks(axs[3],25)
    axs[3].set_title("Relation candidate logit lens")
    axs[3].legend(frameon=False,fontsize=7)
    axs[3].grid(axis="y",alpha=.15)
    save(fig,out/"01_repeated_macro_stages")


def example_plot(res,out):
    vals=partition(table(res/"single_item_trajectories.csv"),("task",))
    fig,axs=plt.subplots(2,3,figsize=(10.5,5.3),layout="constrained",sharex=True)
    for j,t in enumerate(TASKS):
        rows=vals[t,]
        for k,key in enumerate(("S","M","Y")):
            margin=np.array([float(r[key+"_margin"]) for r in rows])
            den=max(np.quantile(np.abs(margin),.9),1e-6)
            axs[0,j].plot(margin/den,color=PALETTE[k],lw=1.1,label=key)
            axs[1,j].plot([int(r[key+"_hit"])*.75+k for r in rows],color=PALETTE[k],lw=.9)
        axs[0,j].axhline(0,color="#89939E",lw=.75,ls="--")
        axs[0,j].set_title(TITLES[j]+" · "+rows[0]["id"].split("/")[-1])
        axs[1,j].set_yticks((.38,1.38,2.38),("Source","Rule","Result"))
        for ax in axs[:,j]:
            virtual_ticks(ax,25)
    axs[0,0].legend(frameon=False,ncols=3,fontsize=8)
    axs[0,0].set_ylabel("Probe margin / within-case scale")
    axs[1,0].set_ylabel("Decoded designated value")
    save(fig,out/"02_semantic_trajectory_cases")


def availability(res,out,s,c):
    fig,axs=plt.subplots(2,3,figsize=(10,5.2),layout="constrained")
    for j,(t,title) in enumerate(zip(TASKS,TITLES)):
        for k,(sem,role,name) in enumerate((("S","source","Source at input role"),
                                           ("S","mapping","Source in map"),
                                           ("Y","readout","Result at readout"))):
            d=[s[t,"open",SPLIT,sem,"full_width",str(l),"24",role] for l in range(1,5)]
            err(axs[0,j],np.arange(1,5)+(k-1)*.07,d,PALETTE[k],field="accuracy",label=name)
        for k,(metric,name) in enumerate((("source_over_nuisance","Cut cost over nuisance"),
                                          ("same_receiver_rescue","Map-state rescue"))):
            d=[c[t,"open",SPLIT,str(l),str(l),metric] for l in (2,3,4)]
            err(axs[1,j],np.arange(2,5)+(k-.5)*.10,d,PALETTE[k],label=name)
        axs[0,j].set_title(title)
        axs[0,j].set_ylim(0,1.01)
        axs[0,j].set_xticks(range(1,5),["K1","K2","K3","K4"])
        axs[1,j].set_xticks(range(2,5),["K2","K3","K4"])
        axs[1,j].axhline(0,color="#919AA4",lw=.7,ls="--")
    axs[0,0].set_ylabel("Full-width linear probe accuracy")
    axs[1,0].set_ylabel("Native designated margin effect (nats)")
    axs[0,2].legend(frameon=False,fontsize=7)
    axs[1,2].legend(frameon=False,fontsize=7)
    save(fig,out/"03_decodability_and_causal_use")


def outcomes(res,out):
    ds=table(res/"decision_predictions.csv")
    c=lookup(res/"behavior_summary.csv",("task","protocol","split","loop","metric"))
    fig,axs=plt.subplots(2,3,figsize=(10,5.2),layout="constrained")
    for j,(t,title) in enumerate(zip(TASKS,TITLES)):
        for k,(p,name) in enumerate((("open","Open stem"),("choice","Options shown"))):
            sets=[c[t,p,SPLIT,str(l),"correct"] for l in range(1,5)]
            err(axs[0,j],np.arange(1,5)+(k-.5)*.08,sets,PALETTE[k],label=name)
        axs[0,j].set_title(title)
        axs[0,j].set_ylim(0,1)
        axs[0,j].set_xticks(range(1,5),["K1","K2","K3","K4"])
        d=[r for r in ds if (r["task"],r["protocol"],r["split"],r["loop"])==(t,"open",SPLIT,"1")]
        for a,b,k,name in ((1,1,0,"Stay correct"),(0,0,1,"Stay wrong"),(0,1,2,"Correction"),(1,0,3,"Regression")):
            q=[r for r in d if (int(r["current_correct"]),int(r["next_correct"]))==(a,b)]
            axs[1,j].scatter([float(r["confidence"]) for r in q],[float(r["Y_evidence"]) for r in q],
                  s=13,color=PALETTE[k],alpha=.6,edgecolors="none",label=f"{name} ({len(q)})")
        axs[1,j].axvline(0,color="#B3BBC3",lw=.7)
        axs[1,j].set_xlabel("K1 native correct-answer margin")
    axs[0,0].set_ylabel("Candidate accuracy ± group CI")
    axs[1,0].set_ylabel("K1 Y evidence from probe")
    axs[0,2].legend(frameon=False,fontsize=7)
    for ax in axs[1]:
        ax.legend(frameon=False,fontsize=6.4)
    save(fig,out/"04_success_failure_and_regressions")


def mechanisms(res,out,s,c):
    d=lookup(res/"decision_summary.csv",("task","protocol","split","loop"))
    fig,axs=plt.subplots(1,3,figsize=(11,3.8),layout="constrained")
    for j,(t,title) in enumerate(zip(TASKS,TITLES)):
        for k,method in enumerate(("full_width","full_width_K1_decoder")):
            v=[s[t,"open",SPLIT,"Y",method,str(l),"24","readout"] for l in range(1,5)]
            line=[float(r["accuracy"]) for r in v]
            axs[0].plot(np.arange(1,5)+(j-1)*.05,line,marker="o" if k==0 else "s",ms=3.4,
               color=PALETTE[j],ls="-" if k==0 else "--",label=title if k==0 else None)
        v=[c[t,"open",SPLIT,str(k),str(k),"matched_over_same_Y"] for k in (2,3,4)]
        err(axs[1],np.arange(2,5)+(j-1)*.09,v,PALETTE[j],label=title)
        v=[d[t,"open",SPLIT,str(k)] for k in (1,2,3)]
        err(axs[2],np.arange(1,4)+(j-1)*.09,v,PALETTE[j],label=title)
    axs[0].axhline(.25,color="#89939E",lw=.7,ls=":")
    axs[0].set_ylabel("Y decoding accuracy")
    axs[0].set_title("Per-loop vs K1-fit decoder")
    axs[0].set_xticks(range(1,5),["K1","K2","K3","K4"])
    axs[1].axhline(0,color="#89939E",lw=.7,ls=":")
    axs[1].set_ylabel("Matched minus same-Y donor (nats)")
    axs[1].set_xticks(range(2,5),["K2","K3","K4"])
    axs[1].set_title("Context-controlled map interchange")
    axs[2].axhline(0,color="#89939E",lw=.7,ls=":")
    axs[2].set_ylabel("Brier loss reduction with semantics")
    axs[2].set_xticks(range(1,4),["K1→2","K2→3","K3→4"])
    axs[2].set_title("Increment beyond native evidence")
    axs[2].legend(frameon=False,fontsize=7)
    save(fig,out/"05_competing_mechanisms")


def across(root,res,out,c):
    profiles=lookup(res/"macro.csv",("task","protocol","split","loop","layer","metric"))
    fig,axs=plt.subplots(1,2,figsize=(9.8,3.8),layout="constrained")
    for j,(t,title) in enumerate(zip(TASKS,TITLES)):
        layers=[float(profiles[t,"open",SPLIT,str(k),str(l),"mixing"]["mean"])
                for k in range(1,5) for l in range(24)]
        axs[0].plot(layers,color=PALETTE[j],lw=1.07,label=title)
        tab=lookup(root/"scope"/t/"summary.csv",("split","tier","output_loop","metric"))
        keys=[("early","cost"),("complement","cost"),("whole","over_noise"),("whole","rescue")]
        for i,(tier,kind) in enumerate(keys):
            err(axs[1],[j+(i-1.5)*.16],[tab[SPLIT,tier,"4",kind]],PALETTE[i],
                ("o","s","^","D")[i],label=("Early cut","Outside window cut","Full cut over nuisance","Full cut rescue")[i]
                if j==0 else None)
    virtual_ticks(axs[0],24)
    axs[0].set_ylabel("Attention mixing")
    axs[0].legend(frameon=False,fontsize=7)
    axs[1].set_ylabel("K4 margin after K2 cuts (nats)")
    axs[1].set_xticks(range(3),TITLES)
    axs[1].axhline(0,color="#89939E",lw=.7,ls=":")
    axs[1].legend(frameon=False,fontsize=7)
    save(fig,out/"06_cross_task_semantic_routes")


def run(root):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,
        "axes.titleweight":"bold","axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#AFB7C0",
        "pdf.fonttype":42,"savefig.facecolor":"white","axes.axisbelow":True})
    res=root/"results"
    out=res/"figures"
    out.mkdir(exist_ok=True)
    s=lookup(res/"semantic.csv",("task","protocol","split","semantic","method","loop","layer","role"))
    c=lookup(res/"causal_summary.csv",("task","protocol","split","loop","horizon","metric"))
    macro_plot(root,res,out)
    example_plot(res,out)
    availability(res,out,s,c)
    outcomes(res,out)
    mechanisms(res,out,s,c)
    across(root,res,out,c)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    run(p.parse_args().root)
