"""Generate the seven registered publication figures from SUMMARY.json."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from native import load


COLORS=("#186a7c","#a45948","#82768f")
MODES=("attention","mlp")
PHASES=("confirm_arc","confirm_external","controlled")
TITLES=("ARC-Challenge","OpenBookQA","Structured")


def export(fig,root,i,name):
    for suffix in ("pdf","png"):
        fig.savefig(root/f"figure_{i}_{name}.{suffix}",dpi=250,bbox_inches="tight")
    plt.close(fig)


def interval(ax,x,t,color,**kwargs):
    y=t["mean"]
    lo,hi=t["ci"]
    ax.errorbar(x,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],color=color,fmt="o",ms=4,capsize=3,**kwargs)


def plot(s,r):
    phases=s["phases"]
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),sharey=True,constrained_layout=True)
    for ax,ph,title,color in zip(axes,PHASES,TITLES,COLORS,strict=True):
        for key,style,name in (("gen","-","Open generation"),("tf","--","Full candidate mean LL"),("letter",":","Option letter")):
            t=phases[ph]["behavior"][key]
            ax.plot([1,2,3,4],np.array(t["correct"])/t["n"],style,color=color,label=name,marker="o",ms=3)
        ax.set(title=title,xlabel="Native loop budget",xticks=[1,2,3,4])
    axes[0].set_ylabel("Accuracy by protocol");axes[-1].legend(frameon=False,fontsize=7)
    export(fig,r,1,"native_behavior")

    for idx,mode,labels in ((2,"attention",("Route","Value","Interaction")),(3,"mlp",("Gate","Up","Interaction"))):
        a=phases["dev"]["atlas"]["metrics"]
        fig,axes=plt.subplots(2,3,figsize=(10.2,5.4),sharex=True,constrained_layout=True)
        for i,t in enumerate(("1to2","2to3","3to4")):
            d=[x[mode] for x in a[t]]
            for panel,key in enumerate(("pre","projected")):
                ax=axes[panel,i]
                for j in range(3):
                    ax.plot(range(1,25),[q[key][j] for q in d],color=COLORS[j],label=labels[j])
                ax.set(yscale="log",title=t.replace("to"," → ") if panel==0 else None)
            axes[1,i].set_xlabel("Physical layer")
        axes[0,0].set_ylabel("Before projection · RMS")
        axes[1,0].set_ylabel("After projection · RMS")
        axes[0,-1].legend(frameon=False,fontsize=8)
        export(fig,r,idx,f"{mode}_decomposition")

    fig,axes=plt.subplots(1,2,figsize=(8.5,3.2),constrained_layout=True)
    for j,(phase,name,color) in enumerate(zip(PHASES,TITLES,COLORS,strict=True)):
        t=phases[phase]["causal"]["response"]["fixed"]
        for ax,key,x in ((axes[0],"gain",[1,2,3,4]),(axes[1],"cosine",[2,3,4])):
            ax.plot(x,[z["mean"] for z in t[key]],color=color,marker="o",ms=4,label=name)
            ax.fill_between(x,[q["ci"][0] for q in t[key]],[q["ci"][1] for q in t[key]],color=color,alpha=.14)
    axes[0].set(xlabel="Natural receiver pass",ylabel="State response / perturbation norm",xticks=[1,2,3,4])
    axes[1].set(xlabel="Successive receiver pass",ylabel="Response cosine to previous pass",xticks=[2,3,4])
    axes[1].legend(frameon=False,fontsize=8)
    export(fig,r,4,"same_perturbation")

    fig,axes=plt.subplots(1,2,figsize=(9.6,3.7),sharey=True,constrained_layout=True)
    names=("factor","content","joint","random","other_tokens")
    for ax,mode in zip(axes,MODES,strict=True):
        for k,tr in enumerate(("1from2","2from1")):
            for j,phase in enumerate(PHASES[:2]):
                src=phases[phase]["causal"]["predictions"][tr][mode]["arms"]
                for i,name in enumerate(names):
                    interval(ax,i+(.12 if k else -.12)+(.035 if j else -.035),src[name]["progress"],COLORS[j],alpha=.95 if k else .45)
        ax.axhline(0,color="#777",lw=.7)
        ax.set(title=mode,xticks=range(len(names)),xticklabels=["Factor","Content","Joint","Random","Shifted"])
    axes[0].set_ylabel("Centered donor-alignment (mean, 95% CI)")
    fig.supxlabel("Dark markers show K2 recipients. Pale markers show K1. Blue is ARC and red is OpenBookQA.",fontsize=8)
    export(fig,r,5,"native_swaps")

    forms=s.get("formation",{})
    fig,axes=plt.subplots(1,2,figsize=(9.8,3.8),constrained_layout=True)
    for col,(mode,ax) in enumerate(zip(MODES,axes,strict=True)):
        if not forms:
            ax.text(.5,.5,"No path admitted",ha="center",va="center",transform=ax.transAxes)
            continue
        names=("drop","matched_random","shifted_tokens","other_write","mediator_only")
        for j,phase in enumerate(PHASES[:2]):
            a=forms[phase][mode]["arms"]
            for k,name in enumerate(names):
                interval(ax,k+(.08 if j else -.08),a[name]["factor_loss"],COLORS[j])
        w=s["origin_lock"]["mode"][mode]["selected"]
        ax.set(title=f"{mode} · writer {w}",xticks=range(len(names)),
               xticklabels=("Drop","Random","Shifted","Other write","Mediator only"))
        ax.tick_params(axis="x",labelrotation=18)
        ax.axhline(0,color="#777",lw=.7)
    axes[0].set_ylabel("Loss of native K1→K2 factor progression")
    export(fig,r,6,"predecessor_writes")

    fig,ax=plt.subplots(figsize=(10.2,3.8),constrained_layout=True)
    ax.axis("off")
    def box(x,y,text,color):
        ax.text(x,y,text,ha="center",va="center",fontsize=9,transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=.52",fc="white",ec=color,lw=1))
    def arrow(x,y,xx,style,color):
        ax.annotate("",(xx,y),(x,y),xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->",color=color,ls=style,lw=1.4))
    for i,mode in enumerate(MODES):
        y=.72-(i*.46)
        wr=s.get("origin_lock",{}).get("mode",{}).get(mode,{}).get("selected","unselected")
        l=s["candidates"][f"candidate_{mode}_layer"]
        color=COLORS[i]
        box(.12,y,f"Native K1 write\n{wr}",color)
        box(.41,y,f"K2 L{l}\n{'attention factor' if i==0 else 'MLP gate'}",color)
        box(.69,y,"Recipient computation\n(equal native suffix)",color)
        box(.90,y,"Task\nanswer",color)
        supported=bool(forms) and all(forms[k][mode]["origin_specificity"]["matched_random"]["ci"][0]>0
                                      and forms[k][mode]["arms"]["drop"]["factor_loss"]["ci"][0]>0
                                      for k in PHASES[:2])
        arrow(.21,y,.31,"-" if supported else "--",color)
        arrow(.51,y,.58,"-",color)
        arrow(.79,y,.85,"--",color)
    ax.text(.5,.07,"Dashed endpoint edges denote a task-level mechanism that requires evidence beyond factor formation.",
            ha="center",fontsize=8,transform=ax.transAxes)
    export(fig,r,7,"causal_chain")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("results",type=Path)
    a=p.parse_args()
    r=a.results/"figures"
    r.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"font.family":"DejaVu Sans"})
    plot(load(a.results/"SUMMARY.json"),r)

