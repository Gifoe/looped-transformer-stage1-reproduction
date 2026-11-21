"""Plot the two-state mediation and readout-preserving context tests."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from native import load

F=("openbookqa","arc_challenge")
N=("OpenBookQA","ARC-Challenge")
C=("#216678","#9c573e")


def err(ax,i,q,color):
    if q is None:
        return
    y=q['mean']
    lo,hi=q['ci']
    ax.errorbar(i,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt="o",ms=4,capsize=3,color=color)


def save(fig,p,i,name):
    for t in ("png","pdf"):
        fig.savefig(p/f"figure_{i}_{name}.{t}",bbox_inches="tight",dpi=240)
    plt.close(fig)


def draw(s,x,p):
    up=s["full_candidate"]
    fig,ax=plt.subplots(1,2,figsize=(8.5,3.1),sharey=True,constrained_layout=True)
    for j,group in enumerate(("ALL","WC")):
        z=ax[j]
        for a,(f,c) in enumerate(zip(F,C,strict=True)):
            q=up[f]["mean"][group]["advance"]["drop"]
            for b,k in enumerate(("K1_loss","K2_loss","increment_loss")):
                err(z,b+(a-.5)*.16,q[k],c)
        z.axhline(0,color="#888",lw=.65)
        z.set(xticks=[0,1,2],xticklabels=["K1","K2","K2−K1"],title=("All items" if group=="ALL" else "Native K1 wrong → K2 correct"))
    ax[0].set_ylabel("Natural writer removal loss (nats/token)")
    save(fig,p,1,"early_and_late_writer_loss")

    fig,ax=plt.subplots(1,2,figsize=(9.5,3.3),sharey=True,constrained_layout=True)
    keys=("up","up_norm_control","product_norm_control","gate","gate_and_up")
    labels=("Up","Raw-dose control","Product-dose control","Gate","Gate + Up")
    for k,(background,sign) in enumerate((("drop",1),("clean",-1))):
        z=ax[k]
        for a,(f,c) in enumerate(zip(F,C,strict=True)):
            for b,key in enumerate(keys):
                q=up[f]["mean"]["ALL"]["arms"][background][key]["effect"]
                err(z,b+(a-.5)*.16,dict(mean=sign*q["mean"],ci=sorted(sign*y for y in q["ci"])),c)
        z.axhline(0,color="#888",lw=.65)
        z.set(xticks=range(len(keys)),xticklabels=labels,title=("Restore in writer-deleted state" if sign==1 else "Reverse in native state"))
        z.tick_params(axis="x",rotation=20)
    ax[0].set_ylabel("Oriented answer-margin effect")
    save(fig,p,2,"up_bidirectional")

    fig,ax=plt.subplots(1,2,figsize=(9.0,3.2),sharey=True,constrained_layout=True)
    for i,f in enumerate(F):
        m=up[f]["mean"]["ALL"]["arms"]["drop"]
        for j,k in enumerate(("inherited","fresh_attention","up")):
            err(ax[0],j+(i-.5)*.18,m[k]["effect"],C[i])
        h=x["support"][f]["ALL"]["background"]["drop"]
        for j,k in enumerate(("earlier","readout","all")):
            err(ax[1],j+(i-.5)*.18,h[k]["versus_matched"],C[i])
    for z in ax:
        z.axhline(0,color="#888",lw=.65)
    ax[0].set(xticks=range(3),xticklabels=["Inherited state","K2 attention write","Full clean Up"],title="Up construction")
    ax[1].set(xticks=range(3),xticklabels=["Earlier","Last prompt","All prompt"],title="Up minus same product-dose control")
    ax[0].set_ylabel("Answer-margin effect in writer-deleted state")
    save(fig,p,3,"up_provenance_and_support")

    fig,ax=plt.subplots(1,2,figsize=(9.5,3.4),sharey=True,constrained_layout=True)
    keys=("body","body_rolled","body_scaled","body_direction","boundary","whole_prompt")
    tick=("Context","Rolled","Norm only","Direction only","Readout","Whole prompt")
    for k,side in enumerate(("drop","clean")):
        z=ax[k]
        for i,f in enumerate(F):
            m=x["body"][f]["ALL"]["tables"][side]
            for j,key in enumerate(keys):
                err(z,j+(i-.5)*.19,m[key]["effect"],C[i])
        z.axhline(0,color="#888",lw=.65)
        z.set(xticks=range(6),xticklabels=tick,title=("Forward restoration" if side=="drop" else "Reverse damage"))
        z.tick_params(axis="x",rotation=18)
    ax[0].set_ylabel("Oriented K2 answer-margin effect")
    fig.supxlabel("Context and its controls preserve every K1 candidate readout. Blue: OpenBookQA. Orange: ARC.",fontsize=8)
    save(fig,p,4,"context_with_fixed_k1_readout")

    fig,ax=plt.subplots(1,2,figsize=(9.0,3.1),sharey=True,constrained_layout=True)
    for f,z,title,col in zip(F,ax,N,C,strict=True):
        g=s["free_decode"][f]
        keys=("K1","K2","drop","restore","reverse","product_control")
        bars=[g["accuracy"][k] for k in keys]
        z.bar(range(6),bars,color=col)
        for i,n in enumerate(bars):
            z.text(i,n+.08,str(n),ha="center",fontsize=8)
        z.set(xticks=range(6),xticklabels=["K1","K2","Drop","Restore","Reverse","Control"],title=title)
    ax[0].set_ylim(0,max(2,max(y for f in F for y in s["free_decode"][f]["accuracy"].values())+1))
    ax[0].set_ylabel("Strict full-text match")
    save(fig,p,5,"open_generation")


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("r",type=Path)
    r=a.parse_args().r
    p=r/"figures"; p.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42})
    draw(load(r/"SUMMARY.json"),load(r/"EXTRA_SUMMARY.json"),p)

