"""Visualize the four evidence tiers without pooling different units."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from native import load

PH=("openbookqa","arc_challenge")
LAB=("OpenBookQA","ARC-Challenge")
COL=("#20657f","#a95d41")


def figfile(fig,d,n,name):
    for ext in ("png","pdf"):
        fig.savefig(d/f"figure_{n}_{name}.{ext}",dpi=240,bbox_inches="tight")
    plt.close(fig)


def error(ax,axis,x,z,c):
    if z is None:
        return
    y=z["mean"]
    lo,hi=z["ci"]
    if axis=="x":
        ax.errorbar(y,x,xerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt="o",color=c,ms=4,capsize=3)
    else:
        ax.errorbar(x,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt="o",color=c,ms=4,capsize=3)


def plots(s,m,b,d):
    a=s["structure"]["confirmation"]
    fig,ax=plt.subplots(1,2,figsize=(9.0,3.1),gridspec_kw={"width_ratios":[2,1]},constrained_layout=True)
    methods=("native_values","scramble_values","natural_receivers","scramble_receivers","other_heads_receivers")
    name=("A×V, fixed prefix","Row exchange, fixed prefix","A across contexts","Row exchange across contexts","Other heads across contexts")
    for k,(method,y) in enumerate(zip(methods,name,strict=True)):
        for l,(scope,c) in enumerate(zip(("rules","reverse"),COL,strict=True)):
            error(ax[0],"x",k+(l-.5)*.20,a[scope]["panels"][method]["interaction"],c)
    ax[0].axvline(0,color="#777",lw=.8)
    ax[0].set(yticks=range(len(name)),yticklabels=name,xlabel="Recipient-specific I (nats/token)")
    ax[0].invert_yaxis()
    for l,(scope,c) in enumerate(zip(("rules","reverse"),COL,strict=True)):
        error(ax[1],"x",l,a[scope]["panels"]["native_selector_oracle"]["interaction"],c)
    ax[1].set(yticks=[0,1],yticklabels=["rules","reverse"],title="Native query change",xlabel="Same interaction contrast")
    ax[1].axvline(0,color="#777",lw=.8)
    figfile(fig,d,1,"recipient_interaction")

    fig,ax=plt.subplots(1,2,figsize=(9,3.3),sharey=True,constrained_layout=True)
    for p,c,title,z in zip(PH,COL,LAB,ax,strict=True):
        rec=s["natural"][p]["mean"]
        q=rec["panels"]["WC"]["transitions"]
        x=[q["native_wc"],q["rollback_damages"],q["dropped_late_correct"],q["restore_after_drop"]]
        z.bar(range(4),x,color=[c,c,"#717384","#748344"])
        for j,k in enumerate(x):
            z.text(j,k+.35,str(k),ha="center",fontsize=8)
        z.set(xticks=range(4),xticklabels=["Corrected","Rollback\ndamaged","Writer drop\ndamaged","Route restore\nrescued"],title=title)
    ax[0].set_ylabel("Count among native K1 wrong → K2 correct")
    figfile(fig,d,2,"native_corrections")

    fig,axes=plt.subplots(1,2,figsize=(8.7,3.2),constrained_layout=True)
    keys=("pos0","rest","all","pos0_permuted","rest_permuted")
    tick=("Pos. 0", "Rest", "All", "Pos. 0\ncontrol", "Rest\ncontrol")
    for i,p in enumerate(PH):
        q=m[p]["writer"]["standard"]["panels"]["ALL"]["arms"]
        for k,key in enumerate(keys):
            for ax,field in zip(axes,("mass","margin"),strict=True):
                error(ax,"y",k+(i-.5)*.20,q[key][field],COL[i])
    for a in axes:
        a.axhline(0,color="#777",lw=.8)
        a.set(xticks=range(len(tick)),xticklabels=tick,xlabel="K1 L1 attention write support")
    axes[0].set_ylabel("Loss of K2 position-0 attention mass")
    axes[1].set_ylabel("Effect on true answer margin")
    figfile(fig,d,3,"writer_support")

    fig,axes=plt.subplots(1,2,figsize=(9.5,3.4),sharey=True,constrained_layout=True)
    groups=("L6_four_heads","L6_all_heads","producer_deleted","L1_mlp")
    kinds=("first","second","both")
    for p,a,title in zip(PH,axes,LAB,strict=True):
        for j,group in enumerate(groups):
            q=m[p]["factorials"][group]["ALL"]
            for i,k in enumerate(kinds):
                error(a,"y",j+(i-1)*.18,q["arms"][k]["effect"],(COL[0],COL[1],"#777383")[i])
        a.axhline(0,color="#777",lw=.8)
        a.set(title=title,xticks=range(4),xticklabels=("L6 four","L6 all","Writer drop","L1 MLP"))
    axes[0].set_ylabel("Change in gold vs. fixed foil (nats/token)")
    fig.supxlabel("Blue: first factor. Orange: second factor. Gray: both factors.",fontsize=8)
    figfile(fig,d,4,"native_factorials")


    fig,axes=plt.subplots(1,2,figsize=(8.9,3.1),sharey=True,constrained_layout=True)
    modes=("total","product","downstream","product_vs_same_norm")
    for z,key,title in zip(axes,("L6_attention","L1_mlp"),("L6 attention","L1 SwiGLU"),strict=True):
        for i,p in enumerate(PH):
            x=b[p][key]["groups"]["ALL"]
            for j,k in enumerate(modes):
                error(z,"y",j+(i-.5)*.19,x[k],COL[i])
        z.axhline(0,color="#777",lw=.8)
        z.set(xticks=range(4),xticklabels=("Full","Product","Suffix","Product −\nmatched"),title=title)
    axes[0].set_ylabel("Answer margin interaction (nats/token)")
    figfile(fig,d,5,"operator_product")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("o",type=Path)
    a=p.parse_args().o
    d=a/"figures"
    d.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    plots(load(a/"SUMMARY.json"),load(a/"MECHANISM_SUMMARY.json"),load(a/"BILINEAR_SUMMARY.json"),d)

