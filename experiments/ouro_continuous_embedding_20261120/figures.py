"""Produce the seven requested figures directly from the GPU records."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import LOCAL, aggregate


TRACKS=("full_loop","branch_flow","bea2","chart","chart_anderson","hybrid","hybrid_shuffled","ridge","prefix_ridge")
LABELS=dict(full_loop="Full-loop ODE",branch_flow="Branch ODE",bea2="BEA2",chart="Picard chart",
            chart_anderson="Anderson chart",hybrid="Residual-gated hybrid",hybrid_shuffled="Shuffled hybrid",
            ridge="Tokenwise ridge",prefix_ridge="Causal-prefix ridge")
COLORS=dict(zip(TRACKS,("#806891","#b35d41","#8a875f","#7186a4","#196c8b","#0b8a68","#777777","#3e3b41","#bf7637")))


def export(out,num,slug,fig):
    for ext in ("png","pdf"):
        fig.savefig(out/f"figure_{num}_{slug}.{ext}",dpi=250,bbox_inches="tight")
    plt.close(fig)


def project(out,s,pilot,operators,natural,causal):
    f,axs=plt.subplots(1,2,figsize=(9.3,3.45),constrained_layout=True)
    for ax,phase in zip(axs,("K1_K2","K2_K3")):
        n=s["natural"]["float32"][phase]
        for k in ("branch_flow","bea2","chart","chart_anderson","hybrid"):
            a=n["pooled"][k]["branch_path"]
            ax.semilogy(range(1,49),np.clip(a,1e-9,None),color=COLORS[k],label=LABELS[k])
        ax.scatter(24,n["linear_oracle_midpoint"]["mean"],marker="x",color="#25272a",label="Oracle linear midpoint")
        ax.set(title=phase.replace("_"," → "),xlabel="Branch boundary (pre terminal norm)",
               ylabel="Mean relative distance to native boundary")
    axs[1].legend(frameon=False,fontsize=7)
    export(out,1,"embeddings_and_native",f)

    f,axs=plt.subplots(1,2,figsize=(9.3,3.45),constrained_layout=True)
    colors=("#a94836","#647e92","#217596","#118567")
    keys=sorted(s["local"],key=float)
    for k,c,label in zip(LOCAL,colors,("Unmodified","BEA2","BEA3","Picard midpoint")):
        series=[s["local"][a]["endpoint"][k] for a in keys]
        axs[0].loglog([float(x) for x in keys],np.clip([v["median"] for v in series],1e-9,None),
                      color=c,marker="o",label=label)
        idx=LOCAL.index(k)
        axs[1].scatter([idx,idx],[series[-1]["median"],series[-1]["maximum"]],
                       color=c,marker="o")
        axs[1].vlines(idx,series[-1]["median"],series[-1]["maximum"],color=c)
    axs[0].set(xlabel="Scaled residual update α",ylabel="Median local relative endpoint error")
    axs[0].legend(frameon=False,fontsize=8)
    axs[1].set(yscale="log",ylabel="Unit-step relative endpoint error",title="Median to worst tested branch",
               xticks=np.arange(4),xticklabels=["Base","BEA2","BEA3","Picard"])
    export(out,2,"modified_equation",f)

    f,axs=plt.subplots(1,2,figsize=(9.3,3.45),constrained_layout=True)
    for i,k in enumerate(("2","3")):
        v=s["order"]["loops"][k]
        a=v["scale"]
        c=("#227194","#a8503c")[i]
        axs[0].semilogx([float(j) for j in a],[x["cos"]["median"] for x in a.values()],marker="o",color=c,
                       label=f"Loop {k}")
        axs[1].plot(range(1,25),v["layer_swap"],color=c,label=f"Loop {k}")
    axs[0].set(xlabel="Scaled residual α",ylabel="Swap direction / JVP bracket cosine",ylim=(.8,1.02))
    axs[1].set(xlabel="Swapped physical layer",ylabel="Relative final state change")
    axs[1].legend(frameon=False)
    export(out,3,"attention_mlp_order",f)

    f,axs=plt.subplots(1,2,figsize=(9.4,3.5),constrained_layout=True)
    target=s["natural"]["float32"]["K1_K2"]["pooled"]
    selected=("branch_flow","bea2","chart","chart_anderson","hybrid")
    matrix=np.array([target[k]["branch_path"] for k in selected])
    im=axs[0].imshow(np.log10(np.clip(matrix,1e-8,None)),aspect="auto",interpolation="nearest",cmap="viridis",
                     extent=(1,48,len(selected)-.5,-.5))
    axs[0].set(yticks=range(len(selected)),yticklabels=[LABELS[k] for k in selected],xlabel="Branch")
    f.colorbar(im,ax=axs[0],label="log10 mean boundary error")
    for family,g in operators.items():
        rows=g["loops"]["2"]
        axs[1].plot(range(1,25),[np.mean(v["raw_norm"]) for v in rows],label=family)
    axs[1].set(xlabel="Physical layer",ylabel="Raw local norm of order change")
    axs[1].legend(frameon=False)
    export(out,4,"branch_error_sources",f)

    f,axs=plt.subplots(1,2,figsize=(10.1,4),sharey=True,constrained_layout=True)
    for ax,dt in zip(axs,("float32","bfloat16")):
        source=s["natural"][dt]["K1_K2"]["pooled"]
        keys=[k for k in TRACKS if k in source]
        ys=np.arange(len(TRACKS))
        for k in keys:
            d=source[k]["full_error"]
            j=TRACKS.index(k)
            ax.scatter(d["median"],j,color=COLORS[k],zorder=3)
            ax.hlines(j,max(1e-8,d["minimum"]),max(1e-8,d["maximum"]),color=COLORS[k],alpha=.7)
        ax.set(xscale="log",xlabel="Relative endpoint L2 (median and min–max)",title=dt,
               yticks=ys,yticklabels=[LABELS[k] for k in TRACKS])
    axs[0].invert_yaxis()
    export(out,5,"untouched_state_prediction",f)

    f,axs=plt.subplots(1,3,figsize=(10.6,3.35),sharex=True,sharey=True,constrained_layout=True)
    rows=[(family,r) for family,d in causal["float32"].items() for r in d["runs"]]
    context="prefix_ridge" if "prefix_ridge" in rows[0][1]["methods"] else "ridge"
    for ax,k in zip(axs,("branch_flow","chart_anderson",context)):
        xx=[]
        yy=[]
        for family,r in rows:
            q=r["methods"][k]
            x=q["native_target_margin_response"]
            y=q["target_margin_response"]
            cc={"tableau":"#276d92","twohop":"#a95440","arc":"#786794"}[family]
            ax.scatter(x[1:3],y[1:3],color=cc,alpha=.55,s=14)
            xx.extend(x[1:3])
            yy.extend(y[1:3])
        mins=min(xx+yy)
        maxs=max(xx+yy)
        ax.plot([mins,maxs],[mins,maxs],":",color="#858585",linewidth=1)
        ax.set(title=LABELS[k],xlabel="Native target margin change")
    axs[0].set_ylabel("Predicted target margin change")
    export(out,6,"counterfactual_responses",f)

    f,axs=plt.subplots(1,2,figsize=(9.7,3.65),constrained_layout=True)
    for k in TRACKS:
        if k not in s["natural"]["float32"]["K1_K2"]["pooled"]:
            continue
        vals=[s["natural"]["float32"][q]["pooled"][k]["full_error"]["median"]
              for q in ("K1_K2","K2_K3")]
        axs[0].semilogy([0,1],vals,marker="o",color=COLORS[k],label=LABELS[k])
        v=[s["causal"]["float32"][q]["pooled"]["semantic"][k]["logit_response_rmse"]["median"]
           for q in ("K1_K2","K2_K3")]
        axs[1].semilogy([0,1],np.clip(v,1e-9,None),marker="o",color=COLORS[k],label=LABELS[k])
    for ax,label in zip(axs,("Natural endpoint error","Counterfactual logit response RMSE")):
        ax.set(xticks=(0,1),xticklabels=("K1 → K2","K2 → K3"),ylabel=label)
    axs[-1].legend(frameon=False,fontsize=7)
    export(out,7,"cross_loop",f)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("results",type=Path)
    a=p.parse_args()
    s,pilot,operators,dev,natural,causal=aggregate(a.results)
    dst=a.results/"figures"
    dst.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42})
    project(dst,s,pilot,operators,natural,causal)
    (a.results/"SUMMARY.json").write_text(json.dumps(s,indent=2,allow_nan=False)+"\n")
