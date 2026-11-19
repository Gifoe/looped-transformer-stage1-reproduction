"""Create six vector/PDF and high-resolution PNG evidence figures."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_composition import TASKS, EFFECTS, table, read

PALETTE=("#235789","#E37730","#158277","#9A4063","#8793A1")
LABELS=("Relation","Logic","Arithmetic")
SPLIT="fresh_structure"
METHOD_GRID=("mean-natural-bilinear","mean-mixed-bilinear","token-natural-bilinear","token-mixed-bilinear")


def cfg():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
        "axes.spines.right":False,"axes.linewidth":.8,"axes.edgecolor":"#85909F","axes.titlesize":10,
        "axes.titleweight":"bold","axes.axisbelow":True,"pdf.fonttype":42,"savefig.facecolor":"white"})


def indexed(path,fields):
    return {tuple(r[k] for k in fields):r for r in table(path)}


def errs(ax,x,y,color,fmt="o",**kw):
    me,lo,hi=(np.array([float(r[k]) for r in y]) for k in ("mean","lower","upper"))
    ax.errorbar(x,me,yerr=np.maximum(0,np.stack((me-lo,hi-me))),
                fmt=fmt,color=color,lw=1.05,ms=4,capsize=2,**kw)


def save(fig,path):
    for ext in (".pdf",".png"):
        fig.savefig(path.with_suffix(ext),dpi=320,bbox_inches="tight",pad_inches=.1)
    plt.close(fig)


def legends(fig,ax,columns):
    h,l=ax.get_legend_handles_labels()
    fig.legend(h,l,loc="outside upper center",ncols=columns,frameon=False,fontsize=8)


def prediction(out,nat,causal,gates):
    fig,axes=plt.subplots(2,3,figsize=(10.5,5.3),layout="constrained")
    for j,(t,title) in enumerate(zip(TASKS,LABELS)):
        ax=axes[0,j]
        methods=("identity","natural-full","natural-mean","natural-token")
        for k,m in enumerate(methods):
            for h,s in enumerate(("fresh_same","fresh_structure")):
                errs(ax,[k+(h-.5)*.25],[nat[t,s,m,"state_mse"]],PALETTE[k],
                     ("o","s")[h],mfc=("white",PALETTE[k])[h],
                     label=("Fresh familiar","Fresh structure")[h] if k==0 else None)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks(range(4),["Identity","Full ridge","Mean ridge","Token ridge"], rotation=23,ha="right")
        ax=axes[1,j]
        for k,m in enumerate(("zero","full-mixed-full",gates[t])):
            data=[causal[t,SPLIT,m,a,"response_mse","all"] for a in EFFECTS]
            errs(ax,np.arange(4)+(k-1)*.19,data,PALETTE[k],"o",
                 label=("Zero","Full ridge + edits","Selected model")[k])
        ax.set_xticks(range(4),["S","M","S+M","Interaction"])
        for top in axes[:,j]:
            top.grid(axis="y",alpha=.18)
    axes[0,0].set_ylabel("Natural K2 MSE")
    axes[1,0].set_ylabel("Physical K2 response MSE")
    axes[0,0].legend(frameon=False,fontsize=7)
    axes[1,-1].legend(frameon=False,fontsize=7)
    save(fig,out/"01_state_prediction")


def atlas_plot(out,atlas):
    fig,axes=plt.subplots(3,2,figsize=(10,6.6),layout="constrained",sharex=True)
    for j,(task,title) in enumerate(zip(TASKS,LABELS)):
        for k,role in enumerate(("mapping","readout")):
            ax=axes[j,k]
            for idx,(p,arm,name) in enumerate((("patch","S","Source"),("patch","M","Map"),
              ("patch","SM","Joint"),("patch","I","Physical interaction"),("natural","I","Natural interaction"))):
                y=[float(atlas[task,SPLIT,p,str(k1),str(layer),role,arm]["mean"])
                   for k1 in range(1,5) for layer in range(25)]
                ax.plot(y,color=PALETTE[idx],lw=1.1,ls=("-", "-", "-", "-", "--")[idx],label=name)
            for at in (24,49,74):
                ax.axvline(at,color="#B4BCC6",lw=.65)
            ax.set_title(f"{title} · {role}")
            ax.grid(axis="y",alpha=.15)
            ax.set_xticks((12,37,62,87),["K1","K2","K3","K4"])
    fig.supylabel("Role perturbation RMS, fixed 32D projection",fontsize=9)
    legends(fig,axes[0,0],5)
    save(fig,out/"02_interaction_by_loop_and_layer")


def grid_plot(out,causal):
    fig,axes=plt.subplots(2,3,figsize=(10,5.6),layout="constrained")
    for j,t in enumerate(TASKS):
        for k,arm in enumerate(("SM","I")):
            ax=axes[k,j]
            for i,method in enumerate(METHOD_GRID):
                row=causal[t,SPLIT,method,arm,"response_mse","all"]
                errs(ax,[i],[row],PALETTE[int(method.startswith("token"))],"s" if "mixed" in method else "o",
                     mfc=PALETTE[int(method.startswith("token"))] if "mixed" in method else "white")
            base=float(causal[t,SPLIT,"zero",arm,"response_mse","all"]["mean"])
            ax.axhline(base,color="#6B7582",lw=.9,ls="--",label="Zero response")
            ax.set_xticks(range(4),["Mean\nnatural","Mean\n+ edits","Token\nnatural","Token\n+ edits"])
            ax.grid(axis="y",alpha=.15)
            if k==0:
                ax.set_title(LABELS[j])
    axes[0,0].set_ylabel("Joint response MSE")
    axes[1,0].set_ylabel("Interaction MSE")
    axes[0,2].legend(frameon=False,fontsize=7)
    save(fig,out/"03_representation_by_supervision")


def semantic_plot(out,eval_root,gates):
    fig,axes=plt.subplots(2,3,figsize=(9.8,5.4),layout="constrained")
    for j,t in enumerate(TASKS):
        rows=table(eval_root/t/"causal.csv")
        for i,(arm,metric) in enumerate((("S","mid"),("I","comp"))):
            ax=axes[i,j]
            actual,prediction=(np.array([float(r[f"{metric}_{key}"]) for r in rows if
                              r["split"]==SPLIT and r["arm"]==arm and r["method"]==gates[t]])
                              for key in ("true","pred"))
            limit=max(np.abs(actual).max(),np.abs(prediction).max(),1e-7)*1.08
            ax.axhline(0,color="#C0C8D0",lw=.6)
            ax.axvline(0,color="#C0C8D0",lw=.6)
            ax.plot((-limit,limit),(-limit,limit),color="#7B8793",lw=.75,ls="--")
            ax.scatter(actual,prediction,c=PALETTE[j],alpha=.52,edgecolors="none",s=13)
            ax.set(xlim=(-limit,limit),ylim=(-limit,limit))
            if i==0:
                ax.set_title(LABELS[j])
    axes[0,0].set_ylabel("Predicted intermediate change")
    axes[1,0].set_ylabel("Predicted final interaction")
    fig.supxlabel("Measured diagnostic change · oracle K2 anchor",fontsize=9)
    save(fig,out/"04_semantic_response")


def behavior_plot(out,behavior,tie_aware):
    fig,axes=plt.subplots(2,3,figsize=(10,5.5),layout="constrained",sharex=True)
    for j,task in enumerate(TASKS):
        for k,arm in enumerate(EFFECTS):
            info=[behavior[task,SPLIT,arm,str(l),"contrast_change"] for l in range(1,5)]
            errs(axes[0,j],np.arange(1,5)+(k-1.5)*.04,info,PALETTE[k],"o-",label=arm)
        for k,arm in enumerate(("S","M","SM")):
            info=[tie_aware[task,SPLIT,arm,str(l),"flipped_into_target"] for l in range(1,5)]
            errs(axes[1,j],np.arange(1,5)+(k-1)*.06,info,PALETTE[k],"o-",label=arm)
        axes[0,j].set_title(LABELS[j])
        axes[0,j].axhline(0,color="#6B7582",lw=.7,ls=":")
        for ax in axes[:,j]:
            ax.grid(axis="y",alpha=.15)
            ax.set_xticks(range(1,5),["K1","K2","K3","K4"])
    axes[0,0].set_ylabel("Signed native margin effect (nats)")
    axes[1,0].set_ylabel("Fraction flipping to unique arm target")
    legends(fig,axes[0,0],4)
    save(fig,out/"05_downstream_behavior")


def comparison(out,root):
    rows=table(root/"baseline_comparisons.csv")
    controls=table(root/"interaction_controls.csv")
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.8),layout="constrained")
    for j,t in enumerate(TASKS):
        for k,metric in enumerate(("response_mse","comp_error","k4_error")):
            arm=("pooled","I","pooled")[k]
            c=min((r for r in rows if r["task"]==t and r["split"]==SPLIT and r["arm"]==arm
                   and r["metric"]==metric),key=lambda r:float(r["baseline_error"]))
            v={dest:100*float(c[src]) for dest,src in zip(("mean","lower","upper"),
                                   ("relative_reduction","lower","upper"))}
            errs(axes[0],[j+(k-1)*.22],[v],PALETTE[k],("o","s","^")[k],
                 label=("K2 geometry","Composition","K4 margin")[k] if j==0 else None)
        for k,name in enumerate(("random","irrelevant","mismatch")):
            v=next(r for r in controls if r["split"]==SPLIT and r["task"]==t and r["loop"]=="2" and
                   r["layer"]=="24" and r["role"]=="readout" and r["reference"]==name)
            errs(axes[1],[j+(k-1)*.22],[{**v,"mean":v["difference"]}],PALETTE[k],("o","s","^")[k],
                 label=("Random","Irrelevant","Mismatched")[k] if j==0 else None)
    for ax in axes:
        ax.axhline(0,color="#738090",ls="--",lw=.9)
        ax.set_xticks(range(3),LABELS)
        ax.grid(axis="y",alpha=.2)
        ax.legend(frameon=False,fontsize=7)
    axes[0].set_ylabel("Error reduction vs best simple baseline (%)")
    axes[1].set_ylabel("K2 interaction RMS, patch minus control")
    save(fig,out/"06_cross_task_comparison")


def run(root):
    cfg()
    result=root/"results"
    out=result/"figures"
    out.mkdir(exist_ok=True)
    gates={r["task"]:r["candidate"] for r in read(result/"gate.json")["tasks"]}
    nat=indexed(result/"natural_summary.csv",("task","split","method","metric"))
    causal=indexed(result/"causal_summary.csv",("task","split","method","arm","metric","difficulty"))
    atlas=indexed(result/"interaction_atlas.csv",("task","split","condition","loop","layer","role","effect"))
    behavior=indexed(result/"behavior_summary.csv",("task","split","arm","loop","metric"))
    tie_aware=indexed(result/"tie_aware_behavior_summary.csv",("task","split","arm","loop","metric"))
    prediction(out,nat,causal,gates)
    atlas_plot(out,atlas)
    grid_plot(out,causal)
    semantic_plot(out,root/"evaluation",gates)
    behavior_plot(out,behavior,tie_aware)
    comparison(out,result)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    run(p.parse_args().root)
