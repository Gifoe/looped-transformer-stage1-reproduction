"""Render the state, semantics, causal-gate, distribution and case study figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_transition import TASKS, partition, table

COLORS = ["#345995", "#E07A33", "#15847C", "#9298A4", "#A04461"]
ROLES = ("Source", "Bridge", "Mapping", "Query", "Readout")
TASK_LABELS = ("Relation", "Logic", "Arithmetic")
METHODS = ("ridge_raw", "ridge", "mlp", "dictionary", "context")
LABELS = ("Full-coordinate ridge", "Latent ridge", "Residual MLP", "Operator mixture", "Context only")


def plot_style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
        "axes.spines.right":False,"axes.labelcolor":"#242F3E","axes.titlesize":11,"axes.titleweight":"bold",
        "axes.axisbelow":True,"pdf.fonttype":42,"ps.fonttype":42,"savefig.facecolor":"white",
        "axes.edgecolor":"#85909F","xtick.color":"#364252","ytick.color":"#364252"})


def save(fig, path):
    for suffix in (".png",".pdf"):
        fig.savefig(path.with_suffix(suffix),dpi=330,bbox_inches="tight",pad_inches=.1)
    plt.close(fig)


def lookup(rows, key):
    return {tuple(r[c] for c in key):r for r in rows}


def errplot(ax, x, rows, color, fields=("mean","lower","upper"), **kw):
    mid,lo,hi = (np.array([float(r[k]) for r in rows]) for k in fields)
    ax.errorbar(x,mid,yerr=(mid-lo,hi-mid),color=color,lw=1.1,ms=4,capsize=2,**kw)


def natural(out, measurements):
    fig,axs=plt.subplots(1,3,figsize=(10,3.35),layout="constrained",sharey=True)
    for t,label,ax in zip(TASKS,TASK_LABELS,axs):
        for j,m in enumerate(METHODS):
            for i,s in enumerate(("confirm","transfer")):
                ref=float(measurements["2048",t,s,"identity_raw","natural","state_mse"]["mean"])
                data=measurements["2048",t,s,m,"natural","state_mse"]
                ratio={k:float(data[k])/ref for k in ("mean","lower","upper")}
                errplot(ax,[j+(i-.5)*.28],[ratio],COLORS[j],fmt=("o","s")[i],
                        mfc=("white",COLORS[j])[i],label=("Confirmation","Structural transfer")[i] if j==0 else None)
        ax.axhline(1,color="#687384",lw=.9,ls="--")
        ax.set_yscale("log")
        ax.set_title(label)
        ax.set_xticks(range(5),["Full ridge","Latent ridge","MLP","Mixture","Context"],
                      rotation=32,ha="right")
        ax.grid(axis="y",alpha=.2)
    axs[0].set_ylabel("K2 state MSE / identity MSE")
    axs[1].legend(frameon=False,fontsize=8)
    save(fig,out/"01_natural_prediction")


def semantics(root, out):
    d=lookup(table(root/"diagnostic_summary.csv"),
             ("width","task","split","target","state","role"))
    fig,axs=plt.subplots(2,3,figsize=(10,5.6),layout="constrained",sharex=True,sharey=True)
    for row,tr in enumerate(("intermediate","final")):
        for ax,t,label in zip(axs[row],TASKS,TASK_LABELS):
            for i,(mode,key) in enumerate(zip(("K1 native","K2 native","K1 decoder on K2"),
                             ("native_K1","native_K2","K1_decoder_on_K2"))):
                data=[d["2048",t,"transfer",tr,key,str(r)] for r in range(5)]
                errplot(ax,np.arange(5)+(i-1)*.15,data,COLORS[i],fmt="o-",label=mode)
            ax.set_ylim(0,1.02)
            ax.grid(axis="y",alpha=.2)
            ax.set_xticks(range(5),ROLES,rotation=30,ha="right")
            if row==0:
                ax.set_title(label)
        axs[row,0].set_ylabel(("Intermediate","Final")[row]+" probe accuracy")
    axs[0,1].legend(frameon=False,fontsize=7,loc="upper left")
    save(fig,out/"02_semantic_readouts")


def causal(root, out):
    rows=table(root/"paired_comparisons.csv")
    fig,axs=plt.subplots(1,3,figsize=(11,3.9),layout="constrained")
    for ax,(metric,title) in zip(axs,(("response_mse","Hidden-state change"),("semantic_error","Intermediate contrast"),
                                     ("behavior_sqerr","Answer-margin change"))):
        for t in TASKS:
            j=TASKS.index(t)
            chosen=[r for r in rows if r["width"]=="2048" and r["task"]==t and r["metric"]==metric]
            for i,split in enumerate(("confirm","transfer")):
                competing=[r for r in chosen if r["split"]==split]
                best=min(competing,key=lambda r:float(r["baseline_error"]))
                row={k:100*float(best[k]) for k in ("relative_improvement","lower","upper")}
                errplot(ax,[j+(i-.5)*.3],[row],COLORS[j],fmt=("o","s")[i],
                        mfc=("white",COLORS[j])[i],fields=("relative_improvement","lower","upper"),
                        label=("Confirmation","Structural transfer")[i] if j==0 else None)
        ax.axhline(0,color="#687384",ls="--",lw=.9)
        if metric=="response_mse":
            ax.axhline(10,color="#A04461",ls=":",lw=1.2)
        ax.set_title(title)
        ax.set_xticks(range(3),TASK_LABELS,rotation=22,ha="right")
        ax.grid(axis="y",alpha=.15)
    fig.supylabel("Error reduction vs strongest simple comparator (%)",size=9)
    axs[-1].legend(frameon=False,fontsize=8,loc="lower left")
    save(fig,out/"03_causal_prediction")


def shift(root,out):
    d=lookup(table(root/"shift_summary.csv"),("task","split","input","loop","role","metric"))
    fig,axs=plt.subplots(1,3,figsize=(10,3.3),layout="constrained",sharey=True)
    for ax,t,label in zip(axs,TASKS,TASK_LABELS):
        for i,(key,name) in enumerate((("natural_world_difference","Natural paired-world difference"),
                                       ("physical_patch","Physical source patch"))):
            bottom=np.zeros(2)
            for role,region in enumerate(ROLES):
                values=[float(d[t,"transfer",key,str(k),str(role),"energy_share"]["mean"]) for k in (1,2)]
                ax.bar(np.arange(2)+(i-.5)*.36, values,.33,bottom=bottom,
                       facecolor=COLORS[role],hatch=("","///")[i],
                       edgecolor="white",lw=.45,label=region if i==0 else None)
                bottom+=values
        ax.set_xticks((-.18,.18,.82,1.18),["K1\nnatural","K1\npatch","K2\nnatural","K2\npatch"])
        ax.set_title(label)
        ax.set_ylim(0,1)
    axs[0].set_ylabel("Share of mean normalized Δ energy")
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncols=5,frameon=False)
    save(fig,out/"04_intervention_distribution")


def cases(root,out,case_file):
    resource=json.loads(case_file.read_text(encoding="utf-8"))
    cohort=table(root/"natural_accuracy_paths.csv")
    fig,axs=plt.subplots(2,3,figsize=(10,5.7),layout="constrained")
    for j,(t,label) in enumerate(zip(TASKS,TASK_LABELS)):
        rec=next(r for r in resource["cases"] if r["task"]==t and r["split"]=="transfer")
        signs=np.array(rec["edited_scores"])-np.array(rec["original_scores"])
        margin=signs[:,rec["donor_gold"]]-signs[:,rec["gold"]]
        ax=axs[0,j]
        for key,color in zip(("mapping","query","readout"),(COLORS[2],COLORS[3],COLORS[4])):
            ax.plot(rec["full_role_deltas"][key],color=color,label=key.capitalize())
        for x in (24,49,74,99):
            ax.axvline(x,color="#748398",lw=.65,alpha=.45)
        ax.set_xticks((24,49,74,99),[f"K{k}" for k in range(1,5)])
        ax.set_title(label)
        aux=ax.twinx()
        aux.spines["right"].set_visible(True)
        aux.plot((24,49,74,99),margin,color="#1B2434",marker="o",mfc="white",ls="--",
                 label="Answer-margin effect")
        if j==2:
            aux.set_ylabel("Donor answer-margin effect (nats)")
        ax=axs[1,j]
        paths=partition([r for r in cohort if r["task"]==t and r["split"]=="transfer"],("category",))
        for key,name,color in zip(("correction","regression","stay_correct","stay_wrong"),
                ("Correction","Regression","Stay correct","Stay wrong"),COLORS):
            r=paths[(key,)]
            y=[np.mean([float(v[f"k{k}_margin"]) for v in r]) for k in (1,2)]
            ax.plot((1,2),y,marker="o",color=color,label=f"{name} ({len(r)})")
        ax.axhline(0,color="#687384",lw=.8,ls=":")
        ax.set_xticks((1,2),("K1","K2"))
        ax.grid(axis="y",alpha=.2)
        ax.legend(frameon=False,fontsize=6.7,loc="best")
    axs[0,0].set_ylabel("Role-state change (L2)")
    axs[1,0].set_ylabel("Natural true-answer margin (nats)")
    axs[0,0].legend(frameon=False,fontsize=7)
    save(fig,out/"05_native_trajectories_and_cases")


def run(root, case_file):
    plot_style()
    out=root/"figures"
    out.mkdir(exist_ok=True)
    collective=lookup(table(root/"summary.csv"),("width","task","split","method","arm","metric"))
    natural(out,collective)
    semantics(root,out)
    causal(root,out)
    shift(root,out)
    cases(root,out,case_file)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("results",type=Path)
    p.add_argument("cases",type=Path)
    a=p.parse_args()
    run(a.results,a.cases)
