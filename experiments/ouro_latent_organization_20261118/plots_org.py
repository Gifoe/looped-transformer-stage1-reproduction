"""Produce six quantitative figures with a shared Loop × Layer coordinate system."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

TASKS=("relation","logic","arithmetic")
NAME={"relation":"Relation","logic":"Logic","arithmetic":"Arithmetic"}
COLORS=("#0F4D92","#B64342","#42949E","#767676","#A87736")


def read(p):
    with p.open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def nums(rows,key):
    return np.array([float(r[key]) for r in rows])


def figsave(fig,out,p):
    for ext in ("png","pdf"):
        fig.savefig(out/f"{p}.{ext}",dpi=300,bbox_inches="tight",pad_inches=.12)
    plt.close(fig)


def loop_ticks(ax):
    ax.set_yticks(range(4),["K1","K2","K3","K4"])
    ax.set_xticks([0,5,11,17,23,24],["0","5","11","17","23","N"])
    ax.set_xlabel("Physical layer · N = post-loop norm")


def trajectories(root,out):
    full=root/"full_token"
    idx=json.loads((full/"catalog.json").read_text())
    values=np.load(full/"figure_values.npz")
    fig,axes=plt.subplots(3,2,figsize=(8.6,8.1),layout="constrained")
    for task,row in zip(TASKS,axes):
        cohort=[next(i for i,r in enumerate(idx) if r["task"]==task and r["trajectory"]==label)
                for label in ("corrected","always_wrong")]
        pair=[values["mid_margin"][i] for i in cohort]
        lim=max(max(np.abs(x).max() for x in pair),.1)
        for a,i,v in zip(row,cohort,pair):
            case=idx[i]
            im=a.imshow(v,aspect="auto",cmap="RdBu_r",vmin=-lim,vmax=lim)
            outcome=" ".join("✓" if case[f"K{k}"] else "×" for k in range(1,5))
            a.set_title(f"{NAME[task]} · {case['trajectory'].replace('_',' ')}  {outcome}",
                        loc="left",fontweight="semibold")
            loop_ticks(a)
        fig.colorbar(im,ax=list(row),shrink=.8,label="Correct intermediate margin")
    figsave(fig,out,"01_latent_trajectories")


def reorganization(root,out):
    raw=read(root/"paired_distances.csv")
    rdm=read(root/"rdm_adjusted.csv")
    fig,axes=plt.subplots(2,3,figsize=(11.5,5.6),layout="constrained")
    keys=(("rx","Intermediate + answer change"),("ry","Same intermediate"),
          ("rd","Same answer"),("rn","Nuisance"))
    for col,t in enumerate(TASKS):
        a,b=axes[:,col]
        for j,(key,label) in enumerate(keys):
            s=[r for r in raw if r["task"]==t and r["split"]=="confirm"
               and r["role"]=="readout" and r["layer"]=="23" and r["pair"]==key]
            y,lo,hi=[nums(s,x) for x in ("mean","lower","upper")]
            a.errorbar(range(1,5),y,yerr=np.maximum([y-lo,hi-y],0),color=COLORS[j],
                       marker="o",ms=3,capsize=2,lw=1.2,label=label)
        a.set_title(NAME[t],loc="left",fontweight="semibold")
        s=[r for r in rdm if r["task"]==t and r["split"]=="confirm" and
           r["role"]=="mapping" and r["layer"]=="23"]
        for j,(k,label) in enumerate((("intermediate","Intermediate"),("final","Final"),
                                     ("map_factor","Mapping factor"))):
            b.plot(range(1,5),nums(s,k),color=COLORS[j],marker="o",ms=3,lw=1.25,label=label)
        b.axhline(0,color="#999999",ls=":",lw=.8)
        for p in (a,b):
            p.set_xticks(range(1,5))
        b.set_xlabel("Recurrence depth")
    axes[0,0].set_ylabel("Readout · projected cosine distance")
    axes[1,0].set_ylabel("Mapping · adjusted coefficient")
    axes[0,-1].legend(fontsize=7,frameon=False)
    axes[1,-1].legend(fontsize=7,frameon=False)
    figsave(fig,out,"02_semantic_reorganization")


def emergence(root,out):
    raw=read(root/"readouts.csv")
    fig,axes=plt.subplots(3,4,figsize=(12,7.2),layout="constrained")
    for t,line in zip(TASKS,axes):
        for role,a in zip(("source","mapping","query","readout"),line):
            s=[r for r in raw if r["task"]==t and r["split"]=="transfer"
               and r["target"]=="intermediate" and r["role"]==role]
            chance=.2 if t=="relation" else .5
            m=nums(s,"accuracy").reshape(4,25)-chance
            im=a.imshow(m,aspect="auto",norm=TwoSlopeNorm(vmin=-.15,vcenter=0,vmax=.55),
                        cmap="RdBu_r",interpolation="nearest")
            a.set_title(f"{NAME[t]} · {role}",loc="left",fontsize=9.4,fontweight="semibold")
            loop_ticks(a)
            if a!=line[0]:
                a.tick_params(labelleft=False)
    fig.colorbar(im,ax=axes,shrink=.8,label="Held-out intermediate accuracy above chance")
    figsave(fig,out,"03_information_redistribution")


def outcomes(root,out):
    raw=read(root/"success_failure.csv")
    fig,axes=plt.subplots(1,3,figsize=(11.7,3.55),layout="constrained")
    for t,a in zip(TASKS,axes):
        for j,label in enumerate(("always_correct","corrected","regressed","always_wrong","transient")):
            s=[x for x in raw if x["task"]==t and x["split"]=="confirm" and x["trajectory"]==label]
            if not s:
                continue
            v,lo,hi=[nums(s,k) for k in ("mid_margin_mean","mid_margin_lower","mid_margin_upper")]
            a.plot(range(1,5),v,lw=1.35,marker="o",ms=3,color=COLORS[j],
                   label=f"{label.replace('_',' ')} ({s[0]['n']})")
            a.fill_between(range(1,5),lo,hi,color=COLORS[j],alpha=.14,lw=0)
        r=next(x for x in raw if x["task"]==t)
        a.set_title(f"{NAME[t]} · {r['role']} L{r['layer']}",loc="left",fontweight="semibold")
        a.set(xticks=range(1,5),xlabel="Recurrence depth")
        a.axhline(0,color="#888888",ls=":",lw=.7)
        a.legend(fontsize=7.3,frameon=False)
    axes[0].set_ylabel("Correct intermediate margin")
    figsave(fig,out,"04_success_failure")


def compare(root,out):
    s=read(root/"shared_dynamics.csv")
    f=read(root/"cross_loop.csv")
    fig,axes=plt.subplots(1,4,figsize=(13.2,3.45),layout="constrained")
    a=axes[0]
    a.axvline(0,color="#999999",ls=":",lw=.7)
    for idx,t in enumerate(TASKS[::-1]):
        for j,split in enumerate(("confirm","transfer")):
            r=next(q for q in s if q["task"]==t and q["split"]==split)
            y=idx+(.12 if j else -.12)
            v,lo,hi=(float(r[k]) for k in ("gain","lower","upper"))
            a.errorbar(v,y,xerr=[[max(v-lo,0)],[max(hi-v,0)]],color=COLORS[j],
                       marker="o",ms=3.4,capsize=2.5,label=split if idx==0 else None)
    a.set_yticks(range(3),[NAME[t] for t in TASKS[::-1]])
    a.set(xlabel="K4 − K1 accuracy",title=f"Frozen role · {s[0]['role']} L{s[0]['layer']}")
    a.legend(fontsize=7.4,frameon=False)
    for t,a in zip(TASKS,axes[1:]):
        r=[q for q in f if q["task"]==t and q["split"]=="transfer" and q["layer"]=="23"
           and q["train_role"]=="mapping" and q["test_role"]=="mapping"]
        m=nums(r,"accuracy").reshape(4,4)
        im=a.imshow(m,aspect="equal",vmin=.4,vmax=1,cmap="viridis")
        a.set_title(NAME[t],loc="left",fontweight="semibold")
        a.set_xticks(range(4),range(1,5)); a.set_yticks(range(4),range(1,5))
        a.set(xlabel="Test K",ylabel="Train K")
    fig.colorbar(im,ax=list(axes[1:]),shrink=.78,label="Frozen decoder accuracy")
    figsave(fig,out,"05_cross_task_comparison")


def case_studies(root,out):
    fig,axes=plt.subplots(3,2,figsize=(9.4,8.15),layout="constrained")
    for task,(a,b) in zip(TASKS,axes):
        file=root/"causal"/task/"cases.json"
        r=json.loads(file.read_text())
        c=r.get("median_correct_flip",r["median"])
        goal,orig=c["counterfactual"],c["original"]
        for j,(name,scores) in enumerate((("Native",c["native"]),("Full donor",c["arms"]["full"]["scores"]),
                              ("Decoder-aligned",c["arms"]["aligned"]["scores"]),
                              ("Early source",c["arms"]["source_control"]["scores"]))):
            a.plot(range(1,5),[p[goal]-p[orig] for p in scores],"o-",lw=1.25,ms=3,
                   color=COLORS[j],label=name)
        a.axhline(0,ls=":",color="#777777",lw=.7)
        a.set_title(f"{NAME[task]} · {c['role']} K{c['site'][0]}/L{c['site'][1]}",
                    loc="left",fontweight="semibold")
        a.set(xticks=range(1,5),xlabel="Recurrence depth",ylabel="Counterfactual margin")
        a.legend(frameon=False,fontsize=7.4)
        v=np.array(c["trace"]["full"]["readout"])
        im=b.imshow(v,aspect="auto",cmap="magma",interpolation="nearest")
        loop_ticks(b)
        b.set_title("Full patch · downstream readout L2",loc="left",fontweight="semibold")
        fig.colorbar(im,ax=b,shrink=.8)
    figsave(fig,out,"06_mechanistic_cases")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("study",type=Path)
    a=p.parse_args()
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.4,"axes.spines.top":False,
                        "axes.spines.right":False,"axes.linewidth":.85,"pdf.fonttype":42})
    out=a.study/"figures"
    out.mkdir(exist_ok=True)
    trajectories(a.study,out)
    reorganization(a.study,out)
    emergence(a.study,out)
    outcomes(a.study,out)
    compare(a.study,out)
    case_studies(a.study,out)
