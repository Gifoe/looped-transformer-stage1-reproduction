"""Draw the seven numbered figures from released numerical measurements."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize import FAMILIES, load, pair_dict, summarize, write


COLORS={"tableau":"#27688a","twohop":"#ab523d","arc":"#796392",
        "A":"#28759d","B":"#ab523d","native":"#343536","control":"#7b7b7b"}
MSET=(16,32,64,128,256,512)


def figure(out,n,slug,fig):
    for suffix in ("pdf","png"):
        fig.savefig(out/f"figure_{n}_{slug}.{suffix}",dpi=250,bbox_inches="tight")
    plt.close(fig)


def project(out,summary,raw,diag,order):
    d=raw["primary"]
    fig,axes=plt.subplots(2,3,figsize=(11.2,5.5),sharex=True,constrained_layout=True)
    for row,kind in enumerate(("A","B")):
        for ax,f in zip(axes[row],FAMILIES):
            for m,c in zip(MSET,plt.cm.viridis(np.linspace(.12,.87,len(MSET)))):
                q=d[f][kind]["geometry"][f"euler{m}"]
                ax.plot(q["t"],np.mean(q["to_initial"],axis=1),color=c,label=str(m))
            ax.set_title(f"{kind} · {f}")
            if row==1:
                ax.set_xlabel("Constructed progress")
    axes[0,0].set_ylabel("Distance from K1 (relative)")
    axes[1,0].set_ylabel("Distance from K1 (pre-norm)")
    axes[0,-1].legend(title="Euler subdivisions",frameon=False,ncols=2)
    figure(out,1,"halving_trajectories",fig)

    fig,axes=plt.subplots(2,2,figsize=(8.6,6),constrained_layout=True)
    for row,method in enumerate(("A","B")):
        for j,(stage,tag) in enumerate((("primary","BF16 field"),("precision_fp32","FP32 field"))):
            ax=axes[row,j]
            for f in FAMILIES:
                pairs=summary[stage][f][method]
                if stage=="precision_fp32":
                    pairs={**pairs,**summary["reference"][f][method]}
                for solver,style in (("euler","-"),("rk4","--")):
                    coords=[]
                    for m in MSET:
                        k=f"{solver}{m}:{solver}{m*2}"
                        if k in pairs:
                            coords.append((m*(4 if solver=="rk4" else 1),pairs[k]["end"]["mean"]))
                    if coords:
                        ax.loglog(*zip(*coords),style,marker="o" if solver=="euler" else "^",
                                  ms=4,color=COLORS[f],label=f"{f} · {solver}")
            ax.set(title=f"{method} · {tag}",xlabel="Coarser path field evaluations per loop/branch",
                   ylabel="Endpoint self-difference")
    axes[0,0].legend(frameon=False,ncols=2,fontsize=7)
    figure(out,2,"euler_rk4",fig)

    fig,axes=plt.subplots(2,1,figsize=(9.2,5.4),sharex=True,constrained_layout=True)
    for f,g in d.items():
        stats=summary["primary"][f]["B"]["euler256:euler512"]
        branch=np.arange(1,49)
        axes[0].semilogy(branch,np.array(stats["path_mean"])[4::4],color=COLORS[f],label=f"{f} · propagated")
        matched=[np.mean(q["matched_paths"]["256"][-1]["full"]) for q in g["B_local"]]
        axes[0].semilogy(branch,matched,ls="--",color=COLORS[f],label=f"{f} · same start")
        defect=[np.mean(q["native_vs_fine"]["full"]) for q in g["B_local"]]
        axes[1].plot(branch,defect,color=COLORS[f],label=f)
    axes[0].set_ylabel("Relative error, 256 vs 512")
    axes[1].set(xlabel="Residual branch (1–48)",ylabel="Relative local defect\nof one unit Euler step")
    axes[0].legend(frameon=False,ncols=3,fontsize=7)
    figure(out,3,"branch_errors",fig)

    endpoints=("K1_vs_native_K2","A_vs_native_K2","B_post_vs_native_K2",
               "B_pre_vs_native_pre","A_vs_B_pre","A_vs_B_post")
    names=("K1 to native K2","A to native K2","B post-norm to native K2",
           "B pre-norm to native pre-norm","A to B pre-norm","A to B post-norm")
    fig,ax=plt.subplots(figsize=(8.5,3.9),constrained_layout=True)
    for j,f in enumerate(FAMILIES):
        e=summary["primary"][f]["endpoints"]
        cells=[e[k]["state"] for k in endpoints]
        y=np.arange(len(cells))+(j-1)*.17
        ax.errorbar([v["mean"] for v in cells],y,xerr=[
            [v["mean"]-v["min"] for v in cells],[v["max"]-v["mean"] for v in cells]],
            marker="o",ls="none",color=COLORS[f],label=f,capsize=2)
    ax.set(xlabel="Relative full-state L2 (mean and observed min–max)",yticks=np.arange(len(names)),yticklabels=names)
    ax.invert_yaxis()
    ax.legend(frameon=False)
    figure(out,4,"native_and_continuous_endpoints",fig)

    fig,axes=plt.subplots(1,2,figsize=(9.5,3.45),constrained_layout=True)
    for f,g in d.items():
        for ax,key in zip(axes,("cross","B_vs_native_path")):
            q=g[key]
            ax.plot(q["t"],[np.mean(v["full"]) for v in q["state"]],color=COLORS[f],label=f)
    for ax,title in zip(axes,("A vs B (pre-norm)","B vs native branches (pre-norm)")):
        ax.set(title=title,xlabel="Normalized progress",ylabel="Relative full-state distance")
    axes[-1].legend(frameon=False)
    figure(out,5,"dynamical_disagreement",fig)

    fig,axes=plt.subplots(2,3,figsize=(11.5,5.6),constrained_layout=True)
    for row,kind in enumerate(("A","B")):
        for f,g in d.items():
            q=g[kind]["geometry"]["euler512"]
            t=np.asarray(q["t"])
            mids=(t[:-1]+t[1:])/2
            speed=np.mean(q["speed"],axis=1)
            turn=np.degrees(np.arccos(np.clip(np.mean(q["turn_cos"],axis=1),-1,1)))
            curve=np.mean(q["curvature"],axis=1)
            for ax,z,x in zip(axes[row],(speed,turn,curve),(mids,t[1:-1],t[1:-1])):
                ax.plot(x,z,color=COLORS[f],label=f)
                ax.set_xlabel("Constructed progress")
        for ax,label in zip(axes[row],("Speed (L2 / progress)","Turn (degrees)","Curvature (1 / L2)")):
            ax.set_title(f"{kind} · {label}")
    axes[0,0].legend(frameon=False)
    figure(out,6,"speed_turn_curvature",fig)

    fig,axes=plt.subplots(1,3,figsize=(11.4,4),constrained_layout=True)
    specs=(("Finite step and norm",
            (("Native final Norm","native_norm"),("B final Norm","B_norm"),("B vs native pre-Norm","B_pre_vs_native_pre"))),
           ("Precision (common K1)",
            (("BF16 vs FP32 A256","BF16_vs_FP32_A256"),("BF16 vs FP32 B256 pre","BF16_vs_FP32_B256_pre"),
             ("BF16 vs FP32 B256 post","BF16_vs_FP32_B256_post"),("Same-input B Norm","B_norm_only_BF16_vs_FP32_same_input"))),
           ("Norm in the A vector field",
            (("Omit loop Norm","pre"),("Append Norm at endpoint","post"))))
    for i,(ax,(title,labels)) in enumerate(zip(axes,specs,strict=True)):
        for j,f in enumerate(FAMILIES):
            if i==0:
                vals=[summary["primary"][f]["endpoints"][k]["state"]["mean"] for _,k in labels]
            elif i==1:
                vals=[summary["precision_fp32"][f]["endpoints"][k]["state"]["mean"] for _,k in labels]
            else:
                vals=[summary["primary"][f]["A_norm_intervention"][k]["mean"] for _,k in labels]
            ax.scatter(vals,np.arange(len(vals))+(j-1)*.17,color=COLORS[f],label=f,marker=("o","^","s")[j])
        ax.set(xscale="log",title=title,xlabel="Relative L2",yticks=np.arange(len(labels)),
               yticklabels=[x for x,_ in labels])
        ax.invert_yaxis()
    axes[0].legend(frameon=False,loc="lower left",fontsize=7)
    figure(out,7,"norm_precision_finite_steps",fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    a=p.parse_args()
    s,r,d,o=summarize(a.root)
    write(a.root,"SUMMARY",s)
    out=a.root/"figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42})
    project(out,s,r,d,o)
