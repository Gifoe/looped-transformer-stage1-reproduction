"""Six figures from sealed prediction artifacts and native outputs."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import END, LABELS, aggregate, collect_manifest, load, phase_rows
from artifact_json import write


P={"full":"#176f80","base":"#ad5744","taylor":"#77758f","prefix_anderson":"#476a42",
   "tableau":"#26698d","twohop":"#ab5842","random":"#85838b","other":"#856794"}


def save(out,n,name,fig):
    for x in ("pdf","png"):
        fig.savefig(out/f"figure_{n}_{name}.{x}",dpi=250,bbox_inches="tight")
    plt.close(fig)


def plots(out,root,s):
    rows=load(root/"LOCAL.json")["rows"]
    alpha=(.0625,.125,.25,.5,1.)
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.3),constrained_layout=True)
    for kind in ("base","full"):
        for col,x in enumerate(("model_error","solver_error")):
            for grid,ls in ((2,"-"),(4,"--")):
                vals=[]
                for a in alpha:
                    ys=[q[x] for q in rows if q["method"]==kind and q["alpha"]==a and q["m"]==grid
                        and q["scheme"]=="rk4" and q["layer"] in (1,24)]
                    vals.append(np.median(ys))
                axes[col].loglog(alpha,vals,color=P[kind],ls=ls,marker="o",label=f"{LABELS[kind]}, m={grid}")
    axes[0].set(xlabel="Residual amplitude α",ylabel="Relative state error",title="Error to the discrete pair")
    axes[1].set(xlabel="Residual amplitude α",title="Difference from same-field RK4 m=16")
    axes[-1].legend(frameon=False,fontsize=7)
    save(out,1,"alpha_grid",fig)

    fig,ax=plt.subplots(figsize=(7.2,3.65),constrained_layout=True)
    kinds=("base","self","order","full","reverse","taylor","prefix_anderson")
    r=[np.array([e for q in rows if q["method"]==k and q["alpha"]==1.0 and q["m"]==2 and q["scheme"]=="rk4"
                     for e in q["model_error"]]) for k in kinds]
    ds=[np.median(z) for z in r]
    xs=np.arange(len(kinds))
    ax.scatter(ds,xs,c=[P.get(k,"#8f7c5d") for k in kinds])
    for j,z in enumerate(r):
        lo,hi=np.quantile(z,[.1,.9])
        ax.hlines(j,max(lo,1e-8),max(hi,1e-8),color=P.get(kinds[j],"#8f7c5d"),alpha=.66)
    ax.set(xscale="log",xlabel="Relative pair-endpoint L2 (median and 10–90%)",
           yticks=xs,yticklabels=[LABELS[k] for k in kinds])
    ax.invert_yaxis()
    save(out,2,"unit_amplitude_pair",fig)

    manifest,lookup=collect_manifest(root)
    native=load(root/"NATIVE_confirmation.json")["rows"]
    keys=["initial"]+[f"{k}/{l}" for k in range(1,4) for l in ("L1","L6","L12","L18","L24","norm")]
    fig,axes=plt.subplots(1,2,figsize=(9.7,3.5),sharey=True,constrained_layout=True)
    for ax,f in zip(axes,("tableau","twohop"),strict=True):
        for mode,c in (("whole",P[f]),("readout","#35333c"),("source","#319341"),("rule","#7f7090")):
            y=[]
            for time in keys:
                vals=[]
                for row in native:
                    if row["family"]!=f:
                        continue
                    r=row["initial"] if time=="initial" else row["methods"]["native"][time]
                    meta=lookup[row["group"]]
                    role=meta["source"] if mode=="source" else meta["rule"] if mode=="rule" else [meta["worlds"][0]["read"]]
                    vals.append(r["mixed_l2"] if mode=="whole" else np.linalg.norm(np.array(r["token_mixed_norm"])[role]))
                y.append(np.mean(vals))
            ax.plot(range(len(keys)),y,color=c,label=mode)
        ax.set(xticks=[0,6,12,18],xticklabels=["K1","K2","K3","K4"],title=f,xlabel="Native boundaries")
    axes[0].set_ylabel("Four-world interaction L2")
    axes[1].legend(frameon=False,ncols=2)
    save(out,3,"joint_state_trajectory",fig)

    chosen=s["decision"]["choice"]
    phase=s["decision"]["phase"]
    state,_,_=phase_rows(root,phase,lookup)
    fig,axes=plt.subplots(1,2,figsize=(8.9,3.65),sharex=True,sharey=True,constrained_layout=True)
    for j,key in enumerate((END[0],END[2])):
        ax=axes[j]
        r=[q for q in state if chosen in q["endpoints"][key]["pred"]]
        for f in ("tableau","twohop"):
            rr=[q["endpoints"][key] for q in r if q["family"]==f]
            ax.scatter([q["iq"] for q in rr],[q["pred"][chosen] for q in rr],s=14,alpha=.6,c=P[f],label=f)
        obs=[t for q in r for t in (q["endpoints"][key]["iq"],q["endpoints"][key]["pred"][chosen])]
        ax.plot([min(obs),max(obs)],[min(obs),max(obs)],":",color="#777777")
        ax.set(xlabel="Native signed Iq",title=("K2" if j==0 else "K4"))
    axes[0].set_ylabel(f"{LABELS[chosen]} prediction")
    axes[1].legend(frameon=False)
    save(out,4,"signed_response_forecast",fig)

    order=s["order"]["validation"]["rows"]
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.4),sharey=True,constrained_layout=True)
    for ax,k in zip(axes,(2,4)):
        q=[x["truth"] for x in order if x["loop"]==k]
        obs=np.mean([abs(x["ba"]) for x in q])
        for arm,c in (("BA_comp",P["full"]),("BA_random",P["random"]),("BA_other",P["other"])):
            delta=[obs]+[np.mean([abs(x["ba"]+x[f"{arm}/{a}"]) for x in q]) for a in (.25,.5,1.)]
            ax.plot([0,.25,.5,1],delta,marker="o",color=c,label=arm.removeprefix("BA_"))
        ax.set(xlabel="Dose (relative to projected bracket)",title=f"K{k}")
    axes[0].set_ylabel("Mean absolute distance from AM Iq")
    axes[-1].legend(frameon=False)
    save(out,5,"native_compensation",fig)

    fig,axes=plt.subplots(1,3,figsize=(11.3,3.45),constrained_layout=True)
    for f in ("tableau","twohop"):
        vals=[s["free"][phase][f][k][chosen]["error"]["mean"] for k in END]
        axes[0].plot([2,3,4],vals,marker="o",color=P[f],label=f)
    axes[0].set(xticks=[2,3,4],xlabel="Loop",ylabel="Held-out Iq MAE")
    axes[0].legend(frameon=False)
    for m in ("full","base","taylor","prefix_anderson"):
        vals=[s["arc"][k][m]["eq"]/s["arc"][k][m]["n"] for k in END]
        axes[1].plot([2,3,4],vals,marker="o",color=P[m],label=LABELS[m])
    axes[1].set(xticks=[2,3,4],xlabel="Loop",ylabel="ARC native-choice agreement")
    axes[1].legend(frameon=False,fontsize=7)
    clocks=load(root/"CLOCKS_AND_BETA.json")["rows"]
    for clock,c in (("slow_fast","#1c6b82"),("fast_slow","#926759")):
        a=[np.median([v for r in clocks for v in r["methods"][f"{m}/{clock}"]["distance_same_grid"]]) for m in (2,4,8)]
        axes[2].loglog([2,4,8],a,color=c,marker="o",label=clock.replace("_","–"))
    axes[2].set(xlabel="Fixed RK4 m",ylabel="Clock endpoint difference")
    axes[2].legend(frameon=False,fontsize=7)
    save(out,6,"transfer_and_clocks",fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    a=p.parse_args()
    summary=aggregate(a.out)
    write(a.out/"SUMMARY.json",summary)
    title=a.out/"figures"
    title.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42})
    plots(title,a.out,summary)
