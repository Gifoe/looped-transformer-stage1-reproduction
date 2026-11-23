"""Make a compact evidence figure from the frozen confirmation audit."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from archive_audit import read


def spread(records):
    y=np.array([r["mean"] for r in records])
    lo,hi=np.array([r["ci95"] for r in records]).T
    return y,lo,hi


def main():
    out=Path("results")
    data=read(out/"DYNAMICAL_AUDIT.json")["results"]
    color={"arc":"#115476","csqa":"#AD5B29"}
    labels={"arc":"ARC-Easy","csqa":"CSQA"}
    fig,axes=plt.subplots(1,3,figsize=(12.0,3.6),layout="constrained")

    for domain in ("arc","csqa"):
        x=np.arange(4)
        for k in ("1","2"):
            cells=data[f"HORIZON_{domain}"][k]["horizons"]
            y,lo,hi=spread([cells[str(m)]["finite"]["1.0"] for m in x])
            axes[0].fill_between(x,lo,hi,color=color[domain],alpha=.12)
            axes[0].plot(x,y,color=color[domain],linestyle="-" if k=="1" else "--",
                         marker="o" if k=="1" else "x",label=f"{labels[domain]}, h{k}")
    axes[0].set(title="Fixed-state horizon",xlabel="Remaining original loops",ylabel="Δ correct-answer log odds",xticks=range(4))
    axes[0].legend(frameon=False,fontsize=7.8)

    sequence=[(domain,k,item["summary"]["transport"]) for domain in ("arc","csqa")
              for k,item in data[f"E2_FRESH_{domain}"].items()]
    x=np.arange(len(sequence))
    for i,(domain,_,v) in enumerate(sequence):
        parallel=v["parallel_task"]["mean"]
        residual=v["residual_task"]["mean"]
        axes[1].bar(i,parallel,color=color[domain],alpha=.90,label="Along late write" if i==0 else None)
        axes[1].bar(i,residual,bottom=parallel,color=color[domain],alpha=.29,
                    label="Orthogonal residual" if i==0 else None)
        axes[1].scatter(i,v["a_late"]["mean"],color="#172435",marker="_",s=180,zorder=4,
                        label="Direct late write" if i==0 else None)
    axes[1].set(title="What the next loop transports",ylabel="Signed first-order task projection",
                xticks=x,xticklabels=[labels[d]+"\n"+k.replace("_"," ") for d,k,_ in sequence])
    axes[1].tick_params(axis='x',labelsize=7.4)
    axes[1].legend(frameon=False,fontsize=7.8)

    offset=np.arange(4)
    cells=[(domain,k,part) for domain in ("arc","csqa") for k,part in data[f"E3_FRESH_{domain}"].items()]
    for treatment,shift,marker,label in (("restore_query_key",-.10,"o","Restore Q/K"),("restore_value",.11,"s","Restore V")):
        values=[part["results"][treatment]["vs_early"] for _,_,part in cells]
        y,lo,hi=spread(values)
        axes[2].errorbar(offset+shift,y,yerr=[y-lo,hi-y],fmt=marker,capsize=2.5,label=label)
    axes[2].set(title="Matched-dose path restoration",ylabel="Δ log odds relative to early write",
                xticks=offset,xticklabels=[labels[d]+"\n"+k.replace("_"," ") for d,k,_ in cells])
    axes[2].tick_params(axis='x',labelsize=7.4)
    axes[2].legend(frameon=False,fontsize=7.8)
    for ax in axes:
        ax.axhline(0,color="#858F95",linewidth=.8,zorder=0)
        ax.spines[["top","right"]].set_visible(False)
    for extension in ("png","pdf"):
        fig.savefig(out/f"SIGNED_DYNAMICS.{extension}",dpi=200)
    plt.close(fig)


if __name__=="__main__":
    main()
