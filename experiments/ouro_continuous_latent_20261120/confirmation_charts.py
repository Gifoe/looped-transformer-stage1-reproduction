"""Add groundable semantic and fully dose-matched causal figure panels."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze import (COLORS, behaviour_prediction, choice, dump, figure,
                     mean_ci, plt, read, semantic_checks)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    root=p.parse_args().root
    source=root/"confirmation"
    summary=json.loads((source/"SUMMARY.json").read_text())
    semantic=read(source/"semantic_predictions.jsonl")
    data=read(source/"grounded_candidate_scores.jsonl")
    rows=[dict(r,family="twohop",tracks={name:{"scores":z,"time":r["time"]}
          for name,z in r["scores"].items()}) for r in data]
    checks=semantic_checks(semantic,rows)
    checks["prediction"]=behaviour_prediction(semantic,rows)
    checks["native_endpoint"]={c:{
        "n":len(subset),
        "k1":sum(choice(r["scores"]["native"][0],r["y"]) for r in subset),
        "k2":sum(choice(r["scores"]["native"][-1],r["y"]) for r in subset)}
        for c in ("development","new_structure")
        for subset in ([r for r in data if r["cohort"]==c],)}
    edits=read(source/"matched_dose_causal.jsonl")
    errors=[abs(d-r["dose"])/r["dose"] for r in edits if r["dose"]>0 for d in r["random_doses"]]
    checks["relative_dose_error"]={"median":float(np.median(errors)),"max":max(errors)}
    dump(source/"CHECKS.json",checks)

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42})
    figs=root/"figures"
    fig,axes=plt.subplots(1,4,figsize=(12,3.15),sharey=True,constrained_layout=True)
    for ax,method in zip(axes,("native","A32","B32","linear")):
        q=summary["semantic"][f"twohop/new_structure/{method}"]
        for role,z in q["roles"].items():
            ax.plot(q["time"],z["mean"],label=role.upper(),color=COLORS[role])
            ax.fill_between(q["time"],z["low"],z["high"],color=COLORS[role],alpha=.13)
        ax.axhline(1/3,ls=":",color="#555555")
        ax.set(title=method,xlabel="Time",ylim=(.15,1))
    axes[0].set_ylabel("Grounded probe accuracy")
    axes[-1].legend(frameon=False)
    figure(figs,4,"semantic_readout",fig)

    fig,axes=plt.subplots(1,3,figsize=(10,3.2),sharey=True,constrained_layout=True)
    ts=(.25,.5,.75)
    ca=summary["causal"]
    for ax,m in zip(axes,("native","A16","B16")):
        for arm,label,c in (("semantic","Same rule, new bridge",COLORS["A32"]),
                            ("same_y","Same answer, new bridge",COLORS["B32"]),
                            ("same_b","Same bridge, new source",COLORS["linear"])):
            cells=[ca[f"new_structure/{m}/{t}/{arm}"]["edit_minus_random"] for t in ts]
            ax.plot(ts,[v["mean"] for v in cells],marker=".",label=label,color=c)
            ax.fill_between(ts,[v["low"] for v in cells],[v["high"] for v in cells],color=c,alpha=.13)
        ax.axhline(0,ls=":",color="#777777")
        ax.set(title=m,xlabel="Edit time",xticks=ts)
    axes[0].set_ylabel("Target logit margin over matched random")
    axes[-1].legend(frameon=False,fontsize=8)
    figure(figs,6,"causal_edits",fig)


if __name__=="__main__":
    main()
