"""Describe native layer-input mixed differences independently of output rescue."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from native import Forward, load, save, setup, sha
from readout import core, mixed, plan

LABELS=("K1","K2","cross_loop")


class LayerReader:
    def __init__(self,p):
        self.p=p
        self.which=torch.arange(len(p),device=p.device)
        self.r={}

    def put(self,key,name,x):
        if name=="input":
            self.r[key]=x[self.which,self.p].detach()


def stats(raw):
    tri={}
    for label in ("repair","safe"):
        pool=[r["norm"] for r in raw if r["kind"]==label]
        arr=np.asarray(pool)
        tri[label]=dict(n=len(pool),statistics={name:np.quantile(arr[:,p,:],[.25,.5,.75],axis=0).tolist()
                                               for p,name in enumerate(LABELS)})
    return tri


def figure(report,path):
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,
                         "axes.labelsize":10,"axes.titlesize":11,"xtick.labelsize":9,"ytick.labelsize":9})
    fig,axes=plt.subplots(1,2,figsize=(9.6,3.55),sharey=True)
    domain=np.arange(1,25)
    legend=[("K1","#126782","Loop 1"),("K2","#B43A27","Loop 2"),("cross_loop","#7660A7","Across-loop change")]
    for axis,title in zip(axes,("repair","safe"),strict=True):
        data=report[title]
        for k,color,name in legend:
            lo,avg,hi=data["statistics"][k]
            axis.plot(domain,avg,color=color,lw=1.7,label=name)
            axis.fill_between(domain,lo,hi,color=color,alpha=.12)
        axis.set_title(("Borrow repairs" if title=="repair" else "Same-answer no-borrow")+f" (n={data['n']})",
                       loc="left",fontweight="semibold")
        axis.set_xticks([1,5,9,14,17,21,24])
        axis.axvline(17,color="#767E89",ls=":",lw=1)
        axis.set_xlabel("Physical layer input")
        axis.grid(axis="y",alpha=.17)
        axis.set_xlim(1,24)
    axes[0].set_ylabel("Readout interaction norm")
    axes[1].legend(frameon=False,loc="upper left",fontsize=8)
    fig.tight_layout(pad=1.4)
    fig.savefig(path,dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)


@torch.inference_mode()
def run(model,root):
    mod,tok=setup(model)
    f=Forward(mod)
    raw=root/"NATIVE.json"
    _,bank,_=plan(tok,load(raw)["rows"])
    aligned=[r for r in bank if r["aligned"]]
    out=[]
    for j in range(0,len(aligned),6):
        group=[c for z in aligned[j:j+6] for c in z["pair"]]
        _,_,dat,_=core(f,group,LayerReader)
        one=torch.stack([mixed(dat.r[0,l]) for l in range(24)],dim=1)
        two=torch.stack([mixed(dat.r[1,l]) for l in range(24)],dim=1)
        curves=[x.norm(dim=-1).tolist() for x in (one,two,two-one)]
        assert all(v[0]==0 for v in curves[0])
        for i,c in enumerate(group):
            out.append(dict(id=c["id"],kind=c["kind"],norm=[x[i] for x in curves]))
        print("state profile",len(out),2*len(aligned),flush=True)
    save(root/"TRAJECTORY_RAW.json",dict(rows=out,native_sha256=sha(raw)))
    tr=stats(out)
    save(root/"TRAJECTORY_SUMMARY.json",tr)
    figure(tr,root/"TRAJECTORY.png")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)

