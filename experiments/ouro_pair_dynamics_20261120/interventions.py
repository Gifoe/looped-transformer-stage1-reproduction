"""Prospective localized A/M order interventions with a declared native suffix."""
import argparse
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from causal_worlds import masked, sha
from dynamics_embedding import read_json, write_json
from pair_fields import PairOps, bracket, pair, rel, timed, load_runtime
from predictor import batches, edit, sample


CANDIDATES=(11,17,23)
ORDER_METHODS=("base","self","order","full","reverse","taylor","prefix_anderson")
DOSES=(.25,.5,1.)


def qvalue(args,target):
    z=[w[target]-sum(x for j,x in enumerate(w) if j!=target)/(len(w)-1) for w in args]
    return (z[3]-z[1])-(z[2]-z[0])


def anchor(op,h,l,transition):
    for _ in range(transition-1):
        for i in range(24):
            h=pair(op,i,h,"native")
        h=op.norm(h)
    for i in range(l):
        h=pair(op,i,h,"native")
    return h


def tail(op,h,l):
    for j in range(l+1,24):
        h=pair(op,j,h,"native")
    return op.norm(h)


def doses(op,l,x):
    d=bracket(op,l,x).detach()
    support=torch.zeros(d.shape[:2],device=d.device,dtype=torch.bool)
    support[op.rows,op.positions]=True
    d=d*support[...,None]
    gen=torch.Generator(device=x.device).manual_seed(416164)
    random=torch.randn(x.shape,device=x.device,generator=gen)*support[...,None]
    other=bracket(op,(l+7)%24,x).detach()*support[...,None]
    def matched(z):
        width=z.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        return z/width*d.norm(dim=-1,keepdim=True)
    return d,matched(random),matched(other)


def scores(op,h,pack):
    return sample(op,h,pack,True)


def prospective(model,m,phase,layers,loops,limit,truth,pred_path,receipt_path):
    if truth:
        receipt=read_json(receipt_path)
        assert sha(pred_path)==receipt["sha256"]
        frozen=read_json(pred_path)
        assert frozen["layers"]==list(layers) and frozen["loops"]==list(loops)
    timing=[]
    all_rows=[]
    with sdpa_kernel(SDPBackend.MATH):
        for family,cohort in m["groups"][phase].items():
            cohort=cohort[:limit] if limit else cohort
            for pack in batches(cohort,4):
                op=PairOps(model,[masked(w) for g in pack for w in g["worlds"]])
                start=edit(op,pack)
                for transition in loops:
                    for l in layers:
                        with timed(op,f"{phase}/{family}/K{transition+1}/L{l+1}",timing):
                            x=anchor(op,start,l,transition)
                            c,r,a=doses(op,l,x)
                            p={}
                            if truth:
                                am=pair(op,l,x,"native")
                                ba=pair(op,l,x,"native",swap=True)
                                p["native"]={"am":scores(op,tail(op,am,l),pack),"ba":scores(op,tail(op,ba,l),pack)}
                                for eta in DOSES:
                                    p["native"].update({
                                        f"AM_weak/{eta}":scores(op,tail(op,am-eta*c,l),pack),
                                        f"BA_comp/{eta}":scores(op,tail(op,ba+eta*c,l),pack),
                                        f"BA_random/{eta}":scores(op,tail(op,ba+eta*r,l),pack),
                                        f"BA_other/{eta}":scores(op,tail(op,ba+eta*a,l),pack)})
                            else:
                                for method in ORDER_METHODS:
                                    am=pair(op,l,x,method)
                                    ba=pair(op,l,x,method,swap=True)
                                    p[method]={"am":scores(op,tail(op,am,l),pack),"ba":scores(op,tail(op,ba,l),pack)}
                                    if method=="full":
                                        for eta in DOSES:
                                            p[method].update({
                                                f"AM_weak/{eta}":scores(op,tail(op,am-eta*c,l),pack),
                                                f"BA_comp/{eta}":scores(op,tail(op,ba+eta*c,l),pack),
                                                f"BA_random/{eta}":scores(op,tail(op,ba+eta*r,l),pack),
                                                f"BA_other/{eta}":scores(op,tail(op,ba+eta*a,l),pack)})
                            for j,g in enumerate(pack):
                                all_rows.append(dict(group=g["group"],family=family,layer=l+1,loop=transition+1,
                                    doses=c[4*j:4*j+4].norm(dim=-1).sum(-1).tolist(),
                                    methods={k:{arm:items[j] for arm,items in states.items()} for k,states in p.items()}))
    result=dict(rows=all_rows,timing=timing,phase=phase,layers=layers,loops=loops,limit=limit)
    if truth:
        assert {(q["group"],q["layer"],q["loop"]) for q in frozen["rows"]}=={
            (q["group"],q["layer"],q["loop"]) for q in all_rows}
        result["prediction_sha256"]=receipt["sha256"]
    return result


def choose(out):
    manifest=read_json(out/"MANIFEST.json")
    targets={q["group"]:q["targets"][-1] for cohort in manifest["groups"]["development"].values() for q in cohort}
    actual=read_json(out/"LOCAL_NATIVE_development.json")
    data={}
    for q in actual["rows"]:
        states=q["methods"]["native"]
        y=targets[q["group"]]
        effect=qvalue(states["ba"]["scores"],y)-qvalue(states["am"]["scores"],y)
        data.setdefault(q["layer"],[]).append(abs(effect))
    means={k:sum(v)/len(v) for k,v in data.items()}
    layer=max(means,key=means.get)
    write_json(out/"LOCAL_FREEZE.json",dict(layers=[layer-1],loops=[1,3],limit=24,
        selector="development mean absolute difference of four-world fixed Iq under true native BA vs AM",
        development_native_sha256=sha(out/"LOCAL_NATIVE_development.json"),
        development_mean_abs=means,support="answer token on each world",doses=DOSES,
        methods=ORDER_METHODS,integrator="RK4",m=2))


def clocks_and_beta(model,manifest,out,layer):
    panels=[]
    timing=[]
    with sdpa_kernel(SDPBackend.MATH):
        for family,items in manifest["groups"]["development"].items():
            pack=items[:4]
            op=PairOps(model,[masked(w) for g in pack for w in g["worlds"]])
            for b_s,b_r in ((.25,.25),(.5,.5),(1.,.5),(.5,1.),(1.,1.)):
                x=anchor(op,edit(op,pack,b_s,b_r),layer,1)
                native=scores(op,tail(op,pair(op,layer,x,"native"),layer),pack)
                method={}
                reference=None
                for m in (2,4,8):
                    for clock in ("linear","slow_fast","fast_slow"):
                        with timed(op,f"clock/{family}/{b_s}/{b_r}/{m}/{clock}",timing):
                            z=pair(op,layer,x,"full",steps=m,clock=clock)
                        if clock=="linear":
                            reference=z
                        method[f"{m}/{clock}"]={"distance_same_grid":rel(op,z,reference).tolist(),
                                  "records":scores(op,tail(op,z,layer),pack)}
                panels.append(dict(family=family,groups=[g["group"] for g in pack],beta_s=b_s,beta_r=b_r,
                                   native=native,methods=method,layer=layer+1))
    write_json(out/"CLOCKS_AND_BETA.json",dict(rows=panels,timing=timing))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("stage",choices=("predict","truth","freeze","clocks"))
    p.add_argument("phase",nargs="?",default="development")
    a=p.parse_args()
    if a.stage=="freeze":
        choose(a.out)
        return
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model,_,_=load_runtime(a.model)
    model.requires_grad_(False).float()
    m=read_json(a.out/"MANIFEST.json")
    if a.phase=="development":
        layers,loops,limit=CANDIDATES,(3,),8
    else:
        freeze=read_json(a.out/"LOCAL_FREEZE.json")
        layers,loops,limit=freeze["layers"],freeze["loops"],freeze["limit"]
    if a.stage=="clocks":
        clocks_and_beta(model,m,a.out,read_json(a.out/"LOCAL_FREEZE.json")["layers"][0])
        return
    pred=a.out/f"LOCAL_PREDICTIONS_{a.phase}.json"
    receipt=a.out/f"LOCAL_PREDICTION_SHA_{a.phase}.json"
    x=prospective(model,m,a.phase,layers,loops,limit,a.stage=="truth",pred,receipt)
    if a.stage=="predict":
        write_json(pred,x)
        write_json(receipt,dict(sha256=sha(pred),script_sha256=sha(Path(__file__))))
    else:
        write_json(a.out/f"LOCAL_NATIVE_{a.phase}.json",x)


if __name__=="__main__":
    main()
