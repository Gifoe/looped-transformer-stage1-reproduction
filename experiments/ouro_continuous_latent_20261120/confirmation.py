"""Ground probe labels in visible prompt positions and match each causal edit dose."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze import mean_ci, semantic_summary
from causal import TIMES, put_dose
from dynamics import NativeBatch, interpolation, random_walk
from semantics import fit_and_measure
from study import GRID, dump, load_runtime, make_batches, policy_seed, role_tokenize, write_row


def anchor(tok, world):
    text=world["text"].removesuffix("\nAnswer:")
    source,bridge,_=world["within"]
    source_order=sorted(range(3),key=lambda j:text.index(f"The {source[j]} key opens"))
    bridge_order=sorted(range(3),key=lambda j:text.index(f"The {bridge[j]} cabinet contains"))
    grounded=dict(world,s=source_order.index(world["s"]),
                  m=bridge_order.index((source_order[0]+world["m"])%3),
                  b=bridge_order.index(world["b"]),y_index=world["y"])
    note="\nOptions: " + ", ".join(world["choices"]) + "."
    encoded=role_tokenize(tok,[(None,text+note),("readout","\nAnswer:")],world["choices"],grounded)
    old_roles={k:v for k,v in world["roles"].items() if k!="readout"}
    encoded["roles"]={**old_roles,**encoded["roles"]}
    return encoded


def repeat_causal(model, worlds, f):
    wants=((0,0,0),(1,0,0),(1,0,2),(1,2,0))
    four=[next(w for w in worlds if tuple(w[k] for k in ("s","m","r"))==key) for key in wants]
    rec, donor, same_y, same_b=four
    assert donor["b"]==same_y["b"]!=rec["b"]==same_b["b"]
    read=rec["roles"]["readout"][-1]
    g=torch.Generator(device="cuda").manual_seed(policy_seed("matched_dose",rec["group"]))
    batch=NativeBatch(model,four+[rec]*12,trace=True)
    names=("clean","semantic","same_y","same_b")
    for method, states, suffix in (
            ("native",batch.native,lambda t,h:batch.branch_path(1,start=h,from_branch=int(48*t))),
            ("A16",batch.euler(16),lambda t,h:batch.euler(16,start=h,from_index=int(16*t))),
            ("B16",batch.branch_path(16),lambda t,h:batch.branch_path(16,start=h,from_branch=int(48*t)))):
        for t in TIMES:
            h=states[t]
            edited=h[:1].repeat(len(h),1,1)
            edited[1:4,read]=h[1:4,read]
            for i in range(1,4):
                v=h[i,read].float()-h[0,read].float()
                for j in range(4):
                    direction=torch.randn(v.shape,device="cuda",generator=g)
                    edited[4+(i-1)*4+j,read]=put_dose(h[0,read],direction,v.norm())
            end=suffix(t,edited)[1.]
            assert torch.equal(end[0],states[1.][0])
            logits=batch.head(end).cpu().tolist()
            dose=(edited[:,read].float()-h[0,read].float()).norm(dim=-1).tolist()
            targets=[x["y"] for x in four]
            target=targets[1]
            margins=[row[target]-max(x for j,x in enumerate(row) if j!=target) for row in logits]
            for i in range(1,4):
                edit=i
                first=i*4
                control=margins[first:first+4]
                write_row(f,dict(group=rec["group"],cohort=rec["cohort"],method=method,time=t,
                               worlds=[w["item"] for w in four],arm=names[i],target=target,scores=logits,
                               dose=dose[edit],random_doses=dose[first:first+4],
                               edit_minus_clean=margins[edit]-margins[0],
                               edit_minus_random=margins[edit]-float(np.mean(control))))
    return batch.counts


def gather(model, group):
    n=NativeBatch(model,group,trace=True)
    paths={"native":n.native,"linear":interpolation(n.k1,n.k2,GRID),
           "norm_linear":interpolation(n.k1,n.k2,GRID,match_norm=True)}
    for m in (8,16,32):
        paths[f"A{m}"]=n.euler(m)
        paths[f"B{m}"]=n.branch_path(m)
    g=torch.Generator(device="cuda").manual_seed(policy_seed("grounded_random",group[0]["group"]))
    paths["random"]=random_walk(paths["A32"],g)
    rows=n.rows
    states={k:(list(GRID),torch.stack([v[t][rows,n.lengths-1].half() for t in GRID]).cpu().numpy())
            for k,v in paths.items()}
    traces={k:torch.stack([n.head(v[t]) for t in GRID]).transpose(0,1).cpu().tolist() for k,v in paths.items()}
    return states,traces,n.counts


def causal_aggregate(rows):
    result={}
    for cohort in ("development","new_structure"):
        for method in ("native","A16","B16"):
            windows={}
            for t in TIMES:
                for arm in ("semantic","same_y","same_b"):
                    q=[r for r in rows if (r["cohort"],r["method"],r["time"],r["arm"])
                       == (cohort,method,t,arm)]
                    result[f"{cohort}/{method}/{t}/{arm}"]={
                        k:mean_ci([r[k] for r in q],[r["group"] for r in q])
                        for k in ("edit_minus_clean","edit_minus_random")}
                    if arm=="same_y":
                        windows[t]={r["group"]:r["edit_minus_random"] for r in q}
            result[f"{cohort}/{method}/same_y_late_minus_early"]=mean_ci(
                [windows[.75][g]-windows[.25][g] for g in windows[.25]],list(windows[.25]))
    return result


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for arg in ("model","tableau","arc","out"):
        p.add_argument(arg,type=Path)
    a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    model,tok,runtime=load_runtime(a.model)
    all_groups,_=make_batches(tok,a.tableau,a.arc,"formal")
    groups=[g for g in all_groups if g[0]["family"]=="twohop"]
    costs=[]
    bank=[]
    start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (a.out/"manifest.jsonl").open("w",encoding="utf-8") as m, \
         (a.out/"grounded_candidate_scores.jsonl").open("w",encoding="utf-8") as scores:
        for g in groups:
            new=[anchor(tok,w) for w in g]
            states,traces,cost=gather(model,new)
            bank.append((new,states))
            costs.append(cost)
            for i,w in enumerate(new):
                write_row(m,w)
                write_row(scores,dict(item=w["item"],group=w["group"],cohort=w["cohort"],
                          y=w["y"],time=GRID,scores={k:v[i] for k,v in traces.items()}))
            print("grounded",g[0]["group"],flush=True)
    # Reuse the unchanged joint probe fitter. Its tableau arm remains native-only.
    for g in all_groups:
        if (g[0]["family"],g[0]["cohort"])==("tableau","development"):
            n=NativeBatch(model,g,trace=True)
            h=torch.stack([n.native[t][n.rows,n.lengths-1].half() for t in GRID])
            bank.append((g,{"native":(list(GRID),h.cpu().numpy())}))
            costs.append(n.counts)
    fit_and_measure(bank,a.out)
    with (a.out/"semantic_predictions.jsonl").open() as f:
        semantic=semantic_summary([json.loads(line) for line in f])
    with (a.out/"matched_dose_causal.jsonl").open("w",encoding="utf-8") as f:
        for g in groups:
            costs.append(repeat_causal(model,g,f))
            print("matched dose",g[0]["group"],flush=True)
    with (a.out/"matched_dose_causal.jsonl").open() as f:
        causal=causal_aggregate([json.loads(line) for line in f])
    torch.cuda.synchronize()
    dump(a.out/"SUMMARY.json",dict(semantic=semantic,causal=causal,runtime=runtime,
        synchronized_wall_s=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated(),
        counters={k:sum(x[k] for x in costs) for k in costs[0]}))


if __name__=="__main__":
    main()
