"""Test a discovery-frozen intermediate direction in held-out complete prompts."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze_org import TaskData, export, group_ci, ridge, ROLES
from capture_org import capture, positions, projection
from manifest_org import TASKS, digest, dump_line, get_encoding, load_lines, write_json
from collect import load_runtime, predictions
from run import dose_matched, fast_forward

DIRECTIONS=(("r","d","x","n","y"),("d","r","y","m","x"))


def get_readout(data,k,l,j,out):
    x=data.feature(k,l,j)
    f=data.data_mask("fit")
    mid=data.labels("intermediate")
    final=data.labels("final")
    MW,cen,intercept=ridge(x[f],mid.y[f])
    FW,_,_=ridge(x[f],final.y[f])
    u,s,_=np.linalg.svd(FW,full_matrices=False)
    keep=(s>max(s[0],1e-6)*.0001)
    Fbasis=u[:,keep]
    reduced=MW-Fbasis@(Fbasis.T@MW)
    np.savez_compressed(out/"decoder.npz",mid_w=MW,final_w=FW,intermediate_residual=reduced,
                        center=cen,bias=intercept)
    write_json(out/"decoder_metadata.json",{"intermediate_vocab":getattr(mid,"vocab",None),
               "final_subspace_rank":int(keep.sum()),"fit_worlds":int(f.sum()),
               "normalization":"unit projected L2", "rule":"remove final decoder column space from mid contrast"})
    return mid,reduced


def score(model,enc,site,pos,v):
    return capture(model,enc,[],edits={site:(pos,v)},tap_layers=())


def bank(model,enc,site,pos):
    trace=capture(model,enc,pos,tap_layers={site[1]})
    where=[trace[4].index(j) for j in pos]
    return trace[0],trace[1],trace[2][site][where]


def norm(v):
    return float(v.float().norm())


def trace_norm(a,b,seq,where):
    r={}
    idx={v:i for i,v in enumerate(where)}
    for name,p in seq.items():
        order=[idx[v] for v in p]
        values=(torch.stack([b[t][order].float()-a[t][order].float() for t in a])
                .square().sum(-1).mean(-1).sqrt().tolist())
        r[name]=[values[k*25:(k+1)*25] for k in range(4)]
    return r


def eval_group(model,tok,g,mid,weights,frame,site,role):
    enc={n:get_encoding(tok,g,w) for n,w in g["worlds"].items()}
    for r,d,x,n,m in DIRECTIONS:
        receiver=enc[r]
        roles=positions(g,receiver)
        pos=roles[role]
        noise=receiver["roles"]["noise"]
        tracked=sorted(set(pos+noise))
        receipt,head,values=bank(model,receiver,site,tracked)
        ref=values[[tracked.index(p) for p in pos]]
        ref_noise=values[[tracked.index(p) for p in noise]]
        donor,_,new=bank(model,enc[d],site,pos)
        noise_v=bank(model,enc[n],site,noise)[2]
        map_only=bank(model,enc[m],site,pos)[2]
        rr,dd=g["worlds"][r],g["worlds"][d]
        inter,other=rr["semantic"]["intermediate"],dd["semantic"]["intermediate"]
        if g["task"]=="arithmetic":
            v=weights[:,0]*np.sign(int(other)-int(inter))
        else:
            v=weights[:,mid.vocab[other]]-weights[:,mid.vocab[inter]]
        axis=frame@torch.from_numpy(v.astype(np.float32,copy=False)).cuda()
        final_site=(site[0],site[1])
        delta=new.float()-ref.float()
        axis=axis/axis.norm().clamp_min(1e-10)
        along=(delta@axis)[:,None]*axis
        complement=delta-along
        dose=norm(along)
        rand=torch.randn(ref.shape,device=ref.device,generator=torch.Generator(device="cuda").manual_seed(
                    int.from_bytes(hashlib.sha256(f"{g['id']}/{r}".encode()).digest()[:8],"big")))
        replacements={
            "full":(pos,ref,new),
            "aligned":(pos,ref,(ref.float()+along).to(ref.dtype)),
            "reverse_aligned":(pos,ref,(ref.float()-along).to(ref.dtype)),
            "orthogonal":(pos,ref,dose_matched(ref,complement,dose)),
            "random":(pos,ref,dose_matched(ref,rand,dose)),
            "irrelevant":(noise,ref_noise,dose_matched(ref_noise,noise_v.float()-ref_noise.float(),dose)),
            "map_only":(pos,ref,map_only),
            "self":(pos,ref,ref),
        }
        arms={}
        dest=g["worlds"][x]["gold"]
        for key,(where,base,vec) in replacements.items():
            y,h,_,_,_=score(model,receiver,final_site,where,vec)
            if key=="self":
                assert torch.equal(h,head)
            assert torch.equal(h[:site[0]-1],head[:site[0]-1])
            actual=vec.float()-base.float()
            arms[key]={"scores":y,"pred":predictions(y),"dose":norm(actual),
                       "mean_axis_delta":float((actual@axis).mean()) if where==pos else None}
        # Prior effective source location is an operator-positive control.
        src=roles["source"]
        loc=(2,11 if g["task"]=="logic" else 5)
        source=bank(model,enc[d],loc,src)[2]
        native_source=bank(model,receiver,loc,src)[2]
        y,h,_,_,_=score(model,receiver,loc,src,source)
        arms["source_control"]={"scores":y,"pred":predictions(y),
                                "dose":norm(source.float()-native_source.float())}
        trace_pos=sorted(set(v for seq in roles.values() for v in seq))
        before=capture(model,receiver,trace_pos)
        traces={}
        for name in ("full","aligned"):
            where,_,vec=replacements[name]
            after=capture(model,receiver,trace_pos,edits={final_site:(where,vec)})
            traces[name]=trace_norm(before[2],after[2],roles,before[4])
        assert rr["gold"]==dd["gold"]!=dest and inter!=other
        yield dict(id=g["id"],split=g["split"],task=g["task"],family=g["family"],
            receive=r,donor=d,site=site,role=role,token_positions=pos,original=rr["gold"],counterfactual=dest,
            mid_receiver=inter,mid_donor=other,native=receipt,donor_native=donor,
            cf_input_native=fast_forward(model,enc[x])[0],arms=arms,trace=traces)


def stats(out):
    records=load_lines(out/"rows.jsonl")
    def rd(s,r,k):
        return s[k][r["counterfactual"]]-s[k][r["original"]]
    report=[]
    arms=("full","aligned","reverse_aligned","orthogonal","irrelevant","random","map_only","source_control")
    for part in ("all","confirm","transfer"):
        keep=[r for r in records if part=="all" or r["split"]==part]
        assert len(keep)%2==0
        for k in range(4):
            for a in arms:
                values=[rd(r["arms"][a]["scores"],r,k)-rd(r["native"],r,k) for r in keep]
                p,lo,hi=group_ci(np.array(values).reshape(-1,2).mean(1))
                report.append(dict(task=keep[0]["task"],split=part,K=k+1,arm=a,reference="native",
                                   groups=len(keep)//2,estimate=p,lower=lo,upper=hi))
            for b in ("orthogonal","random","irrelevant","reverse_aligned"):
                values=[rd(r["arms"]["aligned"]["scores"],r,k)-rd(r["arms"][b]["scores"],r,k) for r in keep]
                p,lo,hi=group_ci(np.array(values).reshape(-1,2).mean(1))
                report.append(dict(task=keep[0]["task"],split=part,K=k+1,arm="aligned",reference=b,
                                   groups=len(keep)//2,estimate=p,lower=lo,upper=hi))
    export(out/"effects.csv",report)
    def effect(r,arm):
        return rd(r["arms"][arm]["scores"],r,3)-rd(r["native"],r,3)
    ordered=sorted(records,key=lambda r:effect(r,"full"))
    qualified=[r for r in ordered if len({r["original"],predictions(r["native"])[3],
                            predictions(r["donor_native"])[3]})==1 and
                            predictions(r["cf_input_native"])[3]==r["counterfactual"]]
    flips=[r for r in qualified if r["arms"]["full"]["pred"][3]==r["counterfactual"]]
    cases={"median":ordered[len(ordered)//2],
           "naturally_correct_dirs":len(qualified),"full_patch_flips_naturally_correct":len(flips)}
    if flips:
        cases["median_correct_flip"]=flips[len(flips)//2]
    write_json(out/"cases.json",cases)


def run(study,model_dir,task):
    frozen=json.loads((study/"analysis_freeze.json").read_text())
    assert frozen["inputs_sha256"]==digest(study/"groups.jsonl")
    selected=frozen["selected_for_causality"][task]
    site=selected["k"],selected["layer"]
    role=selected["role"]
    j=ROLES.index(role)
    out=study/"causal"/task
    out.mkdir(parents=True)
    groups={g["id"]:g for g in load_lines(study/"groups.jsonl")}
    data=TaskData(study,task,groups)
    mid,w=get_readout(data,site[0]-1,site[1],j,out)
    model,tok,runtime=load_runtime(model_dir)
    write_json(out/"runtime.json",runtime)
    frame=torch.from_numpy(projection()).cuda()
    subset=[]
    for split,n in (("confirm",60),("transfer",30)):
        pool=[g for g in groups.values() if (g["task"],g["split"])==(task,split)]
        subset+=sorted(pool,key=lambda g:hashlib.sha256(("causal-org/"+g["id"]).encode()).digest())[:n]
    write_json(out/"protocol.json",{"site":site,"role":role,"dataset_id":[g["id"] for g in subset],
        "decoder_ridge_fit":"fit only","analysis_freeze_sha256":digest(study/"analysis_freeze.json"),
        "implementation_sha256":digest(Path(__file__))})
    n=0
    start=time.perf_counter()
    with torch.inference_mode(), (out/"rows.jsonl").open("x",encoding="utf-8") as f:
        for g in subset:
            for r in eval_group(model,tok,g,mid,w,frame,site,role):
                dump_line(f,r); n+=1
            if n%20==0:
                print(task,n,round(time.perf_counter()-start,1),flush=True)
                f.flush()
    torch.cuda.synchronize()
    write_json(out/"complete.json",{"groups":len(subset),"directions":n,
               "rows_sha256":digest(out/"rows.jsonl"),"gpu_phase_seconds":time.perf_counter()-start,
               "peak_allocated_bytes":torch.cuda.max_memory_allocated()})
    stats(out)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("study",type=Path)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--task",choices=TASKS,required=True)
    a=p.parse_args()
    run(a.study,a.model,a.task)
