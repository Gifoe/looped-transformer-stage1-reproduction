"""Restore native attention, Value, Query-Key or MLP paths and replay at one boundary."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import minimal
from archive_audit import dump, read, sha
from control_transport import calibrate, grouped_sum, seq_rms, sequence_cosine, rows_write
from e2_signed import load_control, partition, run, score, write
from forward_control import scope_indices
from functional_operator import batches, selected_rows
from native_to_rank1 import gradient_utility
from state_horizon import score_arrays


PATHS = {
    "attention": ("attention",),
    "value": ("v",),
    "query_key": ("q","k"),
    "mlp": ("mlp",),
    "attention_mlp": ("attention","mlp"),
}


def full_vector(values, valid, base):
    return torch.zeros_like(base,dtype=values.dtype).index_put(valid,values)


def controls(net, base, delta, packed, items, mean_sign):
    """Use independent Train sign or exchange relative scalar amplitudes across questions."""
    row,pos=packed[1:3]
    source=base[row,pos].float()
    nominal=net.delta(source)
    u=net.out.weight[:,0]
    u=u/u.square().mean().sqrt().clamp_min(1e-12)
    scale=source.square().mean(-1).sqrt().clamp_min(1e-6)
    signed=(nominal*u).mean(-1)/scale
    fixed=mean_sign*signed.abs()
    roles, qids = [], []
    for _,n,q,_ in items:
        for j in range(n):
            roles.append(min(j,2))
            qids.append(q)
    donors=list(range(len(qids)))
    for role in range(3):
        same=[i for i in range(len(qids)) if roles[i]==role]
        for j,i in enumerate(same):
            different=next((same[(j+t)%len(same)] for t in range(1,len(same)) if qids[same[(j+t)%len(same)]]!=qids[i]),i)
            donors[i]=different
    offset=torch.tensor(donors,device=source.device,dtype=torch.long)
    exchanged=signed[offset]
    out={}
    for name,d in (("mean_sign",fixed),("question_shuffle",exchanged)):
        desired=(d*scale)[:,None]*u
        actual=(source+desired).to(base.dtype).float()-source
        out[name]=full_vector(actual,(row,pos),base)
    return out,float((torch.sign(signed)!=torch.sign(fixed)).float().mean()),float((offset!=torch.arange(len(offset),device=offset.device)).float().mean())


def batch(archive,model,net,packed,items,info,k,end,mean_sign,keep,check_exact,records):
    receive=k+1
    selection=scope_indices(packed,"all_valid")
    owner,position=selection
    n=len(items)
    lengths=torch.bincount(owner,minlength=n)
    branch_names=set(x for name in keep for x in PATHS[name])
    with torch.enable_grad():
        native, raw, leaf, branch=run(model,packed,end,capture=(k,receive),leaf_anchor=receive,
                                     branches={"loop":receive,"paths":branch_names})
        utility,_=gradient_utility(model,native,packed,items,info)
        gradient,=torch.autograd.grad(utility.sum(),leaf)
    with torch.no_grad():
        a,h=raw[k].detach(),raw[receive].detach()
        baseline=h[selection].float()
        early_write=write(net,a,packed)
        late=write(net,h,packed)
        first, state, _, _=run(model,packed,end,capture=(receive,),edits={k:lambda _:early_write})
        displacement=state[receive][selection].float()-baseline
        estimates={"early":displacement,"late":late[selection]}
        p=(grouped_sum((displacement*estimates["late"]).sum(-1),owner,n)/
           grouped_sum(estimates["late"].square().sum(-1),owner,n).clamp_min(1e-20))
        estimates["parallel"]=p[owner,None]*estimates["late"]
        estimates["orthogonal"]=displacement-estimates["parallel"]
        for name in keep:
            restore={key:val for key,val in branch.items() if key[1] in PATHS[name]}
            if len(restore)!=24*len(PATHS[name]):
                raise RuntimeError(f"Route {name} was not captured for all native layers")
            _, changed, _, _=run(model,packed,receive,capture=(receive,),edits={k:lambda _:early_write},
                            branches={"loop":receive,"paths":PATHS[name],"restore":restore})
            estimates[f"restore_{name}"]=changed[receive][selection].float()-baseline
        mix, flip_rate, donors=controls(net,h,late,packed,items,mean_sign)
        estimates.update({name:value[selection] for name,value in mix.items()})
        if check_exact:
            exact=state[receive][selection].double()-h[selection].double()
            replay,_,_,_=run(model,packed,end,edits={receive:lambda _:full_vector(exact,selection,h)})
            if not torch.equal(first,replay):
                raise RuntimeError("Common receiving-state replay failed bitwise validation")
        target=torch.minimum(seq_rms(estimates["early"],owner,lengths),seq_rms(estimates["late"],owner,lengths))
        score(archive,model,packed,items,info,native.detach(),"native",records)
        for label,disp in estimates.items():
            update,_,support=calibrate(baseline,disp,owner,lengths,h.dtype,target)
            hidden, acted, _, _=run(model,packed,end,capture=(receive,),
                                    edits={receive:lambda _:full_vector(update,selection,h)})
            measured=acted[receive][selection].float()-baseline
            actual=seq_rms(measured,owner,lengths)
            error=(actual-target).abs()/target.clamp_min(1e-10)
            dot=grouped_sum((gradient.detach()[selection]*measured).sum(-1),owner,n)
            cos=sequence_cosine(measured,estimates["late"],owner,n)
            score(archive,model,packed,items,info,hidden,label,records)
            for q,indices in partition(items).items():
                rec=records[q].setdefault("diagnostics",{})
                rec[label]=dict(actual_mean_rms=float(actual[indices].mean()),task_dot=float(dot[indices].sum()),
                                late_cosine=float(cos[indices].mean()),all_rms_matched=bool(((error<=.01)&support)[indices].all()),
                                maximum_relative_rms_error=float(error[indices].max()))
        return dict(native_early_replay_exact=check_exact,constant_sign_flip_fraction=flip_rate,
                    shuffled_cross_question_fraction=donors)


def paired(rows,seed):
    arms=tuple(rows[0]["scores"])
    baseline="early"
    scores={name:[score_arrays(r["scores"][name],r["gold"],r["characters"]) for r in rows] for name in arms}
    rng=np.random.default_rng(seed)
    draws=rng.integers(len(rows),size=(5000,len(rows)))
    def summary(d):
        return dict(mean=float(d.mean()),ci95=np.quantile(d[draws].mean(1),[.025,.975]).tolist())
    values={}
    for arm in arms:
        t=np.array([x["utility"] for x in scores[arm]])
        b=np.array([x["utility"] for x in scores[baseline]])
        native=np.array([x["utility"] for x in scores["native"]])
        result=dict(vs_early=summary(t-b),vs_native=summary(t-native),
                    acc_norm=int(sum(x["acc_norm"] for x in scores[arm])))
        if arm!="native":
            ok=np.array([r["diagnostics"][arm]["all_rms_matched"] and r["diagnostics"]["early"]["all_rms_matched"]
                         for r in rows])
            gd=np.array([r["diagnostics"][arm]["task_dot"] for r in rows])
            result.update(matched_n=int(ok.sum()),match_only_effect=float((t-b)[ok].mean()) if ok.any() else None,
                          mean_alignment=float(gd.mean()),
                          mean_late_cos=float(np.mean([r["diagnostics"][arm]["late_cosine"] for r in rows])))
            pseudo=[]
            for r in rows:
                wrong=(r["gold"]+1)%len(r["labels"])
                a=score_arrays(r["scores"][arm],wrong,r["characters"])["utility"]
                z=score_arrays(r["scores"]["native"],wrong,r["characters"])["utility"]
                pseudo.append(a-z)
            result["wrong_label_control_vs_native"]=summary(np.array(pseudo))
        values[arm]=result
    for treatment in ("mean_sign","question_shuffle"):
        t=np.array([x["utility"] for x in scores[treatment]])
        l=np.array([x["utility"] for x in scores["late"]])
        values[treatment]["vs_late"]=summary(t-l)
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("archive","previous","model","rank1","data","transport","out"):
        p.add_argument(f"--{name}",required=True,type=Path)
    p.add_argument("--domain",choices=("arc","csqa"),required=True)
    p.add_argument("--cell",action="append",required=True)
    p.add_argument("--n",type=int,default=48)
    p.add_argument("--batch",type=int,default=6)
    p.add_argument("--paths",nargs="+",choices=PATHS,default=list(PATHS))
    p.add_argument("--forecast-lock",type=Path)
    a=p.parse_args()
    if "transport-confirm-v1" in a.data.as_posix():
        raise ValueError("Specify a folded dataset file")
    if a.data.parent.name=="data" and a.data.parent.parent.name.startswith("ouro-dynamical"):
        if a.forecast_lock is None: p.error("Fresh confirmation requires the sealed forecast")
        locked=read(a.forecast_lock)
        if locked["data_sha256"][a.domain]!=sha(a.data) or sha(a.rank1) not in locked["weights_sha256"]:
            raise ValueError("Fresh confirmation bytes or controller differ from the forecast")
    a.out.mkdir(parents=True,exist_ok=True)
    archive,model,tok,rt=minimal.runtime(a.archive,a.previous,a.model)
    net=load_control(model,a.rank1)
    rows=selected_rows(a.data,a.n,261123359)
    info,work=batches(archive,tok,rows,a.batch)
    manifest=dict(runtime=rt,controller_sha256=sha(a.rank1),data_sha256=sha(a.data),ids=[r["id"] for r in rows],
                  code_sha256=sha(Path(__file__)),paths=a.paths,cells=a.cell,n=a.n,
                  forecast_sha256=sha(a.forecast_lock) if a.forecast_lock else None)
    location=a.out/"MANIFEST.json"
    if location.exists() and read(location)!=json.loads(json.dumps(manifest)):
        raise ValueError("E3 historical lock changed")
    dump(location,manifest)
    for spec in a.cell:
        k,end=map(int,spec.split(":"))
        if not 1<=k<end:
            raise ValueError("Specify k:K with a receiving boundary")
        key=f"k{k}_K{end}"
        detail=a.out/f"{key}.json"
        raw=detail.with_suffix(".jsonl")
        if detail.is_file():
            if sha(raw)!=read(detail)["rows_sha256"]: raise RuntimeError("E3 receipt mismatch")
            continue
        path=a.transport/"results/moments"/a.domain/f"h{min(k+1,3)}_m1.pt"
        state=torch.load(path,weights_only=True,map_location="cpu")["mean_state"].to(net.encode.weight.device)
        sign=1 if float(net.encode(state).squeeze())>=0 else -1
        start=time.monotonic()
        records={i:dict(row,scores={}) for i,row in enumerate(info)}
        diag=[]
        for j,items in enumerate(work):
            packed=archive.pack(items)
            diag.append(batch(archive,model,net,packed,items,info,k,end,sign,a.paths,j==0,records))
        recorded=list(records.values())
        rows_write(raw,recorded)
        report=dict(results=paired(recorded,26112331+k+end),diagnostics=diag,seconds=time.monotonic()-start,
                    rows_sha256=sha(raw),meanfield_direction_source=sha(path))
        dump(detail,report)
        print(spec,"late vs early",report["results"]["late"]["vs_early"],flush=True)


if __name__=="__main__":
    main()
