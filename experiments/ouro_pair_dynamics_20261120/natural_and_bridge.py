"""Separate natural ARC transfer and a lexical bridge diagnostic."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from causal_worlds import masked, sha
from pair_fields import PairOps, pair, timed, load_runtime
from dynamics_embedding import read_json, write_json
from predictor import batches, trajectory


def arc(model,m,root,stage):
    merged=[]
    clocks=[]
    source=root/"ARC_PREDICTIONS.json"
    receipt=root/"ARC_PREDICTION_SHA.json"
    if stage=="arc_truth":
        assert sha(source)==read_json(receipt)["sha256"]
    readout_states={}
    for pack in batches(m["arc_confirmation"],8):
        op=PairOps(model,[masked(w) for w in pack])
        items=[dict(item=w["item"],methods={}) for w in pack]
        names=("native",) if stage=="arc_truth" else ("full","taylor","prefix_anderson","base")
        for name in names:
            def output(k,h,readout):
                if not readout:
                    return
                vals=op.scores(h).tolist()
                lengths=[float(a) for a in h.float().square().sum(dim=-1).mul(op.valid).sum(-1).sqrt()]
                for q,v,z in zip(items,vals,lengths,strict=True):
                    q["methods"].setdefault(name,{})[k]=dict(scores=v,state_l2=z)
                readout_states.setdefault(f"{name}_{k.replace('/','_')}",[]).append(
                    h[op.rows,op.positions].float().cpu().numpy())
            with timed(op,"arc/"+stage+"/"+name,clocks):
                trajectory(op,op.start,name,3,2,record=output)
        merged.extend(items)
    np.savez_compressed(root/f"{stage.upper()}_READOUT.npz",**{k:np.concatenate(v) for k,v in readout_states.items()})
    if stage=="arc_truth":
        write_json(root/"ARC_NATIVE.json",dict(rows=merged,timing=clocks,prediction_sha256=sha(source)))
    else:
        write_json(source,dict(rows=merged,timing=clocks))
        write_json(receipt,dict(sha256=sha(source)))


def read_bridge(op,h,positions,bridge_ids):
    v=h[op.rows,positions].to(op.dtype)
    return op.model.lm_head(v).float().gather(-1,bridge_ids)


def bridge(model,tok,m,root):
    raw=[]
    times=[]
    omitted=[]
    for phase in ("development","validation","confirmation"):
        groups=m["groups"][phase]["twohop"]
        keep=[]
        for g in groups:
            options=[tok.encode(" "+v,add_special_tokens=False) for v in g["worlds"][0]["within"][1]]
            if any(len(t)!=1 for t in options) or len({tuple(t) for t in options})!=3:
                omitted.append(g["group"])
            else:
                keep.append(g)
        groups=keep
        for pack in batches(groups,6):
            recs=[g["worlds"][0] for g in pack]
            op=PairOps(model,[masked(w) for w in recs])
            positions=[max(w["roles"]["bridge"]) for w in recs]
            positions=torch.tensor(positions,device="cuda")
            options=[[tok.encode(" "+v,add_special_tokens=False) for v in w["within"][1]] for w in recs]
            assert all(all(len(t)==1 for t in q) for q in options)
            bridge_ids=torch.tensor([[t[0] for t in q] for q in options],device="cuda")
            h=op.start
            items=[dict(group=g["group"],phase=phase,bridge_label=g["worlds"][0]["b"],loops={}) for g in pack]
            for loop in (1,2,3,4):
                before=read_bridge(op,h,positions,bridge_ids).tolist()
                after=read_bridge(op,h,op.positions,bridge_ids).tolist()
                for q,u,v in zip(items,before,after,strict=True):
                    q["loops"][loop]=dict(pre_D=u,answer_pos=v)
                if loop!=4:
                    with timed(op,phase+"/bridge/K"+str(loop),times):
                        h=op.native_suffix(h)
            raw.extend(items)
    swap=[]
    for pack in batches(m["groups"]["validation"]["twohop"][:16],4):
        tokens=[masked(w) for g in pack for w in (g["worlds"][0],g["extra"]["same_bridge_other_answer"],
                                                 g["extra"]["same_answer_other_bridge"])]
        op=PairOps(model,tokens)
        base=op.start
        for roi in ("pre_D","readout"):
            h=base[::3].repeat_interleave(3,dim=0)
            target=[]
            for i,g in enumerate(pack):
                rec=g["worlds"][0]
                read=rec["read"] if roi=="readout" else max(rec["roles"]["bridge"])
                h[3*i+1,read]=base[3*i+1,read]
                h[3*i+2,read]=base[3*i+2,read]
                donor=g["extra"]["same_answer_other_bridge"]
                all_b,all_y=rec["within"][1:3]
                expected=rec["choices"].index(all_y[(donor["b"]+rec["r"])%3])
                target.append([rec["y"],rec["y"],expected])
            data=[]
            for k in (2,3,4):
                with timed(op,"bridge_swap/"+roi+"/"+str(k),times):
                    h=op.native_suffix(h)
                data.append(op.scores(h).tolist())
            for i,g in enumerate(pack):
                swap.append(dict(group=g["group"],roi=roi,targets=target[i],
                                 scores=[x[3*i:3*i+3] for x in data]))
    write_json(root/"BRIDGE.json",dict(rows=raw,swaps=swap,timing=times,omitted_non_single_token=omitted,
                                     probe="frozen lexical head, 3 actual cabinet tokens"))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("stage",choices=("arc_predict","arc_truth","bridge"))
    a=p.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model,tok,_=load_runtime(a.model)
    model.requires_grad_(False).float()
    m=read_json(a.out/"MANIFEST.json")
    with sdpa_kernel(SDPBackend.MATH):
        if a.stage=="bridge":
            bridge(model,tok,m,a.out)
        else:
            arc(model,m,a.out,a.stage)


if __name__=="__main__":
    main()
