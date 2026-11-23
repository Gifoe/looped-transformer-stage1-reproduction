"""Measure a real boundary Jacobian, its adjoint, and the finite causal contrast."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import minimal
from archive_audit import dump, read, sha
from forward_control import scope_indices
from functional_operator import batches, selected_rows
from native_to_rank1 import gradient_utility
from normalized_geometry import rms
from state_horizon import score_arrays


def load_control(model, path):
    raw = torch.load(path, weights_only=True, map_location="cpu")
    if set(raw["state"]) != {"out.weight","encode.weight"}:
        raise ValueError("Expected the archived two-factor rank-one operator")
    net = minimal.Minimal("linear",1,.45)
    net.load_state_dict(raw["state"])
    return net.to(next(model.parameters()).device).eval().requires_grad_(False)


def run(model, packed, limit, *, capture=(), leaf_anchor=None, edits=None, branches=None):
    """Native no-cache recurrence, with optional raw-state and branch hooks."""
    core = model.model
    prefix, groups, pos, _ = packed
    ts = 0
    saved = {}
    edits = {} if edits is None else edits
    branch_outputs = {}
    handles = []

    def before_norm(_, args):
        nonlocal ts, leaf
        ts += 1
        base = args[0]
        if ts == leaf_anchor:
            leaf = base.detach().float().requires_grad_(True)
            base = leaf.to(base.dtype)
        if ts in edits:
            base = (base.float() + edits[ts](base)).to(base.dtype)
        if ts in capture:
            saved[ts] = base
        if base is not args[0]:
            return (base, *args[1:])

    def branch_hook(index, label):
        def wrapped(_m, _a, output):
            if ts+1 != branches["loop"]:
                return None
            key = (index, label)
            other = branches.get("restore", {}).get(key)
            if other is not None:
                return other
            branch_outputs[key] = output.detach()
            return None
        return wrapped

    leaf = None
    handles.append(core.norm.register_forward_pre_hook(before_norm))
    if branches:
        for i, layer in enumerate(core.layers):
            mods = {"attention":layer.input_layernorm_2, "mlp":layer.post_attention_layernorm_2,
                    "v":layer.self_attn.v_proj, "q":layer.self_attn.q_proj, "k":layer.self_attn.k_proj}
            for label in branches["paths"]:
                handles.append(mods[label].register_forward_hook(branch_hook(i,label)))
    prior = core.total_ut_steps
    try:
        core.total_ut_steps = limit
        _, states, _ = core(input_ids=prefix, use_cache=False)
    finally:
        core.total_ut_steps = prior
        for h in handles:
            h.remove()
    if ts!=limit:
        raise RuntimeError("Missing native recurrent boundary")
    return states[-1][groups,pos], saved, leaf, branch_outputs


def write(net, base, packed):
    row, t = packed[1:3]
    value = base[row,t].float()
    delta = (value+net.delta(value)).to(base.dtype).float()-value
    return torch.zeros_like(base,dtype=torch.float32).index_put((row,t),delta)


def jvp_reverse(y, leaf, delta):
    cotangent = torch.zeros_like(y,dtype=torch.float32,requires_grad=True)
    vjp, = torch.autograd.grad(y,leaf,cotangent.to(y.dtype),create_graph=True,retain_graph=True)
    tangent, = torch.autograd.grad(vjp,cotangent,delta,retain_graph=True)
    return tangent.detach().float()


def score(archive, model, packed, items, info, terminal, name, rows):
    values = archive.scores(model.lm_head(terminal).float(), packed[1].cpu().numpy(), packed[-1],len(items))
    for (_, _, q, a), value in zip(items,values,strict=True):
        row = rows[q]
        row["scores"].setdefault(name,[None]*len(info[q]["labels"]))[a] = float(value)


def partition(items):
    groups = {}
    for row, (_, _, q, _) in enumerate(items):
        groups.setdefault(q,[]).append(row)
    return groups


def pair_rows(items, valid, g1, g2, d1, d2, tangent, central, transported):
    for q, subset in partition(items).items():
        keep = torch.isin(valid[0],valid[0].new_tensor(subset))
        idx = valid[0][keep], valid[1][keep]
        v = tangent[idx]
        d = d2[idx]
        f = transported[idx]
        b = central[idx]
        grad = g2[idx]
        local = float((g1[idx]*d1[idx]).sum())
        passed = float((grad*v).sum())
        late = float((grad*d).sum())
        defect = v-d
        axis_gain = (v*d).sum()/d.square().sum().clamp_min(1e-20)
        off_axis = v - axis_gain*d
        eps = max(float(grad.norm()*v.norm()),1e-12)
        yield q,dict(
            a_early=local,a_transported=passed,a_late=late,linear_defect=passed-late,
            adjoint_discrepancy=passed-local,relative_adjoint_discrepancy=abs(passed-local)/eps,
            functional_transport_ratio=passed/late if abs(late)>1e-8 else None,
            jvp_vs_central_relative=float((v-b).norm()/v.norm().clamp_min(1e-8)),
            jvp_vs_central_cosine=float(torch.nn.functional.cosine_similarity(v.flatten(),b.flatten(),dim=0)),
            jvp_late_cosine=float(torch.nn.functional.cosine_similarity(v.flatten(),d.flatten(),dim=0)),
            jvp_finite_cosine=float(torch.nn.functional.cosine_similarity(v.flatten(),f.flatten(),dim=0)),
            tangent_rms=float(v.square().mean().sqrt()),late_rms=float(d.square().mean().sqrt()),
            relative_defect=float(defect.norm()/d.norm().clamp_min(1e-8)),
            parallel_coefficient=float(axis_gain),parallel_task=float(axis_gain)*late,
            residual_task=float((grad*off_axis).sum()),
            finite_gradient_defect=float((grad*(f-d)).sum()),central_gradient_defect=float((grad*(b-d)).sum()),
        )


def intervene(model, packed, k, end, delta, dose, capture):
    return run(model,packed,end,capture=capture,edits={k:lambda _:dose*delta})


def run_cell(archive,model,net,info,work,k,end):
    rows = {j:dict(row,scores={}) for j,row in enumerate(info)}
    receiving = k+1
    small = .1
    for items in work:
        packed=archive.pack(items)
        valid=scope_indices(packed,"all_valid")
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            terminal,graph,leaf,_ = run(model,packed,end,capture=(k,receiving),leaf_anchor=k)
            q, _=gradient_utility(model,terminal,packed,items,info)
            y=graph[receiving]
            g1,g2=torch.autograd.grad(q.sum(),(leaf,y),retain_graph=True)
            d1=write(net,graph[k].detach(),packed)
            d2=write(net,y.detach(),packed)
            tangent=jvp_reverse(y,leaf,d1)
        with torch.no_grad():
            with sdpa_kernel(SDPBackend.MATH):
                _, plus,_,_=intervene(model,packed,k,receiving,d1,small,(receiving,))
                _, minus,_,_=intervene(model,packed,k,receiving,d1,-small,(receiving,))
            central=(plus[receiving].float()-minus[receiving].float())/(2*small)
            ordinary, states,_,_=run(model,packed,end,capture=(k,receiving))
            math_score=terminal.detach()
            native_d1=write(net,states[k],packed)
            native_d2=write(net,states[receiving],packed)
            early, actual,_,_=intervene(model,packed,k,end,native_d1,1.,(receiving,))
            late, _,_,_=intervene(model,packed,receiving,end,native_d2,1.,())
            early_small, _,_,_=intervene(model,packed,k,end,native_d1,small,())
            late_small, _,_,_=intervene(model,packed,receiving,end,native_d2,small,())
            early_minus, _,_,_=intervene(model,packed,k,end,native_d1,-small,())
            late_minus, _,_,_=intervene(model,packed,receiving,end,native_d2,-small,())
            for name, hidden in (("native",ordinary),("math_native",math_score),("early",early),("late",late),
                                 ("early_small",early_small),("late_small",late_small),
                                 ("early_minus",early_minus),("late_minus",late_minus)):
                score(archive,model,packed,items,info,hidden,name,rows)
            actual_transport=actual[receiving].float()-states[receiving].float()
            for qid, diagnostics in pair_rows(items,valid,g1.detach().float(),g2.detach().float(),d1,d2,tangent,
                                              central,actual_transport):
                rows[qid]["diagnostics"]=diagnostics
        del terminal, graph, y, leaf, q
    return list(rows.values())


def paired(rows,k,end):
    vals = {}
    primal=np.array([r["diagnostics"]["linear_defect"] for r in rows])
    for name in rows[0]["scores"]:
        vals[name]=np.array([[v["utility"], v["acc_norm"]] for row in rows
                      for v in [score_arrays(row["scores"][name],row["gold"],row["characters"])]])
    rng=np.random.default_rng(482026+k+end)
    draw=rng.integers(len(rows),size=(4000,len(rows)))

    def summarize(a):
        a=np.asarray(a,dtype=float)
        return dict(mean=float(a.mean()),ci95=np.quantile(a[draw].mean(1),[.025,.975]).tolist())

    small=((vals["early_small"]-vals["late_small"])-(vals["early_minus"]-vals["late_minus"]))[:,0]/.2
    finite=(vals["early"]-vals["late"])[:,0]
    d=rows[0]["diagnostics"]
    effects={name:summarize(np.array([r["diagnostics"][name] for r in rows
                                      if r["diagnostics"][name] is not None]))
             for name in d if all(row["diagnostics"][name] is not None for row in rows)}
    return dict(transport=effects,finite_early_minus_late=summarize(finite),small_signed_defect=summarize(small),
                finite_nonlinear_error=summarize(finite-primal),small_signed_error=summarize(small-primal),
                math_native_score_delta=summarize((vals["math_native"]-vals["native"])[:,0]),
                endpoint={name:dict(acc_norm=int(value[:,1].sum()),
                                    rescued=int(((value[:,1]-vals["native"][:,1])>0).sum()),
                                    damaged=int(((value[:,1]-vals["native"][:,1])<0).sum()),
                                    delta_utility=summarize((value-vals["native"])[:,0]))
                          for name,value in vals.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("archive","previous","model","rank1","data","out"):
        p.add_argument(f"--{name}",required=True,type=Path)
    p.add_argument("--cell",action="append",required=True)
    p.add_argument("--n",type=int,default=32)
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--seed",type=int,default=261123159)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    archive,model,tok,rt=minimal.runtime(args.archive,args.previous,args.model)
    net=load_control(model,args.rank1)
    rows=selected_rows(args.data,args.n,args.seed)
    info,work=batches(archive,tok,rows,args.batch)
    manifest=dict(runtime=rt,ids=[r["id"] for r in rows],data_sha256=sha(args.data),controller_sha256=sha(args.rank1),
                  code_sha256=sha(Path(__file__)),paired_cells=args.cell,n=args.n,batch=args.batch,small=.1)
    path=args.out/"MANIFEST.json"
    manifest_json=json.loads(json.dumps(manifest))
    if path.exists() and read(path)!=manifest_json:
        raise ValueError("Numerical run lock changed")
    dump(path,manifest)
    for spec in args.cell:
        k,end=map(int,spec.split(":"))
        if not 1<=k<end:
            raise ValueError("Expected k:K with space for a full transport loop")
        detail=args.out/f"k{k}_K{end}.json"
        raw=detail.with_suffix(".jsonl")
        if detail.is_file():
            if read(detail)["rows_sha256"]!=sha(raw):
                raise RuntimeError("E2 raw rows no longer match their receipt")
            continue
        start=time.monotonic()
        prediction=run_cell(archive,model,net,info,work,k,end)
        temp=raw.with_suffix(".part")
        archive.write_predictions(temp,prediction)
        temp.replace(raw)
        summary=paired(prediction,k,end)
        dump(detail,dict(**summary,rows_sha256=sha(raw),seconds=time.monotonic()-start))
        print(spec,summary["transport"]["linear_defect"],summary["finite_early_minus_late"],flush=True)


if __name__=="__main__":
    main()
