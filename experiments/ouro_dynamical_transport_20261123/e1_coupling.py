"""Rebuild historical moments and contrast paired question-block nulls."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import minimal
from archive_audit import dump, read, sha, weights
from forward_control import forward_control
from functional_operator import batches, selected_rows
from native_to_rank1 import gradient_utility


CELLS = ((1,0), (1,1), (1,3), (2,1), (3,1))


def collect(model, archive, info, work, k, m, axes):
    device = next(model.parameters()).device
    d = model.config.hidden_size
    matrix = torch.zeros(d,d,device=device)
    per_question = defaultdict(list)
    u = torch.stack([x[0] for x in axes]).float().to(device)
    w = torch.stack([x[1] for x in axes]).float().to(device)
    for items in work:
        packed = archive.pack(items)
        with torch.enable_grad():
            terminal, trace = forward_control(model, packed, k+m, anchor_loop=k, leaf=True)
            q, _ = gradient_utility(model, terminal, packed, items, info)
            grad, = torch.autograd.grad(q.sum(), trace["leaf"])
        h, g = trace["anchor"].float(), grad.detach().float()
        matrix.addmm_(g.T,h)
        l, r = (g @ u.T).cpu().numpy(), (h @ w.T).cpu().numpy()
        left = 0
        for _, size, qid, _ in items:
            for t in range(size):
                # Candidate-relative position deliberately ignores the answer label.
                per_question[qid].append((min(t,2), l[left], r[left]))
                left += 1
    return matrix.cpu()/len(info), per_question


def permute(data, seed):
    groups = sorted(data)
    rng = np.random.default_rng(seed)
    order = np.asarray([rng.permutation(len(groups)) for _ in range(512)])
    # Arrays [role, question, axis] use gradient sums and state means per role.
    g_example = data[groups[0]][0][1]
    shp = (3,len(groups),len(g_example))
    g, h = np.zeros(shp),np.zeros(shp)
    present = np.zeros(shp[:2],dtype=bool)
    joint = np.zeros(shp[-1])
    single_g = np.zeros(shp[1:])
    single_h = np.zeros(shp[1:])
    n = len(groups)
    for j,group in enumerate(groups):
        full = data[group]
        single_g[j] = np.sum([x[1] for x in full],axis=0)
        single_h[j] = np.mean([x[2] for x in full],axis=0)
        joint += np.sum([x[1]*x[2] for x in full],axis=0)/n
        for role in range(3):
            pairs = [(a,b) for p,a,b in full if p==role]
            if not pairs:
                continue
            present[role,j] = True
            g[role,j] = np.sum([p[0] for p in pairs],axis=0)
            h[role,j] = np.mean([p[1] for p in pairs],axis=0)
    question = (single_g*single_h).sum(0)/n
    unrestricted = (single_g[None]*single_h[order]).sum(1)/n
    role_observed = (g*h).sum((0,1))/n
    role_null = np.zeros_like(unrestricted)
    for role in range(3):
        g_role, h_role = g[role,present[role]], h[role,present[role]]
        for j in range(len(role_null)):
            role_null[j] += (g_role*h_role[rng.permutation(len(h_role))]).sum(0)/n
    return joint,question,unrestricted,role_observed,role_null


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("archive","previous","model","transport","t1","out"):
        p.add_argument(f"--{name}",required=True,type=Path)
    a=p.parse_args()
    results=a.transport/"results"
    archive,model,tok,runtime=minimal.runtime(a.archive,a.previous,a.model)
    fixed={"t1":weights(a.t1),"qasc":weights(results/"HOLDOUT/qasc/source/s261123137/k234/controller.pt"),
           "arc_k2":weights(results/"HOLDOUT/arc/depth/s261123137/k2/controller.pt")}
    a.out.mkdir(parents=True,exist_ok=True)
    report={}
    for domain in ("arc","csqa","qasc"):
        d=results/"moments"/domain
        lock=read(d/"INPUT_LOCK.json")
        file=a.transport/("data/optional" if domain=="qasc" else "data")/f"{domain}_train.jsonl"
        if sha(file)!=lock["input_sha256"]:
            raise RuntimeError("E1 input does not match published corpus")
        rows=selected_rows(file,lock["n"],lock["seed"])
        if [r["id"] for r in rows]!=lock["ids"]:
            raise RuntimeError("Archive subset identity differs")
        info,work=batches(archive,tok,rows,lock["batch"])
        report[domain]={}
        for k,m in CELLS:
            key=f"h{k}_m{m}"
            name=d/f"{key}.pt"
            expected=torch.load(name,weights_only=True,map_location="cpu")
            probes={"top":(expected["u"][:,0],expected["w"][:,0]),**fixed}
            matrix,data=collect(model,archive,info,work,k,m,list(probes.values()))
            relative=float((matrix-expected["matrix"]).norm()/expected["matrix"].norm())
            if relative>5e-4:
                raise RuntimeError(f"Historical gradient/matrix did not rebuild: {domain}/{key}: {relative}")
            joint,qs,null,roles,role_null=permute(data,2611230+k*10+m)
            options={}
            for j,arm in enumerate(probes):
                dist=role_null[:,j]
                role=roles[j]
                options[arm]=dict(joint=float(joint[j]),question_mean=float(qs[j]),
                                  within_question=float(joint[j]-qs[j]),role_observed=float(role),
                                  within_role=float(joint[j]-role),role_permutation_null=float(dist.mean()),
                                  matched_role_ci95=np.quantile(dist,[.025,.975]).tolist(),
                                  matched_role_rank=float((np.count_nonzero(dist<=role)+1)/(len(dist)+1)),
                                  unstratified_ci95=np.quantile(null[:,j],[.025,.975]).tolist(),
                                  role_permutation_samples=dist.tolist())
            receipt=dict(n=len(info),matrix_sha256=sha(name),matrix_relative_error=relative,projections=options)
            report[domain][key]=receipt
            dump(a.out/f"{domain}_{key}.json",receipt)
            print(domain,key,relative,options["top"]["matched_role_rank"],flush=True)
    dump(a.out/"E1.json",dict(conditions=report,runtime=runtime,
                             null="Question role means permuted across questions with position role present; "
                                  "role and total matrix estimates are distinct. No higher-mode claims from projection tests."))


if __name__=="__main__":
    main()
