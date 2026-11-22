"""Measure Oracle/prediction alignment and remove access to the operand features."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from csc_bank import KINDS, load_bank, metadata
from csc_data import repair
from csc_learn import cap, compress, infer, load_operator
from csc_oracle import stat
from native import Forward, load, save, setup


def alignment(bank, rows, bundle, slot, depth):
    op, enc, s = bundle
    cache = [bank[r["id"], "joint", ()] for r in rows if r["repair"]]
    feature = torch.stack([x["features"][depth-1] for x in cache]).cuda()
    oracle = torch.stack([x["updates"][slot] for x in cache]).cuda()
    guessed = cap(op(compress(feature, enc)), s["limit"]) * s["gain"]
    ratio = guessed.norm(dim=-1) / oracle.norm(dim=-1).clamp_min(1e-9)
    cos = F.cosine_similarity(guessed, oracle, dim=-1)
    return dict(n=len(cache), cosine_median=float(cos.median()), norm_ratio_median=float(ratio.median()),
                relative_rmse=float((guessed-oracle).norm()/oracle.norm()),
                cosine=cos.tolist(), norm_ratio=ratio.tolist())


@torch.inference_mode()
def run(model, root):
    m, tok = setup(model)
    f = Forward(m)
    raw = load(root / "NATIVE.json")["rows"]
    bank = load_bank(root)
    slot = load(root / "ORACLE_LOCK.json")["slot"]
    cases = [dict(metadata(tok,r,k),repair=k=="joint" and repair(r))
             for r in raw if (r["form"],r["split"]) == ("borrow","test") for k in KINDS]
    swapped = [dict(r,roles=(r["roles"][1],r["roles"][0],*r["roles"][2:])) for r in cases]
    out = {}
    all_rows = {}
    for k in (1,2):
        ref = {(r["id"],r["kind"]):r["pred"] for r in load(root / f"LEARNED_RAW_K{k}.json")["rows"]}
        bundle = load_operator(root / f"operator_causal_K{k}.pt", "cuda")
        op, enc, s = bundle
        blank = [(u,torch.zeros_like(v),w) for u,v,w in enc[:2]] + enc[2:]
        swaps = infer(f,tok,swapped,op,enc,s["gain"],s["limit"],k,s["branch"])
        blanks = infer(f,tok,cases,op,blank,s["gain"],s["limit"],k,s["branch"])
        rows = [dict(id=r["id"],kind=r["kind"],target=r["target"],
                     pred=dict(native=r["native"][k-1],unmodified=ref[r["id"],r["kind"]]["causal"],
                               state_distill=ref[r["id"],r["kind"]]["state_distill"],swap=a,no_content=b))
                for r,a,b in zip(cases,swaps,blanks,strict=True)]
        counts = {a:sum(r["pred"][a]["number"]==r["target"] for r in rows) for a in rows[0]["pred"]}
        distill = load_operator(root / f"operator_state_distill_K{k}.pt", "cuda")
        repairs = [r for r,c in zip(rows,cases,strict=True) if c["repair"]]
        repair_counts = {a:sum(r["pred"][a]["number"]==r["target"] for r in repairs) for a in counts}
        out[f"K{k}"] = dict(patch_geometry=alignment(bank,cases,bundle,slot,k),
                            distill_geometry=alignment(bank,cases,distill,slot,k),
                            exact_repair_outputs=dict(n=len(repairs),**repair_counts),
                            free_results=dict(n=len(rows),**counts),
                            versus_unmodified={a:stat(rows,a,"unmodified") for a in ("swap","no_content")})
        all_rows[f"K{k}"] = rows
    save(root / "OPERATOR_ABLATION_RAW.json",all_rows)
    save(root / "OPERATOR_DIAGNOSTIC.json",out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model",type=Path)
    parser.add_argument("root",type=Path)
    args = parser.parse_args()
    run(args.model,args.root)
