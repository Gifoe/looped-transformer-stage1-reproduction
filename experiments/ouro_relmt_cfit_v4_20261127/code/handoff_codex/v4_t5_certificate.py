"""Measure the V4 three-objective capacity certificate and preserve checked witnesses."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from capacity import certify
from capacity_bounds_cli import multi_start
from v4_common import (EXT_DEV_DIR, UNLABELED_FIT, extract_q_opts_gold,
    forward_with_hook_all_tokens, load_base_runtime, load_jsonl, save_json, sha256_file)


V3=Path("/root/ouro-v3-controllability-20261126")
GROUPS=("arc_cloze","mmlu_letters","wiki_token_nll")


def save_pt(path, value):
    stage=path.with_suffix(".partial")
    torch.save(value,stage)
    stage.replace(path)


class FullProbe:
    def __init__(self, model):
        self.matrix=torch.zeros((2048,2048),device=model.device,requires_grad=True)

    def delta(self, hidden):
        return hidden.float() @ self.matrix.T


def collect(model, tokenizer, source, letters):
    probe=FullProbe(model)
    accum=torch.zeros_like(probe.matrix,dtype=torch.float64)
    letters_ids=[tokenizer.encode(" "+chr(65+i),add_special_tokens=False) for i in range(4)] if letters else []
    if letters and any(len(item)!=1 for item in letters_ids):
        raise ValueError("The locked letter candidates must be exactly one token")
    for index,row in enumerate(source):
        if letters:
            question, options, gold=extract_q_opts_gold(row)
            lines="\n".join(f"{chr(65+i)}. {text}" for i,text in enumerate(options))
            topic=row.get("subject","").replace("_"," ")
            text=f"The following are multiple choice questions (with answers) about {topic}.\n\n{question}\n{lines}\nAnswer:"
            tokens=tokenizer.encode(text,add_special_tokens=False)[-1024:]
        else:
            tokens=tokenizer.encode(row["text"],add_special_tokens=False)[:192]
        ids=torch.tensor(tokens,device=model.device).unsqueeze(0)
        with torch.enable_grad():
            output=forward_with_hook_all_tokens(model,ids,torch.ones_like(ids),4,probe)
            if letters:
                logits=model.lm_head(output[0,-1]).float()[[item[0] for item in letters_ids]]
                loss=-F.cross_entropy(logits[None,:],torch.tensor([gold],device=model.device))
            else:
                logits=model.lm_head(output[0,:-1]).float()
                loss=-F.cross_entropy(logits,ids[0,1:])
            grad,=torch.autograd.grad(loss,probe.matrix)
        accum.add_(grad.double())
        if (index+1)%32==0 or index+1==len(source):
            print("T5_CAPACITY_MOMENT",letters,index+1,len(source),flush=True)
    return accum.cpu()/len(source)


def run(root, phase):
    out=root/"exp_t5_mixing"
    out.mkdir(exist_ok=True)
    result=out/"CAPACITY_CERTIFICATE.json"
    if result.exists():
        return
    env_file=root/"SIGMA_H.pt"
    arc_plan=json.loads((V3/"stage4_d3/FORECAST_INPUTS.json").read_text())
    path=out/"T5_THREE_OBJECTIVE_MOMENTS.pt"
    if not path.exists():
        if phase=="solve":
            raise ValueError("Collect the three gradient matrices before CPU certification")
        sigma=torch.load(env_file,map_location="cpu",weights_only=False)["sigma_unlab"].double()
        eigen,axes=torch.linalg.eigh((sigma+sigma.T)/2)
        sf=(axes*(eigen.clamp_min(0)+float(eigen.mean())*.05).rsqrt())@axes.T
        arc=torch.zeros_like(sigma)
        n=0
        retained={}
        for address in arc_plan["arc_challenge"]["shards"]:
            p=Path(address)
            ck=torch.load(p,map_location="cpu",weights_only=False)
            arc+=ck["sums"]["ordinary"].double()
            n+=ck["processed"]
            retained[str(p)]=sha256_file(p)
        _,_,model,tokenizer=load_base_runtime()
        mmlu_path=EXT_DEV_DIR/"mmlu_validation.jsonl"
        mmlu=collect(model,tokenizer,load_jsonl(mmlu_path)[:256],True)
        prose=collect(model,tokenizer,load_jsonl(UNLABELED_FIT)[:64],False)
        matrices=torch.stack([value@sf for value in (arc/n,mmlu,prose)])
        matrices=matrices/matrices.flatten(1).norm(dim=1)[:,None,None]
        save_pt(path,{"matrices":matrices,"groups":GROUPS,
            "source_sha256":{**retained,str(env_file):sha256_file(env_file),
            str(mmlu_path):sha256_file(mmlu_path),str(UNLABELED_FIT):sha256_file(UNLABELED_FIT)},
            "n":[n,256,64],"test_used":False})
        del model, tokenizer
        torch.cuda.empty_cache()
    if phase=="collect":
        return
    received=torch.load(path,map_location="cpu",weights_only=False)
    matrices=received["matrices"].double()
    print("T5_CAPACITY_CERTIFICATE_START",sha256_file(path),flush=True)
    bound=certify(matrices.numpy(),tolerance=1e-5)
    left,right=zip(*bound.atoms,strict=True)
    witness = out/"T5_CAPACITY_NUCLEAR_WITNESS.pt"
    save_pt(witness,dict(simplex=bound.simplex,atom_weights=bound.atom_weights,left=np.stack(left),right=np.stack(right)))
    lower={}
    for rank in (1,2):
        _,factors=multi_start(matrices,rank,"cpu" if phase=="solve" else "cuda",restarts=32,steps=300)
        values=factors["s"].double()
        values=values/(values.norm(dim=-1,keepdim=True).clamp_min(1.)*(1+2048*16*torch.finfo(torch.float64).eps))
        support=torch.einsum("bir,nij,bjr,br->bn",factors["u"],matrices,factors["v"],values)
        best=max(0.,float(support.min(-1).values.max())-2048**2*16*torch.finfo(torch.float64).eps)
        artifact=out/f"T5_CAPACITY_RANK{rank}_WITNESSES.pt"
        save_pt(artifact,dict(u=factors["u"],v=factors["v"],s=values,losses=support))
        lower[str(rank)]={"objective":best,"witness_sha256":sha256_file(artifact)}
    gap=bound.upper-bound.lower
    # Rank one has coincident Frobenius and nuclear norm. Rank two is certified within the same unit Frobenius ball.
    trigger=gap<=1e-5 and lower["2"]["objective"]>bound.upper+.01
    save_json(result,dict(groups=GROUPS,inputs_sha256=sha256_file(path),count=received["n"],
                lower_nuclear=bound.lower,upper_nuclear=bound.upper,dual_gap=gap,
                iterations=bound.iterations,certified_tolerance=gap<=1e-5,
                rank_feasible_lower=lower,nuclear_witness_sha256=sha256_file(witness),
                expansion_triggered=trigger,required_rank2_margin=.01,official_test_used=False))
    print("T5_CAPACITY_CLOSED",bound.upper,gap,trigger,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--phase",choices=("both","collect","solve"),default="both")
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    run(args.root,args.phase)
