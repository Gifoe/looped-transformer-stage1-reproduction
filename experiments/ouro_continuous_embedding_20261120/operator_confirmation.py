"""Recheck the identical operator interventions on six unseen inputs."""
import argparse
import hashlib
from pathlib import Path

import torch

from embedding_run import load_runtime, read_json, write_json
from embedding_experiments import forecast, operator_phase
from dynamics_embedding import EmbeddingOperators, rel, timed, weight_hash


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("results",type=Path)
    a=p.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    source=read_json(a.results/"MANIFEST.json")
    chosen={k:{"dev":v["test"][:2]} for k,v in source.items()}
    dst=a.results/"operator_confirmation"
    dst.mkdir(exist_ok=True,parents=True)
    script=Path(__file__).resolve().parent
    code={k:hashlib.sha256((script/k).read_bytes()).hexdigest()
          for k in ("operator_confirmation.py","embedding_experiments.py","dynamics_embedding.py")}
    write_json(dst/"FREEZE.json",dict(code=code,ids={k:[w["item"] for w in g["dev"]]
                                                       for k,g in chosen.items()}))
    model,_,_=load_runtime(a.model)
    model.requires_grad_(False)
    before=weight_hash(model,torch.bfloat16)
    operator_phase(model,chosen,dst)
    phase_checks={}
    for family,row in chosen.items():
        batch=row["dev"]
        op2=EmbeddingOperators(model,batch,2)
        op3=EmbeddingOperators(model,batch,3)
        log=[]
        with timed(op2,"same_state_different_phase/BF16_values_FP32_compute",log):
            same=[]
            for i in range(48):
                x=op3.actual[i]
                same.append(torch.equal(op2.branch(i,x),op3.branch(i,x)))
            p2,_=forecast(op2,op3.k1,"chart_anderson",None)
            p3,_=forecast(op3,op3.k1,"chart_anderson",None)
        phase_checks[family]=dict(branches_bitwise=sum(same),forecast_bitwise=torch.equal(p2,p3),
                                  relative_endpoint_difference=rel(op3,p2,p3).tolist(),timing=log)
    write_json(dst/"PHASE_CONTEXT.json",phase_checks)
    after=weight_hash(model,torch.bfloat16)
    assert before==after
    write_json(dst/"WEIGHT_AUDIT.json",dict(unchanged=True,bf16_sha256=after))


if __name__=="__main__":
    main()
