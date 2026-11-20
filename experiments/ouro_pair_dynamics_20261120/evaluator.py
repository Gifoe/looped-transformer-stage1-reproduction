"""Evaluate only after the corresponding predictions are frozen."""
import argparse
from pathlib import Path

import torch

from causal_worlds import sha
from dynamics_embedding import read_json, write_json
from pair_fields import load_runtime
from predictor import paths, predict_records


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("phase")
    p.add_argument("--bf16",action="store_true")
    a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    expected=read_json(a.out/"MANIFEST.json")
    predictions,receipt=paths(a.out,a.phase)
    digest=read_json(receipt)
    assert sha(predictions)==digest["sha256"]
    plan=read_json(predictions)
    groups={q["group"] for q in plan["rows"]}
    for name,items in expected["groups"][a.phase].items():
        expected["groups"][a.phase][name]=[q for q in items if q["group"] in groups]
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model,_,_=load_runtime(a.model)
    model.requires_grad_(False)
    if not a.bf16:
        model.float()
    data=predict_records(model,expected,a.out,a.phase,("native",),plan["loops"],1,0,
                         keep_witness=a.phase=="development",truth=True)
    write_json(a.out/f"NATIVE_{a.phase}{'_BF16' if a.bf16 else ''}.json",
               dict(data,prediction_sha256=digest["sha256"]))


if __name__=="__main__":
    main()
