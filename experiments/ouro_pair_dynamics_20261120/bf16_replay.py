"""Practical BF16 check. AD through reduced-precision casts is diagnostic only."""
import argparse
from pathlib import Path

import torch

from causal_worlds import sha
from dynamics_embedding import read_json, write_json
from pair_fields import load_runtime
from predictor import predict_records


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("stage",choices=("predict","truth"))
    a=p.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model,_,_=load_runtime(a.model)
    model.requires_grad_(False)
    assert next(model.parameters()).dtype==torch.bfloat16
    m=read_json(a.out/"MANIFEST.json")
    pred=a.out/"BF16_PREDICTIONS.json"
    receipt=a.out/"BF16_PREDICTION_SHA.json"
    if a.stage=="predict":
        stats=predict_records(model,m,a.out,"validation",("full","taylor","prefix_anderson"),3,2,8)
        write_json(pred,stats)
        write_json(receipt,dict(sha256=sha(pred),scope="fixed first eight groups per family, validation"))
    else:
        assert sha(pred)==read_json(receipt)["sha256"]
        data=predict_records(model,m,a.out,"validation",("native",),3,1,8)
        write_json(a.out/"BF16_NATIVE.json",dict(data,prediction_sha256=sha(pred)))


if __name__=="__main__":
    main()
