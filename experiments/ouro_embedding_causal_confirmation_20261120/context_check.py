"""Measure native answers to intact donor prompts without choosing new cases."""
import argparse
from pathlib import Path

import torch

from confirm import DTYPES, EmbeddingOperators, LaneView, buckets, load_runtime, read_json, write_json


@torch.no_grad()
def run(model_path,results):
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    freeze=read_json(results/"FREEZE.json")
    model,_,_=load_runtime(model_path)
    model.requires_grad_(False)
    for name,dtype in DTYPES.items():
        model.to(dtype=dtype)
        out={}
        for depth in (2,3,4):
            records=[]
            for family,groups in freeze["primary"]["groups"].items():
                observed=read_json(results/f"causal_{name}_K{depth}_{family}.json")
                by_group={d["group"]:d for d in observed["rows"]}
                active=[r for r in groups if r["worlds"][0]["group"] in by_group]
                for batch in buckets(active,2):
                    lanes=[]
                    for r in batch:
                        clean,source,rule,same_y=r["worlds"]
                        lanes += [clean,source,rule]+[clean]*4+[same_y,clean]
                    op=EmbeddingOperators(model,lanes,depth)
                    for i,r in enumerate(batch):
                        begin=9*i
                        end=begin+9
                        scope=LaneView(op,begin,end).scores(op.k2[begin:end]).tolist()
                        truth=by_group[r["worlds"][0]["group"]]
                        assert scope[0]==truth["methods"]["native"]["scores"][0]
                        records.append(dict(group=truth["group"],cohort=family,
                            targets=[w["y"] for w in r["worlds"]],
                            native_unpack=[scope[j] for j in (0,1,2,7)]))
            out[f"K{depth}"]=records
            print("COMPLETED","untouched donors",name,depth,flush=True)
        write_json(results/f"DONOR_CONTEXT_{name}.json",out)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("results",type=Path)
    a=p.parse_args()
    run(a.model,a.results)
