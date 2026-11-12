"""Hardware/model parity smoke test; never a small-sample scientific accuracy claim."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from core import DEFAULT_DEMOS,DEFAULT_MODEL_DIR,demo_prefix,make_requests,encode_pair,load_ouro,write_json,read_jsonl
from latent_hooks import LayerRecorder,Patch


def forward(model, ids, depth, patch=None):
    model.model.total_ut_steps=depth
    model.config.total_ut_steps=depth
    model.model.config.total_ut_steps=depth
    x=torch.tensor([ids],dtype=torch.long,device="cuda")
    if patch is not None:
        if depth!=4:raise ValueError("Patch parity requires four loops")
        with torch.inference_mode(), LayerRecorder(model,[len(ids)-1],False,patch):
            _,states,_=model.model(input_ids=x,use_cache=False)
    else:
        with torch.inference_mode():
            _,states,_=model.model(input_ids=x,use_cache=False)
    return [v.detach().cpu().clone() for v in states]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--demonstrations",type=Path,default=DEFAULT_DEMOS)
    p.add_argument("--model-dir",type=Path,default=DEFAULT_MODEL_DIR)
    p.add_argument("--out",type=Path,default=Path("analyses/runtime_parity.json"))
    a=p.parse_args()
    tok,model=load_ouro(a.model_dir)
    demo=demo_prefix(json.loads(a.demonstrations.read_text(encoding="utf8")))
    row=next(read_jsonl(a.input))  # structural/numerical smoke only; no labels read
    _,requests=make_requests(row,demo,list(range(len(row["choices"]))))
    prefix,target=encode_pair(tok,*requests[0])
    ids=prefix+target[:-1]
    reference=forward(model,ids,4)
    repeated=forward(model,ids,4)
    if not all(torch.equal(a,b) for a,b in zip(reference,repeated)):
        raise RuntimeError("Two identical uncached model forwards differ; investigate backend")
    identity=forward(model,ids,4,Patch(layer=12,from_loop=1,to_loop=2,fraction=1.0))
    if not all(torch.equal(a,b) for a,b in zip(reference,identity)):
        raise RuntimeError("Identity physical-layer hook changes the model output")
    second=forward(model,ids,2)
    if not all(torch.equal(a,b) for a,b in zip(reference[:2],second)):
        raise RuntimeError("Independent depth-2 and native depth-4 intermediate states disagree")
    model.model.total_ut_steps=4;model.config.total_ut_steps=4;model.model.config.total_ut_steps=4
    report={"pass":True,"id":row["id"],"input_length":len(ids),
            "checks":{"identical_uncached_forward":True,"identity_hook":True,"native_depth2_parity":True},
            "scope":"one input: environment/instrumentation check, never a benchmark outcome"}
    write_json(a.out,report)
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
