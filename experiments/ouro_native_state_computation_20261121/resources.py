"""Derive recurrence and operator invocation counts from completed raw records."""
import argparse
from pathlib import Path

from native import load, save


def count(out):
    rows={}
    for p in sorted(out.glob("ATLAS_*.json")):
        x=load(p)
        rows[p.stem]=dict(recurrences=4*len(x),seconds=sum(z["gpu_seconds"] for z in x),
            extra_projection_calls_per_operator=3*72*len(x))
    for p in sorted(out.glob("CAUSAL_*.json")):
        x=load(p)
        if "rows" not in x:
            continue
        values=x["rows"]
        native=0
        for q in values:
            perturb=4*len(q["response"])+("paired_rewrite" in q["response"])
            interventions=sum(len(t) for k in q["tests"].values() for t in k.values())
            native+=4+perturb+interventions
        rows[p.stem]=dict(recurrences=native,seconds=sum(q["gpu_seconds"] for q in values),
             peak_bytes=x["peak_cuda_bytes"],includes_unsaved_failed_runs=False)
    for p in sorted(out.glob("BEHAVIOR_*.json")):
        x=load(p)
        native=0
        for i in range(0,len(x["rows"]),8):
            pack=x["rows"][i:i+8]
            native+=8
            for j in range(1,5):
                native+=j*max(len(q["k"][str(j)]["generated_tokens"]) for q in pack)
        rows[p.stem]=dict(recurrences=native,seconds=sum(x["seconds_batches_this_run"]),peak_bytes=x["peak_cuda_bytes"])
    for p in sorted(out.glob("FULL_PATCH_*.json")):
        x=load(p)
        rows[p.stem]=dict(recurrences=14*len(x["rows"]),seconds=x["seconds"],peak_bytes=x["peak_cuda_bytes"])
    q=load(out/"SEMANTIC_NATIVE.json")
    rows["SEMANTIC_NATIVE"]=dict(recurrences=sum(2+10*len(x["donors"]) for x in q["rows"]),peak_bytes=q["peak_cuda_bytes"])
    q=load(out/"formation/ORIGIN_SCREEN.json")
    rows["ORIGIN_SCREEN"]=dict(recurrences=98*len(q["rows"]),seconds=sum(x["gpu_seconds"] for x in q["rows"]))
    for p in (out/"formation").glob("ORIGIN_*.json"):
        if p.stem in ("ORIGIN_SCREEN","ORIGIN_PREDICTIONS","ORIGIN_LOCK"):
            continue
        x=load(p)
        native=sum((2 if q["loop_transition"]=="K1toK2" else 3)*
                    (1+sum(len(a) for a in q["mechanisms"].values())) for q in x["rows"])
        rows[p.stem]=dict(recurrences=native,seconds=sum(q["gpu_seconds"] for q in x["rows"]),peak_bytes=x["peak_cuda_bytes"])
    q=load(out/"RESPONSE_CONTROLS.json")
    rows["RESPONSE_CONTROLS"]=dict(recurrences=13*sum(map(len,q["rows"].values())),seconds=q["seconds"],peak_bytes=q["peak_cuda_bytes"])
    for r in rows.values():
        r["attention_calls"]=r["mlp_calls"]=24*r["recurrences"]
    save(out/"RESOURCES.json",dict(scope="batch/operator calls on saved successful runs. CPU/package and technical parity excluded; batches differ in width.",
                rows=rows))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("results",type=Path)
    count(p.parse_args().results)

