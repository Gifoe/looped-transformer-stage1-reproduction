"""Manifest validation and natural K1–K4 evaluation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"ouro_cross_task_20261118"))
from run import encode, fast_forward, load_runtime, dump_line, predictions, write_json
from analyze import load_lines, digest
from factory import benchmarks, controlled


def get_encoding(tok, group, world, probe=False):
    prompt=world["probe"] if probe else world["prompt"]
    choices=group["intermediate_options"] if probe else group["options"]
    enc=encode(tok, {"prompt":prompt, "anchor":world["prefix_chars"] or None}, choices)
    offset=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)["offset_mapping"]
    if not probe:
        enc["roles"] = {}
        for role, spans in world["roles"].items():
            hits={i for start,end in spans for i,(a,b) in enumerate(offset) if a<end and b>start}
            if not hits:
                raise ValueError(f"No tokens for {role}")
            enc["roles"][role]=sorted(hits)
    return enc


def aligned(tok,g):
    encoded=[get_encoding(tok,g,w) for w in g["worlds"].values()]
    a=encoded[0]
    for e in encoded:
        if len(e["ids"])!=len(a["ids"]) or e["anchor"]!=a["anchor"] or e["roles"]!=a["roles"]:
            return False
    if g["mode"]=="context":
        fields=[k for k in a["roles"] if k.startswith("payload_")]
        for e in encoded:
            for field in fields:
                if [e["ids"][j] for j in e["roles"][field]] != [a["ids"][j] for j in a["roles"][field]]:
                    return False
    # Candidate score gains after substitutions use identical single-token labels.
    for w in g["worlds"].values():
        get_encoding(tok,g,w,True)
    return True


def prepare(data, previous, model, out):
    tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True)
    rows=list(controlled(data,lambda g:aligned(tok,g)))+list(benchmarks(data,load_lines(previous)))
    assert len({g["id"] for g in rows})==len(rows)
    counts=Counter(f'{g["task"]}:{g["split"]}' for g in rows)
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/"inputs.jsonl"
    with manifest.open("x",encoding="utf-8") as f:
        for g in rows:
            dump_line(f,g)
    write_json(out/"manifest.json",{"group_counts":dict(counts),"inputs_sha256":digest(manifest),
               "source_hashes":{n:digest(data/n) for n in ("prontoqa.zip","arc.parquet","hellaswag.parquet")},
               "factory_sha256":digest(HERE/"factory.py"),"encoder_sha256":digest(Path(__file__))})
    print("Prepared",len(rows),dict(counts),flush=True)


def selected(g,job):
    return ((job=="calibration" and g["split"]=="dev") or
            (job=="benchmarks" and g["split"]=="benchmark") or
            (job=="controlled" and g["split"] in ("confirm","natural")))


def natural(job,model_path,out):
    folder=out/job
    folder.mkdir()
    model,tok,runtime=load_runtime(model_path)
    write_json(folder/"runtime.json",runtime)
    groups=[g for g in load_lines(out/"inputs.jsonl") if selected(g,job)]
    n=0
    start=time.perf_counter()
    with torch.inference_mode(), (folder/"rows.jsonl").open("w",encoding="utf-8") as file:
        for g in groups:
            for name in ("r","x","y","d"):
                if name not in g["worlds"]:
                    continue
                w=g["worlds"][name]
                enc=get_encoding(tok,g,w)
                position=enc["roles"].get("map",[])+[enc["read"]]
                # Sample at post-layer L23, independently of the output head.
                tap={**enc,"anchor":list(dict.fromkeys(position))}
                scores, bank, _,_,_=fast_forward(model,tap,{(k,23) for k in range(1,5)})
                hidden=[bank[(k,23)].float() for k in range(1,5)]
                geometry=[torch.nn.functional.cosine_similarity(a,b,dim=-1).tolist()
                          for a,b in zip(hidden,hidden[1:])]
                row={"id":g["id"],"task":g["task"],"family":g.get("family"),"world":name,
                     "gold":w["gold"],"scores":scores,"pred":predictions(scores),
                     "length":len(enc["ids"]),"terminal_layer_cosine":geometry}
                if "probe" in w:
                    p=fast_forward(model,get_encoding(tok,g,w,True))[0]
                    row.update(mid_gold=w["mid"],mid_scores=p,mid_pred=predictions(p))
                dump_line(file,row)
                n+=1
            if n%100<4:
                file.flush()
                print(job,n,round(time.perf_counter()-start,1),flush=True)
    torch.cuda.synchronize()
    write_json(folder/"complete.json",{"groups":len(groups),"rows":n,"gpu_phase_seconds":time.perf_counter()-start,
              "peak_allocated_bytes":torch.cuda.max_memory_allocated()})


def main():
    p=argparse.ArgumentParser()
    p.add_argument("job",choices=("prepare","calibration","benchmarks","controlled"))
    p.add_argument("--study",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--data",type=Path)
    p.add_argument("--prior",type=Path)
    args=p.parse_args()
    if args.job=="prepare":
        prepare(args.data,args.prior,args.model,args.study)
    else:
        natural(args.job,args.model,args.study)


if __name__=="__main__":
    main()
