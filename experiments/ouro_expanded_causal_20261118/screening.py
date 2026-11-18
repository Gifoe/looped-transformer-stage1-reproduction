"""Select a fresh behavioral stratum before opening its causal results."""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from collect import aligned, digest, dump_line, get_encoding, load_lines, load_runtime, write_json
from factory import group, rng_for, NAMES
from run import fast_forward, predictions

SIZES={"logic":600,"arithmetic":210}
TARGET=90


def prepare(study,model):
    work=study/"screened"
    work.mkdir()
    tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True)
    reject=set()
    for p in (study/"inputs.jsonl",study/"competence"/"inputs.jsonl"):
        for g in load_lines(p):
            if g["task"] in SIZES:
                reject.add((g["task"],g["family"],json.dumps(g["meta"]["signature"])))
    classes=("lempus","gorpus","vumpus","yumpus","sterpus","jompus","wumpus","rompus")
    def novel(g):
        sign=(g["task"],g["family"],json.dumps(g["meta"]["signature"]))
        if sign in reject or not aligned(tok,g):
            return False
        reject.add(sign)
        return True
    with (work/"pool.jsonl").open("x",encoding="utf-8") as f:
        for task,num in SIZES.items():
            for j in range(num):
                k=3*j+(2 if task=="logic" and j<480 else 0)
                case=None
                source=None
                if task=="logic":
                    rng=rng_for(f"counterfactual-competence-source/{j}")
                    who=rng.choice(NAMES)
                    before,after=rng.sample(classes,2)
                    case={"chain_of_thought":[f"{who} is a {before}.",
                                          f"Every {before} is a {after}.",
                                          f"{who} is a {after}."]}
                    source=f"generated_proof/{j}"
                row=group(task,"screenpool",k,source,case,"lexical",novel)
                dump_line(f,row)
    protocol={"pool_sha256":digest(work/"pool.jsonl"),"sizes":SIZES,
              "rule":"all four natural K4 worlds exactly correct; then take the first 90 per task in seeded hash order",
              "site_selection":"same predeclared K2 sites and source/bridge/prefix interventions",
              "screening_source_sha256":digest(Path(__file__)),
              "intervention_sha256":digest(Path(__file__).with_name("intervene.py"))}
    write_json(work/"screen_rule.json",protocol)


def screen(study,model_dir):
    work=study/"screened"
    rule=json.loads((work/"screen_rule.json").read_text())
    assert digest(work/"pool.jsonl")==rule["pool_sha256"]
    groups=load_lines(work/"pool.jsonl")
    model,tok,runtime=load_runtime(model_dir)
    write_json(work/"runtime.json",runtime)
    selected={t:[] for t in SIZES}
    start=time.perf_counter()
    with torch.inference_mode(), (work/"natural_results.jsonl").open("x",encoding="utf-8") as f:
        for count,g in enumerate(groups,1):
            answers={}
            for world in ("r","x","y","d"):
                case=g["worlds"][world]
                e=get_encoding(tok,g,case)
                scores=fast_forward(model,e)[0]
                answers[world]={"scores":scores,"pred":predictions(scores),"gold":case["gold"]}
            capable=all(x["pred"][3]==x["gold"] for x in answers.values())
            dump_line(f,{"id":g["id"],"task":g["task"],"family":g["family"],
                         "worlds":answers,"eligible_K4":capable})
            if capable:
                selected[g["task"]].append(g)
            if count%60==0:
                f.flush()
                print("Screened",count,{t:len(v) for t,v in selected.items()},
                      round(time.perf_counter()-start,1),flush=True)
    torch.cuda.synchronize()
    path=work/"inputs.jsonl"
    retained={}
    with path.open("x",encoding="utf-8") as f:
        for t,pool in selected.items():
            sort=lambda g:digest_text("screen-list-20261118/"+g["id"])
            chosen=sorted(pool,key=sort)[:TARGET]
            retained[t]=len(chosen)
            for row in chosen:
                row["split"]="C"
                dump_line(f,row)
    write_json(work/"screen_complete.json",{"gpu_phase_seconds":time.perf_counter()-start,
               "eligible":{k:len(v) for k,v in selected.items()},"retained":retained,
               "population":SIZES,"results_sha256":digest(work/"natural_results.jsonl")})
    write_json(work/"frozen_protocol.json",{"input_sha256":digest(path),
                "intervention_sha256":rule["intervention_sha256"],
                "screen_results_sha256":digest(work/"natural_results.jsonl"),
                "naturally_capable_retained":retained,"selected_before_intervention":True})
    print("Qualified / retained", {k:len(v) for k,v in selected.items()},retained,flush=True)


def digest_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("prepare","screen"))
    p.add_argument("--study",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    a=p.parse_args()
    if a.action=="prepare":
        prepare(a.study,a.model)
    else:
        screen(a.study,a.model)
