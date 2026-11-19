"""Measure downstream fork divergence at explicitly aligned roles and free decoding."""
import argparse
from pathlib import Path
import re
import time

import numpy as np
import torch

from mechanism_design import FORKS, dump, encode, load, seed
from mechanism_hotpot import encode_hotpot
from mechanism_run import add_line, load_runtime, norm_at, sha

TAPS = ("attention_update","mlp_update","residual")
METRICS = ("readout_rms_vs_content","query_rms_vs_content","correct_token_lens",
           "gold_new_letter_minus_original", "same_candidate_rms_vs_content")


def hidden_fork(model,tok,g):
    es=[encode(tok,g,0,0,f) for f in FORKS]
    base=FORKS.index("content")
    length=max(len(e["ids"]) for e in es)
    ids=torch.tensor([e["ids"]+[0]*(length-len(e["ids"])) for e in es],device="cuda")
    pos=torch.tensor([e["read"] for e in es],device="cuda")
    rows=torch.arange(len(es),device="cuda")
    query=[e["roles"]["query"] for e in es]
    qi=torch.tensor(query,device="cuda")
    available=[i for i,e in enumerate(es) if "candidate" in e["roles"]]
    positions=es[base]["roles"]["candidate"]
    order=g["order"]
    candidate_indices={i:[es[i]["roles"]["candidate"][(order[1:]+order[:1]).index(k) if FORKS[i]=="order" else order.index(k)] for k in range(3)] for i in available}
    qcmp=torch.tensor([candidate_indices[i] for i in available],device="cuda")
    rr=torch.tensor(available,device="cuda")
    by_content=torch.tensor([e["content"] for e in es],device="cuda")
    la=torch.tensor([e["letter"] for e in es],device="cuda")
    original=la[base,0]
    analytic={k:[] for k in TAPS}
    handle=[]
    def hook(kind):
        def observe(_m,_args,out):
            h=out[rows,pos]
            rms=(h.float()-h[base].float()).square().mean(-1).sqrt()
            q=out[rows[:,None],qi]
            qrms=(q.float()-q[base].float()).square().mean(dim=(-1,-2)).sqrt()
            match=out[rr[:,None],qcmp].float()
            pos_rms=torch.full_like(qrms,float("nan"))
            pos_rms[rr]=(match-match[available.index(base)]).square().mean(dim=(-1,-2)).sqrt()
            if kind=="residual":
                norm=norm_at(model,h)
                target=model.lm_head.weight[by_content]
                cc=torch.einsum("bd,bod->bo",norm,target).float()
                ca=cc[:,0]-cc[:,1:].mean(-1)
                lg=torch.einsum("bd,bd->b",norm,model.lm_head.weight[la[:,0]]-model.lm_head.weight[original]).float()
            else:
                ca=lg=torch.full_like(qrms,float("nan"))
            analytic[kind].append(torch.stack((rms,qrms,ca,lg,pos_rms),dim=-1))
        return observe
    try:
        for layer in model.model.layers:
            handle.extend([layer.input_layernorm_2.register_forward_hook(hook("attention_update")),
                           layer.post_attention_layernorm_2.register_forward_hook(hook("mlp_update")),
                           layer.register_forward_hook(hook("residual"))])
        model.model(input_ids=ids,use_cache=False)
    finally:
        for h in handle:
            h.remove()
    result=np.stack([torch.stack(analytic[k]).cpu().numpy() for k in TAPS])
    if not (result[:,:,:,1]==0).all():
        raise RuntimeError("The shared question region diverged")
    return result,es[base]["binding"] if "binding" in es[base] else es[base]["by_content"]


def generate(model,tok,contexts,depth,steps=8):
    original=model.model.total_ut_steps
    sequences=[list(e["ids"]) for e in contexts]
    emitted=[[] for _ in contexts]
    finished=set()
    try:
        model.model.total_ut_steps=depth
        for _ in range(steps):
            n=max(map(len,sequences))
            ids=torch.tensor([v+[0]*(n-len(v)) for v in sequences],device="cuda")
            positions=torch.tensor([len(v)-1 for v in sequences],device="cuda")
            out=model.model(input_ids=ids,use_cache=False)[0].last_hidden_state
            last=out[torch.arange(len(contexts),device="cuda"),positions]
            nexts=model.lm_head(last).argmax(-1).cpu().tolist()
            for i,k in enumerate(nexts):
                if i in finished:
                    continue
                if k==tok.eos_token_id:
                    finished.add(i)
                    continue
                emitted[i].append(k)
                sequences[i].append(k)
            if len(finished)==len(contexts):
                break
    finally:
        model.model.total_ut_steps=original
    return [(v,tok.decode(v,skip_special_tokens=True)) for v in emitted]


def parsed(text,opts,letter):
    if letter:
        found=re.match(r"^\s*(?:the answer is\s+)?([A-E])(?:\b|\))",text,re.I)
        return found.group(1).upper() if found else None
    keys=sorted(opts,key=len,reverse=True)
    exp=r"(?<!\w)("+"|".join(re.escape(v) for v in keys)+r")(?!\w)"
    got=re.search(exp,text,re.I)
    return got.group(1).casefold() if got else None


def main():
    p=argparse.ArgumentParser()
    for name in ("checkpoint","groups","hotpot","out"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    model,tok,_=load_runtime(a.checkpoint)
    rows=list(load(a.groups))
    starts=time.perf_counter()
    traces=[]
    with torch.inference_mode(), (a.out/"free_generation.jsonl").open("w",encoding="utf-8") as f:
        for g in rows:
            v,_=hidden_fork(model,tok,g)
            traces.append(v)
        print(f"measured {len(rows)} fork atlases",flush=True)
        for split in ("development","new_entities","new_structure","hotpot_bridge_validation_hashed"):
            if split.startswith("hotpot"):
                population=list(load(a.hotpot))
                pop=sorted(population,key=lambda g:seed("free-generation",g["id"]))[:16]
                encode_fn=lambda g,fork:encode_hotpot(tok,g,fork)
            else:
                pop=sorted([g for g in rows if g["split"]==split],key=lambda g:seed("free-generation",g["id"]))[:12]
                encode_fn=lambda g,fork:encode(tok,g,0,0,fork)
            for fork in ("open","content","letter"):
                es=[encode_fn(g,fork) for g in pop]
                for depth in (1,2,3,4):
                    for i,(tokens,text) in enumerate(generate(model,tok,es,depth)):
                        g,e=pop[i],es[i]
                        y=g["y"] if "y" in g else 0
                        candidates=g["options"] if "options" in g else g["answers"][:5]
                        letter=fork=="letter"
                        if letter:
                            expected=chr(65+y) if "y" in g else e["by_content"][y]
                        else:
                            expected=candidates[y].casefold()
                        found=parsed(text,candidates,letter)
                        add_line(f,{"id":g["id"],"split":split,"fork":fork,"k":depth,"tokens":tokens,
                                    "text":text,"gold":expected,"parsed":found,"correct":found==expected})
            f.flush()
            print(f"generation {split}: {len(pop)} rows per fork",flush=True)
    np.savez_compressed(a.out/"fork_atlas.npz",ids=[g["id"] for g in rows],
                        splits=[g["split"] for g in rows],forks=FORKS,taps=TAPS,metrics=METRICS,data=np.stack(traces))
    (a.out/"complete.json").write_text(dump({"groups":len(rows),"manifest_sha256":sha(a.groups),"seconds":time.perf_counter()-starts}),encoding="utf-8")


if __name__=="__main__":
    main()

