"""Post-gate matched standalone cost and reversible full-state loop swaps."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import DynamicCache

from prepare import dump, key, sha
from train_shared import ALPHA, ARMS, DEPTHS, RANK, load_cohort, load_runtime, make_view, states, variants
from evaluate_shared import parser


class DenseBank:
    """Materialize merged Q/V buffers once and swap dense data outside timing."""
    def __init__(self, model, checkpoints):
        self.params=[]
        for i,layer in enumerate(model.model.layers):
            for attr in ("q_proj","v_proj"):
                self.params.append((i,attr,getattr(layer.self_attn,attr).weight))
        self.buffer={"base":{(i,attr):w.detach() for i,attr,w in self.params}}
        for arm, path in checkpoints.items():
            factors=load_file(str(path),device="cpu")
            layer_bank={}
            for i,attr,w in self.params:
                a,b=(factors[f"{i}.{attr}.{name}"].to(w.device) for name in "ab")
                layer_bank[(i,attr)]=(w.float()+(b@a)*(ALPHA/RANK)).to(w.dtype)
                assert layer_bank[(i,attr)].shape==w.shape
            self.buffer[arm]=layer_bank

    def activate(self, name, layer=None):
        r=self.buffer[name]
        for i,attr,w in self.params:
            if layer is None or layer==i:
                w.data=r[(i,attr)]

    @contextmanager
    def intervention(self, model, first, rest):
        handles=[]
        for i,l in enumerate(model.model.layers):
            def on_layer(_, __, args, layer=i):
                arm=first if args["current_ut"]==0 else rest
                self.activate(arm,layer)
            handles.append(l.register_forward_pre_hook(on_layer,with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()
            self.activate("base")


def now():
    torch.cuda.synchronize()
    return time.perf_counter()


def fresh_cache(model,k):
    cache=DynamicCache()
    cache.append_new_layers(k*len(model.model.layers)-1)
    return cache


def step(model,ids,k,method,cache):
    h,raw,actual=states(model,ids,k,capture_raw="hidden" in method,use_cache=True,cache=cache)
    (name,z),=variants(model,[s[:,-1] for s in h],[s[:,-1] for s in raw],{method})
    assert actual is cache and name==method
    return int(z.argmax(-1).item())


def measure(model,tok,view,k,method):
    prompt=torch.tensor([view["ids"]],device="cuda")
    forced=torch.tensor([[view["display"][0]]],device="cuda")
    cache=fresh_cache(model,k)
    torch.cuda.reset_peak_memory_stats()
    common_bytes=torch.cuda.memory_allocated()
    t0=now()
    first=step(model,prompt,k,method,cache)
    t1=now()
    assert cache.get_seq_length()==len(view["ids"])
    step(model,forced,k,method,cache)
    t2=now()
    assert cache.get_seq_length()==len(view["ids"])+1
    peak=torch.cuda.max_memory_allocated()-common_bytes
    del cache
    cache=fresh_cache(model,k)
    output=[]
    t3=now()
    continuation=step(model,prompt,k,method,cache)
    for s in range(12):
        if continuation==tok.eos_token_id:
            break
        output.append(continuation)
        if "\n" in tok.decode(output,skip_special_tokens=False):
            break
        if s+1<12:
            x=torch.tensor([[continuation]],device="cuda")
            continuation=step(model,x,k,method,cache)
    t4=now()
    return {"id":view["id"],"first_token":first,"prefill_ms":1000*(t1-t0),
            "forced_decode_ms":1000*(t2-t1),"equal_work_total_ms":1000*(t2-t0),
            "natural_total_ms":1000*(t4-t3),"natural_generated_ids":output,
            "incremental_allocated_peak_bytes":peak,"cache_layers":24*k,
            "physical_layers_per_token":24*k,
            "vocabulary_projections_per_token":(2 if "fixed" in method or "adaptive" in method else 1),
            "prompt_tokens":len(view["ids"])}


def cost(model,tok,bank,views,out):
    configs=[(a,k,f"native_k{k}") for a in ("base",*ARMS) for k in DEPTHS]
    configs += [("base",k,f"loopcd_{v}_k{k}") for k in DEPTHS[1:] for v in ("fixed","adaptive","hidden")]
    schedule=[]
    assert len(views)==64
    for i,v in enumerate(views):
        for repeat in range(2):
            order=configs.copy()
            random.Random(int(key("measure_sequence",v["id"],str(repeat)),16)).shuffle(order)
            schedule.extend([(a,k,method,i,repeat) for a,k,method in order])
    # One warm-up per configuration with the same execution functions.
    for arm,k,method in configs:
        bank.activate(arm)
        measure(model,tok,views[0],k,method)
    series=defaultdict(list)
    with (out/"STANDALONE_COST_ROWS.jsonl").open("w") as stream:
        for arm,k,method,i,repeat in schedule:
            bank.activate(arm)
            row=measure(model,tok,views[i],k,method)
            row.update(arm=arm,depth=k,method=method,repeat=repeat)
            series[(arm,method)].append(row)
            stream.write(json.dumps(row,ensure_ascii=False)+"\n")
            stream.flush()
            if len(series[(arm,method)])%32==0:
                print(arm,method,len(series[(arm,method)]),flush=True)
    bank.activate("base")
    summary={}
    for (arm,method),rows in series.items():
        n=len(rows)
        assert n==128
        s={}
        for k in ("prefill_ms","forced_decode_ms","equal_work_total_ms","natural_total_ms",
                  "incremental_allocated_peak_bytes"):
            s[k]={"median":float(np.median([r[k] for r in rows])),
                  "p95":float(np.quantile([r[k] for r in rows],.95))}
        summary[f"{arm}/{method}"]=s
    paired={}
    for k in DEPTHS:
        n=f"native_k{k}"
        raw={}
        for arm in ("base","binding"):
            raw[arm]=defaultdict(list)
            for r in series[(arm,n)]:
                raw[arm][r["id"]].append(r["equal_work_total_ms"])
        ratios=np.array([np.median(raw["binding"][q])/np.median(raw["base"][q]) for q in raw["base"]])
        random_indices=np.random.default_rng(101).integers(0,len(ratios),(9999,len(ratios)))
        top=float(np.quantile(np.median(ratios[random_indices],axis=-1),.95))
        paired[k]={"median_prompt_paired_ratio":float(np.median(ratios)),
                   "upper_one_sided_95":top,"within_1_05":top<=1.05}
    dump(out/"STANDALONE_COST.json",{"executions":len(schedule),"warmup_per_configuration":1,
        "independent_preallocated_dense_buffers":"outside timed core, memory reported above common resident bank",
        "equal_length":"One forced single-token decode after identical-length prefill",
        "natural_length":"Unrestricted greedy until EOS, newline, or 12 tokens",
        "costs":summary,"binding_vs_base_same_budget":paired,"all_cost_gates_pass":all(x["within_1_05"] for x in paired.values())})


def recurrence(model,tok,bank,views,out):
    strategies={a:(a,a) for a in ("base","binding","ce")}
    strategies["binding_first_then_base"]=("binding","base")
    strategies["base_first_then_binding"]=("base","binding")
    result=[]
    for ix,v in enumerate(views):
        x=torch.tensor([v["ids"]],device="cuda")
        states_by={}
        exact={}
        for name,(first,rest) in strategies.items():
            with bank.intervention(model,first,rest):
                h,_,_=states(model,x,6)
                states_by[name]=h
                exact[name]={k:parser(tok,int(model.lm_head(h[k-1][:,-1]).argmax(-1)),v["order"])[0]
                             for k in (1,4,6)}
        for a in ("base","binding","ce"):
            bank.activate(a)
            clean,_,_=states(model,x,6)
            assert all(torch.equal(h,r) for h,r in zip(clean,states_by[a])), ("sham",a,v["id"])
        for a,b in (("binding_first_then_base","binding"),("base_first_then_binding","base")):
            assert torch.equal(states_by[a][0],states_by[b][0]), ("first_loop",a,v["id"])
        result.append({"id":v["id"],"gold":v["gold_content"],"prediction":exact,
                       "same_model_sham_full_state":{"base":True,"binding":True,"ce":True}})
        if (ix+1)%16==0:
            print("recurrence",ix+1,flush=True)
    bank.activate("base")
    summary={m:{k:sum(r["prediction"][m][k]==r["gold"] for r in result) for k in (1,4,6)}
             for m in strategies}
    dump(out/"RECURRENT_SWITCH.json",{"rows":result,"correct":summary,"n":len(result)})


def main():
    p=argparse.ArgumentParser()
    for a in ("model","arc","manifest","weights","gate","freeze","out"):
        p.add_argument(a,type=Path)
    p.add_argument("--stage",choices=("cost","recurrence"),required=True)
    a=p.parse_args()
    gate=json.loads(a.gate.read_bytes())
    manifest=json.loads(a.manifest.read_bytes())
    files={m:a.weights/f"adapter.{m}.safetensors" for m in ARMS}
    if not gate["passed"] or gate["cohort"]!="dev" or gate["execution_freeze_sha256"]!=sha(a.freeze):
        raise RuntimeError("Development gate closed")
    assert gate["adapter_hashes"]=={m:sha(p) for m,p in files.items()}
    assert manifest["source"]["arc"]["sha256"]==sha(a.arc)
    a.out.mkdir(parents=True,exist_ok=True)
    model,tok,_=load_runtime(a.model)
    bank=DenseBank(model,files)
    rows=load_cohort(a.arc,"arc",manifest,"dev",label=a.stage=="recurrence")
    if a.stage=="cost":
        rows=sorted(rows,key=lambda r:key("deployment_latency",r["id"]))[:64]
    rendered=[make_view(tok,r,"original") for r in rows]
    dump(a.out/f"POSTGATE_{a.stage.upper()}_MANIFEST.json",{
        "implementation_sha256":sha(Path(__file__)), "ids":[r["id"] for r in rows],
        "frozen_gate_sha256":sha(a.gate),"adapter_hashes":gate["adapter_hashes"]})
    (cost if a.stage=="cost" else recurrence)(model,tok,bank,rendered,a.out)


if __name__=="__main__":
    main()

