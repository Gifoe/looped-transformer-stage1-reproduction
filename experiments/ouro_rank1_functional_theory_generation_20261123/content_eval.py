"""Complete GSM8K and MMLU-Pro verification under the archived greedy protocol."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from controller import load_controller
from generation import bootstrap_pair
from generation_followup_report import harness_match, number
import generation_followup_run_gsm8k as legacy
from minimal import dump, load_jsonl, rms, runtime, sha
from schedule import ORIGINAL, Schedule


class StepFeedback:
    def __init__(self, net, policy, step):
        self.net = net
        self.policy = policy
        self.step = step

    def delta(self, h):
        p = self.policy
        delta = self.net.delta(h)
        ratio = rms(delta) / rms(h)
        cap = p.cap if self.step == 0 or p.after_cap is None else p.after_cap
        active = ratio >= p.min_dose
        if p.headroom_ratio > 0:
            active = active & (ratio <= p.headroom_ratio)
        return delta * (cap / ratio).clamp(max=1.) * active


def greedy(task, model, tok, batch, net, policy, max_tokens, cache):
    inherited = legacy.forward
    step = 0

    def scheduled(model, tokens, valid, indices, arm, net, dose, cache, eligible):
        nonlocal step
        applied = step >= policy.first and (policy.stop is None or step < policy.stop)
        if not applied:
            arm = "native"
        elif policy != ORIGINAL and policy != Schedule(checkpoint="t2"):
            net = StepFeedback(net,policy,step)
        step += 1
        return inherited(model,tokens,valid,indices,arm,net,dose,cache,eligible)

    # Legacy cache positions preserve every previously executed edit while the
    # current iteration only controls the next causal predictor.
    legacy.forward = scheduled
    try:
        rows,_ = legacy.greedy(task,model,tok,batch,"feedback",net,1.,max_tokens,cache)
    finally:
        legacy.forward = inherited
    return rows,step


def grade(task, row):
    value = row["predicted"]
    target = row["gold"]
    if task == "mmlu_pro":
        return {"valid": value is not None, "correct": value == target,
                "strict": row["generated"].strip() == target}
    return dict(valid=number(value) is not None,
                strict=number(value) is not None and number(value)==number(target),
                correct=harness_match(row["generated"],target))


def compare(task, baseline, revised):
    a={r["id"]:grade(task,r) for r in baseline}
    b={r["id"]:grade(task,r) for r in revised}
    assert len(a)==len(baseline)==len(revised)==len(b) and a.keys()==b.keys()
    out={}
    for metric in ("correct","strict","valid"):
        rescue=sum(b[i][metric] and not a[i][metric] for i in a)
        damage=sum(a[i][metric] and not b[i][metric] for i in a)
        n=len(a)
        out[metric]=dict(old=sum(r[metric] for r in a.values()),new=sum(r[metric] for r in b.values()),n=n,
                         rescue=rescue,damage=damage,delta_pp=100*(rescue-damage)/n,
                         ci95_pp=bootstrap_pair(rescue,damage,n))
    both=[i for i in a if a[i]["valid"] and b[i]["valid"]]
    out["both_valid"]={"n":len(both),"old":sum(a[i]["correct"] for i in both),
                       "new":sum(b[i]["correct"] for i in both)}
    out["stop"]={"original":dict(Counter(r["stop"] for r in baseline)),
                 "locked":dict(Counter(r["stop"] for r in revised))}
    out["tokens"]={"original":float(np.mean([len(r["tokens"]) for r in baseline])),
                   "locked":float(np.mean([len(r["tokens"]) for r in revised]))}
    return out


def main(a):
    record=json.loads(a.freeze.read_text())
    tag=record["selected"]["k4"]
    policy=Schedule(**record["policies"][tag])
    weights={"t1":a.control/"results/weights/rank1_t1.pt", "t2":a.control/"results/weights/rank1_t2.pt"}
    if a.new_weight:
        weights["train"]=a.new_weight
    for name in ("t1",policy.checkpoint):
        assert sha(weights[name])==record["weights_sha256"][name]
    source=a.data/f"{a.task}.parquet"
    dataset=json.loads(source.with_suffix(".json").read_text())
    assert sha(source)==dataset["sha256"]
    _,model,tok,info=runtime(a.archive,a.previous,a.model)
    model.eval().requires_grad_(False)
    model.model.total_ut_steps=4
    cuda_cache=legacy.cache_type(model)
    items=legacy.prompt_items(source,a.task,tok)
    assert len(items)==dataset["rows"]
    max_tokens,batch_size=legacy.TASKS[a.task]
    a.out.mkdir(parents=True,exist_ok=True)
    common=dict(task=a.task,dataset=dataset,freeze_sha256=sha(a.freeze),runtime=info,
                greedy=True,seed=0,depth=4,max_new=max_tokens,batch_size=batch_size,capacity_tokens=14500)
    dump(a.out/"LOCK.json",common)
    all_rows={}
    pairs=(('original',ORIGINAL),('locked',policy)) if policy!=ORIGINAL else (('original',ORIGINAL),)
    for name,test in pairs:
        path=a.out/name
        path.mkdir(exist_ok=True)
        net=load_controller(weights[test.checkpoint])
        rows=[]
        for index,batch in enumerate(legacy.batches(items,max_tokens,batch_size,14500)):
            prefix=path/f"{index:04}.jsonl"
            receipt=prefix.with_suffix(".json")
            keys=[x.key for x in batch]
            if prefix.is_file() and receipt.is_file():
                old=json.loads(receipt.read_text())
                assert old["sha256"]==sha(prefix) and old["ids"]==keys and old["policy"]==test.record()
                values=load_jsonl(prefix)
            else:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                values,steps=greedy(a.task,model,tok,batch,net,test,max_tokens,cuda_cache)
                torch.cuda.synchronize()
                legacy.save_jsonl(prefix,values)
                dump(receipt,dict(ids=keys,policy=test.record(),sha256=sha(prefix),
                                  steps=steps,peak_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-start))
            rows.extend(values)
            if index%8==0:
                print(a.task,name,index,len(rows),flush=True)
        legacy.save_jsonl(a.out/f"{name.upper()}.jsonl",rows)
        all_rows[name]=rows
    locked=all_rows.get("locked",all_rows["original"])
    report=dict(actual_policy=tag,exact_reuse=policy==ORIGINAL,
                vs_rank1_t1=compare(a.task,all_rows["original"],locked))
    for name in ("native","fixed","adaptive","feedback"):
        path=a.history/a.task/f"{name}.jsonl"
        if path.is_file():
            report[f"vs_{name}"]=compare(a.task,load_jsonl(path),locked)
    dump(a.out/"SUMMARY.json",report)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",choices=("gsm8k","mmlu_pro"),required=True)
    for key in ("archive","previous","model","control","data","history","freeze","new-weight","out"):
        parser.add_argument("--"+key,type=Path,required=key!="new-weight")
    main(parser.parse_args())
