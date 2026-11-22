"""Score only the confirmed small controller on the archived seven-task inputs."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from benchmark import holm, paired
from controller import load_controller
from generation import write_jsonl
from minimal import dump, forward, load_jsonl, runtime, sha


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(rows, metrics):
    scores = {}
    keys = ("native_k4","loopcd_fixed_k4","loopcd_adaptive_k4","minimal_k4","controller_k4")
    for metric in metrics:
        acc = {}
        for key in keys:
            flags = []
            for r in rows:
                asarray = np.asarray(r["scores"][key])
                if metric=="acc_norm":
                    asarray = asarray/r["denominators"]
                flags.append(int(asarray.argmax())==r["gold"])
            acc[key]=np.asarray(flags)
        scores[metric] = dict(correct={k:int(a.sum()) for k,a in acc.items()},
                              paired={b:paired(acc[b],acc["controller_k4"]) for b in keys[:-1]})
    return scores


@torch.no_grad()
def run(a):
    gate = read(a.freeze.parent/"SEVEN_TASK_GATE.json")
    assert gate["passed"] and gate["freeze_sha256"]==sha(a.freeze)
    freeze = read(a.freeze)
    selected = freeze["weights"][freeze["primary"]]
    checkpoint = a.freeze.parent/"weights"/selected["file"]
    assert sha(checkpoint)==selected["sha256"]
    sys.path.insert(0,str(a.prior/"stage2/code"))
    import stage2_score as s
    _,model,tok,metadata = runtime(a.archive,a.previous,a.model)
    encoder = s.HarnessEncoder(tok,model.config.max_position_embeddings)
    model_info = dict(freeze_sha256=sha(a.freeze),checkpoint_sha256=sha(checkpoint),
                      source_sha256=sha(Path(__file__)),historical_scoring_src_sha256=sha(s.__file__),
                      method=freeze["primary"], model=metadata)
    net=load_controller(checkpoint)
    metrics_by_task={}
    use_tasks=a.tasks or tuple(s.APPLE_BENCHMARKS)
    for task in use_tasks:
        prev=a.prior/"stage2"
        archived=prev/"scores"/task
        target=a.out/task
        target.mkdir(parents=True,exist_ok=True)
        history=load_jsonl(archived/"RESULTS.jsonl")
        input=prev/"inputs"/f"{task}.jsonl"
        meta=input.with_suffix(".jsonl.manifest.json")
        requests, manifest=s.read_export(SimpleNamespace(input=input,manifest=meta,plan=False,benchmarks=(task,)))
        n=len(history)
        assert len(requests)==n==manifest["totals"]["documents"]
        config=read(archived/"LOCK.json")
        assert config["files"][input.name]==sha(input)
        expected=list(zip(requests,history,strict=True))
        assert all((p["id"],p["target"],p["choices"])==(q["id"],q["gold"],q["choices"])
                   for p,q in expected)
        lock=dict(model_info,input_sha256=sha(input),manifest_sha256=sha(meta),
                  prior_result_sha256=sha(archived/"RESULTS.jsonl"),n=n,shots=s.APPLE_BENCHMARKS[task],
                  token_cap=config["token_cap"], attention_cap=config["attention_cap"],
                  head_rows=config["head_rows"],batch_size=config["max_batch"],shard_size=config["shard_size"])
        record=target/"LOCK.json"
        if record.exists():
            assert read(record)==json.loads(json.dumps(lock))
        else:
            dump(record,lock)
        size=config["shard_size"]
        all_rows=[]
        costs=[]
        for i in range(0,n,size):
            shard=target/f"{i//size:04d}.jsonl"
            cost=shard.with_suffix(".cost.json")
            if shard.exists() and cost.exists():
                z,c=load_jsonl(shard),read(cost)
                assert [t["id"] for t in z]==[t["id"] for t in requests[i:i+size]]
                assert c["sha256"]==sha(shard)
            else:
                t=time.perf_counter()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                prompt=s.compact_prefixes(s.encode_rows(requests[i:i+size],encoder))
                z=s.render(requests[i:i+size])
                work=s.batch_work(prompt,config["max_batch"],config["token_cap"],config["attention_cap"])
                v=0
                for pack in work:
                    batch, refs, gather, token_refs, counts=s.pack(pack,"cuda")
                    state,_=forward(model,batch,4,net,1.)
                    heads={"head_calls":0}
                    out=s.score_head(model,[state],(4,),config["head_rows"],gather,token_refs,counts,heads,apple=False)
                    s.record(z,refs,{"controller_k4":out["minimal"]})
                    v+=batch[0].numel()
                for x,y in zip(z,history[i:i+size],strict=True):
                    assert x["id"]==y["id"] and x["gold"]==y["gold"] and x["choices"]==y["choices"]
                    assert all(f is not None for f in x["scores"]["controller_k4"])
                    x["scores"].update(y["scores"])
                torch.cuda.synchronize()
                write_jsonl(shard,z)
                c=dict(
                    n=len(z),seconds=time.perf_counter()-t,sha256=sha(shard),backbone_loop_token_calls=4*v,
                    peak_bytes=torch.cuda.max_memory_allocated(),
                )
                dump(cost,c)
                print(task,i//size,len(z),round(c["seconds"],2),flush=True)
            all_rows.extend(z);costs.append(c)
        assert len(all_rows)==n
        write_jsonl(target/"RESULTS.jsonl",all_rows)
        metrics=sorted(set(manifest["tasks"][0]["metrics"]) & {"acc","acc_norm"})
        results=compare(all_rows,metrics)
        data=dict(n=n,shots=s.APPLE_BENCHMARKS[task],metrics=results,costs=costs)
        dump(target/"SUMMARY.json",data)
        metrics_by_task[task]=data
        print("COMPLETE",task,n,{m:x["correct"]["controller_k4"] for m,x in results.items()},flush=True)
    for b in ("native_k4","loopcd_fixed_k4","loopcd_adaptive_k4","minimal_k4"):
        holm([y["paired"][b] for run in metrics_by_task.values() for y in run["metrics"].values()])
    dump(a.out/"COMPARISON.json",metrics_by_task)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("archive","previous","model","prior","freeze","out"):
        p.add_argument(f"--{k}",type=Path,required=True)
    p.add_argument("--tasks",nargs="+")
    run(p.parse_args())
