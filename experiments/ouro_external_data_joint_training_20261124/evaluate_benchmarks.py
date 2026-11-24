"""Run the original seven K4 candidate tasks and the exact MMLU-Pro decoder."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from controller_io import controller
from evaluate_core import finish, verified
import minimal


def candidate_row(raw, key):
    scores = raw["scores"][key]
    denom = raw["denominators"]
    target = int(raw["gold"])
    norm = np.asarray(scores) / np.asarray(denom)
    return dict(id=raw["id"], choices=raw["choices"], gold=target, chars=denom, scores=scores,
                metrics=dict(acc=int(np.argmax(scores)) == target, acc_norm=int(norm.argmax()) == target))


def k4_plan(folder, benchmark, arm, lock, batch):
    return dict(benchmark=benchmark, mode="mmlu_pro" if benchmark=="mmlu_pro" else "candidate", depth=4,
                arm=arm, checkpoint_sha256=lock["arms"][arm]["sha256"], freeze_sha256=minimal.sha(folder), **batch)


def mmlu_metrics(raw, items, legacy):
    output = []
    for item, row in zip(items, raw, strict=True):
        if row['id'] != item.key or row['gold'] != item.gold or row['predicted'] != legacy.extract('mmlu_pro',row['generated'],item.choices):
            raise ValueError(f"MMLU-Pro prediction does not match the locked parser: {item.key}")
        text = row['generated'].strip()
        output.append(dict(row, metrics=dict(correct=row['predicted']==row['gold'], parseable=row['predicted'] is not None,
                                            strict_format=text in list('ABCDEFGHIJ')[:item.choices], strict=text==row['gold'])))
    return output


def seven(a, lock, model, tokenizer, net, hardware):
    sys.path[:0] = [str(a.unified), str(a.stage2 / "code")]
    import seven_unified as scorer
    import stage2_score as stage

    fingerprints = {name: minimal.sha(a.stage2 / "inputs" / f"{name}.jsonl") for name in scorer.TASKS}
    if fingerprints != lock["seven_requests_sha256"] or minimal.sha(Path(stage.__file__)) != lock["stage2_sha256"]:
        raise ValueError("Fixed Stage2 requests or evaluator changed")
    compared = scorer.json_read(a.stage2 / "COMPARISON.json")
    for name in scorer.TASKS:
        if a.task not in ("seven", name):
            continue
        panel = scorer.check_archive(a, stage, hardware, name, compared)
        cfg = panel.public["archived_scoring_parameters"]
        full = k4_plan(a.freeze, name, a.arm, lock, dict(input_sha256=fingerprints[name], config=cfg))
        target = a.out / f"{name}__{a.arm}__k4__candidate.jsonl"
        ids = [row["id"] for row in panel.requests]
        if verified(target, full, ids):
            continue
        if a.arm == "native":
            rows = [candidate_row(row, "native_k4") for row in panel.references]
            reused = dict(source="Stage2 exact archived Native K4 row", result_sha256=panel.public["historic_result_uncompressed_sha256"],
                          historical_joint_scoring_cost=panel.public["archived_joint_scoring_cost"], native_standalone_seconds=None)
            finish(target, rows, full, reused, hardware)
            continue
        shards = a.out / "shards" / target.stem
        shards.mkdir(parents=True, exist_ok=True)
        encoder = stage.HarnessEncoder(tokenizer, cfg["max_length"])
        size = cfg["shard_size"]
        combined = []
        costs = Counter()
        for i in range(0, len(ids), size):
            target_shard = shards / f"{i//size:05d}.jsonl"
            spec = dict(full, start=i, stop=min(i+size,len(ids)))
            wanted = ids[i:i+size]
            saved = verified(target_shard, spec, wanted)
            if saved:
                scored, cost = saved
            else:
                raw, cost = scorer.shard_score(stage, minimal, model, encoder, net, panel.requests[i:i+size], cfg,
                                               next(model.parameters()).device)
                for row, ref in zip(raw, panel.references[i:i+size], strict=True):
                    if (row["id"],row["gold"],row["requests"],row["denominators"]) != (ref["id"],ref["gold"],ref["requests"],ref["denominators"]):
                        raise ValueError(f"Stage2 raw paired requests differ for {row['id']}")
                scored = [candidate_row(row, scorer.UNIFIED) for row in raw]
                finish(target_shard, scored, spec, cost, hardware)
            for key, val in cost.items():
                costs[key] = max(costs[key], val) if key.startswith("peak_") else costs[key]+val
            combined.extend(scored)
        finish(target, combined, full, dict(costs), hardware)
        print("COMPLETE",name,a.arm,len(combined),flush=True)


def mmlu_pro(a, lock, model, tokenizer, net, hardware):
    sys.path[:0] = [str(a.followup),str(a.unified)]
    import generation_followup_run as old

    source = a.followup / "data/mmlu_pro.parquet"
    if minimal.sha(source) != lock["mmlu_pro_sha256"] or minimal.sha(Path(old.__file__)) != lock["mmlu_pro_code_sha256"]:
        raise ValueError("MMLU-Pro data or decoder changed")
    model.model.total_ut_steps = 4
    items = old.prompt_items(source, "mmlu_pro", tokenizer)
    max_new, batch_size = old.TASKS["mmlu_pro"]
    capacity = 14500
    spec = k4_plan(a.freeze,"mmlu_pro",a.arm,lock,dict(input_sha256=minimal.sha(source),batch_size=batch_size,
                                                       capacity_tokens=capacity,max_new=max_new))
    batches = list(old.batches(items,max_new,batch_size,capacity))
    all_ids = [row.key for block in batches for row in block]
    target = a.out / f"mmlu_pro__{a.arm}__k4__generated.jsonl"
    if verified(target,spec,all_ids):
        return
    if a.arm in ('native','t1','t2'):
        import unified_content as prior
        archive, runtime, history = prior.read_history(a,'mmlu_pro',json.loads(source.with_suffix('.json').read_text()),
                                                       prior.PROTOCOL['mmlu_pro'])
        if runtime['files_sha256'] != hardware['files_sha256'] or runtime['model_sha256'] != hardware['model_sha256']:
            raise ValueError('Archived full MMLU-Pro differs from the frozen model or tokenizer')
        name={'native':'native','t1':'rank1_t1','t2':'rank1_t2'}[a.arm]
        if a.arm!='native' and lock['arms'][a.arm]['sha256'] != prior.WEIGHTS[name]:
            raise ValueError('MMLU-Pro checkpoint ancestry differs from the archived runs')
        artifact=archive[name]
        if [len(batch) for batch in batches] != [shard['rows'] for shard in artifact['receipts']]:
            raise ValueError('Archived MMLU-Pro batching differs')
        rows=mmlu_metrics(artifact['rows'],[item for batch in batches for item in batch],old)
        finish(target,rows,spec,dict(artifact['cost'],reuse=True,source_sha256=artifact['sha256'],historical=history),hardware)
        print('REUSED','mmlu_pro',a.arm,len(rows),flush=True)
        return
    shards = a.out / "shards" / target.stem
    shards.mkdir(parents=True,exist_ok=True)
    cache = old.cache_type(model)
    rows = []
    total = Counter()
    for i, batch in enumerate(batches):
        path = shards / f"{i:05d}.jsonl"
        piece = dict(spec,batch_index=i)
        stored = verified(path,piece,[row.key for row in batch])
        if stored:
            scored, cost = stored
        else:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            raw, _ = old.greedy("mmlu_pro",model,tokenizer,batch,"feedback" if net else "native",net,1.,max_new,cache)
            torch.cuda.synchronize()
            cost=dict(seconds=time.perf_counter()-started, generated_tokens=sum(len(row["tokens"]) for row in raw),
                      model_calls=max(len(row["tokens"]) for row in raw), cuda_peak=torch.cuda.max_memory_allocated())
            scored=mmlu_metrics(raw,batch,old)
            finish(path,scored,piece,cost,hardware)
        for key,val in cost.items():
            total[key]=max(total[key],val) if key=="cuda_peak" else total[key]+val
        rows.extend(scored)
    finish(target,rows,spec,dict(total),hardware)
    print("COMPLETE","mmlu_pro",a.arm,len(rows),flush=True)


@torch.inference_mode()
def run(a):
    plan = json.loads(a.freeze.read_text())
    if minimal.sha(Path(__file__)) != plan["large_benchmark_scorer_sha256"]:
        raise ValueError("The final benchmark runner was changed")
    slot=plan["arms"][a.arm]
    weight = Path(slot["path"]) if slot["path"] is not None else None
    if weight is not None and minimal.sha(weight) != slot["sha256"]:
        raise ValueError("The frozen Rank-1 checkpoint was changed")
    _, model, tokenizer, hardware = minimal.runtime(a.archive,a.previous,a.model)
    if hardware["model_sha256"] != plan["model_sha256"]:
        raise ValueError("Ouro checkpoint changed")
    net = controller(weight,next(model.parameters()).device)
    a.out.mkdir(parents=True,exist_ok=True)
    if a.task=="mmlu_pro":
        mmlu_pro(a,plan,model,tokenizer,net,hardware)
    else:
        seven(a,plan,model,tokenizer,net,hardware)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("archive","previous","model","minimal","stage2","unified","followup","historical","out","freeze"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--arm",required=True)
    parser.add_argument("--task",default="seven",choices=("seven","mmlu_pro","arc_challenge","arc_easy","sciq","mmlu","hellaswag","winogrande","piqa"))
    run(parser.parse_args())
