"""Select a shared generation control policy without inspecting confirmation splits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

import generation
from generation import Arm, run_slice, write_jsonl
from controller import load_controller
from minimal import DEPTHS, dump, label_prompt, load_jsonl, runtime, sha
from schedule import ORIGINAL, Schedule, for_generation


DEV_POLICIES = {
    "t1": ORIGINAL,
    "t2": Schedule(checkpoint="t2"),
    "decision": Schedule(stop=1),
    "early_two": Schedule(stop=2),
    "tail_only": Schedule(first=1),
    "cap25": Schedule(cap=.25),
    "decision25": Schedule(stop=1, cap=.25),
    "phased": Schedule(after_cap=.15),
    "unsaturated": Schedule(headroom_ratio=.43),
}
COMPLEXITY = {name: i for i, name in enumerate(DEV_POLICIES)}


def prepare(raw, tokenizer):
    prepared = []
    for row in raw:
        normalized = dict(row, question=row.get("question", row.get("question_stem")))
        prompt, options, gold = label_prompt(normalized)
        prepared.append(dict(id=row["id"], prompt=prompt, prompt_ids=tokenizer.encode(prompt, add_special_tokens=False),
                             options=options, gold=gold, official_labels=row["choices"]["label"]))
    assert len(prepared) == len({x["id"] for x in prepared})
    return sorted(prepared, key=lambda row: (len(row["prompt_ids"]), row["id"]))


def paired(baseline, treatment, field):
    old = {row["id"]: row for row in baseline}
    assert old.keys() == {row["id"] for row in treatment} and len(old) == len(treatment)
    prefix_differences = Counter()
    rescue = damage = 0
    for row in treatment:
        peer = old[row["id"]]
        assert all(row[k] == peer[k] for k in ("prompt", "gold", "options"))
        rescue += bool(row[field] and not peer[field])
        damage += bool(peer[field] and not row[field])
        if row["tokens"] != peer["tokens"]:
            i = next((i for i, (x, y) in enumerate(zip(row["tokens"], peer["tokens"])) if x != y),
                     min(len(row["tokens"]), len(peer["tokens"])))
            prefix_differences[i] += 1
    n = len(treatment)
    return dict(n=n, correct=sum(row[field] for row in treatment), rescue=rescue, damage=damage,
                delta_pp=100*(rescue-damage)/n, ci95_pp=generation.bootstrap_pair(rescue,damage,n),
                first_token_disagreement=dict(prefix_differences))


def run_policy(out, split, name, depth, rows, model, tokenizer, checkpoint, policy):
    stem = out / f"{split}__{name}__k{depth}"
    path, receipt = stem.with_suffix(".jsonl"), stem.with_suffix(".json")
    if path.exists() and receipt.exists():
        cached = json.loads(receipt.read_text())
        assert cached["records_sha256"] == sha(path) and cached["policy"] == policy.record()
        return load_jsonl(path), cached
    original = generation.model_next

    def next_tokens(model, feedback, arm, depth, packed, last):
        arm = Arm("minimal", for_generation(arm.net, policy, packed), 1.)
        return original(model, feedback, arm, depth, packed, last)

    generation.model_next = next_tokens
    try:
        values, cost = run_slice(split, name, depth, rows, 16, 6, model, tokenizer, None,
                                 Arm("minimal", checkpoint, 1.0))
    finally:
        generation.model_next = original
    write_jsonl(path, values)
    evidence = dict(records_sha256=sha(path), policy=policy.record(), cost=cost)
    dump(receipt, evidence)
    return values, evidence


def choose(table, names):
    decisions = {}
    for depth in DEPTHS:
        eligible = []
        for name in names:
            if name == "t1":
                continue
            rows = [table[split][f"k{depth}"][name] for split in table]
            correctness = [r["correct"]["delta_pp"] for r in rows]
            validity = [r["valid"]["delta_pp"] for r in rows]
            if min(correctness + validity) >= 0 and max(correctness) >= 1.0:
                eligible.append((min(correctness), float(np.mean(correctness)),
                                 -COMPLEXITY.get(name, 100), name))
        decisions[f"k{depth}"] = max(eligible)[-1] if eligible else "t1"
    return decisions


@torch.inference_mode()
def main(args):
    old, model, tokenizer, info = runtime(args.archive,args.previous,args.model)
    model.eval().requires_grad_(False)
    weight_paths = {"t1": args.control/"results/weights/rank1_t1.pt", "t2": args.control/"results/weights/rank1_t2.pt"}
    if args.new_weight:
        weight_paths["train"] = args.new_weight
    weights = {k: load_controller(v) for k,v in weight_paths.items()}
    weights_sha = {k: sha(p) for k,p in weight_paths.items()}
    if args.phase == "dev":
        p = args.ob_validation
        assert sha(p) == "35370b9cfee8c1ff325ccc74adc434d12c47ca0ac3244aa87f3fa77069285206"
        arc = sorted(old.read_data(args.corpus,"train"), key=lambda r: hashlib.sha256(
            ("rank1-dev-20261123:"+r["id"]).encode()).digest())[:256]
        ob = pq.read_table(p).to_pylist()
        assert len(arc) == 256 and len(ob) == 500
        data = {"arc_train_dev": prepare(arc,tokenizer), "openbookqa_validation": prepare(ob,tokenizer)}
        table = {split: {} for split in data}
        policies = dict(DEV_POLICIES)
        if args.new_weight:
            policies.update(train=Schedule(checkpoint="train"), train_first=Schedule(checkpoint="train", stop=1))
        dump(args.out/"DEV_LOCK.json", dict(cells={k: sorted(r["id"] for r in v) for k,v in data.items()},
                                           weights_sha256=weights_sha, policies={k:p.record() for k,p in policies.items()}, runtime=info))
        for split, rows in data.items():
            for depth in DEPTHS:
                cell = table[split][f"k{depth}"] = {}
                for name, policy in policies.items():
                    values,cost = run_policy(args.out,split,name,depth,rows,model,tokenizer,weights[policy.checkpoint],policy)
                    if name == "t1":
                        baseline = values
                    cell[name] = {key:paired(baseline,values,key) for key in ("correct","valid","strict_format")}
                    cell[name]["cost"] = cost["cost"]["wall_s"]
                    print(split,depth,name,cell[name]["correct"]["correct"],cell[name]["valid"]["correct"],flush=True)
        selected = choose(table,policies)
        dump(args.out/"DEV_RESULTS.json",table)
        dump(args.out/"DEV_PROPOSAL.json",dict(selected=selected,policies={k:v.record() for k,v in policies.items()},
                                              weights_sha256=weights_sha,validation_results_sha256=sha(args.out/"DEV_RESULTS.json")))
    else:
        lock = json.loads(args.freeze.read_text())
        assert all(weights_sha[key] == digest for key,digest in lock["weights_sha256"].items())
        data,_ = generation.read_datasets(old,tokenizer,args.corpus,args.external)
        baseline_dir = args.control/"results/GENERATION_RESULTS"
        history_rows = load_jsonl(baseline_dir/"GENERATION_RESULTS.jsonl")
        summary = {}
        for split, rows in data.items():
            summary[split] = {}
            for depth in DEPTHS:
                alias = lock["selected"][f"k{depth}"]
                policy = Schedule(**lock["policies"][alias])
                baseline = load_jsonl(baseline_dir/f"{split}__rank1_t1__k{depth}.jsonl")
                if alias == "t1":
                    values,cost = baseline,dict(reused_sha256=sha(baseline_dir/f"{split}__rank1_t1__k{depth}.jsonl"))
                    write_jsonl(args.out/f"{split}__locked__k{depth}.jsonl",values)
                else:
                    values,cost = run_policy(args.out,split,"locked",depth,rows,model,tokenizer,weights[policy.checkpoint],policy)
                history = {name: [] for name in ("native","linear8_t1","linear8_t2")}
                for row in history_rows:
                    if row["split"] == split and row["depth"] == depth and row["method"] in history:
                        history[row["method"]].append(row)
                contrasts = {name:{k:paired(peer,values,k) for k in ("correct","valid","strict_format")}
                             for name,peer in dict(t1=baseline,**history).items()}
                summary[split][f"k{depth}"] = dict(policy=alias,comparisons=contrasts,cost=cost,
                                                  invalid=dict(Counter(x["invalid_category"] for x in values if not x["valid"])))
                print(split,depth,alias,contrasts["t1"]["correct"],flush=True)
        dump(args.out/"CONFIRM.json",summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("dev","confirm"))
    for key in ("archive","previous","model","corpus","control","ob-validation","external","out","freeze","new-weight"):
        parser.add_argument("--"+key,type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    main(args)
