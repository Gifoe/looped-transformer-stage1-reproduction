"""Full ARC/OpenBook confirmation and genuine generation of frozen controller families."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from benchmark import METRICS, holm, merge, paired, statistic
from controller import load_controller
from generation import Arm, read_datasets, run_slice, write_jsonl
from minimal import DEPTHS, dump, load_jsonl, runtime, sha
from train import evaluate

REFERENCE = ("native", "linear8_t1", "linear8_t2")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def archived_scores(prior, name):
    source = prior / "results" / f"RESULTS_{name}.jsonl"
    rows = load_jsonl(source)
    for row in rows:
        row["scores"] = {k: v for k,v in row["scores"].items() if k.startswith(
            ("native_", "loopcd_", "linear8_t1_", "linear8_t2_", "feedback_only_", "readout_plus_feedback_",
             "adapter8_t1_")
        )}
    return rows, sha(source)


def archive_generations(prior):
    source = prior/"results/generation/GENERATION_RESULTS.jsonl"
    records = load_jsonl(source)
    indexed = {}
    for row in records:
        if row["method"] not in REFERENCE:
            continue
        indexed.setdefault((row["split"],row["method"],row["depth"]),[]).append(row)
    return indexed, sha(source)


def cluster(base, new):
    delta = np.asarray(new, dtype=float)-np.asarray(base, dtype=float)
    n = len(delta)
    means = delta.mean(axis=1)
    rng = np.random.default_rng(261123377)
    sims = np.concatenate([means[rng.integers(n,size=(500,n))].mean(axis=1) for _ in range(20)])
    l2,l5,hi = np.quantile(sims,(.025,.05,.975))
    return dict(n_questions=n,n_outcomes=delta.size, net_pp=100*float(delta.mean()),
                ci95_pp=[100*float(l2),100*float(hi)], lower_one_sided_95_pp=100*float(l5),
                rescue=int((delta>0).sum()), damage=int((delta<0).sum()))


def candidate_correct(rows, method, metric):
    mask = []
    for r in rows:
        results = []
        for k in (2,3,4):
            score = np.asarray(r["scores"][f"{method}_k{k}"])
            if metric == "acc_norm":
                score = score/r["characters"]
            results.append(int(score.argmax()) == r["gold"])
        mask.append(results)
    return np.asarray(mask)


def paired_candidate(rows, best):
    result = {}
    for metric in METRICS:
        a,nat,b = [candidate_correct(rows,name,metric) for name in ("linear8_t1","native",best)]
        rg = int(a.sum())-int(nat.sum())
        bg = int(b.sum())-int(nat.sum())
        pair = cluster(a,b)
        pair["gain_retention"] = bg/rg if rg>0 else None
        pair["rank8_gain_answers"] = rg
        pair["controller_gain_answers"] = bg
        result[metric] = pair
    return result


def rank8_comparisons(rows, summary, winners, methods):
    gold = np.array([r["gold"] for r in rows])
    for method in methods:
        for reference in REFERENCE[1:]:
            contrasted = []
            for metric in METRICS:
                for k in DEPTHS:
                    tag = f"{method}_k{k}"
                    old = f"{reference}_k{k}"
                    masks = [np.array([r[metric][n] for r in winners])==gold for n in (old,tag)]
                    pair = paired(*masks)
                    summary["metrics"][metric]["methods"][tag][f"vs_{reference}"] = pair
                    contrasted.append(pair)
            holm(contrasted)


def collect_generation(split, tag, depth, data, model, tok, net, work, source):
    root = work/"GENERATION_RESULTS"
    named = root/f"{split}__{tag}__k{depth}"
    path = named.with_suffix(".jsonl")
    cost_path = named.with_suffix(".json")
    if path.exists() and cost_path.exists():
        return load_jsonl(path), read(cost_path)
    if split == "arc_validation":
        old = work/"validation"/f"{tag}_generation_k{depth}.jsonl"
        rows = load_jsonl(old)
        c = source[tag]["generation"]["K1_K4"][str(depth)]["cost"]
        assert len(rows) == 299
    else:
        rows,c = run_slice(split,tag,depth,data,16,6,model,tok,None,Arm("minimal",net,1.))
    with path.open("w",encoding="utf-8") as output:
        for r in rows:
            output.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n")
    dump(cost_path,c)
    return rows,c


def check_pairs(generated):
    align = {k: {x["id"]:x for x in v} for k,v in generated.items()}
    native = align["native"]
    for m, rows in align.items():
        assert rows.keys() == native.keys() and len(rows)==len(generated[m])
        assert all(all(x[k] == native[i][k] for k in ("prompt","gold","options")) for i,x in rows.items())
    return align


def generation_statistics(all_runs, methods, historical_reference):
    records = {}
    primary = {}
    for split, by_depth in all_runs.items():
        depths = {d:check_pairs(rows) for d,rows in by_depth.items()}
        records[split] = {}
        primary[split] = {}
        ids = sorted(depths[1]["native"])
        for d, arms in depths.items():
            s = records[split][f"k{d}"] = {}
            for method, outputs in arms.items():
                u = s[method] = dict(
                    n=len(ids), correct=sum(x["correct"] for x in outputs.values()),
                    valid=sum(x["valid"] for x in outputs.values()),
                    mean_tokens=float(np.mean([len(x["tokens"]) for x in outputs.values()])),
                    invalid_types=dict(Counter(x["invalid_category"] for x in outputs.values() if not x["valid"])),
                )
                if method in methods:
                    for field in ("correct","valid"):
                        for base in REFERENCE:
                            u[f"{field}_vs_{base}"] = paired(
                                np.array([arms[base][i][field] for i in ids]),
                                np.array([outputs[i][field] for i in ids]),
                            )
        for name in methods:
            base = historical_reference
            primary[split][name] = {a:cluster(
                [[depths[k][base][i][a] for k in (2,3,4)] for i in ids],
                [[depths[k][name][i][a] for k in (2,3,4)] for i in ids],
            ) for a in ("correct","valid")}
    return records,primary


@torch.no_grad()
def run(a):
    seal = read(a.out/"FREEZE.json")
    names = list(seal["weights"])
    nets = {}
    for name, m in seal["weights"].items():
        p = a.out/"weights"/m["file"]
        assert sha(p)==m["sha256"]
        nets[name] = load_controller(p)
    best = seal["primary"]
    old, model, tok, metadata = runtime(a.archive,a.previous,a.model)
    phases = {}
    timing = {}
    for key in ("BENCHMARK_RESULTS","GENERATION_RESULTS","test_progress"):
        (a.out/key).mkdir(parents=True,exist_ok=True)
    bench_path = a.out/"BENCHMARK_RESULTS"
    for name,raw,count in (("ARC",old.read_data(a.corpus,"test"),1172),
                           ("OPENBOOKQA",pq.read_table(a.external).to_pylist(),500)):
        assert len(raw)==count
        archive, digest = archived_scores(a.prior,name)
        output = [archive]
        timing[name] = {}
        for method,net in nets.items():
            p = a.out/"test_progress"/f"{name}_{method}.jsonl"
            cp = p.with_suffix(".json")
            if p.exists() and cp.exists():
                records,c = load_jsonl(p), read(cp)
            else:
                records,c = evaluate(old,model,tok,raw,net,1.)
                write_jsonl(p,records)
                dump(cp,c)
            timing[name][method] = c
            for row in records:
                row["scores"] = {f"{method}_{tag}":s for tag,s in row["scores"].items()}
            output.append(records)
            print(name,method,len(records),flush=True)
        full = merge(output)
        summary, winners = statistic(full,names)
        rank8_comparisons(full,summary,winners,names)
        for row,pred in zip(full,winners,strict=True):
            row["predictions"] = {m:pred[m] for m in METRICS}
        write_jsonl(bench_path/f"RESULTS_{name}.jsonl",full)
        phases[name] = dict(summary=summary,primary=paired_candidate(full,best),prior_scores_sha256=digest)
    dump(bench_path/"BENCHMARK.json",phases)
    dump(bench_path/"COST.json",timing)
    datasets, data_receipt = read_datasets(old,tok,a.corpus,a.external)
    archive, gen_sha = archive_generations(a.prior)
    screens = read(a.out/"VAL_SCREEN.json")
    total = {}
    all_rows = []
    costs = {}
    gen_path = a.out/"GENERATION_RESULTS"
    generation_ref = max(("linear8_t1","linear8_t2"),key=lambda n:seal["reference"][n]["correct"])
    for split,rows in datasets.items():
        total[split] = {}
        for depth in DEPTHS:
            witnesses = {method: archive[(split,method,depth)] for method in REFERENCE}
            for method,net in nets.items():
                output,c = collect_generation(split,method,depth,rows,model,tok,net,a.out,screens)
                witnesses[method] = output
                costs[f"{split}__{method}__k{depth}"] = c
                print("generate",split,method,depth,len(output),flush=True)
            total[split][depth] = witnesses
            all_rows.extend(r for rows in witnesses.values() for r in rows)
    by_depth, pairs = generation_statistics(total,names,generation_ref)
    write_jsonl(gen_path/"GENERATION_RESULTS.jsonl",all_rows)
    dump(gen_path/"GENERATION_SUMMARY.json",dict(by_depth=by_depth,paired_with_prior=pairs,
         prior_generation_sha256=gen_sha, pooled_reference=generation_ref, data_receipt=data_receipt))
    dump(gen_path/"COST.json",costs)
    gates = []
    for p in phases.values():
        for e in p["primary"].values():
            gates.append(e["gain_retention"] is not None and e["gain_retention"]>=.8
                         and e["lower_one_sided_95_pp"]>-5)
    for split in pairs:
        d = pairs[split][best]
        gates.extend(d[k]["lower_one_sided_95_pp"]>-5 for k in ("correct","valid"))
    dump(a.out/"SEVEN_TASK_GATE.json",dict(primary=best, passed=all(gates),checks=gates,
         priority="Independent transfer & mechanism confirmation precede optional seven-task scoring",
         freeze_sha256=sha(a.out/"FREEZE.json")))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("archive","previous","model","corpus","external","prior","out"):
        p.add_argument(f"--{name}",type=Path,required=True)
    run(p.parse_args())
