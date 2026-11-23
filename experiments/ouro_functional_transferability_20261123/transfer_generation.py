"""Greedy Round 2 option-label diagnostic with one frozen controller per source."""

import argparse
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np
import torch

from dataset_transfer import load_cohort
from depth_holdout import read, read_rows, require_lock, save, sha


DEPTH = 4
MODEL_SEED = 261123137
BOOT_SEED = 261123702
COHORTS = {"arc_round2_confirm": 70, "csqa_round2_confirm": 299}
SUITES = {"arc": "depth", "csqa": "source", "mixed": "source"}
ENDPOINTS = ("correct", "valid", "strict_format", "strict_correct", "strict_single_label", "strict_single_label_correct")
COMPARISONS = (("arc","native"), ("csqa","native"), ("mixed","native"),
               ("csqa","arc"), ("mixed","arc"), ("mixed","csqa"))


def verified(a):
    forecast = a.results / "forecast_round2"
    forecast_lock, prediction = forecast / "FORECAST_LOCK.json", forecast / "PREDICTIONS.json"
    seal = read(forecast_lock)
    if sha(prediction) != seal["predictions_sha256"]:
        raise RuntimeError("The Round 2 forecast changed after lock")
    train_sets, receipts, paths = {}, {}, {}
    for arm, suite in SUITES.items():
        source = a.results / "HOLDOUT" / arm / suite / f"s{MODEL_SEED}"
        packet_path = source / "TRAIN_PACKET.json"
        lock_path = source / "TRAIN_LOCK.json"
        fit = source / "k234"
        paths[arm] = fit / "controller.pt"
        fit_receipt = read(fit / "TRAIN.json")
        packet = read(packet_path)
        population = read(lock_path)
        digest = sha(paths[arm])
        if (digest not in seal["controller_sha256s"] or digest != fit_receipt["checkpoint_sha256"]
                or digest != packet["checkpoints"]["k234"]["checkpoint_sha256"]
                or packet["train_lock_sha256"] != sha(lock_path)
                or fit_receipt["lock_sha256"] != sha(lock_path)
                or fit_receipt["source"] != arm or fit_receipt["seed"] != MODEL_SEED
                or fit_receipt["trained_depths"] != [2,3,4] or fit_receipt["rank_one_parameters"] != 4096
                or fit_receipt["actual_unique_questions"] != 1119):
            raise RuntimeError(f"{arm} is not the sealed Round 2 scratch checkpoint")
        train_sets[arm] = set(population["training_ids"])
        receipts[arm] = dict(checkpoint_sha256=digest, train_receipt_sha256=sha(fit/"TRAIN.json"),
            train_packet_sha256=sha(packet_path), train_lock_sha256=sha(lock_path),
            model_sha256=population["runtime"]["model_sha256"], steps=fit_receipt["steps"],
            initial_state_sha256=fit_receipt["initial_state_sha256"])
    for field in ("initial_state_sha256","model_sha256","steps"):
        if len({r[field] for r in receipts.values()}) != 1:
            raise RuntimeError(f"The three scratch arms do not share the intended {field}")
    data = {}
    manifest = read(a.data_manifest)
    for name, count in COHORTS.items():
        item = manifest["cohorts"][name]
        if item["n"] != count or item["sha256"] != seal["evaluation_sha256s"][name]:
            raise RuntimeError(f"Round 2 population changed for {name}")
        raw = load_cohort(a.data_manifest, name)
        if any({row["id"] for row in raw} & group for group in train_sets.values()):
            raise RuntimeError(f"Round 2 generation overlaps a controller's training set: {name}")
        data[name] = raw
    frozen_data = dict(data_manifest_sha256=sha(a.data_manifest), data={n: manifest["cohorts"][n] for n in COHORTS},
        round2_forecast_lock_sha256=sha(forecast_lock), round2_predictions_sha256=sha(prediction), checkpoints=receipts)
    return data, paths, frozen_data


def prompt_rows(raw, tokenizer, minimal):
    items = []
    for row in raw:
        prompt, labels, gold = minimal.label_prompt(row)
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        if not encoded:
            raise RuntimeError(f"Empty canonical input for {row['id']}")
        items.append(dict(id=row["id"],prompt=prompt,prompt_ids=encoded,options=labels,gold=gold,
                          official_labels=row["choices"]["label"]))
    return sorted(items,key=lambda row:(len(row["prompt_ids"]),row["id"]))


def strict_metrics(row):
    row = dict(row)
    literal = row["text"].strip()
    row["strict_single_label"] = literal in row["options"]
    row["strict_single_label_correct"] = literal==row["gold"]
    row["strict_correct"] = row["strict_format"] and row["correct"]
    return row


def paired_metrics(rows, peer):
    reference = {row["id"]:row for row in peer}
    if len(reference) != len(peer) or len(rows) != len(peer) or {row["id"] for row in rows} != reference.keys():
        raise RuntimeError("Generation has mismatched unique paired questions")
    aligned = [reference[row["id"]] for row in rows]
    for a,b in zip(rows,aligned,strict=True):
        if any(a[key]!=b[key] for key in ("gold","options","prompt","official_labels")):
            raise RuntimeError(f"Paired greedy prompt or official answer changed: {a['id']}")
    n = len(rows)
    rng = np.random.default_rng(BOOT_SEED)
    boot = rng.integers(0,n,size=(10_000,n),dtype=np.int32)
    output = {}
    for name in ENDPOINTS:
        actual = np.asarray([r[name] for r in rows],dtype=np.int8)
        baseline = np.asarray([r[name] for r in aligned],dtype=np.int8)
        delta = actual-baseline
        output[name] = dict(reference_count=int(baseline.sum()), target_count=int(actual.sum()),
            rescues=int((delta>0).sum()),damages=int((delta<0).sum()),delta_pp=100*float(delta.mean()),
            paired_ci95_pp=(100*np.quantile(delta[boot].mean(1),[.025,.975])).tolist())
    overlap = np.asarray([a["valid"] and b["valid"] for a,b in zip(rows,aligned,strict=True)],dtype=bool)
    output["both_valid_descriptive"] = dict(n=int(overlap.sum()),
        target_correct=sum(row["correct"] for row,keep in zip(rows,overlap,strict=True) if keep),
        reference_correct=sum(row["correct"] for row,keep in zip(aligned,overlap,strict=True) if keep))
    output["identical_greedy_tokens"] = sum(a["tokens"]==b["tokens"] for a,b in zip(rows,aligned,strict=True))
    return output


def report(methods, receipt):
    return {arm:dict(n=len(rows),endpoints={m:sum(r[m] for r in rows) for m in ENDPOINTS},
        invalid_categories=dict(Counter(r["invalid_category"] for r in rows if not r["valid"])),
        stop_reasons=dict(Counter(r["stop"] for r in rows)), receipt=receipt[arm])
        for arm,rows in methods.items()}


@torch.inference_mode()
def run(a, data, checkpoints, provenance):
    sys.path.insert(0,str(a.minimal.resolve()))
    import generation
    import minimal

    old,model,tokenizer,runtime = minimal.runtime(a.archive,a.previous,a.model)
    model.eval().requires_grad_(False)
    for receipt in provenance["checkpoints"].values():
        if receipt["model_sha256"] != runtime["model_sha256"]:
            raise RuntimeError("Generation uses a different Ouro than scratch training")
    arms = {"native":generation.Arm("native")}
    for name, path in checkpoints.items():
        net = minimal.load_net(path).requires_grad_(False)
        if (net.kind,net.rank,net.bound,sum(p.numel() for p in net.parameters())) != ("linear",1,.45,4096):
            raise RuntimeError(f"Unexpected scratch-controller shape for {name}")
        arms[name] = generation.Arm("minimal",net,1.)
    manifest = a.out/"GENERATION_LOCK.json"
    a.out.mkdir(parents=True,exist_ok=True)
    require_lock(manifest,dict(**provenance, runtime=runtime, depth=DEPTH, seed=MODEL_SEED,
        scope="Unrestricted vocabulary greedy, no cache, inherited EOS and internal-newline stops.",
        literal_strict="Decoded string stripped of whitespace exactly equals one available uppercase option letter.",
        batch_size=a.batch_size,max_new_tokens=a.max_new,code_sha256={p.name:sha(p) for p in
            (Path(__file__),Path(generation.__file__),Path(minimal.__file__),
             Path(__file__).with_name("dataset_transfer.py"),Path(__file__).with_name("depth_holdout.py"))}))
    outcomes = {}
    began = time.perf_counter()
    for split, raw in data.items():
        prompts = prompt_rows(raw,tokenizer,minimal)
        methods, receipts = {}, {}
        for name, arm in arms.items():
            out = a.out / f"{split}__{name}__k{DEPTH}.jsonl"
            receipt = out.with_suffix(".json")
            if receipt.exists():
                costs=read(receipt)
                if costs["prediction_sha256"]!=sha(out):
                    raise RuntimeError(f"Corrupt resumed generation output {out}")
                rows=read_rows(out)
            else:
                rows, cost = generation.run_slice(split,name,DEPTH,prompts,a.batch_size,a.max_new,model,tokenizer,None,arm)
                rows=[strict_metrics(row) for row in rows]
                generation.write_jsonl(out,rows)
                costs=dict(prediction_sha256=sha(out),cost=cost)
                save(receipt,costs)
            if [q["id"] for q in rows] != [q["id"] for q in prompts]:
                raise RuntimeError(f"Greedy question coverage differs at {split}/{name}")
            methods[name],receipts[name]=rows,costs
            print("transfer_greedy",split,name,sum(x["correct"] for x in rows),len(rows),flush=True)
        comparisons={f"{left}_vs_{right}":paired_metrics(methods[left],methods[right]) for left,right in COMPARISONS}
        outcomes[split]=dict(arms=report(methods,receipts),comparisons=comparisons)
        save(a.out/"GENERATION_SUMMARY.json",dict(lock_sha256=sha(manifest),elapsed_seconds=time.perf_counter()-began,
             results=outcomes,boot_draws=10_000,all_cohorts_complete=len(outcomes)==len(COHORTS)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for option in ("archive","previous","model","minimal","results","data-manifest","out"):
        p.add_argument(f"--{option}",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--max-new",type=int,choices=range(8,13),default=8)
    p.add_argument("--verify-only",action="store_true")
    a=p.parse_args()
    if a.batch_size<1:
        p.error("Batch size must be positive")
    data,checkpoints,provenance=verified(a)
    if a.verify_only:
        print("CPU verified", {name:len(rows) for name,rows in data.items()},
              {name:value["checkpoint_sha256"] for name,value in provenance["checkpoints"].items()},flush=True)
    else:
        run(a,data,checkpoints,provenance)


if __name__ == "__main__":
    main()
