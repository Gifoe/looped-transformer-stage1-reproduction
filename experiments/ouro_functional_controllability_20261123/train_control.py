"""Fit bounded 2K–4K controls through the real frozen suffix, then freeze on ARC Val."""
import argparse
from pathlib import Path
import time
import json

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from controller import CAP, controller_size, load_controller, new_controller, save_controller
from generation import Arm, run_slice, write_jsonl
from minimal import DEPTHS, dump, forward, label_prompt, runtime, sha
from train import BATCH, SEED, count, evaluate, letter_targets, logprobs, partitions, ranking

FORMS = (
    ("fixed25", "fixed", "none", .25),
    ("fixed35", "fixed", "none", .35),
    ("fixed45", "fixed", "none", .45),
    ("hard_t1", "hard", "t1", CAP),
    ("smooth_t1", "smooth", "t1", CAP),
    ("smooth_t2", "smooth", "t2", CAP),
    ("rank1_t1", "rank1", "t1", CAP),
    ("rank1_t2", "rank1", "t2", CAP),
)


def prep_generation(raw, tokenizer):
    data = []
    for row in raw:
        prompt, options, gold = label_prompt(row)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        data.append(dict(id=row["id"], prompt=prompt, prompt_ids=ids, options=options,
                         gold=gold, official_labels=row["choices"]["label"]))
    return sorted(data, key=lambda row: (len(row["prompt_ids"]), row["id"]))


def fit(model, old, items, meta, labels, net, objective):
    if objective == "none":
        return dict(controller_size(net), steps=0, seconds=0.0, network_calls=0, executed_loop_calls=0)
    torch.manual_seed(SEED)
    order = np.random.default_rng(SEED).permutation(len(items)).tolist()
    for i in range(0, len(order), 64):
        order[i:i+64] = sorted(order[i:i+64], key=lambda q: max(len(x[0]) for x in items[q]))
    parameters = [x for x in net.parameters() if x.requires_grad]
    opt = torch.optim.AdamW(parameters, lr=.001, weight_decay=.03)
    calls = 0
    loop_calls = 0
    hist = []
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.enable_grad():
        for step, i in enumerate(range(0, len(order), BATCH)):
            selected = order[i:i+BATCH]
            depth = 1 + step % 4
            packed = old.pack([member for q in selected for member in items[q]])
            state, _ = forward(model, packed, depth, net)
            likelihood, logits = logprobs(model, state, packed[-1])
            del logits
            task_loss, content = ranking(likelihood, packed[1], meta, selected)
            loss = task_loss
            if objective == "t2":
                official = old.pack([labels[q] for q in selected])
                gen_state, _ = forward(model, official, depth, net)
                nll, logits = logprobs(model, gen_state, official[-1])
                del logits
                loss = .35 * content - .65 * nll.mean() + .3 * task_loss
                calls += 1
                loop_calls += depth
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(norm):
                raise RuntimeError(f"Nonfinite controller gradient at step {step}")
            opt.step()
            hist.append((loss.detach(), norm.detach()))
            calls += 1
            loop_calls += depth
    torch.cuda.synchronize()
    assert len(order) == 1119 and len(hist) == 280
    losses, norms = zip(*hist)
    return dict(controller_size(net), steps=len(hist), objective=objective,
                mean_loss=float(torch.stack(losses).mean()), mean_gradient_norm=float(torch.stack(norms).mean()),
                seconds=time.perf_counter()-start, network_calls=calls, executed_loop_calls=loop_calls,
                peak_allocated_bytes=torch.cuda.max_memory_allocated())


def archive_generation(prior):
    path = prior / "results/generation/GENERATION_SUMMARY.json"
    payload = json.loads(path.read_text(encoding="utf-8"))["runs"]
    methods = ("linear8_t1", "linear8_t2", "linear8_t2kl")
    result = {}
    for name in methods:
        x = [payload[f"arc_validation__{name}__k{k}"] for k in (2, 3, 4)]
        assert all(r["n"] == 299 for r in x)
        result[name] = {a: sum(r[a] for r in x)/(3*299) for a in ("correct", "valid")}
    upper = {field: max(x[field] for x in result.values()) for field in ("correct", "valid")}
    return upper, result, sha(path)


def freeze(a, trials, floors, history, source):
    def utility(value):
        x, y = value["candidate"], value["generation"]
        ranking_score = sum(sum(x[m][1:]) for m in ("acc", "acc_norm")) / (6 * 299)
        return .50*ranking_score + .35*y["correct"] + .15*y["valid"]

    def eligible(value):
        return (value["fit"]["stored_scalars"] <= 4096
                and all(value["generation"][k] >= floors[k] - .05 for k in floors))

    def pick(keys):
        return max(keys, key=lambda name: (eligible(trials[name]), utility(trials[name]), name))

    best = pick(trials)
    families = {
        "fixed": pick([x for x in trials if x.startswith("fixed")]),
        "adaptive": pick([x for x in trials if x.startswith(("hard", "smooth"))]),
        "rank1": pick([x for x in trials if x.startswith("rank1")]),
    }
    names = list(dict.fromkeys([best, *families.values()]))
    weights = {name: dict(file=f"{name}.pt", sha256=sha(a.out/"weights"/f"{name}.pt"),
                          form=trials[name]["form"], params=trials[name]["fit"]) for name in names}
    dump(a.out/"FREEZE.json", dict(
        primary=best, class_representatives=families, selection_eligible=eligible(trials[best]),
        weights=weights, reference=history, reference_sha256=source, predeclared_theory_sha256=sha(a.theory),
        prior_train_direction_sha256=sha(a.axis),
        utility={name: {"score": utility(item), "eligible": eligible(item)} for name,item in trials.items()},
        selection_data={"split":"ARC Validation", "n":299, "K2_to_K4_reference":floors,
                        "sha256":sha(a.corpus/"validation.parquet")},
    ))
    print("FROZEN", best, families, "eligible", eligible(trials[best]), flush=True)


@torch.no_grad()
def gen_trials(model, tok, raw, tag, net, location):
    data = prep_generation(raw, tok)
    detailed = {}
    for depth in DEPTHS:
        scores, cost = run_slice("arc_validation", tag, depth, data, 16, 6, model, tok, None, Arm("minimal",net,1.))
        assert len(scores) == 299
        write_jsonl(location/f"{tag}_generation_k{depth}.jsonl", scores)
        detailed[str(depth)] = dict(cost=cost, correct=sum(r["correct"] for r in scores),
                                   valid=sum(r["valid"] for r in scores), n=len(scores))
    pooled = {k: sum(detailed[str(d)][k] for d in (2,3,4))/(3*299) for k in ("correct","valid")}
    return dict(pooled, K1_K4=detailed)


def run(a):
    for folder in ("weights", "validation"):
        (a.out/folder).mkdir(parents=True,exist_ok=True)
    old, model, tok, info = runtime(a.archive, a.previous, a.model)
    dump(a.out/"MODEL_CONFIG.json", info)
    train = old.read_data(a.corpus, "train")
    val = old.read_data(a.corpus, "validation")
    assert len(train) == 1119 and len(val) == 299
    items, metadata = partitions(old,tok,train)
    labels = letter_targets(tok,train)
    threshold, archive, src = archive_generation(a.prior)
    results_path = a.out/"VAL_SCREEN.json"
    trials = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    for tag, form, objective, dose in FORMS:
        if tag in trials:
            continue
        start = time.perf_counter()
        weights = a.out/"weights"/f"{tag}.pt"
        net = new_controller(form,a.axis,dose).cuda()
        measured = fit(model,old,items,metadata,labels,net,objective)
        result, candidate_cost = evaluate(old,model,tok,val,net,1.)
        cand = count(result)
        write_jsonl(a.out/"validation"/f"{tag}_candidates.jsonl",result)
        gen = gen_trials(model,tok,val,tag,net,a.out/"validation")
        save_controller(weights,net,measured)
        trials[tag] = dict(form=form, objective=objective, candidate=cand, candidate_cost=candidate_cost,
                           generation=gen, fit=measured, total_seconds=time.perf_counter()-start)
        dump(results_path,trials)
        print(tag, cand, "generation", round(gen["correct"],4),round(gen["valid"],4),
              "seconds",round(trials[tag]["total_seconds"],2),flush=True)
    freeze(a,trials,threshold,archive,src)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for item in ("archive", "previous", "model", "corpus", "prior", "axis", "theory", "out"):
        p.add_argument(f"--{item}",type=Path,required=True)
    run(p.parse_args())
