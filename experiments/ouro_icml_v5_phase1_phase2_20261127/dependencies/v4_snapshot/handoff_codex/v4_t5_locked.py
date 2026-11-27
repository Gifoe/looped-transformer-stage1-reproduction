"""Execute the unfinished V4 T5 Rank-1 design using the existing frozen-model runtime."""
import argparse
import bisect
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

import minimal
import v4_wave3_mixing_and_cert as original
from v4_common import (
    DEPTHS, EXT_DEV_DIR, RECIPE, UNLABELED_AUDIT, UNLABELED_FIT, FastUnifiedDevEvaluator,
    cloze_option_logits_differentiable, evaluate_unlabeled_manifold, extract_q_opts_gold,
    forward_with_hook_all_tokens, load_base_runtime, load_jsonl,
    load_minimal_controller, make_candidate_row, save_json, save_rank1_controller, sha256_file,
)

V3 = Path("/root/ouro-v3-controllability-20261126")
SOURCES = ("arc_challenge","boolq","csqa","hellaswag","medmcqa","obqa","piqa","qasc",
           "race","scienceqa","social_iqa","winogrande")
TOP = ("arc_challenge","social_iqa","obqa","race","scienceqa","piqa")
NONARC = ("social_iqa","obqa","race","scienceqa","piqa","hellaswag")
SEEDS = (26112801,26112802)
STEPS = 1050
FOR_SLOTS = {
    "old0": ("T5_Uniform_All12","T5_CFIT_Top6_4096","T5_Poison_MedOnly_Raw"),
    "dual0": ("T5_CFIT_BLUE_All12","T5_CFIT_Top6_Trust_4096","T5_CFIT_Top6_PlusMMLUCloze"),
    "dual1": ("T5_CFIT_NonARC_4096","T5_Poison_Raw","T5_Poison_CFIT_Rescued"),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(root):
    folder = root / "handoff_codex"
    target = folder / "T5_TRAINING_LOCK.json"
    if target.exists():
        existing = read(target)
        if existing["runner_sha256"] != sha256_file(Path(__file__)):
            raise ValueError("The sealed T5 runner has changed")
        return
    features = root / "cfit_14_features/CFIT_14_SOURCE_FEATURES.json"
    scores = read(features)["per_source"]
    operator_file = root / "cfit_14_features/SOURCE_OPERATORS_14.pt"
    ops = torch.load(operator_file, map_location="cpu", weights_only=False)
    sigma_file = root / "SIGMA_H.pt"
    sigma = torch.load(sigma_file, map_location="cpu", weights_only=False)["sigma_unlab"].double()
    eigen, axes = torch.linalg.eigh((sigma + sigma.T) / 2)
    eig = eigen.clamp_min(0) + float(eigen.mean()) * .05
    if eig.min() <= 0:
        raise ValueError("The archived covariance is degenerate")
    anchor_file = root / "V4_SELF_ANCHOR.pt"
    anchor = torch.load(anchor_file, map_location="cpu", weights_only=False)
    if anchor["human_labels_used"]:
        raise ValueError("T5 BLUE must use the sealed zero-label anchor")
    u_self, w_self = (anchor[name].double().flatten() for name in ("u_self","w_self"))
    nrm = float(u_self.norm() * ((axes.T @ w_self).square() @ eig).sqrt())
    if nrm <= 0:
        raise ValueError("The pre-registered zero-label anchor is degenerate")
    fold_plan = read(V3 / "stage4_d3/FORECAST_INPUTS.json")
    fitted = {}
    for source in SOURCES:
        item = ops[source]
        u, w = item["u_ord"].double(), item["w_ord"].double()
        magnitude = float(item["s_ord"])
        beta = magnitude * float(u @ u_self) * float(w @ w_self) / nrm
        norm_sq = (magnitude * float(u.norm())) ** 2 * float(((axes.T @ w).square() / eig).sum())
        eta_sq = max(0., norm_sq - beta*beta)
        folds = []
        for name in fold_plan[source]["shards"]:
            part = torch.load(name, map_location="cpu", weights_only=False)
            n = int(part["processed"])
            mean = part["sums"]["ordinary"].double() / n
            projection = float(u_self @ mean @ w_self) / nrm
            folds.append((n, projection))
        n_total = sum(n for n, _ in folds)
        center = sum(n * value for n, value in folds) / n_total
        variance = sum(n * (value - center) ** 2 for n, value in folds) / (len(folds)-1) / n_total
        proxy = max(0., scores[source]["P_deconf_op_sig"]) * scores[source]["M_pres_ord"]
        weight = max(0., beta) / max(eta_sq + variance, 1e-12) if proxy > .01 else 0.
        pool = original.resolve_pool_path(source)
        regime = root / "cfit_14_features/regime3_samples" / f"{source}.jsonl"
        fitted[source] = dict(beta_hat=beta, eta_sq_hat=eta_sq, sampling_variance_hat=variance, used=n_total,
                              proxy=proxy, normalized_candidate=weight, pool_sha256=sha256_file(pool),
                              regime_sha256=sha256_file(regime))
    total = sum(v["normalized_candidate"] for v in fitted.values())
    if total <= 0:
        raise ValueError("No eligible positive BLUE estimate")
    for entry in fitted.values():
        entry["blue_weight"] = entry.pop("normalized_candidate") / total
    audit_path = folder / "T5_NONARC_AUDIT.json"
    if not audit_path.exists():
        raise ValueError("The zero-ARC cohort must be locked before launch")
    data = {
        "nonarc_audit_sha256":sha256_file(audit_path),
        "status":"FROZEN_BEFORE_T5_TRAINING", "runner_sha256":sha256_file(Path(__file__)),
        "inputs_sha256":{path.name:sha256_file(path) for path in (features,operator_file,sigma_file,anchor_file)},
        "protocol":"Rank-1 cloze CE, matched 1050 updates and effective batch 4, adaptive microbatch, two fixed seeds",
        "sources":fitted, "arms":FOR_SLOTS,"seeds":SEEDS,"steps":STEPS,
        "beta":"signed projection of archived compact ordinary moment onto the normalized zero-label anchor",
        "eta":"orthogonal norm under right covariance whitening of the same compact moment",
        "sampling_variance":"four-fold mean variation of the full ordinary moment projected on the zero-label anchor",
        "assumption":"diagonal contamination/noise working model; denominator estimates are proxies",
        "eligible":"archived CFIT zero-reference proxy > .01 and positive beta for BLUE",
        "blue_sample_scope":"archived Regime-III probed cohort only; every sampled row weighted by its recorded w_iii_mc",
        "selection":"mean over seeds on common dev macro with MMLU-158 replacing overlapping MMLU-658; choose the stronger seed inside the winning eligible arm",
        "official_test_used":False,
    }
    save_json(target, data)
    print("V4_T5_RECIPE_FROZEN",sha256_file(target),flush=True)


def raw_item(row, source):
    question, options, gold = extract_q_opts_gold(row)
    return {"id":row["sample_id"],"question":question,"options":options,"gold":gold,"dataset":source}


def setup_arm(root, lock, name):
    weights = lock["sources"]
    known = {}

    def pool(source, mode="full"):
        key = (source, mode)
        if key in known:
            return known[key]
        path = original.resolve_pool_path(source)
        if sha256_file(path) != weights[source]["pool_sha256"]:
            raise ValueError("A registered source population changed")
        samples = {row["sample_id"]:row for row in load_jsonl(path)}
        if name=="T5_CFIT_NonARC_4096":
            audit=root/"handoff_codex/T5_NONARC_AUDIT.json"
            if sha256_file(audit)!=lock["nonarc_audit_sha256"]:
                raise ValueError("The zero-ARC exclusion set has changed")
            record=read(audit)["sources"][source]
            if record["pool_sha256"]!=weights[source]["pool_sha256"] or record["n"]!=len(samples):
                raise ValueError("The zero-ARC exclusion population is inconsistent")
            for row in record["excluded"]:
                del samples[row["id"]]
        if mode != "full":
            p = root / "cfit_14_features/regime3_samples" / f"{source}.jsonl"
            if sha256_file(p) != weights[source]["regime_sha256"]:
                raise ValueError("The Regime-III population fingerprint changed")
            observed = load_jsonl(p)
            if mode == "top512":
                observed = sorted(observed,key=lambda row:(-row["w_iii_mc"],row["sample_id"]))[:512]
            selected = [(samples[row["sample_id"]],row["w_iii_mc"] if mode=="weighted" else 1.) for row in observed]
        else:
            selected = [(row,1.) for row in samples.values()]
        items = [(raw_item(row,source),weight) for row,weight in selected]
        items = [(item,weight) for item,weight in items if weight>0 and 2<=len(item["options"])<=10]
        if not items:
            raise ValueError(f"No eligible rows in {source}/{mode}")
        known[key] = items
        return items

    if name == "T5_Uniform_All12":
        chosen = [(source,float(len(pool(source))),pool(source)) for source in SOURCES]
    elif name == "T5_CFIT_BLUE_All12":
        chosen = [(source,values["blue_weight"],pool(source,"weighted"))
                  for source,values in weights.items() if values["blue_weight"]>0]
    elif name.startswith("T5_CFIT_"):
        allowed = NONARC if name=="T5_CFIT_NonARC_4096" else TOP
        chosen = [(source,weights[source]["proxy"],pool(source)) for source in allowed]
        if name=="T5_CFIT_Top6_PlusMMLUCloze":
            practice = [raw_item(row,"mmlu500_clz") for row in load_jsonl(EXT_DEV_DIR/"mmlu_validation.jsonl")]
            chosen.append(("mmlu500_clz",.20*sum(w for _,w,_ in chosen),[(row,1.) for row in practice]))
    elif name in ("T5_Poison_MedOnly_Raw","T5_Poison_CFIT_Rescued"):
        mode = "top512" if name.endswith("Rescued") else "full"
        chosen = [("arc_challenge",.25,pool("arc_challenge")),("csqa",.25,pool("csqa")),
                  ("medmcqa",.50,pool("medmcqa",mode))]
    elif name=="T5_Poison_Raw":
        chosen = [(source,weight,pool(source)) for source,weight in
                  zip(("arc_challenge","csqa","boolq","medmcqa","winogrande"),(.25,.25,.17,.17,.16))]
    else:
        raise ValueError(f"Unknown registered Rank-1 arm {name}")
    trust = {"T5_CFIT_BLUE_All12":.25, "T5_CFIT_Top6_PlusMMLUCloze":.25,
             "T5_CFIT_Top6_Trust_4096":.5}.get(name,0.)
    return [(s,w,items) for s,w,items in chosen if w>0],trust


def train(archive, model, tokenizer, bundle, arm, seed, trust, text, checkpoint, lock_sha):
    torch.manual_seed(seed)
    controller = minimal.Minimal("linear",1,.45).to(model.device).requires_grad_(True)
    optimizer = torch.optim.AdamW(controller.parameters(),lr=RECIPE["learning_rate"],weight_decay=RECIPE["weight_decay"])
    matched = arm in ("T5_Poison_MedOnly_Raw","T5_Poison_CFIT_Rescued")
    stream = "medmcqa_matched_poison" if matched else arm
    source_rng = random.Random(f"t5/{stream}/{seed}/source")
    item_rngs = [random.Random(f"t5/{stream}/{seed}/{source}") for source,_,_ in bundle]
    groups, cumulative = [], []
    for _,weight,items in bundle:
        item_rows, item_weights = zip(*items)
        item_cum = []
        for value in item_weights:
            item_cum.append((item_cum[-1] if item_cum else 0.) + value)
        groups.append((item_rows,item_cum))
        cumulative.append((cumulative[-1] if cumulative else 0.)+weight)
    start, elapsed, losses, n_forwards = 0, 0., [], 0
    if checkpoint.exists():
        pre = torch.load(checkpoint,map_location="cpu",weights_only=False)
        if pre["lock_sha256"]!=lock_sha:
            raise ValueError("Refuse to resume a T5 job from different inputs")
        controller.load_state_dict(pre["state"])
        optimizer.load_state_dict(pre["optim"])
        source_rng.setstate(pre["source_rng"])
        for rng,state in zip(item_rngs,pre["item_rngs"],strict=True):
            rng.setstate(state)
        start,elapsed,losses,n_forwards = pre["step"],pre["seconds"],pre["losses"],pre["score_microbatches"]
    teacher_cache = {}

    def penalty(seq, depth):
        ids = torch.tensor(seq,device=model.device,dtype=torch.long).unsqueeze(0)
        mask = torch.ones_like(ids)
        target = ids[:,1:].flatten()
        key = depth,tuple(seq)
        if key not in teacher_cache:
            with torch.no_grad():
                hidden = forward_with_hook_all_tokens(model,ids,mask,depth,net=None)
                logits = model.lm_head(hidden[:,:-1]).float().squeeze(0)
                teacher_cache[key] = (float(F.cross_entropy(logits,target)),
                                      logits.softmax(-1)[:,tokenizer.eos_token_id].cpu())
        hidden = forward_with_hook_all_tokens(model,ids,mask,depth,net=controller)
        logits = model.lm_head(hidden[:,:-1]).float().squeeze(0)
        default, native_eos = teacher_cache[key]
        margin = .50 if arm=="T5_CFIT_Top6_Trust_4096" else .55
        change = (logits.softmax(-1)[:,tokenizer.eos_token_id]-native_eos.to(logits.device)).abs().mean()
        return F.relu(F.cross_entropy(logits,target)-default-margin) + change

    begin=time.perf_counter()
    for step in range(start+1,STEPS+1):
        chosen = []
        for _ in range(RECIPE["effective_batch"]):
            pos=bisect.bisect_right(cumulative,source_rng.random()*cumulative[-1])
            items, weights=groups[pos]
            selected=bisect.bisect_right(weights,item_rngs[pos].random()*weights[-1])
            chosen.append(items[selected])
        length=max(len(it["question"])+max(map(len,it["options"])) for it in chosen)
        options=max(len(it["options"]) for it in chosen)
        size=1 if length>1400 or length*options>5200 else (2 if length>650 or length*options>2600 else len(chosen))
        depth=DEPTHS[(step-1)%len(DEPTHS)]
        optimizer.zero_grad(set_to_none=True)
        total=0.
        with torch.enable_grad():
            for start_row in range(0,len(chosen),size):
                subset=chosen[start_row:start_row+size]
                crows=[make_candidate_row(it["id"],it["question"],it["options"],it["gold"],dataset=it["dataset"]) for it in subset]
                logits,_,_=cloze_option_logits_differentiable(archive,model,tokenizer,crows,depth,controller)
                loss=sum(F.cross_entropy(x.unsqueeze(0),torch.tensor([it["gold"]],device=x.device))
                         for x,it in zip(logits,subset,strict=True))/len(chosen)
                loss.backward()
                n_forwards+=1
                total+=float(loss.detach())
            if trust and step%3==0:
                loss=trust*penalty(text[step%len(text)],depth)
                loss.backward()
                total+=float(loss.detach())
        grad=torch.nn.utils.clip_grad_norm_(controller.parameters(),RECIPE["clip"])
        if not grad.isfinite().item():
            raise FloatingPointError("A registered Rank-1 arm has non-finite gradients")
        optimizer.step()
        losses=(losses+[total])[-70:]
        if step%70==0 or step==STEPS:
            staging=checkpoint.with_suffix(".partial")
            torch.save(dict(state=controller.state_dict(),optim=optimizer.state_dict(),source_rng=source_rng.getstate(),
                            item_rngs=[rng.getstate() for rng in item_rngs],step=step,seconds=elapsed+time.perf_counter()-begin,
                            losses=losses,score_microbatches=n_forwards,lock_sha256=lock_sha),staging)
            staging.replace(checkpoint)
            print("T5_TRAIN",arm,seed,step,STEPS,round(elapsed+time.perf_counter()-begin,1),flush=True)
    return controller,dict(seconds=elapsed+time.perf_counter()-begin,steps=STEPS,n_params=sum(p.numel() for p in controller.parameters()),
                           score_microbatches=n_forwards,loss_final_mean=sum(losses)/len(losses))


def protected(scores):
    return (3*scores["cloze3"]+scores["mmlu_dev158"]["acc"]+scores["mmlu_pro_val"]["correct"])/5


def run(root, slot, arms=None):
    location=root/"handoff_codex/T5_TRAINING_LOCK.json"
    receipt=read(location)
    if receipt["status"]!="FROZEN_BEFORE_T5_TRAINING" or receipt["runner_sha256"]!=sha256_file(Path(__file__)):
        raise ValueError("A sealed, development-only T5 protocol is required")
    lock=sha256_file(location)
    out=root/"exp_t5_mixing"
    out.mkdir(exist_ok=True)
    cfg,archive,model,tokenizer=load_base_runtime()
    evaluator=FastUnifiedDevEvaluator(cfg,archive,model,tokenizer)
    native=evaluator.evaluate(None,include_mmlu_dev158=True)
    text=[tokenizer.encode(row["text"],add_special_tokens=False)[:192] for row in load_jsonl(UNLABELED_FIT)[:128]]
    for arm in arms or FOR_SLOTS[slot]:
        target=out/f"T5_{arm}.json"
        current=read(target) if target.exists() else {"arm":arm,"native":native,"lock_sha256":lock,"seeds":{}}
        if current["lock_sha256"]!=lock:
            raise ValueError("Do not overwrite an existing result from another T5 recipe")
        if all(f"{arm}.s{seed}" in current["seeds"] for seed in SEEDS):
            continue
        bundle,trust=setup_arm(root,receipt,arm)
        for seed in SEEDS:
            tag=f"{arm}.s{seed}"
            if tag in current["seeds"]:
                continue
            weights=out/f"{tag}.pt"
            saved=out/f"{tag}.TRAIN.json"
            if weights.is_file() and saved.is_file():
                run_details=read(saved)
                if sha256_file(weights)!=run_details["sha256"] or run_details["lock_sha256"]!=lock:
                    raise ValueError("Retained T5 checkpoint differs from its recorded training")
                controller=load_minimal_controller(weights,device=model.device)
            else:
                checkpoint=out/f"{tag}.resume.pt"
                controller,run_details=train(archive,model,tokenizer,bundle,arm,seed,trust,text,checkpoint,lock)
                fingerprint=save_rank1_controller(weights,controller,
                    {"arm":tag,"rank":1,"steps":STEPS,"seed":seed,"training_lock_sha256":lock})
                run_details.update(sha256=fingerprint,lock_sha256=lock)
                save_json(saved,run_details)
            scores=evaluator.evaluate(controller,include_mmlu_dev158=True)
            unlab=evaluate_unlabeled_manifold(model,tokenizer,controller,UNLABELED_AUDIT,max_docs=64)
            current["seeds"][tag]=dict(seed=seed,train=run_details,eval=scores,unlabeled=unlab,
                                       delta_macro5_pp=(scores["macro5"]-native["macro5"])*100,
                                       protected_macro5=protected(scores),delta_protected_pp=(protected(scores)-protected(native))*100,
                                       delta_cloze3_pp=(scores["cloze3"]-native["cloze3"])*100)
            save_json(target,current)
            print("T5_EVALUATED",tag,round(current["seeds"][tag]["delta_protected_pp"],4),flush=True)
        current["mean_protected_macro5"]=sum(seeded["protected_macro5"] for seeded in current["seeds"].values())/len(SEEDS)
        save_json(target,current)
    print("V4_T5_SLOT_DONE",slot,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--slot",choices=tuple(FOR_SLOTS))
    parser.add_argument("--freeze",action="store_true")
    parser.add_argument("--arms",nargs="+",choices=[arm for arms in FOR_SLOTS.values() for arm in arms])
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.freeze:
        freeze(args.root)
    else:
        run(args.root,args.slot,args.arms)


if __name__=="__main__":
    main()
