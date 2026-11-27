"""Run only the locked V5 1A diagnostic, with no QA training and no gold export."""
import hashlib
import math
import random
import re
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from generation_followup_run import cache_type, forward as decode_forward, prepare
from v4_common import (load_base_runtime, load_jsonl, make_candidate_row, save_json, sha256_file,
    score_cloze_probs, score_mc_letter_rotation_probs, forward_with_hook_all_tokens)
from p1_gpu1_csbp_pure_prose import (
    format_stem, jaccard, kl_div, make_shuffled_quads, score_natural_cloze_probs,
)
from v5_prose_corpus import PRE, save_jsonl

DRAWS = 12
SAMPLE_SIZE = 300
MAX_NEW = 24
SEED = 26112761
TEMPLATES = ("T1_Direct", "T2_Blank", "T3_Passage")
CLOZE = ("ClozeQA", "ClozeNat")


def keyed(probe):
    return hashlib.sha256(f"{SEED}/{probe['id']}".encode()).digest()


def doc_sample(rows):
    docs, selected = set(), []
    for row in sorted(rows, key=keyed):
        if row["doc_id"] in docs:
            continue
        selected.append(row)
        docs.add(row["doc_id"])
        if len(selected) == SAMPLE_SIZE:
            break
    if len(selected) != SAMPLE_SIZE:
        raise ValueError("The corpus cannot support 300 document-separated samples")
    return selected


def active_logits(model, tokens, valid, cache, width, step):
    if step == 0:
        idx = torch.arange(width, device=tokens.device)
        return decode_forward(model, tokens, valid, idx, "native", None, 1., cache, slice(None))
    idx = torch.tensor([valid.shape[1]-1], dtype=torch.long, device=tokens.device)
    return decode_forward(model, tokens, valid, idx, "native", None, 1., cache, slice(None))


@torch.inference_mode()
def sample_tails(model, tokenizer, stubs, seed):
    create_cache = cache_type(model)
    items = [SimpleNamespace(ids=tokenizer.encode(stub, add_special_tokens=False))
             for stub in stubs for _ in range(DRAWS)]
    tokens, valid = prepare(items, tokenizer.eos_token_id)
    width = tokens.shape[1]
    cache = create_cache(width+MAX_NEW)
    model.model.total_ut_steps = 4
    rng = torch.Generator(device=tokens.device)
    rng.manual_seed(seed)
    ended = [False] * len(items)
    candidates, logp = [[] for _ in items], [[] for _ in items]
    for step in range(MAX_NEW):
        next_logit = active_logits(model, tokens, valid, cache, width, step)
        values, order = torch.sort(next_logit/.85, dim=-1, descending=True)
        p = values.softmax(-1)
        p = p.masked_fill(p.cumsum(-1)-p > .92, 0) 
        taken = order.gather(1, torch.multinomial(p, 1, generator=rng))
        sampled = taken[:, 0].tolist()
        prior = next_logit.log_softmax(-1).gather(1, taken)[:, 0].tolist()
        for i, token in enumerate(sampled):
            if ended[i]:
                continue
            candidates[i].append(token)
            logp[i].append(prior[i])
            text = tokenizer.decode(candidates[i], skip_special_tokens=True)
            ended[i] = token == tokenizer.eos_token_id or bool(re.search(r"[.;\n]", text))
        if all(ended):
            break
        tokens = taken
        valid = torch.cat((valid, torch.ones_like(taken, dtype=torch.bool)), dim=1)
    return [(tokenizer.decode(ids, skip_special_tokens=True), sum(lp)) for ids, lp in zip(candidates, logp, strict=True)]


def plausible_tail(text, original):
    leading = text.lstrip()
    match = re.match(r"^([^.;\n\r]+[.;])", leading)
    if not match:
        return None
    candidate = " ".join(match.group(1).split())
    if not .6*len(original) <= len(candidate) <= 1.6*len(original):
        return None
    if re.search(r'[\[\]{}@=<>•#]', candidate) or re.match(r"\d+[.)]|[A-D][.)]\s", candidate):
        return None
    if jaccard(original,candidate) > .45 or original.lower() == candidate.lower():
        return None
    return candidate


@torch.inference_mode()
def blind(model, tokenizer, probes, route):
    path = PRE / f"DIAG_{route}_G1_K4Blind.jsonl"
    current = load_jsonl(path) if path.exists() else []
    progress_file = PRE / f"DIAG_{route}_G1_K4Blind.resume.json"
    audit = PRE / "CORPUS_DEDUP_AUDIT.json"
    input_id = sha256_file(audit)
    at = 0
    if progress_file.exists():
        resume = __import__("json").loads(progress_file.read_text())
        if resume["corpus_audit_sha256"] != input_id:
            raise ValueError("Cannot mix blind candidates from different corpus audits")
        at = resume["completed"]
        kept = set(row["id"] for row in probes[:at])
        current = [row for row in current if row["id"] in kept]
    clock = time.monotonic()
    batch = 8
    for start in range(at,len(probes),batch):
        group = probes[start:start+batch]
        sampled = sample_tails(model,tokenizer,[row["stub"] for row in group],SEED+10000*(route=="B")+start)
        score_items, memberships = [], []
        for j, probe in enumerate(group):
            tails = {}
            for text, _ in sampled[j*DRAWS:(j+1)*DRAWS]:
                cand = plausible_tail(text,probe["c0"])
                if cand and cand.lower() not in tails:
                    tails[cand.lower()] = cand
            if len(tails)>=3:
                score_items.append((probe["stub"],list(tails.values())))
                memberships.append(probe)
        # Recompute each literal suffix under its K4 short tail. The posterior ranks the summed
        # continuation log probability divided by the character count through the first punctuation.
        if score_items:
            probabilities = score_natural_cloze_probs(model,tokenizer,score_items,depth=4,batch_size=4)
        else:
            probabilities = []
        for probe, (_, tails), ranks in zip(memberships, score_items, probabilities, strict=True):
            selected = []
            for t, _ in sorted(zip(tails,ranks,strict=True),key=lambda x:(-x[1],x[0])):
                if all(jaccard(t, prior)<=.45 for prior in selected):
                    selected.append(t)
                if len(selected)==3:
                    current.append({**probe,"distractors":selected})
                    break
        save_jsonl(path,current)
        stop = start+len(group)
        save_json(progress_file,dict(corpus_audit_sha256=input_id,completed=stop,retained=len(current)))
        print("V5_1A_BLIND",route,stop,len(probes),len(current),round(time.monotonic()-clock,1),flush=True)
    return current


_ART = {"a", "an", "the"}
_PRE = {"of","with","from","in","on","for","as","by","at","to","into","over","under","through"}
_PRO = {"it","its","his","her","their","this","that","these","those","they","he","she","which"}


def morph(text):
    first = re.sub(r"[^a-z]","",text.split()[0].lower())
    for group, name in ((_ART,"article"),(_PRE,"preposition"),(_PRO,"pronoun")):
        if first in group:
            return name
    return "gerund" if first.endswith("ing") else "nominal"


@torch.inference_mode()
def embed(model, tokenizer, rows):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    blocks = []
    for i in range(0,len(rows),48):
        texts = [row["c0"] for row in rows[i:i+48]]
        data = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt").to(model.device)
        h = forward_with_hook_all_tokens(model, data["input_ids"],data["attention_mask"],depth=4)
        mask = data["attention_mask"][:,:,None].float()
        value = (h.float()*mask).sum(1)/mask.sum(1).clamp_min(1.)
        blocks.append(torch.nn.functional.normalize(value,dim=-1).cpu())
    return torch.cat(blocks)


def retrieval(model, tokenizer, probes, complete, route):
    path = PRE / f"DIAG_{route}_G2_HumanRetr.jsonl"
    if path.exists():
        return load_jsonl(path)
    embeddings = embed(model, tokenizer, complete)
    matches = embed(model, tokenizer, probes) @ embeddings.mT
    records = []
    for i, probe in enumerate(probes):
        selected = []
        ref = probe["c0"]
        for idx in matches[i].argsort(descending=True,stable=True).tolist():
            item=complete[idx]
            c=item["c0"]
            if (item["doc_id"] == probe["doc_id"] or morph(c)!=morph(ref)
                    or not .7*len(ref)<=len(c)<=1.4*len(ref)
                    or c.lower()==ref.lower() or jaccard(c,ref)>.4
                    or any(jaccard(c,t)>.4 for t in selected)):
                continue
            selected.append(c)
            if len(selected)==3:
                records.append({**probe,"distractors":selected})
                break
    save_jsonl(path,records)
    print("V5_1A_RETRIEVAL",route,len(records),len(probes),flush=True)
    return records


def accumulate(quads, mc, clo):
    keep = []
    ag_mc, ag_cl = [], []
    confidences, kl = [], []
    for row, mp, cp in zip(quads,mc,clo,strict=True):
        order=sorted(mp,reverse=True)
        div=kl_div(mp,cp)
        confidences.append(order[0])
        kl.append(div)
        ag_mc.append(max(range(4),key=mp.__getitem__)==row["_orig_idx"])
        ag_cl.append(max(range(4),key=cp.__getitem__)==row["_orig_idx"])
        keep.append(order[0]>=.45 and order[0]-order[1]>=.25 and div>=.15)
    n = len(quads)
    k = sum(keep)
    diag = {"n":n,"n_filtered":k}
    if n:
        count_mc,count_cl=sum(ag_mc),sum(ag_cl)
        k_mc=sum(a and b for a,b in zip(keep,ag_mc,strict=True))
        k_cl=sum(a and b for a,b in zip(keep,ag_cl,strict=True))
        diag.update(raw_mc=count_mc/n,raw_cloze=count_cl/n,raw_gap_pp=100*(count_mc-count_cl)/n,
                    mean_conf=sum(confidences)/n,mean_kl=sum(kl)/n,yield_after_generation=k/SAMPLE_SIZE,
                    keep_rate=k/n, filtered_mc=k_mc/k if k else 0.,filtered_gap_pp=100*(k_mc-k_cl)/k if k else 0.)
    return diag


def evaluate(model, tokenizer, archive, probes, choices, route, generator, remaining):
    seed = SEED + 20*(route=="B") + (generator=="G2_HumanRetr")
    quads=make_shuffled_quads(choices,seed=seed)
    questions=[f"{q['context']} {q['lead']}".strip() for q in quads]
    if quads:
        pairs = [(q,row["options"]) for q,row in zip(questions,quads,strict=True)]
        _,qa=score_cloze_probs(archive,model,tokenizer,[make_candidate_row(q["id"],s,o,0) for q,(s,o) in zip(quads,pairs,strict=True)],depth=4,batch_size=8)
        nat=score_natural_cloze_probs(model,tokenizer,pairs,depth=4,batch_size=8)
        own_options=[row["options"] for row in quads]
        option_only=score_mc_letter_rotation_probs(model,tokenizer,[("Choose the best option:",o) for o in own_options],depth=4,batch_size=12)
        shuffle=score_mc_letter_rotation_probs(model,tokenizer,[(questions[(i+17)%len(quads)],o) for i,o in enumerate(own_options)],depth=4,batch_size=12)
        odds=lambda ps: sum(max(range(4),key=p.__getitem__)==q["_orig_idx"] for p,q in zip(ps,quads,strict=True))/len(quads)
        shortcuts={"option_only_original_recovery":odds(option_only),"shuffled_context_original_recovery":odds(shuffle)}
    else:
        qa=nat=[]
        shortcuts={}
    report={}
    for template in TEMPLATES:
        mc=score_mc_letter_rotation_probs(model,tokenizer,[(format_stem(row,template),row["options"]) for row in quads],
                                          depth=4,batch_size=12) if quads else []
        for name,p in zip(CLOZE,(qa,nat),strict=True):
            result=accumulate(quads,mc,p)
            projected=math.floor(remaining*result["n_filtered"]/len(probes))
            result.update(route=route,generator=generator,template=template,cloze=name,diagnostic_input=len(probes),
                          unseen_eligible_stems=remaining,projected_filtered=projected,**shortcuts)
            result["pass"]=(result["n"]>=250 and result.get("filtered_mc",0)>=.78
                            and result.get("filtered_gap_pp",0)>=20 and projected>=1000)
            report[f"{route}/{generator}/{template}/{name}"]=result
            print("V5_1A_DIAGNOSTIC",list(report)[-1],result,flush=True)
    return report


def main():
    torch.set_num_threads(4)
    audit=PRE/"CORPUS_DEDUP_AUDIT.json"
    lock=PRE/"PREFLIGHT_LOCK.json"
    tgt=PRE/"CSBP_PREFLIGHT_VERDICT.json"
    if tgt.exists():
        return
    cfg,archive,model,tokenizer=load_base_runtime()
    loaded={}
    for route in ("A","B"):
        path=PRE/f"DIAGNOSTIC_ONLY_{route}.jsonl"
        loaded[route]=[r for r in load_jsonl(path) if len(tokenizer.encode(r["context"],add_special_tokens=False))<=110]
    diag={route:doc_sample(rows) for route,rows in loaded.items()}
    all_diagnostic_doc={r["doc_id"] for part in diag.values() for r in part}
    remain={route:len({r["id"] for r in rows if r["doc_id"] not in all_diagnostic_doc}) for route,rows in loaded.items()}
    definition=dict(seed=SEED,model_sha256="58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af",
                    audited_corpus_sha256=sha256_file(audit),runner_sha256=sha256_file(Path(__file__)),
                    diagnostic_ids={route:[r["id"] for r in rows] for route,rows in diag.items()},
                    threshold=dict(min_quad=250,purity=.78,gap_pp=20,projected_filtered=1000),
                    unused_pool_sizes=remain,diagnostic_only=True)
    if lock.exists():
        if __import__("json").loads(lock.read_text())!=definition:
            raise ValueError("The locked diagnostic inputs have changed")
    else:
        save_json(lock, definition)
    matrices={}
    for route in ("A","B"):
        g1=blind(model,tokenizer,diag[route],route)
        g2=retrieval(model,tokenizer,diag[route],loaded[route],route)
        for name,rows in (("G1_K4Blind",g1),("G2_HumanRetr",g2)):
            metrics_file = PRE / f"METRICS_{route}_{name}.json"
            if metrics_file.exists():
                metrics = __import__("json").loads(metrics_file.read_text())
            else:
                metrics = evaluate(model,tokenizer,archive,diag[route],rows,route,name,remain[route])
                save_json(metrics_file,metrics)
            matrices.update(metrics)
    def priority(x):
        return (x["pass"], min(x["n"]/250,x.get("filtered_mc",0)/.78,
                               x.get("filtered_gap_pp",0)/20,x["projected_filtered"]/1000),
                x["projected_filtered"],x.get("filtered_gap_pp",0))
    ordered=sorted(matrices,key=lambda name: priority(matrices[name]),reverse=True)
    status="PASS" if matrices[ordered[0]]["pass"] else "FAIL"
    save_json(tgt,dict(status=status,preflight_lock_sha256=sha256_file(lock),
                        winning_recipe=ordered[0] if status=="PASS" else None,
                        nearest=ordered[:3],matrix=matrices,train_enabled=status=="PASS"))
    lines=["# CSBP Phase 1A", "", f"Hard-gate verdict: **{status}**", "",
           f"Diagnostic design: 300 audited text probes per route, {len(matrices)} scored cells (12 generator/template pairs × 2 Cloze interfaces).",
           "", "| Cell | n | Retained | Filtered recovery | Gap pp | Forecast yield |", "|---|---:|---:|---:|---:|---:|"]
    for name in ordered:
        z=matrices[name]
        lines.append(f"| {name} | {z['n']} | {z['n_filtered']} | {z.get('filtered_mc',0):.3f} | {z.get('filtered_gap_pp',0):.2f} | {z['projected_filtered']} |")
    (PRE/"PREFLIGHT_SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("CSBP_PREFLIGHT_"+status,ordered[:3],flush=True)


if __name__=="__main__":
    main()

