"""Create a deduplicated generative training pool and freeze benchmark folds."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from dataset_transfer import canonical
from task_data import encode_target,hash_order,identity,load,multiple_choice,save


QUOTAS={"csqa":6000,"obqa":4000,"squad":10000,"gsm8k":6000,"no_robots":4000,
        "smol-magpie-ultra":4000,"smol-rewrite":2000,"smol-summarize":2000,"smol-constraints":2000}
REVISIONS={"squad":"rajpurkar/squad@7b6d24c440a36b6815f21b70d25016731768db1f",
           "gsm8k":"openai/gsm8k@740312add88f781978c0658806c59bc2815b9866",
           "no_robots":"HuggingFaceH4/no_robots@e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b",
           "smoltalk":"HuggingFaceTB/smoltalk@5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def answer(ex):
    raw=ex["answer"]
    if "####" not in raw:
        raise ValueError("Original GSM8K answer has no final marker")
    return raw.rsplit("####",1)[-1].strip()


def gsm(raw,kind):
    q=raw["question"]
    tag=("Solve the problem and give only the final number after ####." if kind=="final" else
         "Solve the problem step by step. End with the final answer after ####.")
    truth=answer(raw)
    return dict(instruction=tag,context=q,response="#### "+truth if kind=="final" else raw["answer"],
                task_type=f"gsm_{kind}",dataset="gsm8k",sample_id="gsm/"+hash_order(identity(q),"gsmid"),
                prompt=f"Question: {q}\n{tag}\nAnswer:",references=[truth],source_original_question=q)


def squad(raw):
    texts=[x for x in raw["answers"]["text"] if x and x.strip()]
    if not texts:
        return None
    context,q=raw["context"],raw["question"]
    tag="Answer the question with a short span from the passage."
    return dict(instruction=q,context=context,response=texts[0],task_type="span",dataset="squad",
                sample_id=raw["id"],prompt=f"{tag}\nPassage: {context}\nQuestion: {q}\nAnswer:",
                references=texts,source_original_question=q,title=raw["title"])


def conversation(raw,source):
    tray=raw["messages"]
    if len(tray)<2 or tray[-1]["role"]!="assistant" or tray[-2]["role"]!="user":
        return None
    if any(not isinstance(x["content"],str) or not x["content"].strip() for x in tray):
        return None
    if source.startswith("smol"):
        chars=[len(x["content"]) for x in tray]
        if sum(chars)>7680 or chars[-1]>2880:
            return None
    whole="\n".join(f"{x['role'].capitalize()}: {x['content'].strip()}" for x in tray[:-1])
    instruction=tray[-2]["content"].strip()
    content=tray[-1]["content"].strip()
    uid=raw.get("prompt_id") or hash_order(identity(whole),source)
    return dict(instruction=instruction,context=whole,response=content,task_type=raw.get("category") or source,
                dataset=source,sample_id=f"{source}/{uid}",prompt=whole+"\nAssistant:",references=[content])


def parent_rows(path):
    if path.suffix==".parquet":
        return pq.read_table(path).to_pylist()
    return load(path)


def stable_order(records,seed):
    return sorted(records,key=lambda r:hash_order(r["sample_id"],seed))


def normalize_key(record):
    text=record.get("source_original_question",record.get("question",
        record.get("candidate",{}).get("question",record.get("instruction",record.get("context","")))))
    return identity(text)


def canonical_mc(path,domain,part):
    return [canonical(row,domain,domain,part) for row in parent_rows(path)]


def valid(row,tok,max_total,max_answer):
    if not row or not normalize_key(row):
        return None
    try:
        prefix,total,n=encode_target(tok,row)
    except AssertionError:
        return None
    if len(total)>max_total or n>max_answer or len(prefix)<2:
        return None
    return dict(row,input_tokens=len(total)-1,response_tokens=n,prompt_tokens=len(prefix))


def take(candidates,count,tok,seen,salt,max_total=640,max_answer=240):
    picked=[]
    for record in stable_order([x for x in candidates if x],salt):
        key=normalize_key(record)
        if key in seen:
            continue
        selected=valid(record,tok,max_total,max_answer)
        if selected is None:
            continue
        picked.append(selected)
        seen.add(key)
        if len(picked)==count:
            break
    return picked


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for option in ("model","inputs","prior","dynamics","out"):
        p.add_argument(f"--{option}",type=Path,required=True)
    p.add_argument("--reuse-complete-sources",action="store_true")
    a=p.parse_args()
    inp=a.inputs
    old=a.prior/"inputs"
    archived=a.prior/"data"
    output=a.out
    output.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=True)
    used=set()
    provenance={}
    for parent in (archived,a.dynamics/"data"):
        for f in sorted(parent.rglob("*.jsonl")):
            if any(x in f.name for x in ("discovery","confirm","validation","test")):
                for row in load(f):
                    if normalize_key(row):
                        used.add(normalize_key(row))
    # All evaluation folds are reserved before taking any train samples.
    train_exclusion=set(used)
    all_eval={
      "arc_challenge":canonical_mc(old/"upstream_arc_challenge_test.parquet","arc_challenge","test"),
      "arc_easy":canonical_mc(old/"arc_easy_test.parquet","arc_easy","test"),
      "csqa":canonical_mc(archived/"csqa_validation.jsonl","csqa","validation"),
      "obqa":canonical_mc(old/"obqa_test.parquet","obqa","test"),
    }
    dev={
      "csqa":[multiple_choice(x,"csqa","content") for x in canonical_mc(archived/"csqa_discovery.jsonl","csqa","development")],
      "obqa":[multiple_choice(x,"obqa","content") for x in canonical_mc(old/"obqa_validation.parquet","obqa","validation")],
    }
    eval_seen=set()
    for label,rows in all_eval.items():
        save(output/f"test_{label}.jsonl",[multiple_choice(x,label,"content") for x in rows])
        eval_seen.update(normalize_key(row) for row in rows)
    for label,sample in (
        ("squad",[squad(x) for x in parent_rows(inp/"squad_validation.parquet")]),
        ("gsm8k",[gsm(x,"final") for x in parent_rows(inp/"gsm8k_test.parquet")]),
        *((n,[conversation(x,n) for x in parent_rows(inp/f"{n}_test.parquet")]) for n in QUOTAS if n.startswith("smol") or n=="no_robots"),
    ):
        pool=[x for x in sample if x]
        evaluation_seen=set()
        if label=="squad":
            titles=sorted({r["title"] for r in pool},key=lambda s:hash_order(s,"squad-title"))
            dev_set=set(titles[:max(1,len(titles)//4)])
            dev_rows=[r for r in pool if r["title"] in dev_set]
            eval_rows=[r for r in pool if r["title"] not in dev_set]
        else:
            ordered=stable_order(pool,"eval_partition")
            limit=160 if label=="gsm8k" else min(80,len(ordered)//5)
            dev_rows,eval_rows=ordered[:limit],ordered[limit:]
        upper=360 if label=="squad" else 320 if label=="gsm8k" else 64
        dev[label]=take(dev_rows,96 if label in ("squad","gsm8k") else 24,tok,evaluation_seen,
                        "dev_evaluation",max_total=960,max_answer=256)
        test=take(eval_rows,upper,tok,evaluation_seen,"test_evaluation",max_total=960,max_answer=256)
        save(output/f"test_{label}.jsonl",test)
        eval_seen.update(normalize_key(row) for row in pool)
    for label,rows in dev.items():
        if label in ("csqa","obqa"):
            eval_seen.update(normalize_key(row) for row in rows)
            rows=take(rows,160 if label=="csqa" else 96,tok,set(),"dev_main",max_total=960,max_answer=256)
        save(output/f"dev_{label}.jsonl",rows)
    train_exclusion.update(eval_seen)

    sources={
        "csqa":(old/"csqa_train.jsonl",lambda ex:multiple_choice(ex,"csqa",shuffle=True),"mc"),
        "obqa":(old/"obqa_train.parquet",lambda ex:multiple_choice(ex,"obqa",shuffle=True),"mc"),
        "squad":(inp/"squad_train.parquet",squad,"single"),
        "gsm8k":(inp/"gsm8k_train.parquet",lambda ex:gsm(ex,"final" if int(hash_order(ex["question"],"gsm_mode"),16)%2 else "process"),"single"),
    }
    for label in QUOTAS:
        if label not in sources:
            sources[label]=(inp/f"{label}_train.parquet",lambda ex,d=label:conversation(ex,d),"single")
    combined=[]
    by_source={}
    for label,quota in QUOTAS.items():
        path,render,style=sources[label]
        rows=canonical_mc(path,label,"train") if style=="mc" else parent_rows(path)
        output_file=output/f"train_{label}.jsonl"
        reused=a.reuse_complete_sources and output_file.is_file()
        if reused:
            sample=load(output_file)
            keys={normalize_key(r) for r in sample}
            if len(sample)!=quota or len(keys)!=len(sample) or keys&train_exclusion:
                raise ValueError(f"Invalid complete source cache {output_file}")
            if any(r["dataset"]!=label or r["input_tokens"]>=640 or not 1<=r["response_tokens"]<=240 for r in sample):
                raise ValueError(f"Length or data provenance differs {output_file}")
            train_exclusion.update(keys)
        else:
            sample=take([render(x) for x in rows],quota,tok,train_exclusion,"fit",max_total=640,max_answer=240)
        if style=="mc" and not reused:
            # Assign the exact 70/30 split to the selected IDs, preserving their shuffled option order.
            ordered=stable_order(sample,"exact_format")
            preferred={r["sample_id"] for r in ordered[:round(.3*len(ordered))]}
            sample=[valid(multiple_choice(x["candidate"],label,"letter" if x["sample_id"] in preferred else "content"),tok,640,240)
                    for x in sample]
            if any(x is None for x in sample):
                raise RuntimeError("Changing MC format unexpectedly changed the token eligibility")
        combined.extend(sample)
        if not reused:
            save(output_file,sample)
        counts=Counter(r["task_type"] for r in sample)
        by_source[label]=dict(wanted=quota,n=len(sample),raw_n=len(rows),file_sha256=digest(path),
                              task_type_counts=dict(counts),response_tokens=int(sum(r["response_tokens"] for r in sample)),
                              prompt_tokens=int(sum(r["prompt_tokens"] for r in sample)),
                              p_answer_tokens=np.quantile([r["response_tokens"] for r in sample],[0,.5,.9,.99,1]).tolist())
        print(label,len(sample),quota,flush=True)
    combined=stable_order(combined,"mixed_training")
    assert len({normalize_key(r) for r in combined})==len(combined)
    save(output/"mixed_train.jsonl",combined)
    for file in sorted([*inp.glob("*.parquet"),*old.glob("*parquet")]):
        provenance[file.name]=digest(file)
    summary=dict(revisions=REVISIONS,input_build_hashes=provenance,by_source=by_source,
                 unique_train=len(combined),output_hashes={f.name:digest(f) for f in sorted(output.glob("*.jsonl"))},
                 corpus_notes=["Magpie Ultra uses pinned train shard 0 of 6; no descriptions are truncated.",
                               "MC calibration test cohorts use official fixed splits; SQuAD dev/test are divided by title.",
                               "Licenses and upstream pretraining exposure are reported separately."])
    (output/"DATA_LOCK.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
