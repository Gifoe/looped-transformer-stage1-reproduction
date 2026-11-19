"""Test HotpotQA on disjoint bridge problems with context-present wrong options."""
import argparse
from pathlib import Path
import random
import re
import time

import pyarrow.parquet as pq
import torch

from mechanism_design import dump, load, seed
from mechanism_hotpot import SHA, conditions, encode_hotpot, split_docs, target_class
from mechanism_run import add_line, load_runtime, sha


def prepare(tok,rows,seen):
    pool={row["answer"] for row in rows if row["type"]=="bridge" and len(row["answer"].split())==1
          and row["answer"].casefold() not in ("yes","no")}
    pool=sorted(x for x in pool if len(tok.encode(" "+x,add_special_tokens=False))==1)
    normalized={x.casefold():x for x in pool}
    items=[]
    for row in sorted(rows,key=lambda r:seed("hotpot-hard-independent-v1",r["id"])):
        answer=row["answer"]
        if row["id"] in seen or row["type"]!="bridge" or answer not in pool:
            continue
        docs=split_docs(row)
        if not docs:
            continue
        bridge,mapping,distractors=docs
        lower=row["question"].casefold()
        bridge_target=mapping[0].split(" (")[0].casefold()
        if bridge_target in lower or answer.casefold() in (bridge[0]+" "+mapping[0]).casefold().split():
            continue
        rng=random.Random(seed("hotpot-hard-options",row["id"]))
        rng.shuffle(distractors)
        for j, candidate in enumerate(distractors):
            text=" ".join(candidate)
            options={normalized[w.casefold()] for w in re.findall(r"\b\w+\b", text) if w.casefold() in normalized}
            options=sorted(x for x in options if x.casefold()!=answer.casefold() and x.casefold() not in lower
                           and target_class(x)==target_class(answer))
            if len(options)<2:
                continue
            hard=rng.sample(options,2)
            elsewhere=" ".join((*bridge,*mapping,*candidate,row["question"]))
            unseen=[x for x in pool if x.casefold()!=answer.casefold() and target_class(x)==target_class(answer)
                    and re.search(r"(?<!\w)"+re.escape(x)+r"(?!\w)",elsewhere,re.I) is None]
            if len(unseen)<2:
                continue
            opts=[*hard,*rng.sample(unseen,2),answer]
            rng.shuffle(opts)
            g={"id":row["id"],"split":"hotpot_hard_disjoint","question":row["question"],"y":opts.index(answer),
              "options":opts,"hard_options":hard,"source":bridge,"mapping":mapping,"noise":candidate,
              "replacement":next(x for i,x in enumerate(distractors) if i!=j),
              "supporting_facts":row["supporting_facts"],"level":row["level"],"s":0,"m":0}
            try:
                e=encode_hotpot(tok,g,"letter")
            except ValueError:
                continue
            if len(e["ids"])>360:
                continue
            items.append(g)
            break
    return items


def main():
    p=argparse.ArgumentParser()
    for x in ("checkpoint","dataset","previous","out"):
        p.add_argument(x,type=Path)
    a=p.parse_args()
    if sha(a.dataset)!=SHA:
        raise RuntimeError("HotpotQA data hash mismatch")
    model,tok,_=load_runtime(a.checkpoint)
    start=time.perf_counter()
    a.out.mkdir(parents=True,exist_ok=False)
    seen={p["id"] for p in load(a.previous)}
    examples=prepare(tok,pq.read_table(a.dataset).to_pylist(),seen)
    if len(examples)<32:
        raise RuntimeError(f"Only {len(examples)} qualified for context-present distractors")
    chosen=examples[:48]
    print(f"Eligible {len(examples)}; selected {len(chosen)}",flush=True)
    with torch.inference_mode(), (a.out/"hotpot_hard_inputs.jsonl").open("w",encoding="utf-8") as i, (a.out/"hotpot_hard_arms.jsonl").open("w",encoding="utf-8") as o:
        for j,g in enumerate(chosen):
            add_line(i,g)
            for fork in ("open","content","letter"):
                for v in conditions(model,tok,g,fork):
                    add_line(o,v)
            i.flush(); o.flush()
            if j%8==0:
                print(f"hard Hotpot {j+1}/{len(chosen)}",flush=True)
    f=a.out/"hotpot_hard_inputs.jsonl"
    (a.out/"complete.json").write_text(dump({"groups":len(chosen),"eligible":len(examples),"previous_excluded":len(seen),
                   "manifest_sha256":sha(f),"seconds":time.perf_counter()-start}),encoding="utf-8")


if __name__=="__main__":
    main()
