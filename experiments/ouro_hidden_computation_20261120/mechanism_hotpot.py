"""Paired HotpotQA bridge evidence interventions on the official validation split."""
import argparse
from pathlib import Path
import random
import re
import time

import pyarrow.parquet as pq
import torch

from mechanism_design import dump, seed
from mechanism_run import add_line, checks, forward, load_runtime, predict, sha

SHA = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
COUNT = 64


def split_docs(row):
    context = {title: sentences for title,sentences in zip(row["context"]["title"],row["context"]["sentences"])}
    facts = list(dict.fromkeys(zip(row["supporting_facts"]["title"],row["supporting_facts"]["sent_id"])))
    titles = list(dict.fromkeys(t for t,_ in facts))
    if len(titles)!=2 or any(t not in context or i<0 or i>=len(context[t]) for t,i in facts):
        return
    marked = {t: " ".join(context[t][i].strip() for source,i in facts if source==t) for t in titles}
    match = re.compile(r"(?<!\w)"+re.escape(row["answer"])+ r"(?!\w)",re.I)
    carriers = [t for t in titles if match.search(marked[t])]
    if len(carriers)!=1:
        return
    target=carriers[0]
    bridge=next(t for t in titles if t!=target)
    # Require a literal cross-document link surviving the paired supporting extraction.
    bridge_lower = marked[bridge].casefold()
    if not (target.casefold() in bridge_lower or target.split(" (")[0].casefold() in bridge_lower):
        return
    candidates = [(t," ".join(v[:2]).strip()) for t,v in context.items() if t not in titles and v]
    candidates = [(t,v) for t,v in candidates if v and not match.search(v+" "+t)]
    if len(candidates)<2:
        return
    return (bridge,marked[bridge]),(target,marked[target]),candidates


def target_class(answer):
    if answer.isdigit():
        return "number"
    if answer.isupper() or re.fullmatch(r"[A-Z][a-z]+",answer):
        return "capitalized"
    return "lexical"


def manifest(tokenizer, rows):
    pools = {v:set() for v in ("number","capitalized","lexical")}
    target_set = {r["answer"] for r in rows if r["type"]=="bridge" and
                  len(r["answer"].split())==1 and r["answer"].casefold() not in ("yes","no")}
    single = {v for v in target_set if len(tokenizer.encode(" "+v,add_special_tokens=False))==1}
    for v in sorted(single):
        pools[target_class(v)].add(v)
    usable = []
    for row in sorted(rows,key=lambda r:seed("hotpot",r["id"])):
        answer=row["answer"]
        if row["type"]!="bridge" or answer not in single:
            continue
        docs=split_docs(row)
        if not docs:
            continue
        bridge,mapping,distractors=docs
        rng=random.Random(seed("hotpot-options",row["id"]))
        rng.shuffle(distractors)
        pool=sorted(v for v in pools[target_class(answer)] if v.casefold()!=answer.casefold()
                    and re.search(r"(?<!\w)"+re.escape(v)+r"(?!\w)", " ".join((*bridge,*mapping,*distractors[0],row["question"])),re.I) is None)
        if len(pool)<4:
            continue
        options=rng.sample(pool,4)+[answer]
        rng.shuffle(options)
        item={"id":row["id"],"split":"hotpot_bridge_validation_hashed","question":row["question"],
              "y":options.index(answer),"options":options,"source":bridge,"mapping":mapping,
              "noise":distractors[0],"replacement":distractors[1],"supporting_facts":row["supporting_facts"],
              "level":row["level"],"s":0,"m":0}
        try:
            result={fork:encode_hotpot(tokenizer,item,fork) for fork in ("open","content","letter")}
        except ValueError:
            continue
        if max(len(e["ids"]) for e in result.values())>360:
            continue
        usable.append(item)
    if len(usable)<COUNT:
        raise RuntimeError(f"Eligible HotpotQA rows {len(usable)} < {COUNT}")
    return usable[:COUNT],len(usable)


def encode_hotpot(tok,g,fork):
    text,spans="",{}
    for role in ("noise","source","mapping"):
        title,body=g[role]
        old=len(text)
        text+=f"{title}: {body}\n"
        spans[role]=(old,len(text))
    start=len(text)
    text+=f"Question: {g['question']}"
    spans["query"]=(start,len(text))
    prefix=text
    if fork!="open":
        text+="".join(f"\n{chr(65+j)}) {x}" for j,x in enumerate(g["options"]))
        text+="\nReply with the option letter." if fork=="letter" else "\nReply with the answer text."
    text+="\nAnswer:"
    a=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    ids=a["input_ids"]
    p=tok.encode(prefix,add_special_tokens=False)
    roles={role:[j for j,(b,e) in enumerate(a["offset_mapping"]) if b<v and e>u] for role,(u,v) in spans.items()}
    roles["readout"]=[len(ids)-1]
    if ids[:len(p)]!=p:
        raise ValueError("Unstable shared prefix")
    target=lambda v: tok.encode(text+" "+v,add_special_tokens=False)
    content=[target(v) for v in g["options"]]
    letters=[target(chr(65+j)) for j in range(5)]
    if any(v[:-1]!=ids for v in content+letters):
        raise ValueError("Invalid continuation tokenization")
    return {"ids":ids,"text":text,"roles":roles,"read":len(ids)-1,"prefix":len(p),"fork":fork,
            "content":[v[-1] for v in content],"letter":[v[-1] for v in letters]}


def matched_replacement(tok,g,enc,role):
    pos=enc["roles"][role]
    prefix=f"{g['replacement'][0]}: {g['replacement'][1]} "
    alt=tok.encode(prefix,add_special_tokens=False)
    alt=(alt*((len(pos)//len(alt))+1))[:len(pos)]
    ids=enc["ids"].copy()
    for i,v in zip(pos,alt):
        ids[i]=v
    return {**enc,"ids":ids}


def conditions(model,tok,item,fork):
    e=encode_hotpot(tok,item,fork)
    arms={"natural":e}
    for role in ("source","mapping","noise"):
        arms[role+"_replace"]=matched_replacement(tok,item,e,role)
    out=forward(model,list(arms.values()))
    roles=e["roles"]
    source,mapping,noise=[roles[role] for role in ("source","mapping","noise")]
    n=min(len(source),len(noise))
    source=source[-n:]
    noise=noise[-n:]
    cuts={2:[(1,mapping,source),(2,mapping,noise)]}
    sham=forward(model,[e]*4,cuts={})
    route=forward(model,[e]*4,cuts={2:cuts[2]+[(3,mapping,source)]},
                  edits={2:[(3,mapping,sham["boundary"][1][0,mapping])]})
    batches=[(list(arms),out),(["sham","source_cut","noise_cut","rescue"],route)]
    for labels,b in batches:
        for j,tag in enumerate(labels):
            scores=b["scores"][j]
            mode=int(fork=="letter")
            yield {"id":item["id"],"fork":fork,"arm":tag,"y":item["y"],"scores":scores.tolist(),
                   "pred":predict(scores[:,mode,:]),"top_token":b["top_token"][j],
                   "gold_content_token":e["content"][item["y"]],"gold_label_token":e["letter"][item["y"]],
                   "nmatched_cut_keys":n,"positions":len(e["ids"])}


def main():
    ap=argparse.ArgumentParser()
    for field in ("checkpoint","dataset","out"):
        ap.add_argument(field,type=Path)
    ap.add_argument("--pilot",action="store_true")
    args=ap.parse_args()
    if sha(args.dataset)!=SHA:
        raise RuntimeError("Wrong HotpotQA validation file")
    args.out.mkdir(parents=True,exist_ok=False)
    model,tok,runtime=load_runtime(args.checkpoint)
    rows=pq.read_table(args.dataset).to_pylist()
    examples,neligible=manifest(tok,rows)
    examples=examples[:2] if args.pilot else examples
    start=time.perf_counter()
    with torch.inference_mode(), (args.out/"hotpot_inputs.jsonl").open("w",encoding="utf-8") as i, (args.out/"hotpot_arms.jsonl").open("w",encoding="utf-8") as f:
        for j,item in enumerate(examples):
            add_line(i,item)
            if j==0:
                e=encode_hotpot(tok,item,"open")
                print(dump(checks(model,e)),flush=True)
            for fork in ("open","content","letter"):
                for row in conditions(model,tok,item,fork):
                    add_line(f,row)
            i.flush(); f.flush()
            if j%5==0:
                print(f"hotpot {j+1}/{len(examples)} {time.perf_counter()-start:.1f}s",flush=True)
    (args.out/"hotpot_complete.json").write_text(dump({"groups":len(examples),"eligible":neligible,
        "original_split_rows":len(rows),"dataset_sha256":SHA,"manifest_sha256":sha(args.out/"hotpot_inputs.jsonl"),
        "runtime":runtime,"seconds":time.perf_counter()-start}),encoding="utf-8")


if __name__=="__main__":
    main()
