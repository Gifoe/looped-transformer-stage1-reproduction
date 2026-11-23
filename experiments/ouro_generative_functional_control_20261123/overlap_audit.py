"""Trace exact and lexically near train/evaluation overlaps before locking tests."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import minimal
from prepare_pool import normalize_key
from task_data import load,save


def fourgrams(tokens):
    return {tuple(tokens[j:j+4]) for j in range(len(tokens)-3)}


def audit(data,fit,output):
    evaluations={}
    queries=[]
    for path in sorted(data.glob("test_*.jsonl")):
        source=load(path)
        evaluations[path.stem]=source
        for row in source:
            key=normalize_key(row)
            if key:
                queries.append((path.stem,row["sample_id"],key))
    exact={}
    lexicon={}
    index=defaultdict(list)
    for i,(file,key,text) in enumerate(queries):
        exact.setdefault(text,[]).append(i)
        if len(text.split())>=12:
            grams=fourgrams(text.split())
            if len(grams)>=8:
                lexicon[i]=grams
                for gram in grams:
                    index[gram].append(i)
    receipts={}
    active=[]
    for source in fit:
        rows=load(source)
        active.extend(rows)
        receipts[source.name]=dict(n=len(rows),sha256=minimal.sha(source))
    seen=set()
    details=[]
    blocked=set()
    for row in active:
        text=normalize_key(row)
        if text in seen:
            continue
        seen.add(text)
        for i in exact.get(text,[]):
            file,qid,_=queries[i]
            if (file,qid) not in blocked:
                blocked.add((file,qid))
                details.append(dict(file=file,id=qid,train_id=row["sample_id"],degree="normalized_exact",similarity=1.))
        ts=text.split()
        if len(ts)<12:
            continue
        g=fourgrams(ts)
        matches={i for gram in g for i in index.get(gram,[]) if len(index[gram])<100}
        for i in matches:
            file,qid,_=queries[i]
            if (file,qid) in blocked:
                continue
            other=lexicon[i]
            shared=len(g&other)
            union=len(g|other)
            jaccard=shared/max(union,1)
            if jaccard>=.8:
                blocked.add((file,qid))
                details.append(dict(file=file,id=qid,train_id=row["sample_id"],degree="fourgram_near",similarity=jaccard))
    output.mkdir(parents=True,exist_ok=True)
    tests={}
    for file,rows in evaluations.items():
        delete={qid for filename,qid in blocked if filename==file}
        selected=[r for r in rows if r["sample_id"] not in delete]
        canonical=output/f"{file}.jsonl"
        save(canonical,selected)
        tests[file]=dict(n_total=len(rows),n_confirmed=len(selected),flagged_ids=sorted(delete),
                         source_sha256=minimal.sha(data/f"{file}.jsonl"),sha256=minimal.sha(canonical))
    record=dict(train=receipts,evaluation=tests,detected=details,train_unique_normalized=len(seen),
                near_duplicate="Word four-gram Jaccard ≥ .8, minimum twelve tokens. Common shingles cannot index a candidate.",
                scope="Recorded experiment training vs this test population. Unknown pretraining exposure remains.")
    minimal.dump(output/"CONTAMINATION.json",record)
    print("AUDIT",len(seen),len(queries),len(details),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--train",type=Path,action="append",required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    audit(args.data,args.train,args.out)


if __name__=="__main__":
    main()
