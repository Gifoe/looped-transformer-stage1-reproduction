"""Create a previously unused mechanism confirmation fold without using outcomes."""

import argparse
import hashlib
from pathlib import Path

import pyarrow.parquet as pq

from archive_audit import dump, sha
from control_transport import rows_write
from dataset_transfer import canonical, signatures
from minimal import load_jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transport",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--extra-exclusion",type=Path,action="append",default=[])
    p.add_argument("--n",type=int,default=96)
    p.add_argument("--salt",default="transport-confirm-v1")
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    inputs=a.transport/"inputs"
    blocked_ids=set()
    blocked_stems=set()
    documents={}
    old_data=a.transport/"data"
    for path in sorted(old_data.rglob("*.jsonl")):
        for row in load_jsonl(path):
            if isinstance(row.get("question"),str) and "choices" in row:
                i,s,_=signatures(row)
                blocked_ids.add(i)
                blocked_stems.add(s)
        documents[str(path.relative_to(a.transport))]=sha(path)
    for path in a.extra_exclusion:
        for row in load_jsonl(path):
            i,s,_=signatures(row)
            blocked_ids.add(i)
            blocked_stems.add(s)
        documents[path.name]=sha(path)
    arc=inputs/"arc_easy_train.parquet"
    csqa=inputs/"csqa_train.jsonl"
    donors={
        "arc": (arc,[canonical(x,"arc","ARC-Easy","train") for x in pq.read_table(arc).to_pylist()]),
        "csqa": (csqa,[canonical(x,"csqa","author","train") for x in load_jsonl(csqa)]),
    }
    counts={}
    for name,(source,all_rows) in donors.items():
        ordered=sorted(all_rows,key=lambda row:hashlib.sha256(f"{a.salt}/{row['id']}".encode()).digest())
        selected=[]
        for row in ordered:
            i,s,_=signatures(row)
            if i not in blocked_ids and s not in blocked_stems:
                blocked_ids.add(i)
                blocked_stems.add(s)
                selected.append(row)
                if len(selected)==a.n: break
        if len(selected)!=a.n:
            raise RuntimeError("Not enough audited unseen questions")
        out=a.out/f"{name}_confirm.jsonl"
        rows_write(out,selected)
        counts[name]=dict(source_sha256=sha(source),sha256=sha(out),ids=[r["id"] for r in selected])
    dump(a.out/"COHORTS.json",dict(cohorts=counts,blocked_input_sha256=documents,
        limit="ID/stem disjoint from archived transferability data. Both folds are from official Train donor "
              "pools. Upstream pretraining or historical few-shot donor exposure is not excluded.",selection="hash order only"))


if __name__=="__main__":
    main()
