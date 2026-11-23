"""Fill the remaining Magpie quota with additional non-repeated official examples."""

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

import minimal
from prepare_pool import conversation,digest,normalize_key
from task_data import hash_order,load,save


def blocks(path):
    p=pq.ParquetFile(path)
    # Shard names and row-batch order are fixed. The within-batch sampling
    # order is hashed and training shuffles the final whole pool.
    for batch in p.iter_batches(batch_size=512,columns=["messages"]):
        raw=[conversation(r,"smol-magpie-ultra") for r in batch.to_pylist()]
        yield sorted([r for r in raw if r],key=lambda r:hash_order(r["sample_id"],"extra-smol"))


def eligibility(tok,rows):
    if not rows: return
    x=tok([r["prompt"] for r in rows],add_special_tokens=False)["input_ids"]
    y=tok([r["prompt"]+" "+r["response"] for r in rows],add_special_tokens=False)["input_ids"]
    for row,a,b in zip(rows,x,y,strict=True):
        n=len(b)-len(a)+1
        if b[:len(a)]==a and len(b)+1<=640 and 1<n<=240:
            yield dict(row,prompt_tokens=len(a),input_tokens=len(b),response_tokens=n)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("inputs","data","model"):
        p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--minimum-quota",type=int,default=4000)
    a=p.parse_args()
    base=a.data/"mixed_train.jsonl"
    whole=load(base)
    n0=sum(r["dataset"]=="smol-magpie-ultra" for r in whole)
    keys={normalize_key(r) for r in whole}
    for data in a.data.glob("*.jsonl"):
        if data.name.startswith(("dev_","test_")):
            keys.update(normalize_key(r) for r in load(data))
    # Filter the entire official Magpie test configuration, including rows
    # that are too long to be selected by this study's evaluation protocol.
    for raw in blocks(a.inputs/"smol-magpie-ultra_test.parquet"):
        keys.update(normalize_key(r) for r in raw)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=True)
    extra=[]
    files={}
    for file in sorted(a.inputs.glob("smol-magpie-ultra_train_0*.parquet")):
        files[file.name]=digest(file)
        for block in blocks(file):
            eligible=[r for r in block if normalize_key(r) not in keys]
            for row in eligibility(tok,eligible):
                key=normalize_key(row)
                if not key or key in keys: continue
                keys.add(key)
                extra.append(row)
                if len(extra)+n0>=a.minimum_quota: break
            if len(extra)+n0>=a.minimum_quota: break
        print(file.name,"cumulative Magpie",n0+len(extra),flush=True)
        if len(extra)+n0>=a.minimum_quota: break
    full=sorted([*whole,*extra],key=lambda r:hash_order(r["sample_id"],"mixed_training"))
    out=a.data/"full_mixed_train.jsonl"
    save(out,full)
    minimal.dump(a.data/"SCALE_LOCK.json",dict(base_sha256=digest(base),full_sha256=digest(out),
        sources=files,n=len(full),magpie_n=n0+len(extra),added_ids=[r["sample_id"] for r in extra]))
    print("FULL",len(full),n0+len(extra),flush=True)


if __name__=="__main__":
    main()
