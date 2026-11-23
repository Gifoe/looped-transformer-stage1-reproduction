"""Derive the exact matched CSQA generation pool from the historical Rank-1 fit."""

import argparse
import json
from pathlib import Path

from task_data import load,multiple_choice,save,hash_order


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("previous",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    source=a.previous/"data/csqa_train.jsonl"
    raw=load(source)
    lock=json.loads((a.previous/"results/HOLDOUT/csqa/source/s261123137/TRAIN_LOCK.json").read_text())
    frozen=lock["training_ids"]
    by_id={row["id"]:row for row in raw}
    if len(frozen)!=len(by_id) or set(frozen)!=by_id.keys():
        raise RuntimeError("CSQA source differs from the historical Rank-1 fit")
    letter_ids=set(sorted(frozen,key=lambda key:hash_order(key,"b_letter"))[:round(.3*len(frozen))])
    save(a.out,[multiple_choice(by_id[key],"csqa","letter" if key in letter_ids else "content",shuffle=True)
                for key in frozen])
    print("matched CSQA",len(frozen),"letter",len(letter_ids),"content",len(frozen)-len(letter_ids))


if __name__=="__main__":
    main()
