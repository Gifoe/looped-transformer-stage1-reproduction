"""Verify A and B really match training IDs, optimizer updates, and loop depth."""

import argparse
import json
from pathlib import Path

import minimal


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("historical",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    old=read(a.historical)
    ids=[rows["question_ids"] for rows in old["step_records"]]
    depths=[rows["depth"] for rows in old["step_records"]]
    reports={}
    for arm in ("gen_only","gen_rank"):
        phase=a.root/"results"/arm
        report=read(phase/"TRAIN.json")
        steps=report["logs"]
        assert [x["ids"] for x in steps]==ids and [x["depth"] for x in steps]==depths
        assert report["steps"]==282 and report["unique_samples"]==1119
        assert minimal.sha(phase/"controller.pt")==report["checkpoint_sha256"]
        reports[arm]=dict(checkpoint_sha256=report["checkpoint_sha256"],steps=282,unique_samples=1119,
                          same_train_schedule=True,padded_token_loops=report["padded_token_loops"],
                          supervised_tokens=report["supervised_tokens"],seconds=report["seconds"])
    file=a.root/"results/MATCHED_CONTROLS.json"
    minimal.dump(file,dict(historical_train_sha256=minimal.sha(a.historical),exact_optimizer_step_ids_and_depths=True,arms=reports))
    print("Matched",len(ids),len(set(q for batch in ids for q in batch)),flush=True)


if __name__=="__main__":
    main()
