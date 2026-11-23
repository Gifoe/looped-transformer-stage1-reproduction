"""Audit the inherited complete candidate protocol and preserve per-row evidence."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from minimal import dump, load_jsonl, sha


def run(control, out):
    checks = {}
    receipt = json.loads((control/"results/FREEZE.json").read_text())
    frozen = receipt["weights"]["rank1_t1"]
    assert sha(control/"results/weights"/frozen["file"]) == frozen["sha256"]
    out.mkdir(parents=True,exist_ok=True)
    for name,n,rel,key in (
        ('arc',1172,'BENCHMARK_RESULTS/RESULTS_ARC.jsonl','rank1_t1'),
        ('openbookqa',500,'BENCHMARK_RESULTS/RESULTS_OPENBOOKQA.jsonl','rank1_t1'),
        ('commonsenseqa',1221,'CSQA_TRANSFER/PREDICTIONS.jsonl','frozen'),
    ):
        source = control/"results"/rel
        values = load_jsonl(source)
        assert len(values)==len({r["id"] for r in values})==n
        metrics={"acc":[],"acc_norm":[]}
        for k in range(1,5):
            counts={"acc":0,"acc_norm":0}
            for row in values:
                tag=f"{key}_k{k}"
                scores=np.array(row["scores"][tag])
                chars=np.array(row["characters"])
                for metric,arr in (("acc",scores),("acc_norm",scores/chars)):
                    old=row["predictions"][tag][metric] if key=="frozen" else row["predictions"][metric][tag]
                    assert int(arr.argmax())==old
                    counts[metric]+=old==row["gold"]
            for name_, count in counts.items():
                metrics[name_].append(count)
        dest=out/f"INHERITED_{name.upper()}.jsonl"
        shutil.copyfile(source,dest)
        assert sha(source)==sha(dest)
        checks[name]=dict(n=n,**metrics,file=dest.name,sha256=sha(dest),cases_replayed_numeric=0)
    dump(out/"CANDIDATE_INVARIANCE.json",dict(frozen_rank1_sha256=frozen["sha256"],
        evaluator="unchanged train.evaluate + minimal.forward",reference="e2b21c95c1e5ff37ac9be7d9cbc73d0aead8f26c",
        benchmarks=checks))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    run(args.control,args.out)
