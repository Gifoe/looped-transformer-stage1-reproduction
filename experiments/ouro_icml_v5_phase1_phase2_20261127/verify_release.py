"""Verify the immutable release inventory and recompute paired-population scores."""
from pathlib import Path
from statistics import mean
import gzip
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parent
EXPECTED=218845


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda:stream.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()


def audit():
    for line in (ROOT/"SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected,relative=line.split("  ",1)
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or sha(path)!=expected:
            raise ValueError(f"Release asset differs: {relative}")
    status=json.loads((ROOT/"RELEASE_STATUS.json").read_text(encoding="utf-8"))
    pooled={}
    items=files=0
    for method,tasks in status["official"].items():
        scores=[]
        for task,info in tasks.items():
            zipped=ROOT/info["gzip"]
            receipt=json.loads(Path(str(zipped).removesuffix(".jsonl.gz")+".json").read_text(encoding="utf-8"))
            orig=hashlib.sha256()
            roster={}
            score=0
            with gzip.open(zipped,"rb") as stream:
                for line in stream:
                    orig.update(line)
                    row=json.loads(line)
                    if row["id"] in roster:
                        raise ValueError(f"Duplicate evaluation ID in {zipped.name}")
                    roster[row["id"]]=row["gold"]
                    score+=row["metrics"][info["metric"]]
            if (orig.hexdigest()!=info["jsonl_sha256"] or receipt["rows_sha256"]!=orig.hexdigest()
                    or len(roster)!=info["n"] or (task in pooled and roster!=pooled[task])):
                raise ValueError(f"Unmatched official rows in {method}/{task}")
            pooled[task]=roster
            accuracy=score/len(roster)
            if abs(accuracy-info["accuracy"])>1e-12:
                raise ValueError(f"Unmatched official score in {method}/{task}")
            scores.append(accuracy)
            items+=len(roster)
            files+=1
        if abs(mean(scores)-status["official_macro8"][method])>1e-12:
            raise ValueError(f"Unmatched Macro8 in {method}")
        print(f"{method}: {mean(scores):.4%}")
    if items!=EXPECTED or files!=40:
        raise ValueError(f"Unexpected release population: {items} rows, {files} files")
    print(f"Verified {items:,} predictions across {files} official file receipts")


if __name__=="__main__":
    audit()

