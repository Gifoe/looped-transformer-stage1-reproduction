"""Estimate uncompressed activation storage on *entire* manifest before any GPU run."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict
from core import DEFAULT_DEMOS,DEFAULT_MODEL_DIR,demo_prefix,make_requests,encode_pair,read_jsonl
from latent_hooks import capture_positions


def pretty(n):
    return f"{n/(1024**4):,.3f} TiB / {n/(1024**3):,.2f} GiB"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",type=Path,nargs="+",required=True)
    p.add_argument("--output-mount",type=Path,default=Path("."))
    p.add_argument("--demonstrations",type=Path,default=DEFAULT_DEMOS)
    p.add_argument("--model-dir",type=Path,default=DEFAULT_MODEL_DIR)
    p.add_argument("--rotations",choices=("original","cyclic"),default="original")
    p.add_argument("--capture",choices=("none","answer","current","all"),default="current")
    p.add_argument("--save-head",action="store_true")
    args=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(args.model_dir),local_files_only=True,trust_remote_code=True)
    demo=demo_prefix(json.loads(args.demonstrations.read_text(encoding="utf8")))
    stats=defaultdict(lambda:dict(n=0,forwards=0,selected_positions=0,estimated_uncompressed_bytes=0))
    for file in args.inputs:
        for r in read_jsonl(file):
            split=r["split"];stats[split]["n"]+=1
            n=len(r["choices"]);orders=[list(range(n))]
            if args.rotations=="cyclic": orders += [[(j+i)%n for j in range(n)] for i in range(1,n)]
            for order in orders:
                _,requests=make_requests(r,demo,order)
                groups=set()
                for ctx,tar in requests:
                    prefix,target=encode_pair(tok,ctx,tar)
                    pred=tuple(range(len(prefix)-1,len(prefix)+len(target)-1))
                    ids=prefix+target[:-1]
                    key=(tuple(ids),pred)
                    if key in groups:continue
                    groups.add(key)
                    selected=capture_positions(args.capture,ids,list(pred),tok,demo)
                    # raw 24 layers + final loop-normalized residual; 4 loops; BF16 2 bytes.
                    bytes_states=len(selected)*25*4*2048*2 if args.capture!="none" else 0
                    bytes_head=4*49152*4 if args.save_head else 0
                    stats[split]["forwards"]+=1
                    stats[split]["selected_positions"]+=len(selected)
                    stats[split]["estimated_uncompressed_bytes"]+=bytes_states+bytes_head
    usage=shutil.disk_usage(args.output_mount)
    print(json.dumps({"estimation_basis":"uncompressed BF16 snapshots; actual .npz compression and metadata vary",
                      "splits":dict(stats),"total_estimated":pretty(sum(x["estimated_uncompressed_bytes"] for x in stats.values())),
                      "output_mount":str(args.output_mount.resolve()),"free":pretty(usage.free),
                      "disk_total":pretty(usage.total)},indent=2))

if __name__=="__main__":main()
