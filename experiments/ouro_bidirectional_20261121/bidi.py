"""Matched open-generation counterfactuals with a common wrong midpoint."""
import argparse
from collections import Counter
from pathlib import Path
import random

import numpy as np
import torch

from carry import DEMO, PROMPT, decode, numeric
from native import Forward, load, save, setup, sha


def unordered(pair):
    return tuple(sorted(pair))


def prior_pairs(pilot_file, boundary_file):
    first=load(pilot_file)
    second=load(boundary_file)
    seen={unordered(p) for p in DEMO}
    seen.update(unordered(p) for item in first["rows"] for p in item["pairs"])
    seen.update(unordered(p["pair"]) for item in second["rows"] for p in item["conditions"].values())
    return seen


def construct(pilot_file, boundary_file, path):
    rng=random.Random(211121529)
    split_rng=random.Random(211121735)
    seen=prior_pairs(pilot_file,boundary_file)
    dev=set()
    for h in (4,5,6):
        dev.update(split_rng.sample(range(100*h,100*(h+1)),34 if h==4 else 33))
    draw=[]
    for middle in range(400,700):
        hundred,suffix=divmod(middle,100)
        for j in range(8):
            changed=j%2
            prefer=j//2%2
            found=None
            for c in (prefer,1-prefer):
                base=suffix+100*c
                if base > 198:
                    continue
                depth=hundred-c
                possible=[(a,z) for a in range(2,depth) for z in range(max(base-99,0),min(base,99)+1)]
                rng.shuffle(possible)
                for a,z in possible:
                    x,y=100*a+z,100*(depth-a)+base-z
                    lo,hi=(x-100,y),(x+100,y)
                    if changed:
                        lo,hi=lo[::-1],hi[::-1]
                    keys={unordered(lo),unordered(hi)}
                    if len(keys)==2 and not keys.intersection(seen):
                        found=(lo,hi,c)
                        seen.update(keys)
                        break
                if found:
                    break
            # Very small operand-suffix support leaves some strata partially filled.
            if found is None:
                continue
            lo,hi,c=found
            draw.append(dict(id=f"{middle}-{j+1}",midpoint=middle,
                             split="discovery" if middle in dev else "confirmation",
                             changed_operand=changed,carry_to_hundreds=c,
                             pairs=[lo,hi],target=[middle-100,middle+100]))
    # Process both fixed partitions in interleaved midpoint order to keep topic distribution stable.
    rng.shuffle(draw)
    save(path,dict(rows=draw,midpoint_splits={str(y):"discovery" if y in dev else "confirmation"
                    for y in range(400,700)},prompt=PROMPT,
                    sources=[sha(pilot_file),sha(boundary_file)],seed=211121529,split_seed=211121735))


def event(row):
    p=[q["number"] for q in row["pred"][0]]
    q=[x["number"] for x in row["pred"][1]]
    midpoint=row["midpoint"]
    correct=row["target"]
    return dict(eligible=p == [midpoint,midpoint],
                low_k1_mid=p[0]==midpoint,high_k1_mid=p[1]==midpoint,
                k1_correct=int(p[0]==correct[0])+int(p[1]==correct[1]),
                k2_correct=int(q[0]==correct[0])+int(q[1]==correct[1]),
                direct=p == [midpoint,midpoint] and q == correct,
                swap=p == [midpoint,midpoint] and q[::-1] == correct)


def baseline(out,root):
    r=load(root/"DECK.json")
    bins={int(k):v for k,v in r["midpoint_splits"].items()}
    report={}
    for split in ("discovery","confirmation"):
        count=Counter()
        indexed={m:0 for m,s in sorted(bins.items()) if s==split}
        mid_eligible=set()
        mid_direct=set()
        eligible_ids=[]
        direct_ids=[]
        arms={0:Counter(),1:Counter()}
        for x in out:
            if x["split"]!=split:
                continue
            d=event(x)
            count.update(d)
            count["drawn_stems"]+=1
            m=x["midpoint"]
            a=arms[x["changed_operand"]]
            a["drawn_stems"]+=1
            if d["eligible"]:
                a["eligible"]+=1
                a["direct"]+=d["direct"]
                a["swap"]+=d["swap"]
                mid_eligible.add(m)
                eligible_ids.append(x["id"])
                for i,name in enumerate(["minus_100","plus_100"]):
                    count["eligible_k2_"+name]+=x["pred"][1][i]["number"]==x["target"][i]
            if d["direct"]:
                mid_direct.add(m)
                direct_ids.append(x["id"])
            indexed[m]+=int(d["direct"])-int(d["swap"])
        values=np.asarray(list(indexed.values()))
        rng=np.random.default_rng(211121943)
        ix=rng.integers(0,len(values),size=(20000,len(values)))
        interval=np.quantile(values[ix].mean(axis=1),[0.025,0.975]).tolist()
        distinct=len(mid_eligible)
        passed=distinct>=12 and count["direct"]>=12 and interval[0]>0 and all(a["direct"]>a["swap"] for a in arms.values())
        report[split]=dict(count,n_midpoints=len(indexed),distinct_eligible_midpoints=distinct,
                           distinct_direct_midpoints=len(mid_direct),by_changed_operand=arms,
                           eligible_ids=eligible_ids,direct_ids=direct_ids,
                           direct_excess_per_midpoint=float(values.mean()),interval=interval,pass_gate=passed)
    save(root/"NATIVE_SUMMARY.json",report)
    return report


@torch.no_grad()
def run(model, pilot, boundary, root):
    deck=root/"DECK.json"
    if not deck.exists():
        construct(pilot,boundary,deck)
    manifest=load(deck)["rows"]
    raw=root/"NATIVE.json"
    current=load(raw)["rows"] if raw.exists() else []
    assert [r["id"] for r in current]==[r["id"] for r in manifest[:len(current)]]
    f,tok=setup(model)
    f=Forward(f)
    n=len(manifest)
    for i in range(len(current),n,24):
        block=manifest[i:i+24]
        flat=[p for row in block for p in row["pairs"]]
        outputs=[decode(f,tok,flat,k) for k in (1,2)]
        current.extend(dict(row,pred=[x[2*j:2*j+2] for x in outputs]) for j,row in enumerate(block))
        if (i//24)%4==0 or len(current)==n:
            save(raw,dict(deck_sha256=sha(deck),rows=current))
            print("native",len(current),n,flush=True)
    s=baseline(current,root)
    print({k:{a:b for a,b in v.items() if not a.endswith("_ids")} for k,v in s.items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("pilot",type=Path)
    p.add_argument("boundary",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.pilot,a.boundary,a.root)

