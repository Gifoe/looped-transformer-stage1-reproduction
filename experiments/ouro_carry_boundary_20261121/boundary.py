"""Independent validation of candidate-free overpropagation at the tens boundary."""
import argparse
import random
from collections import Counter
from pathlib import Path

import torch

from carry import DEMO,PROMPT,decode
from native import Forward,load,save,setup,sha

CONDITIONS=("stop","continue","safe","same_stop_no_incoming","same_continue_no_incoming","neighbor")


def splits(total):
    return [x for x in range(10) if 0<=total-x<=9]


def pair(rng,ha,hb,st,su):
    a,b=rng.choice(splits(st)),rng.choice(splits(su))
    return (100*ha+10*a+b,100*hb+10*(st-a)+(su-b))


def generate(old_file,out):
    old=load(old_file)
    seen=set(DEMO)
    seen.update(tuple(w) for row in old["rows"] for w in row["pairs"])
    rng=random.Random(211121317)
    rows=[]
    for label,limit in (("discovery",96),("confirmation",192)):
        for i in range(limit):
            for trial in range(3000):
                x=rng.randint(1,6)
                y=rng.randint(1,7-x)
                neighbor=13 if i%2==0 else 15
                found=[pair(rng,x,y,a,b) for a,b in ((8,14),(9,14),(7,14),(9,4),(10,4),(8,neighbor))]
                if not seen.intersection(found):
                    break
            else:
                raise RuntimeError((label,i))
            seen.update(found)
            z={}
            for name,values in zip(CONDITIONS,found,strict=True):
                goal=sum(values)
                wrong=goal-100 if name in ("continue","same_continue_no_incoming") else goal+100
                z[name]=dict(pair=values,target=goal,foil=wrong)
            assert z["stop"]["target"]==z["same_stop_no_incoming"]["target"]
            assert z["continue"]["target"]==z["same_continue_no_incoming"]["target"]
            rows.append(dict(id=f"{label}-{i+1:03d}",split=label,conditions=z))
    save(out,dict(rows=rows,seed=211121317,previous_sha256=sha(old_file),prompt=PROMPT,
                  predeclared_primary="confirmation/stop/first wrong exactly 100 too high then entire correct integer"))


def scores(out):
    r={}
    for split in ("discovery","confirmation"):
        q=[x for x in out if x["split"]==split]
        if not q:
            continue
        a={}
        for name in CONDITIONS:
            rows=[x["conditions"][name] for x in q]
            b=Counter()
            for k in rows:
                z=[y["number"] for y in k["outputs"]]
                target=k["target"]; bad=k["foil"]
                for i,p in enumerate(z):
                    b[f"correct_K{i+1}"]+=p==target
                    b[f"foil_K{i+1}"]+=p==bad
                    b[f"unparsed_K{i+1}"]+=p is None
                b["WC"]+=z[0]!=target and z[1]==target
                b["CW"]+=z[0]==target and z[1]!=target
                b["foil_repaired"]+=z[0]==bad and z[1]==target
            a[name]=dict(n=len(rows),**b)
        r[split]=a
    return r


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("previous",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    root=a.out
    manifest=root/"MANIFEST.json"
    if not manifest.exists():
        generate(a.previous,manifest)
    deck=load(manifest)["rows"]
    f,tok=setup(a.model)
    f=Forward(f)
    path=root/"NATIVE.json"
    out=load(path)["rows"] if path.exists() else []
    assert [x["id"] for x in out]==[x["id"] for x in deck[:len(out)]]
    for i in range(len(out),len(deck),8):
        pack=deck[i:i+8]
        flat=[q["conditions"][n]["pair"] for q in pack for n in CONDITIONS]
        draws=[decode(f,tok,flat,k) for k in (1,2)]
        for j,g in enumerate(pack):
            z={name:dict(g["conditions"][name],outputs=[v[j*len(CONDITIONS)+o] for v in draws])
                        for o,name in enumerate(CONDITIONS)}
            out.append(dict(g,conditions=z))
        save(path,dict(rows=out,manifest_sha256=sha(manifest)))
        print(pack[0]["split"],len(out),len(deck),flush=True)
    save(root/"NATIVE_SUMMARY.json",scores(out))


if __name__=="__main__":
    main()

