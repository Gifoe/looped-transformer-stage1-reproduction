"""Feasibility for a candidate-free, independently auditable carry mechanism."""
import argparse
import random
import re
from pathlib import Path

import torch
from native import Forward,load,output,pad,save,setup,sha

PROMPT="18 + 25 = 43\n36 + 23 = 59\n13 + 71 = 84\n218 + 125 = 343\n326 + 243 = 569\n413 + 371 = 784\n"
DEMO={(18,25),(36,23),(13,71),(218,125),(326,243),(413,371)}
MAX_NEW=6


def build(out):
    rng=random.Random(211121281)
    seen=set(DEMO)
    rows=[]
    for d in (2,3):
        n=0
        for _ in range(20000):
            if n==128:
                break
            ta=rng.randint(1,7); tb=rng.randint(1,8-ta)
            x,y=10*ta,10*tb
            if d==3:
                a=rng.randint(1,7); b=rng.randint(1,8-a)
                x+=100*a; y+=100*b
            u=rng.randrange(5); v=rng.randrange(5,10)
            pairs=[(x+u,y+4-u),(x+v,y+14-v)]
            if seen.intersection(pairs):
                continue
            seen.update(pairs)
            n+=1
            rows.append(dict(id=f"{d}-{n:03}",digits=d,pairs=pairs))
        assert n==128
    save(out,dict(prompt=PROMPT,seed=211121281,rows=rows,minimum_carry_repaired_groups=24))


def numeric(text):
    first=next((l.strip() for l in text.splitlines() if l.strip()),"")
    return int(first) if re.fullmatch(r"[0-9]{1,4}",first) else None


def decode(f,tok,pack,steps):
    ids=[tok.encode(PROMPT+f"{a} + {b} =",add_special_tokens=False) for a,b in pack]
    result=[[] for _ in pack]
    active=list(range(len(pack)))
    for _ in range(MAX_NEW):
        tok_ids,pos=pad([ids[j]+result[j] for j in active])
        z=output(f.model,f.run(tok_ids,steps)[-1],pos).argmax(-1).tolist()
        rest=[]
        for j,v in zip(active,z,strict=True):
            result[j].append(v)
            if v!=tok.eos_token_id and "\n" not in tok.decode(result[j],skip_special_tokens=True).strip():
                rest.append(j)
        active=rest
        if not rest:
            break
    text=[tok.decode(x,skip_special_tokens=True) for x in result]
    return [dict(text=t,number=numeric(t),tokens=v) for t,v in zip(text,result,strict=True)]


def summary(raw):
    result={}
    for digit in (2,3,"all"):
        case=[q for q in raw if q["digits"]==digit or digit=="all"]
        repair=[]
        correct=[0,0,0]; parse=[0,0,0]
        n_gain=n_loss=n_other=0
        for v in case:
            ok=False
            for floor,(a,b) in enumerate(v["pairs"]):
                y=a+b
                wrong=y+(10 if floor==0 else -10)
                z=[h[floor]["number"] for h in v["pred"]]
                for j,p in enumerate(z):
                    correct[j]+=p==y
                    parse[j]+=p is not None
                n_gain+=z[0]!=y and z[1]==y
                n_loss+=z[0]==y and z[1]!=y
                n_other+=z[0]!=y and z[1]==y and z[0]!=wrong
                ok=ok or (z[0]==wrong and z[1]==y)
            if ok:
                repair.append(v["id"])
        result[str(digit)]=dict(groups=len(case),correct_by_k=correct,parsed_by_k=parse,
                               wc=n_gain,cw=n_loss,other_wc=n_other,carry_wc_groups=len(repair),carry_ids=repair)
    return result


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    m=a.root/"MANIFEST.json"
    if not m.exists():
        build(m)
    deck=load(m)
    raw_path=a.root/"NATIVE.json"
    rows=load(raw_path)["rows"] if raw_path.exists() else []
    model,tok=setup(a.model)
    f=Forward(model)
    for i in range(len(rows),len(deck["rows"]),8):
        groups=deck["rows"][i:i+8]
        flat=[z for g in groups for z in g["pairs"]]
        p=[decode(f,tok,flat,k) for k in (1,2,3)]
        rows.extend(dict(g,pred=[x[2*j:2*j+2] for x in p]) for j,g in enumerate(groups))
        save(raw_path,dict(rows=rows,manifest_sha256=sha(m),max_new=MAX_NEW))
        print(len(rows),len(deck["rows"]),flush=True)
    result=summary(rows)
    save(a.root/"SUMMARY.json",dict(result=result,gate_pass=result["all"]["carry_wc_groups"]>=24))
    print(result["all"]["carry_wc_groups"],"threshold=24",flush=True)


if __name__=="__main__":
    main()
