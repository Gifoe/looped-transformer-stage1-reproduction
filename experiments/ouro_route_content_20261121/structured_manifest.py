"""Counterbalance two selector routes and two receiver rule tables."""
import argparse
import random
from pathlib import Path

from transformers import AutoTokenizer

from native import save, sha

KEYS="copper silver bronze golden iron brass jade steel".split()
BOXES="blue yellow green black purple white pink orange".split()
COLORS="red blue green white black brown yellow gray pink purple".split()
OBJECTS="cup bowl plate fork spoon mug vase pot book hat".split()


def atom(tok,word):
    return len(tok.encode(" "+word,add_special_tokens=False))==1


def prompt(layout,keys,boxes,items,first,second,selector):
    if layout=="rules":
        one=[f"The {keys[i]} key opens the {boxes[i]} box." for i in first]
        two=[f"The {boxes[i]} box contains the {items[i]}." for i in second]
        return "\n".join(["Read the facts.",*one,*two,
                         f"Question: Using the {keys[selector]} key, which item can you recover?","Answer:"])
    one=[f"The {keys[i]} key is for the {boxes[i]} box." for i in first]
    two=[f"Inside the {boxes[i]} box is the {items[i]}." for i in second]
    return "\n".join(["Recovery notes:", *two,*one,
                     f"Question: Which item is in the box for the {keys[selector]} key?","Answer:"])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
    materials=[w for w in KEYS if atom(tok,w)]
    hues=[w for w in BOXES if atom(tok,w)]
    items=[f"{q} {w}" for q in COLORS for w in OBJECTS
          if len(tok.encode(" "+q+" "+w,add_special_tokens=False))==2]
    assert len(items)>20
    rng=random.Random(21112117)
    seen=set()
    rows=[]
    counts={}
    encode=lambda s:tok.encode(s,add_special_tokens=False)
    for phase,count in (("pilot",24),("confirmation",128)):
        for layout in ("rules","reverse"):
            n=0
            for attempt in range(3000):
                if n==count//2:
                    break
                keys=rng.sample(materials,4); boxes=rng.sample(hues,4)
                terms=rng.sample(items,8)
                if len({s.split()[1] for s in terms[:6]})!=6:
                    continue
                first=rng.sample(range(4),4); second=rng.sample(range(4),4)
                sig=tuple(keys+boxes+terms)
                if sig in seen:
                    continue
                left=terms[:4]
                right=left[2:]+left[:2]
                donor=terms[4:]
                receivers=[[prompt(layout,keys,boxes,v,first,second,s) for s in range(2)] for v in (left,right)]
                donors=[prompt(layout,keys,boxes,donor,first,second,s) for s in range(2)]
                enc=[encode(s) for s in [*donors,*receivers[0],*receivers[1]]]
                if len({len(z) for z in enc})!=1:
                    continue
                diffs=[{i for i,(u,v) in enumerate(zip(enc[q],enc[q+1],strict=True)) if u!=v} for q in (0,2,4)]
                if len(diffs[0])!=1 or not diffs[0]==diffs[1]==diffs[2]:
                    continue
                seen.add(sig)
                n+=1
                choices=left+donor[:2]
                rng.shuffle(choices)
                y=[[choices.index(v) for v in table[:2]] for table in (left,right)]
                dy=[choices.index(v) for v in donor[:2]]
                assert len({*y[0],*y[1],*dy})==6
                rows.append(dict(id=f"{phase}-{layout}-{n:03}",phase=phase,layout=layout,choices=choices,
                                  receivers=receivers,donors=donors,targets=y,donor_targets=dy,
                                  query_token=next(iter(diffs[0])),prompt_tokens=len(enc[0]),
                                  anchor=(n-1)%2))
            assert n==count//2
            counts[f"{phase}/{layout}"]=n
    save(a.out,dict(rows=rows,expected=counts,seed=21112117,model_config_sha256=sha(a.model/"config.json")))


if __name__=="__main__":
    main()

