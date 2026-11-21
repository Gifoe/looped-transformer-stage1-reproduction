from collections import Counter
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

from bidi import prior_pairs, unordered
from native import load
from operand import make_case

root=Path(sys.argv[1])
model=Path(sys.argv[2])
old=prior_pairs(sys.argv[3],sys.argv[4])
d=load(root/"DECK.json")
t=AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True)
counts=Counter()
for row in d["rows"]:
    a,b=row["pairs"]
    i=row["changed_operand"]
    assert a[1-i]==b[1-i] and b[i]-a[i]==200
    assert a[i]%100==b[i]%100
    assert list(map(sum,row["pairs"]))==row["target"]
    assert row["target"]==[row["midpoint"]-100,row["midpoint"]+100]
    keys={unordered(a),unordered(b)}
    assert len(keys)==2 and not keys.intersection(old)
    old.update(keys)
    assert d["midpoint_splits"][str(row["midpoint"])]==row["split"]
    counts["stems"]+=1
    if i==0:
        row=dict(row,pred=[[None,None],[None,None]])
        for k in range(2):
            c=make_case(t,row,k)
            counts["oriented_first"]+=1
            if c is not None:
                if c["prefix"]==list(range(c["prefix"][0],c["prefix"][-1]+1)):
                    counts["contiguous"]+=1
                counts["aligned"]+=1
                counts[f'slots_{len(c["prefix"])}']+=1
print(json.dumps(counts,sort_keys=True))

