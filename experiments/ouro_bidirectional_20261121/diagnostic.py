"""Descriptive error anatomy for the complete native deck."""
from collections import Counter
from pathlib import Path
import json
from bidi import event
from native import load,save,sha

root=Path("results")
src=root/"NATIVE.json"
r=load(src)
reports={}
for split in ("discovery","confirmation"):
    multi=Counter()
    for row in r["rows"]:
        if row["split"]!=split:
            continue
        p,q=row["pred"]
        for j,y in enumerate(row["target"]):
            n1=p[j]["number"]
            n2=q[j]["number"]
            if n1 is None or n2 is None:
                multi["unparsed"]+=1
            if n1!=y and n2==y:
                multi["repair"]+=1
                multi["repair_low_true" if j==0 else "repair_high_true"]+=1
                multi["K1_offset_at_repair:"+str(n1-y if n1 is not None else None)]+=1
            if n1==y and n2!=y:
                multi["regression"]+=1
            if n1!=y and n2!=y:
                multi["both_wrong"]+=1
    reports[split]=dict(multi)
save(root/"DESCRIPTIVE.json",dict(native_sha256=sha(src),by_split=reports))
for s,p in reports.items():
    print(s, {k:v for k,v in p.items() if not k.startswith("K1_offset")}, 
          sorted(((k.partition(":")[2],v) for k,v in p.items() if k.startswith("K1_offset")),
                 key=lambda u:(-u[1],u[0]))[:8])

