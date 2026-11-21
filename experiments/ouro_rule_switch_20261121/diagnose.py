"""Descriptive native error anatomy. This script changes no gate."""
from collections import Counter
from pathlib import Path
from native import load,save,sha

root=Path("results")
native=root/"NATIVE.json"
rows=load(native)["rows"]
result={}
for s in ("discovery","confirmation"):
    c=Counter()
    cases=[]
    for g in rows:
        if g["split"]!=s:
            continue
        for i in (0,1):
            p,q=[x[2*i:2*i+2] for x in g["native"]]
            first=[v["number"] for v in p]
            second=[v["number"] for v in q]
            y=g["target"][i]
            c["n_orientations"]+=1
            c["same_K1_integer"]+=first[0] is not None and first[0]==first[1]
            c["same_K1_then_correct_different_K2"]+=first[0] is not None and first[0]==first[1] and second==y
            for o in (0,1):
                if first[o]!=y[o] and second[o]==y[o]:
                    c[f"rule{o}_repair"]+=1
                    c[f"rule{o}_wrong_rule_integer"]+=first[o]==y[1-o]
                    c[f"rule{o}_K1_follows_other_completion"]+=first[o]==first[1-o]
                    c[f"rule{o}_K1_hotspot_delta_{None if first[o] is None else first[o]-y[o]}"]+=1
                    if first[o]==y[1-o]:
                        cases.append([g["id"],i,o])
    result[s]=dict(counts=dict(c),exact_wrong_rule_cases=cases)
save(root/"DESCRIPTIVE.json",dict(native_sha256=sha(native),results=result))
for a,b in result.items():
    print(a,{k:n for k,n in b["counts"].items() if "hotspot" not in k})

