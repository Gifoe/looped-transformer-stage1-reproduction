"""Describe whether the native model used the proper transfer into hundreds."""
from collections import Counter
from pathlib import Path
from native import load, save, sha

root=Path("results")
path=root/"NATIVE.json"
out={}
for split in ("discovery","confirmation"):
    result={}
    for o,rule in enumerate(("add","reverse_subtract")):
        group=[Counter(),Counter()]
        for g in load(path)["rows"]:
            if g["split"]!=split:
                continue
            for i,a in enumerate(g["a"]):
                b=g["b"]
                target=g["target"][i][o]
                hundreds_a,lo_a=divmod(a,100)
                hundreds_b,lo_b=divmod(b,100)
                u=(lo_a+lo_b)//100 if o==0 else int(lo_b<lo_a)
                independent_lower=((lo_a+lo_b) if o==0 else (lo_b-lo_a))%100
                if o==0:
                    y=100*(hundreds_a+hundreds_b)+independent_lower
                    wrong=target+100 if not u else y
                else:
                    y=100*(hundreds_b-hundreds_a)+independent_lower
                    wrong=target-100 if not u else y
                if u:
                    assert abs(target-y)==100
                d=group[u]
                d["n"]+=1
                p,q=[v[2*i+o]["number"] for v in g["native"]]
                d["K1_correct"]+=p==target
                d["K2_correct"]+=q==target
                d["stopped_or_spurious_transfer_K1"]+=p==wrong
                d["stopped_or_spurious_transfer_K2"]+=q==wrong
                d["targeted_repair"]+=p==wrong and q==target
                d["all_repairs"]+=p!=target and q==target
        result[rule]=dict(no_transfer_needed=dict(group[0]),transfer_needed=dict(group[1]))
    out[split]=result
save(root/"CARRY_DESCRIPTIVE.json",dict(native_sha256=sha(path),by_split=out))
print(out["confirmation"],flush=True)

