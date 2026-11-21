"""Post-hoc breakdown of causal switches by direction and arithmetic carry."""
from collections import Counter
from pathlib import Path
from native import load,save,sha

root=Path("results")
src=root/"OPERAND_RAW.json"
raw=load(src)
diagnostic={}
for s in ("discovery","confirmation"):
    counts=Counter()
    for row in raw["rows"]:
        if row["split"]!=s or not row["aligned"]:
            continue
        for k,c in enumerate(row["cases"]):
            shift=c["counterfactual"]-c["answer"]
            carry=sum(x%100 for x in c["recipient"])>=100
            key=f"{shift:+d}/carry={carry}"
            counts[key+"/n"]+=1
            for arm in ("native","prefix","digits","prefix_noise","digits_noise"):
                result=row["modes"][arm][k]["number"]
                counts[key+"/"+arm]+=result==c["counterfactual"]
    diagnostic[s]=dict(sorted(counts.items()))
save(root/"OPERAND_DESCRIPTIVE.json",dict(operand_sha256=sha(src),by_split=diagnostic))
print(diagnostic["confirmation"])

