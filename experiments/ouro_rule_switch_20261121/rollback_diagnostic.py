"""Descriptive specificity check by the original hundred-transfer error."""
from collections import Counter
from pathlib import Path
from native import load,save,sha
from boundary_control import needs

root=Path("results")
source=root/"NATURAL_RAW.json"
samples=load(source)["rows"]
index={r["id"]:r for r in load(root/"NATIVE.json")["rows"]}
report={}
for o in (0,1):
    for cat in ("transfer_omitted","other_repair","native_correct"):
        a=Counter()
        for r in samples:
            if r["rule"]!=o or not r["aligned"]:
                continue
            g=index[r["id"]]
            first=g["a"][r["orientation"]]
            b=g["b"]
            foil=r["target"]+(-100 if o==0 else 100)
            specific=needs(first,b,o) and r["wrong"]==foil
            label="native_correct" if r["cohort"]=="control" else ("transfer_omitted" if specific else "other_repair")
            if cat!=label:
                continue
            a["n"]+=1
            for mode,cell in r["outputs"].items():
                output=cell["number"]
                a[mode+":original"]+=output==r["target"]
                a[mode+":exact_K1"]+=r["wrong"] is not None and output==r["wrong"]
                a[mode+":signed_foil"]+=output==foil
        report[f"rule{o}/{cat}"]=dict(a)
save(root/"ROLLBACK_DESCRIPTIVE.json",dict(natural_raw_sha256=sha(source),results=report))
print(report,flush=True)

