"""Describe the forward graft by the donors' and receiver's native K1 answers."""
from collections import Counter
from pathlib import Path
from native import load, save, sha

root=Path("results")
index={r["id"]:r for r in load(root/"NATIVE.json")["rows"]}
path=root/"GRAFT_RAW.json"
summary={}
for split in ("discovery","confirmation"):
    result=Counter()
    for r in load(path)["rows"]:
        x=r["forward"]
        if r["split"]!=split or not x["aligned"]:
            continue
        g=index[r["id"]]
        k1=[o["number"] for o in g["pred"][0]]
        truth=g["answers"]
        aligned=x["pred"]["ab"]["number"]==truth[3]
        b=[k1[i]!=truth[i] for i in range(3)]
        flags={"all_stems":True,"recipient_wrong":b[0],"both_donors_wrong":b[1] and b[2],
               "all_three_wrong":all(b),"native_joint_corrected_exact_foil":k1[3]==g["foil"]}
        for label,on in flags.items():
            result[label+"_n"]+=on
            result[label+"_joint_graft_exact"]+=on and aligned
    summary[split]=dict(result)
save(root/"GRAFT_DESCRIPTIVE.json",dict(summary=summary,graft_sha256=sha(path)))
print(summary,flush=True)

