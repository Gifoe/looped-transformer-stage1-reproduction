"""Describe which original corners preceded an exact native borrowing error."""
from collections import Counter
from pathlib import Path

from factorial import corrected, number
from native import load, save, sha

root=Path("results")
path=root/"NATIVE.json"
rows=load(path)["rows"]
result={}
for split in ("discovery","confirmation"):
    p=Counter()
    for g in rows:
        if g["split"]!=split:
            continue
        p["stems"]+=1
        answer=g["answers"][3]
        y=number(g,0,3)
        clean=all(number(g,0,k)==g["answers"][k] for k in range(3))
        p["K1_three_no_borrow_correct"]+=clean
        p["K1_joint_other_wrong_integer"]+=y is not None and y not in (answer,g["foil"])
        p["K1_joint_same_remainder_wrong_hundred"]+=y is not None and y%100==answer%100 and y!=answer
        if corrected(g,3):
            p["exact_plus100_repair"]+=1
            p["exact_plus100_after_all_three_singles_correct"]+=clean
            p["exact_plus100_amplitude"+str(g["amplitude_arm"])]+=1
    result[split]=dict(p)
save(root/"NATIVE_DESCRIPTIVE.json",dict(summary=result,native_sha256=sha(path)))
print(result,flush=True)

