"""Freeze one experimental checkpoint family and shared per-depth schedules."""
import argparse
import json
from pathlib import Path

from minimal import dump, sha
from screen import choose


def run(source, target):
    provisional = json.loads((source / "DEV_PROPOSAL.json").read_text())
    table = json.loads((source / "DEV_RESULTS.json").read_text())
    assert provisional["validation_results_sha256"] == sha(source / "DEV_RESULTS.json")
    policies = provisional["policies"]
    families = sorted({p["checkpoint"] for p in policies.values()}-{"t1"})
    plans = []
    for name in ["t1", *families]:
        variants = [key for key,p in policies.items() if p["checkpoint"] in ("t1",name)]
        selected = choose(table,variants)
        gains = []
        for depth, key in selected.items():
            increments = [one[depth][key]["correct"]["delta_pp"] for one in table.values()]
            gains.append((min(increments),sum(increments)/len(increments)))
        score = (sum(x[0] for x in gains), sum(x[1] for x in gains), name=="t1")
        plans.append((score,name,selected))
    score, name, selected = max(plans)
    dump(target,dict(provisional,selected=selected,alternate_weight_family=name,
                     all_coherent_plans=[dict(weight=n,policy=x,utility=s) for s,n,x in plans],
                     cross_validation_score=score))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dev",type=Path)
    p.add_argument("target",type=Path)
    a=p.parse_args()
    run(a.dev,a.target)
