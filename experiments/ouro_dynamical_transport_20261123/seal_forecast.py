"""Seal dynamical predictions from discovery before running fresh questions."""

import argparse
from pathlib import Path

import numpy as np

import minimal
from archive_audit import dump, read, sha
from state_horizon import score_arrays


def observed(row):
    return (score_arrays(row["scores"]["early"],row["gold"],row["characters"])["utility"]-
            score_arrays(row["scores"]["late"],row["gold"],row["characters"])["utility"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("results","data","t1","k2","qasc","out"):
        p.add_argument(f"--{name}",required=True,type=Path)
    a=p.parse_args()
    if a.out.is_file():
        raise FileExistsError("Existing forecast is immutable")
    phases=("ARC_T1","CSQA_T1","ARC_K2ONLY","ARC_QASC_RECHECK","E3_FIND_arc","E3_FIND_csqa")
    observed_folders={name:a.results/name for name in phases}
    direct={}
    sources={}
    raw=[]
    for name,folder in observed_folders.items():
        manifest=folder/"MANIFEST.json"
        sources[name]={"manifest_sha256":sha(manifest)}
        direct[name]={}
        for item in sorted(folder.glob("k*_K*.json")):
            result=read(item)
            rows=item.with_suffix(".jsonl")
            if sha(rows)!=result["rows_sha256"]:
                raise RuntimeError(f"Discovery prediction SHA differs: {item}")
            sources[name][item.stem]=sha(item)
            if name.startswith("E3"):
                direct[name][item.stem]={arm:v["vs_early"]["mean"] for arm,v in result["results"].items() if arm!="early"}
            else:
                direct[name][item.stem]=result["finite_early_minus_late"]["mean"]
            if name in ("ARC_T1","ARC_K2ONLY"):
                raw.extend(minimal.load_jsonl(rows))
    if not raw or not all(direct.values()):
        raise ValueError("Incomplete real discovery evidence")
    y=np.array([observed(x) for x in raw])
    cal={}
    for field in ("linear_defect","finite_gradient_defect"):
        x=np.array([row["diagnostics"][field] for row in raw])
        cal[field]=float(x@y / (x@x))
    folds=read(a.data/"COHORTS.json")["cohorts"]
    hashes={domain:sha(a.data/f"{domain}_confirm.jsonl") for domain in folds}
    if any(hashes[k]!=v["sha256"] for k,v in folds.items()):
        raise ValueError("A locked confirmation selection changed")
    lock=dict(
        data_sha256=hashes,weights_sha256=[sha(path) for path in (a.t1,a.k2,a.qasc)],
        discovery_sha256=sources,discovery_values=direct,
        regression=dict(source="ARC discovery only, frozen T1 plus ARC K2-only",n=len(y),zero_intercept=cal),
        predictions=[
            "T1 has later-write advantage at fixed K for both domains at 1→2 with K3 and 2→3 with K4.",
            "One remaining loop maximizes mean finite utility for each fixed starting anchor h1/h2 in both domains.",
            "JVP and natural late field have mean cosine <.3 on ARC and CSQA; test off-axis contribution directly.",
            "ARC K2-only preserves a positive late finite change at K6 on fresh ARC.",
            "Defect prediction using the actual transported direction projected on the native receiving gradient reduces MAE relative to the JVP-only defect.",
            "Out-of-sample ARC h3,m1 archived leading direction exceeds the two-sided role-permutation range.",
            "At least one QASC→ARC condition at k2/K4 or k4/K6 has early>late mean finite effect; both directions kept.",
            "In ARC the native-Q/K restoration does better than native-V restoration at both matched receiving boundaries.",
            "For fixed T1 the mean-sign and cross-question coefficient controls stay within ±0.02 utility of late writing; use paired equivalence intervals.",
            "With the same answer text and frozen control coefficients, unrelated-question score shifts explain at most half of the true-context benefit at m=1. Test target contexts without using their correctness labels to assign donors.",
        ],
        decisions="Report every planned path. A direction predicted from discovery can fail. No selection from confirmation.",
        code_sha256=sha(Path(__file__))
    )
    dump(a.out,lock)
    print("LOCKED",sha(a.out),cal,flush=True)


if __name__=="__main__":
    main()
