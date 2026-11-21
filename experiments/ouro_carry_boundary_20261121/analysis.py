"""Group-paired estimates for the replication and randomized direction controls."""
import argparse
from pathlib import Path

import numpy as np

from boundary import CONDITIONS, scores
from native import load, save, sha


def estimate(values, ci=.95):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(211121901)
    draws = rng.integers(0, len(values), size=(20000, len(values)))
    low, high = np.quantile(values[draws].mean(axis=1), [(1-ci)/2, (1+ci)/2])
    return dict(n=len(values), total=int(values.sum()), mean=float(values.mean()),
                interval=[float(low), float(high)], level=ci)


def repaired(row):
    k1,k2 = [p["number"] for p in row["outputs"]]
    return k1 == row["foil"] and k2 == row["target"]


def behavior(root):
    data = root/"NATIVE.json"
    report = load(data)
    assert report["manifest_sha256"] == sha(root/"MANIFEST.json")
    assert len(report["rows"]) == 288
    raw = [q for q in report["rows"] if q["split"] == "confirmation"]
    assert len(raw) == 192
    diffs = []
    repair = []
    for row in raw:
        q = row["conditions"]
        a = repaired(q["stop"])
        b = repaired(q["same_stop_no_incoming"])
        diffs.append(int(a)-int(b))
        if a:
            repair.append(row["id"])
    ci = estimate(diffs)
    result = dict(native_sha256=sha(data), primary_repaired=len(repair),
                  primary_ids=repair, primary_minus_matched=ci,
                  pass_gate=len(repair) >= 16 and ci["interval"][0] > 0)
    result["pass"] = result.pop("pass_gate")
    save(root/"BEHAVIOR_GATE.json", result)
    print({k:v for k,v in result.items() if k != "primary_ids"}, flush=True)


def causal(root):
    paths = sorted(root.glob("CONFIRMATION_RAW_*.json"))
    assert paths
    native_path = root/"NATIVE.json"
    expected = load(native_path)["rows"]
    native = {r["id"]:r["conditions"] for r in expected if r["split"]=="confirmation"}
    stats = {}
    for path in paths:
        report = load(path)
        assert report["native_sha256"] == sha(native_path)
        assert report["selection_sha256"] == sha(root/"SELECTION.json")
        raw = report["rows"]
        assert len(raw) == len(native) and set(native) == {r["id"] for r in raw}
        parts = {}
        for c in raw[0]["conditions"]:
            checks = [native[r["id"]][c] for r in raw]
            arms = [r["conditions"][c]["outputs"] for r in raw]
            donors = [r["conditions"][c]["donor_target"] for r in raw]
            baseline = [repaired(x) for x in checks]
            is_ok = {m:np.asarray([z[m]["number"] == r["target"] for z,r in zip(arms,checks,strict=True)],dtype=int)
                     for m in arms[0]}
            is_foil = {m:np.asarray([z[m]["number"] == r["foil"] for z,r in zip(arms,checks,strict=True)],dtype=int)
                       for m in arms[0]}
            primary = np.asarray(baseline,dtype=bool)
            assert (np.abs(np.array(donors)-[r["target"] for r in checks])>=100).all()
            events = dict(n=len(raw), primary_repaired=int(primary.sum()),
                          correct={m:int(v.sum()) for m,v in is_ok.items()},
                          arithmetic_foil={m:int(v.sum()) for m,v in is_foil.items()},
                          parsed={m:sum(y[m]["number"] is not None for y in arms) for m in arms[0]},
                          donor_integer_transfer=sum(z["k1_other"]["number"] == donor for z,donor in zip(arms,donors,strict=True)))
            for m,values in is_ok.items():
                events[m+"_correct_in_primary"]=int(values[primary].sum())
                events[m+"_foil_in_primary"]=int(is_foil[m][primary].sum())
            contrasts = {"forward":is_ok["k1_swap"]-is_ok["k1_noise"],
                         "reverse":is_foil["k2_reverse"]-is_foil["k2_noise"],
                         "transfer":is_ok["k1_other"]-is_ok["k1_noise"]}
            events["unconditional"]={m:estimate(z) for m,z in contrasts.items()}
            events["primary_all_groups"]={m:estimate(z*primary, .975) for m,z in contrasts.items()}
            parts[c]=events
        key=path.stem.removeprefix("CONFIRMATION_RAW_")
        stats[key]=parts
    save(root/"CAUSAL_SUMMARY.json",stats)
    print({m:{k:v for k,v in z["stop"].items() if k in {"correct","arithmetic_foil","primary_all_groups"}}
           for m,z in stats.items()}, flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("behavior","causal"))
    p.add_argument("root",type=Path)
    a=p.parse_args()
    {"behavior":behavior,"causal":causal}[a.stage](a.root)

