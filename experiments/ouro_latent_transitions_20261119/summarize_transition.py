"""Aggregate paired group estimates, enforce the advancement gate and audit outputs."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

TASKS = ("relation", "logic", "arithmetic")
WIDTHS = (96, 2048)
SIMPLE = ("identity_raw", "identity", "ridge_raw", "ridge", "rank8", "rank24", "context", "random")


def table(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        out.writeheader()
        out.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def partition(rows, cols):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in cols)].append(r)
    return out


def resample(n):
    return np.random.default_rng(20261119).integers(n, size=(6000,n), dtype=np.int32)


def pairs(x, y):
    assert len(x) == len(y) > 0
    take = resample(len(x))
    den = y[take].mean(1)
    assert np.all(den > 0)
    improvements = 1-x[take].mean(1)/den
    return 1-x.mean()/y.mean(), *np.quantile(improvements, [.025,.975])


def summary(rows, ids, metrics):
    output = []
    for values, members in partition(rows, ids).items():
        metrics_data = {m: np.array([float(r[m]) for r in members]) for m in metrics}
        take = resample(len(members))
        for metric,x in metrics_data.items():
            lo,hi = np.quantile(x[take].mean(1), [.025,.975])
            output.append({**dict(zip(ids,values)), "groups":len(x), "metric":metric,
                           "mean":x.mean(),"lower":lo,"upper":hi})
    return output


def compare(rows, task, width, split, metric, candidate, baselines, arm):
    by = partition([r for r in rows if r["split"] == split and r["arm"] == arm], ("method",))
    cand = {r["id"]:float(r[metric]) for r in by[(candidate,)]}
    result = []
    for ref in baselines:
        base = {r["id"]:float(r[metric]) for r in by[(ref,)]}
        assert cand.keys() == base.keys()
        keys = sorted(cand)
        x = np.array([cand[k] for k in keys])
        y = np.array([base[k] for k in keys])
        point,lo,hi = pairs(x,y)
        result.append(dict(task=task,width=width,split=split,metric=metric,arm=arm,
            candidate=candidate,baseline=ref,groups=len(keys),candidate_error=x.mean(),
            baseline_error=y.mean(),relative_improvement=point,lower=lo,upper=hi))
    return result


def audit(runs, result):
    audit = {"native": {}, "model_configs_match_source": True, "checkpoint": {}}
    for task in TASKS:
        folder = runs/task
        completion = read(folder/"capture.json")
        natural, causal = lines(folder/"natural.jsonl"), lines(folder/"causal.jsonl")
        assert completion["natural_worlds"] == len(natural) == 4000
        assert completion["causal_worlds"] == len(causal) == 496
        assert completion["checks"]["exact_self_and_full_boundary_swap"] == 1
        hash_checks = {k:sha(folder/(k+"_full.npy")) == completion[k+"_sha256"] for k in ("natural","causal")}
        assert all(hash_checks.values())
        split_counts = Counter({k:len(v)//4 for (k,),v in partition(natural, ("split",)).items()})
        assert split_counts == {"fit":300,"discover":100,"confirm":300,"transfer":300}
        for k,cases in partition([r for r in causal if r["arm"] == "semantic"], ("split",)).items():
            groups = partition(cases, ("id",))
            assert k in (("confirm",),("transfer",)) and len(groups) == 60
            assert all({r["world"] for r in q} == {"r","x","y","d"} for q in groups.values())
        audit["native"][task] = {**completion, "array_hashes_match": hash_checks}
        audit["checkpoint"][task] = read(folder/"runtime.json")
        for width in WIDTHS:
            folder = result/str(width)/task
            manifest = read(folder/"freeze.json")
            for source, want in manifest["implementation"].items():
                assert sha(Path(__file__).parent/source) == want
            assert len(table(folder/"natural_by_group.csv")) == 5400
            assert len(table(folder/"causal_by_group.csv")) == 1280
            audit["native"][task][f"fit_{width}_model_archive_sha256"] = sha(folder/"models.pkl")
    audit["status"] = "passed"
    return audit


def run(runs, result):
    audit_data = audit(runs, result)
    summaries, gates, contrasts = [], [], []
    for width in WIDTHS:
        for task in TASKS:
            folder = result/str(width)/task
            freeze = read(folder/"freeze.json")
            base, candidate = freeze["primary_linear"], freeze["primary_nonlinear"]
            natural, causal = [table(folder/f"{stage}_by_group.csv") for stage in ("natural","causal")]
            compact = ("width","task","split","method","arm")
            core = ("response_mse", "delta_energy", "cosine", "norm_ratio", "semantic_error",
                    "behavior_sqerr", "semantic_real_signed", "semantic_pred_signed",
                    "behavior_real_margin", "behavior_pred_margin", "semantic_direction_agreement",
                    "behavior_direction_agreement")
            summaries.extend(summary(causal, compact, core))
            summaries.extend(summary(natural, compact, ("state_mse","delta_energy","k2_agreement",
                                 "intermediate_k2_prediction_accuracy","final_k2_prediction_accuracy")))
            split_gates = {}
            for split in ("confirm","transfer"):
                a = compare(natural,task,width,split,"state_mse",candidate,(base,),"natural")
                b = compare(causal,task,width,split,"response_mse",candidate,SIMPLE,"semantic")
                c = compare(causal,task,width,split,"semantic_error",candidate,SIMPLE,"semantic")
                d = compare(causal,task,width,split,"behavior_sqerr",candidate,SIMPLE,"semantic")
                contrasts.extend(a+b+c+d)
                hardest = min(b, key=lambda r:r["baseline_error"])
                sem = min(c, key=lambda r:r["baseline_error"])
                behavior = min(d, key=lambda r:r["baseline_error"])
                split_gates[split] = {
                    "natural": bool(a[0]["lower"] > 0), "causal": all(
                        r["relative_improvement"] >= .10 and r["lower"] > 0 for r in b),
                    "semantic": bool(sem["lower"] > 0), "behavior": bool(behavior["lower"] > 0),
                    "natural_relative_improvement": float(a[0]["relative_improvement"]),
                    "hardest_causal_baseline": hardest["baseline"],
                    "hardest_causal_relative_improvement": float(hardest["relative_improvement"]),
                }
            passes = width == 2048 and all(s["natural"] and s["causal"] for s in split_gates.values()) and (
                split_gates["transfer"]["semantic"] and split_gates["transfer"]["behavior"])
            gates.append(dict(task=task,width=width,candidate=candidate,natural_baseline=base,
                              splits=split_gates,passes=passes))
    diagnostic = [r for width in WIDTHS for task in TASKS for r in
                  table(result/str(width)/task/"diagnostic_by_group.csv")]
    write(result/"diagnostic_summary.csv", summary(diagnostic, ("width","task","split","target","state","role"),
                                                   ("accuracy",)))
    write(result/"cross_task_summary.csv", summary(table(result/"cross_task_by_group.csv"),
          ("source_tasks","task","method","split","arm"), ("mse","identity_energy","cosine")))
    write(result/"summary.csv", summaries)
    write(result/"paired_comparisons.csv", contrasts)
    (result/"advancement.json").write_text(json.dumps({"task_gates":gates,
        "stage2_authorized_by_results": any(g["passes"] for g in gates)},indent=2)+"\n",encoding="utf-8")
    (result/"execution_audit.json").write_text(json.dumps(audit_data,indent=2)+"\n",encoding="utf-8")
    print([(r["task"],r["width"],r["passes"]) for r in gates], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", type=Path)
    p.add_argument("results", type=Path)
    a = p.parse_args()
    run(a.runs, a.results)
