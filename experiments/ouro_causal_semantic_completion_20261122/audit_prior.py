"""Audit real archived inputs, output records, and performance protocol."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess

from native import load,save,sha
from rule import parse

ROOT=Path("/root")
DATA=ROOT/"autodl-tmp"


def interpreted_outputs(x):
    if isinstance(x,dict):
        if {"tokens","text","number"}<=set(x):
            assert parse(x["text"])==x["number"]
            return 1
        return sum(map(interpreted_outputs,x.values()))
    if isinstance(x,list):
        return sum(map(interpreted_outputs,x))
    return 0


def direct(folder,list_files):
    p=ROOT/f"ouro-{folder}"
    out={}
    for f in list_files:
        path=p/f
        x=load(path)
        v=dict(sha256=sha(path),schema=sorted(x))
        if "rows" in x:
            a=x["rows"]
            v.update(n=len(a),record_fields=sorted(a[0]),scored_sequences=interpreted_outputs(a))
            if {"target","foil","kind"}<=set(a[0]):
                v["natural_repair_records"]=sum(r["kind"]=="repair" for r in a)
        out[f]=v
    return out


def jsonl(path):
    rows=0
    first=None
    with path.open(encoding="utf-8") as f:
        for line in f:
            row=json.loads(line)
            first=first or row
            rows+=1
    return dict(n=rows,sha256=sha(path),fields=sorted(first),site=first.get("site"),shape_score=len(first.get("native",[])))


def git(repo,key):
    z=subprocess.check_output(["git","-C",str(repo),"show",f"HEAD:{key}"],stderr=subprocess.DEVNULL)
    return dict(sha256=hashlib.sha256(z).hexdigest(),bytes=len(z)),z


def audit():
    out={}
    selected={
        "borrow-factorial-20261121":["results/DECK.json","results/GRAFT_RAW.json","results/NATURAL_RAW.json"],
        "borrow-block-20261121":["results/DECK.json","results/BLOCK_RAW.json","results/BRANCH_RAW.json","results/LOCK.json"],
        "rule-switch-20261121":["results/DECK.json","results/GRAFT.json","results/NATURAL_RAW.json"],
        "lookup-destination-20261122":["results/DECK.json","results/DEST_RAW.json","results/PROJECTION.json"],
        "lookup-onset-20261121":["results/DECK.json","results/ONSET_RAW.json","results/PAIRED.json"],
    }
    for folder,files in selected.items():
        out[folder]=direct(folder,files)
    org=DATA/"ouro-latent-organization-20261118"
    out["latent-organization-20261118"]={k:jsonl(org/"causal"/k/"rows.jsonl") for k in ("arithmetic","logic","relation")}
    trans=DATA/"ouro-latent-transitions-20261119"
    o={k:jsonl(trans/"runs"/k/"causal.jsonl") for k in ("arithmetic","logic","relation")}
    for f in ("novel_inputs_summary.csv","cross_task_summary.csv","natural_accuracy_paths.csv"):
        p=trans/"results"/f
        with p.open() as stream:
            data=list(csv.DictReader(stream))
        o[f]=dict(n=len(data),sha256=sha(p),fields=sorted(data[0]))
    out["latent-transitions-20261119"]=o
    arc=DATA/"ouro-arc-pareto-20261119"/"release_git"
    prep="experiments/ouro_arc_pareto_20261119/results/"
    m,p=git(arc,prep+"test_predictions.jsonl")
    d=[json.loads(x) for x in p.splitlines() if x]
    keys=sorted(d[0]["scores"])
    assert all(sorted(x["scores"])==keys for x in d)
    n,z=git(arc,prep+"test_metrics.json")
    metrics=json.loads(z)
    computed={}
    for measure in ("acc", "acc_norm"):
        computed[measure]={}
        for k in keys:
            total=0
            for row in d:
                values=row["scores"][k]
                if measure=="acc_norm":
                    values=[a/b for a,b in zip(values,row["characters"],strict=True)]
                total+=max(range(len(values)),key=values.__getitem__)==row["gold"]
            assert metrics[measure][k]["n"]==len(d) and metrics[measure][k]["correct"]==total
            computed[measure][k]=total
    fr,config=git(arc,prep+"freeze.json")
    freeze=json.loads(config)
    out["arc-pareto-20261119"]=dict(n=len(d),test_historically_used=True,scored="full candidate content",
                                   verified_acc_numerators=computed["acc"],verified_acc_norm_numerators=computed["acc_norm"],
                                   predictions=m,aggregate=n,variants=keys,freeze=fr,
                                   development_selected=freeze["strong_baseline_by_depth"])
    better=DATA/"ouro-better-20261120"/"release_git"
    bp="experiments/ouro_better_20261120/"
    o={}
    for f in ("SPLIT_MANIFEST.json","iteration1/FINAL_VERDICTS.json","iteration1/dev/predictions.binding.json",
              "iteration1/dev/predictions.ce.json","iteration1/dev/predictions.distill.json","iteration1/transfer/predictions.binding.json"):
        m,p=git(better,bp+f)
        x=json.loads(p)
        o[f]=dict(**m,keys=list(x)[:12],n=len(x.get("rows",[])) if isinstance(x,dict) else len(x))
    out["better-20261120"]=o
    return out


if __name__=="__main__":
    save(Path("results/PRIOR_AUDIT.json"),audit())
    print("Prior audits complete")
