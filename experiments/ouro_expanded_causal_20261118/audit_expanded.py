"""Verify populations, disjointness, effect witnesses, and frozen provenance."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from collect import digest, load_lines, write_json
from factory import TASKS

CORE={"relation":240,"logic":180,"arithmetic":240}
BENCH={"arc":299,"prontoqa":270,"hellaswag":150}
JOBS={**{f"main-{t}":120 for t in TASKS},"A-relation":90,"B-relation":90,"C-logic":90,"C-arithmetic":90}


def frozen(path,source):
    f=json.loads((path/"frozen_protocol.json").read_text())
    assert f["input_sha256"]==digest(path/"inputs.jsonl")
    assert f["intervention_sha256"]==digest(source)
    if "calibration_sha256" in f:
        assert f["calibration_sha256"]==digest(path/"calibration"/"rows.jsonl")


def causal(root,job,n,encoding,seen):
    f=root/job
    receipt=json.loads((f/"complete.json").read_text())
    records=load_lines(f/"rows.jsonl")
    assert len(records)==receipt["rows"]==2*n and receipt["groups"]==n
    cases=defaultdict(list)
    for r in records:
        g=encoding[r["id"]]
        assert r["id"] not in seen
        w=g["worlds"]
        receive=r["receive"]
        donor=r["donor"]
        assert w[receive]["gold"]==w[donor]["gold"]==r["original"]
        alt="x" if receive=="r" else "y"
        assert w[alt]["gold"]==r["counterfactual"]!=r["original"]
        assert w[receive]["mid"]!=w[donor]["mid"]
        expected="confirm" if job.startswith("main") else job.split("-")[0]
        assert g["split"]==expected
        cases[r["id"]].append(receive)
        if job=="B-relation":
            assert r["arms"]["self"]["scores"]==r["native"]
        else:
            for role,v in r["arms"].items():
                arms=v["arms"]
                assert arms["self"]["scores"]==arms["map_only"]["scores"]==r["native"]
                assert arms["semantic"]["scores"][0]==r["native"][0]
                if job=="A-relation" and role=="payload":
                    assert v["same_token_ids"]
    assert len(cases)==n and all(set(x)=={"r","d"} for x in cases.values())
    seen.update(cases)
    return {"groups":n,"directions":2*n,"raw_sha256":digest(f/"rows.jsonl")}


def unique_signatures(groups):
    signatures=[(g["task"],g["family"],json.dumps(g["meta"]["signature"]))
                for g in groups if g["task"] in TASKS]
    assert len(signatures)==len(set(signatures))
    return len(signatures)


def natural_population(root):
    expected=Counter()
    base=load_lines(root/"inputs.jsonl")
    for g in base:
        if g["split"] in ("dev","natural","confirm","benchmark"):
            expected[(g["task"],g["id"])]=4 if g["task"] in TASKS else 1
    obtained=Counter((r["task"],r["id"]) for stage in ("benchmarks","calibration","controlled")
                     for r in load_lines(root/stage/"rows.jsonl"))
    assert obtained==expected
    counts=Counter()
    for task,g in obtained:
        counts[task]+=1
    assert counts==Counter(CORE|BENCH)
    return dict(counts)


def verify_screen(root,records):
    d=root/"screened"
    rule=json.loads((d/"screen_rule.json").read_text())
    assert digest(d/"pool.jsonl")==rule["pool_sha256"]
    s=json.loads((d/"screen_complete.json").read_text())
    assert digest(d/"natural_results.jsonl")==s["results_sha256"]
    rows={r["id"]:r for r in load_lines(d/"natural_results.jsonl")}
    c=Counter()
    for g in load_lines(d/"inputs.jsonl"):
        r=rows[g["id"]]
        assert r["eligible_K4"] and all(x["pred"][3]==x["gold"] for x in r["worlds"].values())
        c[g["task"]]+=1
    assert dict(c)==s["retained"]
    for t,n in c.items():
        records[f"screened/C-{t}"]=causal(d,f"C-{t}",n,{g["id"]:g for g in load_lines(d/"inputs.jsonl")},set())
        for row in load_lines(d/f"C-{t}"/"rows.jsonl"):
            truth=rows[row["id"]]["worlds"]
            assert row["native"]==truth[row["receive"]]["scores"]
            assert row["donor_native"]==truth[row["donor"]]["scores"]
    return s


def main(root):
    source=Path(__file__).with_name("intervene.py")
    input_main=load_lines(root/"inputs.jsonl")
    inputs_comp=load_lines(root/"competence"/"inputs.jsonl")
    rename_groups=load_lines(root/"renaming"/"inputs.jsonl")
    screened=load_lines(root/"screened"/"pool.jsonl")
    assert len({g["id"] for g in input_main+inputs_comp+rename_groups+screened})==len(input_main+inputs_comp+rename_groups+screened)
    unique=unique_signatures(input_main+inputs_comp+rename_groups+screened)
    for p in (root,root/"competence",root/"screened",root/"boolean"):
        frozen(p,source)
    protocol=json.loads((root/"manifest.json").read_text())
    assert protocol["factory_sha256"]==digest(source.with_name("factory.py"))
    candidates=(source.with_name("collect.py"),source.parent/"provenance"/"executed_collect.py")
    assert protocol["encoder_sha256"] in {digest(p) for p in candidates if p.exists()}
    assert protocol["inputs_sha256"]==digest(root/"inputs.jsonl")
    rprotocol=json.loads((root/"renaming"/"protocol.json").read_text())
    assert rprotocol["input_sha256"]==digest(root/"renaming"/"inputs.jsonl")
    assert rprotocol["source_sha256"]==digest(source.with_name("renaming.py"))
    rrows=load_lines(root/"renaming"/"rows.jsonl")
    assert len(rrows)==180 and len({r["id"] for r in rrows})==len(rename_groups)==90
    for r in rrows:
        assert r["native"]==r["arms"]["self"]["scores"]
    evidence={}
    for location,groupings,jobs in ((root,input_main,JOBS),(root/"competence",inputs_comp,
            {f"{j}-{t}":90 for j in ("main","C") for t in ("logic","arithmetic")})):
        all_seen=set()
        ix={g["id"]:g for g in groupings}
        for job,n in jobs.items():
            evidence[f"{'expanded' if location==root else 'competence'}/{job}"]=causal(location,job,n,ix,all_seen)
    result={"status":"passed","natural_cohorts":natural_population(root),"causal":evidence,
            "unique_generated_group_signatures":unique,
            "renaming":{"groups":90,"rows":180,"rows_sha256":digest(root/"renaming"/"rows.jsonl")}}
    result["competence_screen"]=verify_screen(root,evidence)
    write_json(root/"execution_audit.json",result)
    print("Audit verified",len(evidence),"causal jobs plus renaming, exact screening-score replay, unique groups",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("study",type=Path)
    main(p.parse_args().study)
