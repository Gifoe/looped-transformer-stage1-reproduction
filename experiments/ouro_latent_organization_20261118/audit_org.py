"""Verify exact native trajectories, dataset partitions and causal receipts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from manifest_org import TASKS,GOALS,SPLITS,digest,load_lines,write_json
from capture_org import PROJECTION_WIDTH,ROLES


def audit(study,allow_server_arrays):
    rows=load_lines(study/"groups.jsonl")
    ids={g["id"]:g for g in rows}
    assert len(ids)==len(rows)
    assert Counter(g["task"] for g in rows)==Counter(GOALS)
    manifests=json.loads((study/"manifest.json").read_text())
    freeze=json.loads((study/"analysis_freeze.json").read_text())
    assert digest(study/"groups.jsonl")==manifests["group_sha256"]==freeze["inputs_sha256"]
    assert digest(study/"discovery_readouts.csv")==freeze["discovery_sha256"]
    assert freeze["analysis_sha256"]==digest(Path(__file__).with_name("analyze_org.py"))
    assert manifests["generator_sha256"]==digest(Path(__file__).with_name("manifest_org.py"))
    slots=json.loads((study/"slot_balance_audit.json").read_text())
    assert slots["previous_counts"]["relation|r|0"]==slots["previous_counts"]["logic|r|0"]==1000
    for task in TASKS:
        counts=Counter(g["worlds"]["r"]["gold"] for g in rows if g["task"]==task)
        assert len(counts)>=2 and max(counts.values()) < 600
    signs=[(g["task"],g["family"],json.dumps(g["meta"]["signature"]))
           for g in rows if g["task"] in TASKS]
    assert len(set(signs))==len(signs)==3000
    for task in TASKS:
        cohort=[g for g in rows if g["task"]==task]
        assert Counter(g["split"] for g in cohort)==Counter(SPLITS)
        assert all((g["family"]==2)==(g["split"]=="transfer") for g in cohort)
        for r in cohort:
            w=r["worlds"]
            assert w["r"]["gold"]==w["d"]["gold"]
            assert w["r"]["mid"]==w["y"]["mid"]!=w["d"]["mid"]
            assert w["r"]["gold"]!=w["y"]["gold"]
    catalog=[]
    natural={}
    keys={}
    for t in GOALS:
        folder=study/"natural"/t
        m=json.loads((folder/"complete.json").read_text())
        assert m["input_sha256"]==manifests["group_sha256"] and m["independent_K1_K4_logits_exact"]
        data=load_lines(folder/"rows.jsonl")
        expected=[(g["id"],w) for g in rows if g["task"]==t for w in g["capture_worlds"]]
        assert [(r["id"],r["world"]) for r in data]==expected
        assert len(data)==m["rows"]
        layer=np.load(folder/"stats.npy",mmap_mode="r")
        assert layer.shape==(len(data),4,25,len(ROLES),4)
        assert np.isfinite(layer[...,0]).all()
        assert np.isfinite(layer[:,1:,:,:,2:]).all()
        for i,r in enumerate(data):
            prompt=ids[r["id"]]["worlds"][r["world"]]["prompt"]
            assert r["i"]==i and r["prompt_sha256"]==hashlib.sha256(prompt.encode()).hexdigest()
            assert max(max(s) for s in r["semantic_roles"].values())<len(r["ids"])
            keys[r["id"],r["world"]]=r["scores"]
        if allow_server_arrays:
            p=folder/"vectors.npy"
            v=np.load(p,mmap_mode="r")
            assert v.shape==(len(data),4,25,len(ROLES),PROJECTION_WIDTH) and np.isfinite(v).all()
            catalog.append(dict(file=str(p.relative_to(study)),bytes=p.stat().st_size,sha256=digest(p)))
        natural[t]={"groups":m["groups"],"worlds":len(data),"runtime_seconds":m["gpu_phase_seconds"]}
    causal={}
    for t in TASKS:
        p=study/"causal"/t
        protocol=json.loads((p/"protocol.json").read_text())
        r=load_lines(p/"rows.jsonl")
        assert len(r)==180 and len(set(x["id"] for x in r))==len(protocol["dataset_id"])==90
        assert set(protocol["dataset_id"])=={x["id"] for x in r}
        assert protocol["analysis_freeze_sha256"]==digest(study/"analysis_freeze.json")
        rcounts=Counter()
        for row in r:
            assert row["native"]==row["arms"]["self"]["scores"]==keys[row["id"],row["receive"]]
            assert row["donor_native"]==keys[row["id"],row["donor"]]
            assert ids[row["id"]]["split"] in ("confirm","transfer")
            rcounts[ids[row["id"]]["split"]]+=1
        assert rcounts==Counter(confirm=120,transfer=60)
        causal[t]={"groups":90,"directions":180,"natural_and_self_score_parity":True}
    full=study/"full_token"
    raw=json.loads((full/"catalog.json").read_text())
    for item in raw:
        assert item["score_parity"]
        if allow_server_arrays:
            p=full/item["file"]
            assert digest(p)==item["sha256"]
            catalog.append(dict(file=str(p.relative_to(study)),bytes=p.stat().st_size,sha256=item["sha256"]))
    repeat=study/"route_retest"
    repeated=load_lines(repeat/"rows.jsonl")
    rp=json.loads((repeat/"protocol.json").read_text())
    tested={r["id"] for r in repeated}
    rel=json.loads((study/"causal"/"relation"/"protocol.json").read_text())
    assert len(repeated)==120 and len(tested)==60 and tested==set(rp["ids"])
    assert tested.isdisjoint(rel["dataset_id"]) and rp["input_sha256"]==manifests["group_sha256"]
    assert all(r["self_exact"] and r["native"]==keys[r["id"],r["receive"]] for r in repeated)
    width=study/"width_control"
    for t in TASKS:
        p=width/f"{t}_rows.jsonl"
        checks=load_lines(p)
        assert len(checks)==560
        assert all(r["native_scores"]==keys[r["id"],r["world"]] for r in checks)
        if allow_server_arrays:
            p=width/f"{t}_bf16_bits.npy"
            obj=np.load(p,mmap_mode="r")
            assert obj.shape==(560,4,2,3,2048) and obj.dtype==np.int16
            catalog.append(dict(file=str(p.relative_to(study)),bytes=p.stat().st_size,sha256=digest(p)))
    result={"status":"passed","unique_groups":len(rows),"natural":natural,"causal":causal,
            "representative_full_token_cases":len(raw),"raw_tensor_inventory_verified":allow_server_arrays,
            "native_prompt_only":True,"route_retest_groups":len(tested),"full_width_readout_worlds":1680}
    if allow_server_arrays:
        write_json(study/"raw_tensor_inventory.json",catalog)
        write_json(study/"execution_audit.json",result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("study",type=Path)
    cli.add_argument("--include-server-tensors",action="store_true")
    args=cli.parse_args()
    audit(args.study,args.include_server_tensors)
