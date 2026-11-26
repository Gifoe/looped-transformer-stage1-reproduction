"""Plan disjoint V4 evaluation jobs against the repository's identical historical K4 protocols."""
import argparse
import json
import os
from pathlib import Path
import sys

from d2_worker import digest,save_json
from v4_official import ALIASES, BASELINES, EXTRA, LEGACY, official

COST={"arc_challenge":130,"arc_easy":170,"sciq":85,"mmlu":450,"hellaswag":350,"winogrande":125,
      "piqa":160,"mmlu_pro":1100,"csqa":240,"obqa":110,"squad":6500,"gsm8k":4400}
DEVICES={"0":{"host":"old","local_index":0},"1":{"host":"dual","local_index":0},"2":{"host":"dual","local_index":1}}
MODES={"csqa":("candidate","content","letter"),"obqa":("candidate","content","letter"),
       "squad":("standard","span"),"gsm8k":("standard","gsm_process")}


def plan(root):
    selection=root/"SELECTION_V4.json"
    authority=official.checked_selection(selection)
    if set(authority["candidates"])!=set(ALIASES):
        raise ValueError("The frozen V4 roster includes two CFIT controls and a zero-label anchor")
    runner=root/"code/v4_official.py"
    folder=root/"official"
    jobs=[]

    def add(name,kind,task,arm):
        path=folder/name
        if kind=="legacy":
            modes=("generated",) if task=="mmlu_pro" else ("candidate",)
        else:
            modes=MODES[task]
        identifier=f"v4official.{name}.{task}.{arm}"
        points=[path/kind/f"{task}__{arm}__k4__{mode}.jsonl" for mode in modes]
        paths=(selection,selection.with_suffix(".sha256"),runner,path/"FREEZE.json")
        jobs.append(dict(job_id=identifier,estimated_seconds=COST[task],
                         cmd=[sys.executable,"-u",str(runner),"--root",str(root),"--phase",kind,
                              "--alias",name,"--task",task,"--arm",arm],
                         input_sha256={str(file):digest(file) for file in paths},
                         done_marker=str(root/"receipts"/f"{identifier}.json"),
                         log_path=str(root/"logs"/f"{identifier}.log"),
                         artifacts=[str(reported) for point in points for reported in (point,point.with_suffix(".json"))]))

    for name in ALIASES:
        for task in LEGACY:
            add(name,"legacy",task,"new")
        for task in EXTRA:
            add(name,"extra",task,"new")
    # The original eight historical outputs are read and verified bytewise during aggregation.
    # The four extensions have no verified complete baseline cache on the two hosts.
    for arm in BASELINES:
        for task in EXTRA:
            add(ALIASES[0],"extra",task,arm)
    if len(jobs)!=64:
        raise ValueError("The official registry should hold 36 new-model and 28 extensions-only baseline jobs")
    duration=dict.fromkeys(DEVICES,0.)
    multipliers={"0":1.55,"1":1.,"2":1.}
    for job in sorted(jobs,key=lambda row:(-row["estimated_seconds"],row["job_id"])):
        span=job["estimated_seconds"]
        device=min(DEVICES,key=lambda k:(duration[k]+span*multipliers[k],k))
        job["preferred_gpu"]=device
        duration[device]+=span*multipliers[device]
    out=root/"V4_OFFICIAL_DISPATCH.json"
    payload=dict(devices=DEVICES,jobs=jobs,selection_sha256=digest(selection),planning_wall_seconds=duration)
    if out.exists():
        if json.loads(out.read_text())!=payload:
            raise ValueError("The official dispatch changed under the frozen selection")
    else:
        save_json(out,payload)
    print("V4_OFFICIAL_DISPATCH_FROZEN",len(jobs),digest(out),duration,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    plan(p.parse_args().root)
