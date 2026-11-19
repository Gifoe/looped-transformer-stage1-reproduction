"""Archive complete native token trajectories for predetermined causal cases."""
import argparse
from pathlib import Path

import numpy as np
import torch

from capture_transition import (TASKS, capture, get_encoding, load_runtime, positions,
                                load_lines, source_region, write_json, digest)


def capture_case(model, tok, g, row, out):
    enc = {w:get_encoding(tok,g,g["worlds"][w]) for w in (row["world"],row["donor"])}
    clean, donor = enc[row["world"]], enc[row["donor"]]
    roles = positions(g,clean)
    src = source_region(g,clean)
    donor_bank = capture(model,donor,src,depth=1,tap_layers=())
    borrowed = donor_bank[2][1,24][[donor_bank[4].index(i) for i in src]]
    traces = [capture(model,clean,[],edits=edit,full=True,depth=4)
              for edit in (None,{(1,24):(src,borrowed)})]
    assert [t[0] for t in traces] == [row["original_scores"],row["edited_scores"]]
    events = list(traces[0][2])
    states = np.stack([
        torch.stack(list(t[2].values())).contiguous().view(torch.int16).cpu().numpy()
        for t in traces
    ])
    name = f"{g['task']}-{row['split']}-{row['world']}.npy"
    path = out/name
    np.save(path,states)
    role_deltas = {name:[] for name in roles}
    for k in events:
        delta = traces[1][2][k].float()-traces[0][2][k].float()
        for name,indices in roles.items():
            role_deltas[name].append(float(delta[indices].mean(0).norm()))
    w,d = g["worlds"][row["world"]],g["worlds"][row["donor"]]
    return {**row, "input_ids":clean["ids"],"roles":roles,"options":g["options"],
            "prompt":w["prompt"], "semantics":{"receiver":w["semantic"],"donor":d["semantic"]},
            "events":events,"full_role_deltas":role_deltas,"tensor":{
                "path":path.name,"bytes":path.stat().st_size,"array_shape":list(states.shape),
                "storage":"BF16 bit pattern in NumPy int16; clean, edited × event × token × channel",
                "sha256":digest(path)}}


def run(past, runs, out, model_path):
    out.mkdir(parents=True)
    groups={g["id"]:g for g in load_lines(past/"groups.jsonl")}
    model,tok,runtime = load_runtime(model_path)
    all_cases = []
    with torch.inference_mode():
        for task in TASKS:
            rows=load_lines(runs/task/"causal.jsonl")
            for split in ("confirm","transfer"):
                eligible=[r for r in rows if r["arm"]=="semantic" and r["world"]=="r"
                          and r["split"]==split and "role_deltas_by_event" in r]
                assert len(eligible)==1
                r=eligible[0]
                all_cases.append(capture_case(model,tok,groups[r["id"]],r,out))
                print(task,split,r["id"],flush=True)
    write_json(out/"cases.json",{"runtime":runtime,"cases":all_cases})


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for arg in ("past","runs","out","model"):
        p.add_argument(arg,type=Path)
    a=p.parse_args()
    run(a.past,a.runs,a.out,a.model)
