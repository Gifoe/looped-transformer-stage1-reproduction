"""Disentangle the queried and unqueried records in the two-hop task."""
import argparse
import hashlib
import re
from pathlib import Path
import time

import torch

from analyze import clustered, digest, interval, load_lines, write_csv
from design import aligned_group
from run import (DIRECTIONS, aligned_inputs, dose_matched, dump_line, encode,
                 fast_forward, load_runtime, margin, norm_delta, predictions,
                 trace_deltas, write_json)

SITE = (2, 5)
BLOCK = (2, 11)
ARMS = {"target_minus_native": ("target", "natural"),
        "target_minus_unqueried": ("target", "unqueried"),
        "target_minus_random": ("target", "random"),
        "unqueried_minus_native": ("unqueried", "natural"),
        "blocked_minus_native": ("blocked", "natural"),
        "target_minus_blocked": ("target", "blocked")}


def record_positions(tok, prompt, anchor):
    subject = re.search(r"destination for (\w+)\.", prompt)[1]
    offsets = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    records = {}
    for item in re.finditer(r"^(\w+) is assigned to the (\w+) box\.$", prompt, re.MULTILINE):
        start, end = item.span(2)
        records[item[1]] = [i for i, (a,b) in enumerate(offsets[:anchor+1])
                             if a < end and b > start]
    assert len(records) == 3 and all(records.values())
    return subject, records


def align(tok, group):
    if not aligned_inputs(tok, group):
        return False
    mapped = []
    for w in group["worlds"].values():
        e = encode(tok, w, group["options"])
        mapped.append(record_positions(tok, w["prompt"], e["anchor"]))
    return all(p == mapped[0] for p in mapped[1:])


def evaluate(model, tok, g):
    worlds = g["worlds"]
    encs = {name: encode(tok, w, g["options"]) for name, w in worlds.items()}
    subject, pos = record_positions(tok, worlds["base"]["prompt"], encs["base"]["anchor"])
    prefix = list(range(encs["base"]["anchor"]+1))
    natural = {name: fast_forward(model, {**e, "anchor": prefix}, {SITE}, trace=True)
               for name, e in encs.items()}
    for rx, same, alt, _, _ in DIRECTIONS:
        selected = pos[subject]
        ids = encs[rx]["ids"]
        counter = encs[same]["ids"]
        changed = [p for p in pos if any(ids[j]!=counter[j] for j in pos[p])]
        assert len(changed) == 2 and subject in changed
        other_name = next(p for p in changed if p != subject)
        other = pos[other_name]
        score, bank, state, branches, head = natural[rx]
        rec, don = bank[SITE], natural[same][1][SITE]
        target = don[selected]
        dose = norm_delta(target, rec[selected])
        deranged = dose_matched(rec[other], don[other].float()-rec[other].float(), dose)
        rng = torch.Generator(device="cuda").manual_seed(int.from_bytes(
            hashlib.sha256(f"{g['id']}/{rx}/record".encode()).digest()[:8], "big"))
        random_value = dose_matched(rec[selected], torch.randn(target.shape, device="cuda", generator=rng), dose)
        readout = (encs[rx]["read"], state[BLOCK])
        replacements = {
            "target": {SITE:(selected, target)},
            "unqueried": {SITE:(other, deranged)},
            "random": {SITE:(selected, random_value)},
            "self": {SITE:(selected,rec[selected])},
            "blocked": {SITE:(selected,target), BLOCK:readout},
            "block_only": {BLOCK:readout},
        }
        orig, goal = worlds[rx]["answer"], worlds[alt]["answer"]
        conditions = {}
        for name, edits in replacements.items():
            result, _, downstream, b, full = fast_forward(model, encs[rx], edits=edits, trace=name=="target")
            assert torch.equal(full[0],head[0])
            if name in ("self","block_only"):
                assert torch.equal(full,head)
            conditions[name] = {"scores":result, "margin":margin(result,goal,orig), "pred":predictions(result)}
            if name == "target":
                conditions[name]["readout_l2"] = trace_deltas(state,downstream)
                conditions[name]["branch_l2"] = trace_deltas(branches,b)
        yield {"id":g["id"],"task":g["task"],"receive":rx,"donor":same,
               "queried":subject,"unqueried":other_name,"queried_positions":selected,
               "unqueried_positions":other,"original":orig,"counterfactual":goal,
               "dose":{"target":dose,"unqueried":norm_delta(deranged,rec[other]),
                       "random":norm_delta(random_value,rec[selected])},
               "natural":{"scores":score,"margin":margin(score,goal,orig),"pred":predictions(score)},
               "donor_pred":predictions(natural[same][0]),"conditions":conditions}


def summarize(out):
    rows = load_lines(out / "rows.jsonl")
    assert len(rows)==60
    records=[]
    for k in range(4):
        for name, arms in ARMS.items():
            level=1-.05/3 if k==3 and name in tuple(ARMS)[:3] else .95
            mean,lo,hi=interval(clustered(rows,*arms,k),level)
            records.append(dict(readout_K=k+1, contrast=name, groups=30,
                                mean_nats=mean, lower=lo, upper=hi, interval_level=level))
    write_csv(out/"causal_effects.csv",records)
    aligned=[r for r in rows if r["natural"]["pred"][3]==r["donor_pred"][3]==r["original"]]
    flips=[r for r in aligned if r["conditions"]["target"]["pred"][3]==r["counterfactual"]]
    order=lambda r:r["conditions"]["target"]["margin"][3]-r["natural"]["margin"][3]
    case=sorted(flips,key=order)[len(flips)//2] if flips else sorted(rows,key=order)[len(rows)//2]
    write_json(out/"case_witness.json",case)
    proof={"eligible_directions":len(aligned),"queried_record_adopts_counterfactual":len(flips),
           "unqueried_record_adopts_counterfactual":sum(r["conditions"]["unqueried"]["pred"][3]==r["counterfactual"] for r in aligned),
           "random_adopts_counterfactual":sum(r["conditions"]["random"]["pred"][3]==r["counterfactual"] for r in aligned),
           "max_dose_relative_error":max(abs(r["dose"][c]/r["dose"]["target"]-1)
                                         for r in rows for c in ("random","unqueried"))}
    write_json(out/"decision_witness.json",proof)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("study",type=Path)
    p.add_argument("--model",type=Path)
    p.add_argument("--analyze-only",action="store_true")
    args=p.parse_args()
    out=args.study/"record_followup"
    if args.analyze_only:
        summarize(out)
        return
    out.mkdir()
    m,t,runtime=load_runtime(args.model)
    groups=[aligned_group("multihop",i,None,lambda g:align(t,g)) for i in range(200,230)]
    for g in groups:
        g["split"]="fresh_queried_record_followup"
    write_json(out/"runtime.json",runtime)
    with (out/"semantic_interventions.jsonl").open("w",encoding="utf-8") as f:
        for g in groups:
            dump_line(f,g)
    start=time.perf_counter()
    with torch.inference_mode(), (out/"rows.jsonl").open("w",encoding="utf-8") as f:
        for g in groups:
            for row in evaluate(m,t,g):
                dump_line(f,row)
            f.flush()
    torch.cuda.synchronize()
    write_json(out/"completed.json",{"groups":len(groups),"rows":2*len(groups),
               "gpu_work_seconds":time.perf_counter()-start,"script_sha256":digest(Path(__file__)),
               "inputs_sha256":digest(out/"semantic_interventions.jsonl"),
               "peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated()})
    summarize(out)
    print("Queried and unqueried records: 30 fresh groups",flush=True)


if __name__ == "__main__":
    main()
