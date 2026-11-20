"""Freeze cohorts, execute the parity gate and capture GPU dynamics."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ouro_cross_task_20261118"))
from run import load_runtime
sys.path.insert(0, str(HERE.parent / "ouro_hidden_computation_20261120"))
from mechanism_design import encode, load

from dynamics import NativeBatch, interpolation, random_walk, relative, step_cosine

GRID = tuple(i / 8 for i in range(9))
LEVELS = (1, 2, 4, 8, 16, 32)
SEED = 57420261120
FACTORS = ("s", "m", "b", "y")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def policy_seed(*keys):
    return int.from_bytes(hashlib.sha256(repr((SEED, keys)).encode()).digest()[:8], "big")


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def write_row(f, item):
    f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


def role_tokenize(tok, parts, candidates, metadata):
    text, spans = "", {}
    for role, frag in parts:
        if role:
            spans.setdefault(role, []).append((len(text), len(text)+len(frag)))
        text += frag
    encoding = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoding["input_ids"]
    roles = {key: sorted({i for a, b in intervals for i, (x, y) in enumerate(encoding["offset_mapping"])
                           if x < b and y > a}) for key, intervals in spans.items()}
    roles["readout"] = [len(ids)-1]
    next_ids = [tok.encode(text + " " + candidate, add_special_tokens=False) for candidate in candidates]
    if not all(z[:-1] == ids for z in next_ids):
        raise ValueError("Candidate token mismatch")
    ids_out = [x[-1] for x in next_ids]
    if len(set(ids_out)) != len(ids_out):
        raise ValueError("Ambiguous candidate list")
    return dict(metadata, text=text, ids=ids, roles=roles, read=len(ids)-1,
                content=ids_out, choices=candidates)


VOCAB = {
    "development": (("Alex", "Laura", "Peter"), ("red", "blue", "green", "white", "black"),
                    ("circle", "square", "star", "diamond", "cross"),
                    ("apple", "peach", "plum", "pear", "orange", "melon")),
    "new_structure": (("Alice", "David", "Sarah"), ("gold", "silver", "purple", "yellow", "gray"),
                      ("triangle", "heart", "moon", "arrow", "bell"),
                      ("horse", "tiger", "zebra", "wolf", "rabbit", "lion")),
}


def twohop(tok, cohort, idx, attempt):
    rng = random.Random(policy_seed("twohop", cohort, idx, attempt))
    names, sources, bridges, objects = VOCAB[cohort]
    name = rng.choice(names)
    s_words, b_words, y_words = (rng.sample(x, 3) for x in (sources, bridges, objects))
    order1, order2, choices = list(range(3)), list(range(3)), rng.sample(y_words, 3)
    rng.shuffle(order1)
    rng.shuffle(order2)
    gid = f"hop-{cohort}-{idx:03d}"
    worlds = []
    for s in range(3):
        for m in range(3):
            for r in range(3):
                b = (s + m) % 3
                y = (b + r) % 3
                parts = [(None, "Read the facts.\nA spare tool is in the"),
                         ("noise", " room"), (None, f".\n{name} holds the"),
                         ("source", " " + s_words[s]), (None, " key.\n")]
                for j in order1:
                    parts.append(("bridge", f"The {s_words[j]} key opens the {b_words[(j+m)%3]} cabinet.\n"))
                for j in order2:
                    parts.append(("mapping", f"The {b_words[j]} cabinet contains the {y_words[(j+r)%3]}.\n"))
                if cohort == "new_structure":
                    parts += [(None, "Ignore the spare tool.\n"),
                              ("query", f"Question: Follow the cabinet from {name}'s key. What is inside?")]
                else:
                    parts.append(("query", f"Question: What object can {name} retrieve?"))
                parts.append(("readout", "\nAnswer:"))
                worlds.append(role_tokenize(tok, parts, choices, dict(
                    s=s, m=m, r=r, b=b, y=choices.index(y_words[y]), y_index=y,
                    group=gid, item=f"{gid}:{s}{m}{r}", cohort=cohort, family="twohop",
                    attempt=attempt, bridge_target=b_words[b], within=[s_words,b_words,y_words],
                    choices_semantic=y_words)))
    if not all(w["roles"] == worlds[0]["roles"] and len(w["ids"]) == len(worlds[0]["ids"])
               for w in worlds):
        raise ValueError("Counterfactual token roles drifted")
    assert {len(w["roles"]["source"]) for w in worlds} == {1}
    return worlds


def make_batches(tok, tableau_path, arc_path, stage):
    native_groups = list(load(tableau_path))
    selected = []
    for phase, count in (("development", 16), ("new_entities", 16), ("new_structure", 20)):
        groups = [g for g in native_groups if g["split"] == phase]
        groups.sort(key=lambda g: policy_seed("tableaux", g["id"]))
        for g in groups[:count]:
            worlds = []
            for s in range(3):
                for m in range(3):
                    w = encode(tok, g, s, m, "open")
                    worlds.append(dict(w, item=f'{g["id"]}:{s}{m}', group=g["id"], cohort=phase,
                                       family="tableau", y_index=w["y"]))
            selected.append(worlds)
    for phase, count in (("development", 10), ("new_structure", 12)):
        seen = set()
        for j in range(count):
            for attempt in range(500):
                try:
                    worlds = twohop(tok, phase, j, attempt)
                except ValueError:
                    continue
                sig = hashlib.sha256(bytes(str(worlds[0]["ids"]), "utf8")).hexdigest()
                if sig not in seen:
                    seen.add(sig)
                    selected.append(worlds)
                    break
            else:
                raise RuntimeError("No aligned two-hop prompt found")
    arc = []
    for row in pq.read_table(arc_path).to_pylist():
        choices = row["choices"]["text"]
        parts = [("query", f'Question: {row["question"]}')]
        parts += [("candidate", "\n" + chr(65+i) + ") " + c) for i,c in enumerate(choices)]
        parts.append(("readout", "\nAnswer:"))
        labels = [chr(65+i) for i in range(len(choices))]
        arc.append(role_tokenize(tok, parts, labels, dict(
            item=row["id"], group=row["id"], cohort="validation", family="arc",
            y=row["choices"]["label"].index(row["answerKey"]))))
    assert len(arc) == len({x["item"] for x in arc}) == 299
    if stage == "pilot":
        return [selected[0][:4], selected[16+16+20][:4],
                [row for row in arc if len(row["content"]) == 4][:4]], []
    return selected, arc


def geometry(batch, path):
    ts = list(sorted(path))
    h1, h2 = batch.k1, batch.k2
    motion = [relative(path[t], path[p], batch.lengths) for p,t in zip(ts, ts[1:])]
    turn = [step_cosine(path[p],path[t],path[t],path[q],batch.lengths) for p,t,q in zip(ts,ts[1:],ts[2:])]
    return {
        "time": ts,
        "to_k1": np.asarray([relative(path[t],h1,batch.lengths) for t in ts]).T.tolist(),
        "to_k2": np.asarray([relative(path[t],h2,batch.lengths) for t in ts]).T.tolist(),
        "speed": (np.asarray(motion).T / np.diff(ts)).tolist(),
        "turn_cos": np.asarray(turn).T.tolist() if turn else [[] for _ in batch.encodings],
    }


def path_record(batch, path):
    geo = geometry(batch, path)
    scores = torch.stack([batch.head(path[t]) for t in geo["time"]]).transpose(0,1).cpu().tolist()
    return [dict(time=geo["time"], scores=scores[i], to_k1=geo["to_k1"][i], to_k2=geo["to_k2"][i],
                 speed=geo["speed"][i], turn_cos=geo["turn_cos"][i])
            for i in range(len(batch.ids))]


def profiles(batch, high=True, pilot=False):
    gen = torch.Generator(device="cuda").manual_seed(policy_seed("random", batch.encodings[0]["item"]))
    paths = {"native": batch.native, "linear": interpolation(batch.k1,batch.k2,GRID),
             "norm_linear": interpolation(batch.k1,batch.k2,GRID,match_norm=True)}
    for family in ("A", "B"):
        for m in (LEVELS if high else LEVELS[:4]):
            paths[f"{family}{m}"] = (batch.euler(m) if family == "A"
                                      else batch.native if m == 1 else batch.branch_path(m,capture_every=m==32))
    ref = paths["A32" if high else "A8"]
    paths["random"] = random_walk(ref, gen)
    if pilot:
        paths["rk4_8"] = batch.euler(8, rk4=True)
        paths["A32bf16"] = batch.euler(32, bf16_accum=True)
    return paths


def comparisons(batch, paths):
    comparisons_for_batch = {}
    for family in ("A", "B"):
        for a,b in zip(LEVELS,LEVELS[1:]):
            if f"{family}{b}" not in paths:
                continue
            comparisons_for_batch[f"{family}{a}/{family}{b}"] = (f"{family}{a}",f"{family}{b}")
    for a,b in (("A32","B32"),("A32","linear"),("B32","linear"),
                ("A32","rk4_8"),("A32","A32bf16")):
        if a in paths and b in paths:
            comparisons_for_batch[f"{a}/{b}"] = (a,b)
    records = {}
    for key,(a,b) in comparisons_for_batch.items():
        shared = sorted(set(paths[a]) & set(paths[b]))
        distances = [relative(paths[a][t],paths[b][t],batch.lengths) for t in shared]
        records[key] = {"time": shared, "rel": np.asarray(distances).T.tolist()}
    return records


def arc_scan(model, rows):
    ref = {}
    for k in range(0,len(rows),16):
        group = rows[k:k+16]
        n = NativeBatch(model,group)
        scores = [n.head(h).argmax(-1).cpu().tolist() for h in (n.k1,n.k2)]
        for i,w in enumerate(group):
            ref[w["item"]] = [scores[0][i],scores[1][i]]
    return ref


def run_batch(model, worlds, f, bank, high, witness_dir=None, pilot=False):
    # ARC answer counts may vary; batches are already sorted into same-width strata.
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    batch = NativeBatch(model,worlds,trace=True,parity=pilot)
    pre_boundary = batch.pre_boundary_norm
    paths = profiles(batch,high=high,pilot=pilot)
    records = {name:path_record(batch,p) for name,p in paths.items()}
    gaps = comparisons(batch,paths)
    norm_step = relative(batch.k2,pre_boundary,batch.lengths)
    before_norm_distance = relative(pre_boundary,batch.k1,batch.lengths)
    before_norm_scores = batch.head(pre_boundary).cpu().tolist()
    for i,w in enumerate(worlds):
        write_row(f, {
            **{k:w[k] for k in ("item","group","cohort","family","s","m","b","y","y_index","native_scan") if k in w},
            "valid_tokens":len(w["ids"]), "tracks":{name:rs[i] for name,rs in records.items()},
            "comparisons":{name:{"time":rs["time"],"rel":rs["rel"][i]} for name,rs in gaps.items()},
            "native_norm_rel":norm_step[i],
            "before_native_norm":{"to_k1":before_norm_distance[i],"scores":before_norm_scores[i]},
        })
    if bank is not None:
        bank.append((worlds,{name:(ts, torch.stack([p[t][batch.rows,batch.lengths-1].half()
                                                       for t in ts]).cpu().numpy())
                            for name,p in paths.items()
                            if (ts:=sorted(set(p) & set(GRID)))}))
    if witness_dir is not None:
        chosen = ["native", "A32", "B32", "rk4_8", "A32bf16"] if pilot else ["native","A32","B32","linear"]
        for name in chosen:
            p = paths[name]
            width = len(worlds[0]["ids"])
            np.savez_compressed(witness_dir / f"{worlds[0]['family']}-{name}.npz", time=list(p),
                                ids=worlds[0]["ids"], hidden=torch.stack([x[0,:width].half()
                                                                        for x in p.values()]).cpu().numpy())
    torch.cuda.synchronize()
    return dict(group=worlds[0]["group"], members=len(worlds), max_tokens=batch.ids.shape[1],
                wall_s=time.perf_counter()-start, peak_allocated=torch.cuda.max_memory_allocated(),
                counters=batch.counts, parity=batch.parity)


def main():
    p = argparse.ArgumentParser()
    for x in ("model","tableau","arc","out"):
        p.add_argument(x,type=Path)
    p.add_argument("stage",choices=("pilot","formal"))
    args = p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.stage=="formal":
        assert json.loads((args.out/"NATIVE_PARITY.json").read_bytes())["status"]=="PASS"
    model,tok,runtime=load_runtime(args.model)
    groups, arc = make_batches(tok,args.tableau,args.arc,args.stage)
    if args.stage=="pilot":
        before=digest(args.model/"model.safetensors")
        receipts = []
        with (args.out/"pilot.jsonl").open("w",encoding="utf-8") as f:
            for group in groups:
                receipts.append(run_batch(model,group,f,None,True, None, True))
        after=digest(args.model/"model.safetensors")
        assert before==after==runtime["model_sha256"]
        dump(args.out/"NATIVE_PARITY.json",dict(status="PASS", gpu=runtime, batches=receipts,
            check="Unmasked right-padding with causal SDPA; use identical native attention masks, cache_position and RoPE tensors. No KV cache.",
            checkpoint_hash_before=before,checkpoint_hash_after=after))
        print("NATIVE_PARITY PASS",flush=True)
        return
    from semantics import fit_and_measure
    from causal import causality
    # Only ARC strata use native outcomes before panel selection.
    by_width = {}
    for item in arc:
        by_width.setdefault(len(item["content"]),[]).append(item)
    ref={}
    for rows in by_width.values():
        ref.update(arc_scan(model,rows))
    strata={}
    for w in arc:
        w["native_scan"]=ref[w["item"]]
        status="".join(str(int(x==w["y"])) for x in w["native_scan"])
        strata.setdefault(status,[]).append(w)
    panel=set()
    for status, rows in strata.items():
        rows.sort(key=lambda x:policy_seed("arc_panel",status,x["item"]))
        panel.update(w["item"] for w in rows[:12])
    dump(args.out/"FREEZE.json",dict(dataset_sha256={k:digest(getattr(args,k)) for k in ("tableau","arc")},
          code_sha256={p.name:digest(p) for p in HERE.glob("*.py") if p.name in
                       {"study.py","dynamics.py","causal.py","semantics.py"}},
          seed=SEED, structured_groups=[g[0]["group"] for g in groups],
          arc_questions=len(arc),arc_native_strata={k:len(v) for k,v in strata.items()},
          arc_high_resolution=sorted(panel),causal_t=[.25,.5,.75],causal_resolution=16))
    book=[]
    witness=args.out/"witness"
    witness.mkdir(exist_ok=True)
    with (args.out/"trajectories.jsonl").open("w",encoding="utf-8") as f, \
         (args.out/"runtime.jsonl").open("w",encoding="utf-8") as log, \
         (args.out/"data_manifest.jsonl").open("w",encoding="utf-8") as manifest:
        for i, group in enumerate(groups):
            for e in group:
                write_row(manifest,e)
            rec=run_batch(model,group,f,book,True,witness if i in (32,52+10) else None)
            write_row(log,rec)
            f.flush()
            if i%8==0:
                print(f"structured {i+1}/{len(groups)}",flush=True)
        for rows in by_width.values():
            for scope in (False,True):
                items=[e for e in rows if (e["item"] in panel)==scope]
                for k in range(0,len(items),12):
                    group=items[k:k+12]
                    rec=run_batch(model,group,f,None,scope)
                    write_row(log,rec)
                    for e in group:
                        write_row(manifest,{k:v for k,v in e.items() if k!="text"})
                    print("ARC",scope, k+len(group),"/",len(items),flush=True)
    fit_and_measure(book,args.out)
    causality(model,groups,args.out)
    dump(args.out/"COMPLETE.json",dict(status="PASS",runtime=runtime,arc_questions=len(arc),
         structured_groups=len(groups),files={p.name:digest(p) for p in args.out.glob("*.jsonl")}))


if __name__=="__main__":
    main()
