"""Run the original no-digit lookup intervention at a later physical layer."""

import argparse
from pathlib import Path

import torch

import csc_lookup
from csc_oracle import stat
from native import Forward, load, save, setup


SITE = 19


@torch.inference_mode()
def run(model, root):
    deck = load(root / "LOOKUP_DECK.json")["rows"]
    raw = load(root / "LOOKUP_RAW.json")["rows"]
    prior = {r["id"]:r for r in raw}
    cases = [r for d in csc_lookup.DOMAINS for r in [x for x in deck if x["domain"]==d][:96]]
    m,tok = setup(model)
    f = Forward(m)
    saved = root / "LOOKUP_LATE_RAW.json"
    original_layer = csc_lookup.LAYER
    csc_lookup.LAYER = SITE
    rows = load(saved)["rows"] if saved.exists() else []
    assert [r["id"] for r in rows] == [r["id"] for r in cases[:len(rows)]]
    try:
        for start in range(len(rows),len(cases),16):
            batch = cases[start:start+16]
            tokenized = [dict(zip(("ids","roles"),csc_lookup.roles(tok,r["text"]),strict=True)) for r in batch]
            updates = {key:csc_lookup.lookup_oracle(f,tok,tokenized,branch) for key,branch in (("attn","a"),("mlp","m"))}
            for i,r in enumerate(batch):
                keep = {a:prior[r["id"]]["pred"][a] for a in ("native_k1","native_k2","oracle","geometry")}
                interventions = {f"{a}_{b}":seq[i] for a,slots in updates.items() for b,seq in slots.items()}
                rows.append(dict(id=r["id"],domain=r["domain"],target=r["target"],pred={**keep,**interventions}))
            save(saved,dict(rows=rows,layer=SITE+1))
            print("CSC late lookup",len(rows),len(cases),flush=True)
    finally:
        csc_lookup.LAYER = original_layer
    summary = {}
    for domain in csc_lookup.DOMAINS:
        group = [r for r in rows if r["domain"]==domain]
        out = {}
        for a in group[0]["pred"]:
            old = [r["pred"]["native_k1"]["number"]==r["target"] for r in group]
            now = [r["pred"][a]["number"]==r["target"] for r in group]
            out[a] = dict(n=len(group),correct=sum(now),rescued=sum(y and not x for x,y in zip(old,now,strict=True)),
                          damaged=sum(x and not y for x,y in zip(old,now,strict=True)))
        summary[domain] = dict(arms=out,vs_geometry={key:stat(group,f"{key}_oracle",f"{key}_geometry")
                                                     for key in ("attn","mlp")})
    save(root / "LOOKUP_LATE_SUMMARY.json",summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model",type=Path)
    parser.add_argument("root",type=Path)
    args = parser.parse_args()
    run(args.model,args.root)
