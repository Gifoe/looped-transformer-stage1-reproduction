"""Build matched conflict worlds and measure native K2 eligibility before any edits."""

import argparse
from pathlib import Path

import torch

from conf_deck import DOMAINS, history, prompt, repair
from edge import short_hash
from native import Forward, load, save, setup, sha
from rule import greedy

SIDES = ("original", "swapped", "unrelated_original", "unrelated_swapped")
STABLE_N = 64


def swap(values,i,j):
    result = list(values)
    result[i],result[j] = result[j],result[i]
    return result


def alternatives(row,kind):
    state = row["example"]["original"]
    vals = state["values"]
    target = row["target"]
    other = [v for v in vals if v!=target]
    old = row["native"]["original"][0]["number"] if kind=="repair" else min(other,key=lambda x:short_hash("conflict stable foil",row["id"],str(x)))
    a,b = vals.index(target),vals.index(old)
    x,y = [j for j in range(4) if j not in (a,b)]
    swapped = swap(vals,a,b)
    candidates = (vals,swapped,swap(vals,x,y),swap(swapped,x,y))
    example = {s:dict(order=state["order"],values=v) for s,v in zip(SIDES,candidates,strict=True)}
    return dict(row,old=old,kind=kind,example=example)


def build(rows):
    stable = set()
    for d in DOMAINS:
        ids = [r["id"] for r in rows if r["domain"]==d and all(a["number"]==r["target"] for a in r["native"]["original"])]
        stable.update(sorted(ids,key=lambda x:short_hash("conflict stable sample",x))[:STABLE_N])
    groups = [alternatives(r,"stable" if r["id"] in stable else "repair") for r in rows if r["id"] in stable or repair(r)]
    seen = history() | {prompt(r,"original") for r in rows}
    for r in groups:
        for side in SIDES[1:]:
            p = prompt(r,side)
            assert p not in seen
            seen.add(p)
    return groups


def spans(tok,r):
    captured = {}
    for side in SIDES:
        q = prompt(r,side)
        expect = r["old"] if side.endswith("swapped") else r["target"]
        anchor = f"{r['key']}: {expect}"
        end = q.rindex(anchor)+len(anchor)
        begin = end-len(str(expect))
        encoded = tok(q,add_special_tokens=False,return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        take = [j for j,(a,b) in enumerate(offsets) if a<end and b>begin]
        if not take:
            return None
        a,b = offsets[take[0]][0],offsets[take[-1]][1]
        if b!=end or q[a:end].strip()!=str(expect) or a>begin:
            return None
        captured[side] = (encoded["input_ids"],offsets,take)
    _,base,pos = captured["original"]
    if any(offsets!=base or take!=pos for _,offsets,take in captured.values()):
        return None
    assert max(pos)<len(base)-1
    return dict(tokens={s:x[0] for s,x in captured.items()},positions=pos,offsets=[list(p) for p in base])


def qualified(r):
    return r["span"] is not None and all(r["pair_native"][s]["number"]==r["old" if s.endswith("swapped") else "target"] for s in SIDES)


def summary(rows):
    out = {}
    for d in DOMAINS:
        out[d] = {}
        for k in ("repair","stable"):
            selected = [r for r in rows if r["domain"]==d and r["kind"]==k]
            out[d][k] = dict(n=len(selected),aligned=sum(r["span"] is not None for r in selected),eligible=sum(map(qualified,selected)))
    out["gate"] = all(out[d]["repair"]["eligible"]>=48 for d in DOMAINS)
    return out


@torch.inference_mode()
def run(model,root):
    native = root/"NATIVE.json"
    plan = build(load(native)["rows"])
    dst = root/"PAIRS.json"
    h = sha(native)
    data = load(dst) if dst.exists() else dict(native_sha256=h,rows=[])
    assert data["native_sha256"]==h
    rows = data["rows"]
    assert [r["id"] for r in rows]==[r["id"] for r in plan[:len(rows)]]
    m,tok = setup(model)
    f = Forward(m)
    for off in range(len(rows),len(plan),10):
        batch = plan[off:off+10]
        ids = [tok.encode(prompt(r,s),add_special_tokens=False) for r in batch for s in SIDES[1:]]
        decoded = greedy(f,tok,ids,2)
        for j,r in enumerate(batch):
            other = dict(zip(SIDES[1:],decoded[3*j:3*j+3],strict=True))
            rows.append(dict(r,span=spans(tok,r),pair_native={"original":r["native"]["original"][1],**other}))
        save(dst,data)
        print("conflict native pairs",len(rows),len(plan),flush=True)
    z = summary(rows)
    save(root/"PAIR_SUMMARY.json",z)
    print(z,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a = p.parse_args()
    run(a.model,a.root)
