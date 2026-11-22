"""Test arithmetic writeback on new lookup prompts with answer-blind role extraction."""

import argparse
from contextlib import contextmanager
from pathlib import Path
from random import Random
import re

import torch

from csc_bank import LAYER, whites
from csc_learn import cap, compress, load_operator
from csc_oracle import Write, stat
from dest_deck import history as earlier_lookup
from interchange import terminated
from native import Forward, load, output, pad, save, setup, sha
from onset_deck import DOMAINS, prompt
from rule import MAX_NEW, greedy, text_results
from value_test import shuffle


SEED = 261134451


def build(parent):
    used = earlier_lookup()
    used.update(prompt(r, "original") for r in load(parent)["rows"])
    rng = Random(SEED)
    rows = []
    for domain in DOMAINS:
        targets = rng.sample(range(100, 1000), 160)
        keys = list(range(1, 5)) if domain == "standard" else list(range(5, 9))
        for j, y in enumerate(targets):
            index = j % 4
            while True:
                values = rng.sample([g for g in range(100, 1000) if g != y], 3)
                values.insert(index, y)
                order = rng.sample(keys, 4)
                row = dict(id=f"{domain}-{j}", domain=domain, key=keys[index], target=y,
                           example=dict(original=dict(order=order, values=values)))
                text = prompt(row, "original")
                if text not in used:
                    used.add(text)
                    rows.append(dict(row, text=text))
                    break
    return dict(seed=SEED, parent_sha256=sha(parent), rows=rows)


def roles(tok, text):
    match = re.search(r"(?:Query|Lookup key) = (\d); (?:Table|Entries) = (.+); Answer:$", text)
    assert match
    data = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets, ids = data["offset_mapping"], data["input_ids"]

    def gather(lo, hi):
        return [i for i, (a, b) in enumerate(offsets) if a < hi and lo < b]

    table_offset = match.start(2)
    values = [gather(table_offset+m.start(1), table_offset+m.end(1))
              for m in re.finditer(r": (\d{3})", match[2])]
    assert len(values) == 4 and all(len(q) == 3 for q in values)
    a = list(zip(*(v[-2:] for v in values[::2]), strict=True))
    b = list(zip(*(v[-2:] for v in values[1::2]), strict=True))
    context = list(range(len(ids)-4, len(ids)-1))
    return ids, (a, b, gather(*match.span(1)), context)


def feature(h, at, positions):
    a, b, rule, ctx = positions
    left, right = [h[torch.as_tensor(x, device=h.device)].mean(1) for x in (a, b)]
    return torch.cat((left, right, h[rule].mean(0, keepdim=True), h[ctx].mean(0, keepdim=True), h[at,None]))


def prediction(h, at, pos, op, encoder, s):
    x = torch.stack([feature(h[i], t, p) for i, (t, p) in enumerate(zip(at, pos, strict=True))])
    return s["gain"] * cap(op(compress(x, encoder)), s["limit"])


@contextmanager
def layer_write(f, at, pos, bundle, previous=None):
    op, enc, s = bundle
    state = {} if previous is None else {"delta": prediction(previous, at, pos, op, enc, s)}
    layer = f.layers[LAYER]

    def read(_, args):
        state["delta"] = prediction(args[0], at, pos, op, enc, s)

    def write(_, args, h):
        y = h.clone()
        y[torch.arange(len(at), device=h.device), at] += state["delta"]
        return y

    branches = {"residual": layer, "mlp": layer.post_attention_layernorm_2, "attention": layer.input_layernorm_2}
    handles = [branches[s["branch"]].register_forward_hook(write)]
    if previous is None:
        handles.append(layer.register_forward_pre_hook(read))
    try:
        yield
    finally:
        for hook in handles:
            hook.remove()


def decode(f, tok, cases, bundle):
    active, done = {i:[] for i in range(len(cases))}, {}
    for _ in range(MAX_NEW):
        batch = list(active)
        if not batch:
            break
        seq = [cases[i]["ids"] + active[i] for i in batch]
        ids, at = pad(seq)
        locus = [len(cases[i]["ids"])+len(whites(tok, {"tokens":active[i]}))-1 for i in batch]
        pos = [cases[i]["roles"] for i in batch]
        start = f.run(ids, 1)[0] if bundle[2]["depth"] == 2 else None
        with layer_write(f, locus, pos, bundle, start):
            h = f.run(ids, 1, start=start)[0]
        for i, token in zip(batch, output(f.model, h, at).argmax(-1).tolist(), strict=True):
            active[i].append(token)
            if terminated(tok, active[i]):
                done[i] = active.pop(i)
    done.update(active)
    return text_results(tok, [done[i] for i in range(len(cases))])


def capture_delta(f, ids, pos, branch):
    taps, block = {"write":[], "base":[]}, f.layers[LAYER]
    module = block.post_attention_layernorm_2 if branch == "m" else block.input_layernorm_2
    row = torch.arange(len(pos), device=ids.device)

    def capture(key):
        def record(_, args, h):
            taps[key].append(h[row,pos])
        return record

    hooks = [module.register_forward_hook(capture("write")), block.register_forward_hook(capture("base"))]
    try:
        f.run(ids, 2)
    finally:
        for hook in hooks:
            hook.remove()
    u1, u2 = taps["write"]
    return u2-u1, taps["base"][0]


def lookup_oracle(f, tok, cases, branch):
    arms = ("oracle", "geometry")
    original = [r["ids"] for r in cases]
    active, done, bank = {(a,i):[] for a in arms for i in range(len(cases))}, {}, {}
    for _ in range(MAX_NEW):
        batch = list(active)
        if not batch:
            break
        needed = {(i,whites(tok,{"tokens":active[a,i]})) for a,i in batch}
        missing = sorted(needed.difference(bank))
        if missing:
            ids, at = pad([original[i]+list(q) for i,q in missing])
            u, base = capture_delta(f, ids, at, branch)
            geom = shuffle(base, u)-base
            bank.update({k:(x,y) for k,x,y in zip(missing,u,geom,strict=True)})
        ids, at = pad([original[i]+active[a,i] for a,i in batch])
        sites = [(i,whites(tok,{"tokens":active[a,i]})) for a,i in batch]
        pos = torch.tensor([len(original[i])+len(q)-1 for i,q in sites],device=ids.device)
        update = torch.stack([bank[key][arms.index(a)] for (a,_),key in zip(batch,sites,strict=True)])
        with Write(f,LAYER,pos,update):
            h = f.run(ids,1)[0]
        for i,token in zip(batch,output(f.model,h,at).argmax(-1).tolist(),strict=True):
            active[i].append(token)
            if terminated(tok,active[i]):
                done[i] = active.pop(i)
    done.update(active)
    return {a:text_results(tok,[done[a,j] for j in range(len(cases))]) for a in arms}


@torch.inference_mode()
def run(checkpoint, root):
    oracle = load(root / "ORACLE_SUMMARY.json")
    assert oracle["gate"]["full"]
    parent = Path("/root/ouro-lookup-destination-20261122/results/DECK.json")
    deck = root / "LOOKUP_DECK.json"
    if not deck.exists():
        save(deck, build(parent))
    m, tok = setup(checkpoint)
    f = Forward(m)
    path = root / "LOOKUP_RAW.json"
    raw = load(path) if path.exists() else dict(deck_sha256=sha(deck), rows=[])
    assert raw["deck_sha256"] == sha(deck)
    rows = load(deck)["rows"]
    assert [r["id"] for r in raw["rows"]] == [r["id"] for r in rows[:len(raw["rows"])]]
    learned = {f"{arm}_k{k}":load_operator(root / f"operator_{arm}_K{k}.pt", "cuda")
               for k in (1,2) for arm in ("causal","answer_only","state_distill")}
    for j in range(len(raw["rows"]),len(rows),16):
        batch = rows[j:j+16]
        cases = [dict(zip(("ids","roles"),roles(tok,r["text"]),strict=True)) for r in batch]
        p = {f"native_k{k}":greedy(f,tok,[r["ids"] for r in cases],k) for k in (1,2)}
        p.update(lookup_oracle(f,tok,cases,"m"))
        p.update({key:decode(f,tok,cases,op) for key,op in learned.items()})
        raw["rows"].extend(dict(id=r["id"],domain=r["domain"],target=r["target"],pred={a:res[i] for a,res in p.items()})
                           for i,r in enumerate(batch))
        save(path,raw)
        print("CSC lookup",len(raw["rows"]),len(rows),flush=True)
    report = {}
    for domain in DOMAINS:
        group = [r for r in raw["rows"] if r["domain"] == domain]
        arms = {}
        for k in group[0]["pred"]:
            baseline = "native_k2" if k.endswith("k2") else "native_k1"
            correct = [r["pred"][k]["number"] == r["target"] for r in group]
            old = [r["pred"][baseline]["number"] == r["target"] for r in group]
            arms[k] = dict(n=len(group), correct=sum(correct),
                           rescued=sum(a and not b for a,b in zip(correct,old,strict=True)),
                           damaged=sum(b and not a for a,b in zip(correct,old,strict=True)),
                           vs_native=stat(group,k,baseline))
        report[domain] = dict(arms=arms, oracle_vs_geometry=stat(group,"oracle","geometry"))
    save(root / "LOOKUP_SUMMARY.json",report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("root",type=Path)
    args = parser.parse_args()
    run(args.checkpoint,args.root)
