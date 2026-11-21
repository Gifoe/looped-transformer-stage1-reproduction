"""Lock independent lookup prompts and collect native greedy outcomes."""

import argparse
from collections import Counter
from pathlib import Path
from random import Random

import torch

from native import Forward, load, save, setup, sha
from onset_deck import DOMAINS, prompt, repair
from av_deck import history as earlier
from rule import greedy

PARENT = Path("/root/ouro-lookup-av-20261121/results")
SEED = 261129117


def history():
    result = earlier()
    for path in ("DECK.json", "NATIVE.json"):
        result.update(prompt(r, v) for r in load(PARENT / path)["rows"] for v in r["example"])
    return result


def build():
    rng = Random(SEED)
    seen = history()
    rows = []
    layouts = [rng.sample(range(100, 1000), 512) for _ in DOMAINS]
    for j, target in enumerate(x for pair in zip(*layouts, strict=True) for x in pair):
        domain = DOMAINS[j % 2]
        keys = list(range(1, 5)) if domain == "standard" else list(range(5, 9))
        index = (j // 2) % 4
        values = rng.sample([x for x in range(100, 1000) if x != target], 3)
        values.insert(index, target)
        row = dict(id=f"lookup-necessity-{j:03}", domain=domain, target=target, key=keys[index],
                   example=dict(original=dict(order=rng.sample(keys, 4), values=values)))
        assert prompt(row, "original") not in seen
        seen.add(prompt(row, "original"))
        rows.append(row)
    return dict(seed=SEED, parent={f:sha(PARENT / f) for f in ("DECK.json", "NATIVE.json")}, rows=rows)


def summary(rows):
    out = {}
    for domain in DOMAINS:
        q = [r for r in rows if r["domain"] == domain]
        cells = Counter("".join(str(int(a["number"] == r["target"])) for a in r["native"]["original"]) for r in q)
        out[domain] = dict(n=len(q), transitions={x:cells[x] for x in ("00", "01", "10", "11")},
                           repairs=sum(map(repair, q)))
    out["gate"] = all(out[d]["repairs"] >= 48 for d in DOMAINS)
    return out


@torch.inference_mode()
def run(model, root):
    deck = root / "DECK.json"
    if not deck.exists():
        save(deck, build())
    design = load(deck)["rows"]
    m, tok = setup(model)
    f = Forward(m)
    dst = root / "NATIVE.json"
    digest = sha(deck)
    saved = load(dst) if dst.exists() else dict(rows=[], deck_sha256=digest)
    assert saved["deck_sha256"] == digest
    rows = saved["rows"]
    assert [r["id"] for r in rows] == [r["id"] for r in design[:len(rows)]]
    for j in range(len(rows), len(design), 32):
        batch = design[j:j+32]
        ids = [tok.encode(prompt(r, "original"), add_special_tokens=False) for r in batch]
        k1, k2 = [greedy(f, tok, ids, k) for k in (1, 2)]
        rows.extend(dict(r, native={"original": [a, b]}) for r, a, b in zip(batch, k1, k2, strict=True))
        save(dst, saved)
        print("necessity native", len(rows), len(design), flush=True)
    z = summary(rows)
    save(root / "NATIVE_SUMMARY.json", z)
    print(z, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    run(a.model, a.root)
