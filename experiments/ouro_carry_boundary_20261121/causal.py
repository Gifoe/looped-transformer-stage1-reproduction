"""Native residual-branch discovery and candidate-free confirmation."""
import argparse
from pathlib import Path

import numpy as np
import torch

from carry import MAX_NEW, PROMPT, numeric
from native import Forward, load, output, pad, save, setup, sha


CONDITIONS = ("stop", "continue", "same_stop_no_incoming")
WRITES = ("a_write", "m_write")
BRANCHES = tuple((i, w) for i in range(24) for w in WRITES)


class Capture:
    def __init__(self, p, branches):
        self.position = p
        self.row = torch.arange(len(p), device=p.device)
        self.branches = set(branches)
        self.data = {}

    def put(self, key, name, value):
        k, layer = key
        if (layer, name) in self.branches:
            self.data[(k, layer, name)] = value[self.row, self.position].detach()


class Swap:
    def __init__(self, k, branch, anchor, value, absolute=False):
        self.key = (k, branch[0])
        self.name = branch[1]
        self.anchor = anchor
        self.value = value
        self.row = torch.arange(len(anchor), device=anchor.device)
        self.absolute = absolute

    def resid(self, k, layer, x):
        return x

    def attn(self, key, a, v):
        return a, v

    def mlp(self, key, z, record):
        return z

    def write(self, key, name, value):
        if key != self.key or name != self.name:
            return value
        y = value.clone()
        if self.absolute:
            y[self.row, self.anchor] = self.value
        else:
            y[self.row, self.anchor] += self.value
        return y


def prompt(row):
    a, b = row["pair"]
    return PROMPT + f"{a} + {b} ="


def diagnostic(tok, row):
    text = prompt(row)
    base = tok.encode(text, add_special_tokens=False)
    paths = [tok.encode(text + " " + str(row[k]) + "\n", add_special_tokens=False)
             for k in ("target", "foil")]
    assert all(p[:len(base)] == base for p in paths)
    for i, (a, b) in enumerate(zip(*paths)):
        if a != b:
            return paths[0][:i], len(base) - 1, a, b
    raise ValueError(row)


def margin(f, state, pos, target, foil):
    g = output(f.model, state, pos)
    r = torch.arange(len(g), device=g.device)
    return g[r, target] - g[r, foil]


@torch.inference_mode()
def screen(f, tok, deck, root):
    pack = [x for x in deck if x["split"] == "discovery"]
    path = root / "DISCOVERY_RAW.json"
    records = load(path)["rows"] if path.exists() else []
    for i in range(len(records), len(pack), 8):
        block = pack[i:i+8]
        dx = [diagnostic(tok, b["conditions"][c]) for b in block for c in CONDITIONS]
        tokens, anchor, yes, no = zip(*dx)
        ids, pos = pad(tokens)
        at, target, foil = [torch.tensor(v, device=ids.device) for v in (anchor, yes, no)]
        trace = Capture(at, BRANCHES)
        h1, h2 = f.run(ids, 2, record=trace)
        assert torch.equal(f.run(ids, 1, start=h1)[0], h2)
        sample = BRANCHES[0]
        identity = Swap(0, sample, at, trace.data[(1, *sample)], absolute=True)
        assert torch.equal(f.run(ids, 1, start=h1, edit=identity)[0], h2)
        native = margin(f, h2, pos, target, foil)
        first = margin(f, h1, pos, target, foil)
        deltas = []
        for branch in BRANCHES:
            edit = Swap(0, branch, at, trace.data[(0, *branch)], absolute=True)
            state = f.run(ids, 1, start=h1, edit=edit)[0]
            deltas.append(native - margin(f, state, pos, target, foil))
        dl = torch.stack(deltas, dim=-1).cpu().tolist()
        n1, n2 = first.tolist(), native.tolist()
        for j, item in enumerate(block):
            current = {}
            for t, c in enumerate(CONDITIONS):
                x = j * len(CONDITIONS) + t
                current[c] = dict(k1_margin=n1[x], k2_margin=n2[x], damage=dl[x],
                                  shared_answer_prefix_tokens=len(tokens[x])-anchor[x]-1)
            records.append(dict(id=item["id"], conditions=current))
        save(path, dict(branches=BRANCHES, rows=records))
        print("screen", len(records), len(pack), flush=True)
    arrays = {c: np.array([r["conditions"][c]["damage"] for r in records]) for c in CONDITIONS}
    raw = []
    for j, (layer, write) in enumerate(BRANCHES):
        row = {c: float(v[:, j].mean()) for c, v in arrays.items()}
        raw.append(dict(layer=layer+1, write=write, **row,
                        score=row["stop"]-row["same_stop_no_incoming"]))
    winners = []
    for write in WRITES:
        candidate = max((x for x in raw if x["write"] == write and x["stop"] > 0),
                        key=lambda x: x["score"], default=None)
        if candidate is not None and candidate["score"] > 0:
            winners.append(candidate)
    save(root/"SELECTION.json", dict(winners=winners, scores=raw, raw_sha256=sha(path),
                                     causal_protocol_sha256=sha(Path("CAUSAL_PROTOCOL.md"))))
    print("selected", winners, flush=True)


def generate(f, tok, prompts, k, branch, values):
    result = [[] for _ in prompts]
    active = list(range(len(prompts)))
    for _ in range(MAX_NEW):
        ids, pos = pad([prompts[j]+result[j] for j in active])
        anchor = torch.tensor([len(prompts[j])-1 for j in active], device=ids.device)
        edit = Swap(k-1, branch, anchor, values[active])
        z = output(f.model, f.run(ids, k, edit=edit)[-1], pos).argmax(-1).tolist()
        rest = []
        for j, v in zip(active, z, strict=True):
            result[j].append(v)
            if v != tok.eos_token_id and "\n" not in tok.decode(result[j], skip_special_tokens=True).strip():
                rest.append(j)
        active = rest
        if not rest:
            break
    text = [tok.decode(x, skip_special_tokens=True) for x in result]
    return [dict(text=t, number=numeric(t), tokens=v) for t, v in zip(text, result, strict=True)]


def dose(x, norm):
    return x * (norm / x.norm(dim=-1, keepdim=True).clamp_min(1e-12))


@torch.inference_mode()
def confirm(f, tok, deck, root):
    for site in load(root/"SELECTION.json")["winners"]:
        branch = (site["layer"]-1, site["write"])
        confirm_site(f, tok, deck, root, branch)


def confirm_site(f, tok, deck, root, branch):
    groups = [x for x in deck if x["split"] == "confirmation"]
    prompts = [tok.encode(prompt(row["conditions"][c]), add_special_tokens=False)
               for row in groups for c in CONDITIONS]

    gaps = []
    for i in range(0, len(prompts), 24):
        ids, pos = pad(prompts[i:i+24])
        trace = Capture(pos, (branch,))
        f.run(ids, 2, record=trace)
        gaps.append(trace.data[(1, *branch)] - trace.data[(0, *branch)])
    gap = torch.cat(gaps)
    noise = torch.Generator(device=gap.device).manual_seed(211121333)
    gauss = dose(torch.randn(gap.shape, generator=noise, device=gap.device), gap.norm(dim=-1, keepdim=True))

    donors = []
    for i, g in enumerate(groups):
        for c in CONDITIONS:
            y = g["conditions"][c]["target"]
            j = next((i+d) % len(groups) for d in range(1, len(groups))
                     if abs(groups[(i+d) % len(groups)]["conditions"]["stop"]["target"] - y) >= 100)
            donors.append(j)
    transfer = dose(gap[[j*len(CONDITIONS) for j in donors]], gap.norm(dim=-1, keepdim=True))
    path = root/f"CONFIRMATION_RAW_L{branch[0]+1:02}_{branch[1]}.json"
    rows = load(path)["rows"] if path.exists() else []
    native = {r["id"]:r for r in load(root/"NATIVE.json")["rows"]}
    modes = dict(k1_native=(1, torch.zeros_like(gap)), k2_native=(2, torch.zeros_like(gap)),
                 k1_swap=(1, gap), k2_reverse=(2, -gap), k1_noise=(1, gauss),
                 k2_noise=(2, -gauss), k1_other=(1, transfer))
    for i in range(len(rows), len(groups), 8):
        batch = groups[i:i+8]
        span = slice(i*len(CONDITIONS), (i+len(batch))*len(CONDITIONS))
        results = {}
        for k in (1, 2):
            which = [name for name,(depth,delta) in modes.items() if depth == k]
            p = prompts[span]
            for name in which:
                results[name] = generate(f, tok, p, k, branch, modes[name][1][span])
        norms = gap[span].norm(dim=-1).tolist()
        for j,g in enumerate(batch):
            tested = {}
            for t,c in enumerate(CONDITIONS):
                n = j*len(CONDITIONS)+t
                donor = groups[donors[i*len(CONDITIONS)+n]]
                outputs = {name:items[n] for name,items in results.items()}
                clean = native[g["id"]]["conditions"][c]["outputs"]
                assert [outputs[x]["tokens"] for x in ("k1_native","k2_native")] == [x["tokens"] for x in clean], (g["id"],c)
                tested[c] = dict(outputs=outputs, delta_norm=norms[n], donor=donor["id"],
                                 donor_target=donor["conditions"]["stop"]["target"])
            rows.append(dict(id=g["id"], conditions=tested))
        save(path, dict(selection_sha256=sha(root/"SELECTION.json"), native_sha256=sha(root/"NATIVE.json"), rows=rows))
        print("confirm", branch, len(rows), len(groups), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("screen", "confirm"))
    p.add_argument("model", type=Path)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    assert load(a.root/"BEHAVIOR_GATE.json")["pass"]
    model, tok = setup(a.model)
    changes = load(a.root/"MANIFEST.json")["rows"]
    {"screen": screen, "confirm": confirm}[a.stage](Forward(model), tok, changes, a.root)


if __name__ == "__main__":
    main()

