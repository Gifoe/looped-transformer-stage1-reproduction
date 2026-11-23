"""Replay paired generated prefixes to separate writeback and trajectory effects on EOS."""

import argparse
from pathlib import Path
import re

import torch

from evaluate_generation import generate
from generation import pack_generation
import minimal
from task_data import hash_order, load, save


MARKER = re.compile(r"####\s*[-+]?\d[\d,]*(?:\.\d+)?")


def split_tokens(row, eos):
    tokens = row["generated_tokens"]
    return tokens[:-1] if tokens and tokens[-1] == eos else tokens


def eligible(tok, first, second, t):
    if len(first) < t or len(second) < t:
        return False
    return not any(MARKER.search(tok.decode(sequence[:t], skip_special_tokens=True)) for sequence in (first, second))


def comparisons(tok, before, after):
    for native, edited in zip(before, after, strict=True):
        if native["id"] != edited["id"]:
            raise RuntimeError("Unpaired generated trajectories")
        left = split_tokens(native, tok.eos_token_id)
        right = split_tokens(edited, tok.eos_token_id)
        for t in (0, 2, 4, 8):
            if eligible(tok, left, right, t):
                yield dict(id=native["id"], position=t, sample="fixed", native=left[:t], controlled=right[:t])
        t = len(right)
        if edited["stop"] == "eos" and eligible(tok, left, right, t):
            yield dict(id=native["id"], position=t, sample="observed_premature_stop", native=left[:t], controlled=right[:t])


def stopping_odds(model, net, tok, prompts, prefixes, depth):
    device = next(model.parameters()).device
    eos = tok.eos_token_id
    values = []
    for offset in range(0, len(prompts), 16):
        selected = prompts[offset:offset+16]
        generated = prefixes[offset:offset+16]
        packed, last, _ = pack_generation(selected, generated, list(range(len(selected))), 0, device)
        hidden, _ = minimal.forward(model, packed, depth, net)
        logits = model.lm_head(hidden[last]).float()
        alternate = torch.cat((logits[:, :eos], logits[:, eos+1:]), dim=1)
        odds = logits[:, eos] - alternate.logsumexp(-1)
        argmax = logits.argmax(-1) == eos
        values.extend(zip(odds.tolist(), argmax.tolist(), strict=True))
    return values


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "previous", "model", "data", "retention", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--depth", type=int, default=4)
    a = p.parse_args()
    archive, model, tok, runtime = minimal.runtime(a.archive, a.previous, a.model)
    controller = minimal.load_net(a.retention).requires_grad_(False)
    rows = sorted(load(a.data), key=lambda x: hash_order(x["sample_id"], "prefix-causality"))[:a.n]
    a.out.mkdir(parents=True, exist_ok=True)
    before, old_cost = generate(archive, model, tok, None, rows, "gsm_process", a.depth, 24, 192)
    after, new_cost = generate(archive, model, tok, controller, rows, "gsm_process", a.depth, 24, 192)
    cases = list(comparisons(tok, before, after))
    prompt_tokens = {row["sample_id"]: tok.encode(row["prompt"], add_special_tokens=False) for row in rows}
    encoded = [prompt_tokens[case["id"]] for case in cases]
    for origin in ("native", "controlled"):
        prefixes = [case[origin] for case in cases]
        for name, net in (("native", None), ("retention", controller)):
            predictions = stopping_odds(model, net, tok, encoded, prefixes, a.depth)
            for row, (odds, top) in zip(cases, predictions, strict=True):
                row[f"{origin}_{name}"] = dict(eos_log_odds=odds, eos_top1=top)
    for row in cases:
        base = row["native_native"]["eos_log_odds"]
        path = row["controlled_native"]["eos_log_odds"]
        net = row["controlled_retention"]["eos_log_odds"]
        row["path_effect"] = path - base
        row["controlled_direct_effect"] = net - path
    save(a.out / "NATIVE.jsonl", before)
    save(a.out / "RETENTION.jsonl", after)
    save(a.out / "PREFIX_REPLAY.jsonl", cases)
    minimal.dump(a.out / "RECEIPT.json", dict(n_trajectories=len(rows), n_replays=len(cases), depth=a.depth,
        native_cost=old_cost, retention_cost=new_cost, model_sha256=runtime["model_sha256"],
        data_sha256=minimal.sha(a.data), checkpoint_sha256=minimal.sha(a.retention),
        conditional_endpoint="observed shortened prefix provided both arms before formal marker at this length"))
    print("DONE", len(rows), len(cases), flush=True)


if __name__ == "__main__":
    main()
