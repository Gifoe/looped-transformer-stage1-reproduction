"""Collect actual premature-EOS prefixes from training inputs and measure continuation odds."""

import argparse
from itertools import islice
from pathlib import Path

import torch

from evaluate_generation import generate
from generation import pack_generation
import minimal
from prefix_causality import MARKER, split_tokens
from task_data import encode_target, hash_order, load, save


def score_prefixes(model, net, tokenizer, rows, prefixes, depth):
    device = next(model.parameters()).device
    text = [tokenizer.encode(row["prompt"], add_special_tokens=False) for row in rows]
    packed, last, width = pack_generation(text, prefixes, list(range(len(rows))), 0, device)
    hidden, _ = minimal.forward(model, packed, depth, net)
    logits = model.lm_head(hidden[last]).float()
    eos = tokenizer.eos_token_id
    alternatives = torch.cat((logits[:, :eos], logits[:, eos+1:]), dim=-1)
    return alternatives.logsumexp(-1) - logits[:, eos], width * len(rows) * depth


def choose_batch(rows, archive):
    return [(i, row, archive[row["sample_id"]]) for i, row in enumerate(rows) if row["sample_id"] in archive]


@torch.inference_mode()
def main():
    from train_advantage import load_fit, schedule

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "previous", "model", "data", "controller", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--steps", type=int, default=336)
    p.add_argument("--n", type=int, default=168)
    p.add_argument("--seed", type=int, default=261124173)
    a = p.parse_args()
    fit = load_fit(a.data)
    planned = [row for _, packet in islice(schedule(fit, 4, a.seed), a.steps)
               for row in packet if row["dataset"] == "gsm8k"]
    rows = sorted(planned, key=lambda x: hash_order(x["sample_id"], "advantage/off-policy"))[:a.n]
    archive, model, tokenizer, runtime = minimal.runtime(a.archive, a.previous, a.model)
    net = minimal.load_net(a.controller).requires_grad_(False)
    predictions, cost = generate(archive, model, tokenizer, net, rows, "gsm_process", 4, 24, 192)
    accepted = []
    for row, prediction in zip(rows, predictions, strict=True):
        if row["sample_id"] != prediction["id"]:
            raise RuntimeError("Different generated training question")
        if prediction["stop"] != "eos" or MARKER.search(prediction["prediction"]):
            continue
        prefix = split_tokens(prediction, tokenizer.eos_token_id)
        _, _, n = encode_target(tokenizer, row)
        stage = "early" if len(prefix) < n // 2 else "middle"
        accepted.append(dict(id=prediction["id"], stage=f"gsm8k/{stage}", prefix=prefix))
    a.out.mkdir(parents=True, exist_ok=True)
    save(a.out / "GENERATED_TRAIN.jsonl", predictions)
    save(a.out / "TRAIN_PREFIXES.jsonl", accepted)
    minimal.dump(a.out / "RECEIPT.json", dict(n_generated=len(rows), n_premature=len(accepted), seconds=cost["seconds"],
        model_sha256=runtime["model_sha256"], source_sha256=minimal.sha(a.data / "gsm8k_fit.jsonl"),
        method_sha256=minimal.sha(a.controller), prefixes_sha256=minimal.sha(a.out / "TRAIN_PREFIXES.jsonl")))
    print("DONE", len(rows), len(accepted), flush=True)


if __name__ == "__main__":
    main()
