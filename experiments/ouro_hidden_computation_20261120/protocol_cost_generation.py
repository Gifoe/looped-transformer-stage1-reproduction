"""Standalone candidate costs and cached, unconstrained greedy ARC generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import time

import torch
from transformers import DynamicCache

from protocol_arc import (
    DEMOS, LOOPCD, NATIVE, TEMPLATE, build_jobs, digest, dump, key, load_runtime,
    plan, question, score_raw, states, tokenize, variants,
)


def synchronize():
    torch.cuda.synchronize()
    return time.perf_counter()


def native_cache(model, depth):
    cache = DynamicCache()
    slots = depth * len(model.model.layers)
    cache.append_new_layers(slots-1)
    assert len(cache.layers) == slots
    return cache


def next_token(model, input_ids, budget, method, cache):
    hidden, raw, cache_out = states(model, input_ids, budget, capture_raw="hidden" in method,
                                    use_cache=True, cache=cache)
    h = [v[:, -1, :] for v in hidden]
    raw = [v[:, -1, :] for v in raw]
    (name, z), = variants(model, h, raw, {method})
    assert name == method and cache_out is cache
    return int(z.argmax(-1).item())


def generated(model, tok, prompt, method, max_new):
    depth = int(method[-1])
    cache = native_cache(model, depth)
    context = tokenize(tok, prompt)
    inp = torch.tensor([context], device="cuda")
    start = synchronize()
    next_id = next_token(model, inp, depth, method, cache)
    n_prompt = cache.get_seq_length()
    prefill = synchronize() - start
    assert n_prompt == len(context)
    ids, decode = [], []
    eos = tok.eos_token_id
    for step in range(max_new):
        if next_id == eos:
            stop = "eos"
            break
        ids.append(next_id)
        text = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if "\n" in text:
            stop = "newline"
            break
        if step + 1 == max_new:
            stop = "token_limit"
            break
        nxt = torch.tensor([[next_id]], device="cuda")
        point = synchronize()
        next_id = next_token(model, nxt, depth, method, cache)
        decode.append(synchronize() - point)
        assert cache.get_seq_length() == len(context)+len(ids)
    return {
        "prefill_seconds": prefill, "decode_seconds": decode, "generated_ids": ids,
        "text": tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
        "context_tokens": len(context), "stop": stop,
    }


def normalize(answer):
    return re.sub(r"\s+", " ", answer.casefold().strip()).rstrip(".!?").strip()


def interpret(raw, choices, spec):
    line = raw.split("\n", 1)[0].strip()
    letter = re.fullmatch(r"\s*([A-Ea-e])(?:[.)])?\s*", line)
    if spec == "letter":
        guess = ord(letter[1].upper()) - 65 if letter else None
        if guess is not None and guess >= len(choices):
            guess = None
        return {"exact": guess, "bare_letter": guess is not None}
    matches = [i for i,text in enumerate(choices) if normalize(text) == normalize(line)]
    return {"exact": matches[0] if len(matches) == 1 else None, "bare_letter": bool(letter)}


def run(model, tok, prefix, rows, protocol, method, max_new):
    spec = TEMPLATE[protocol]
    depth = int(method[-1])
    records = []
    # The hash-first prompt is a separate warmup, excluded from all timing metrics.
    for i, row in enumerate(rows):
        choices = row["choices"]["text"]
        query = question(row["question"], choices if spec[1] else None)
        prompt = prefix[spec[0]] + query
        jobs, details = build_jobs(tok, prefix, protocol, [row], False)
        c0 = synchronize()
        scored_ids, scores = score_raw(model, jobs, depth, {method})
        candidate_s = synchronize() - c0
        generated_rec = generated(model, tok, prompt, method, max_new)
        if i == 0:
            continue
        scores_by_content = [None]*len(choices)
        for (_, a), value in zip(scored_ids, scores[method]):
            scores_by_content[a] = float(value)
        recorded = details[0]
        records.append({
            "id": row["id"], "gold": recorded["gold"], "prompt_sha256": recorded["prompt_sha256"],
            "candidate_seconds": candidate_s, "candidate_scores": scores_by_content,
            "scored_unique_requests": len(jobs),
            "distinct_input_tokens": sum(len(item[0]) for item in jobs),
            "padded_input_tokens": len(jobs)*max(len(item[0]) for item in jobs),
            **generated_rec, **interpret(generated_rec["text"], choices, spec[2]),
        })
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--demos", type=Path, default=DEMOS)
    p.add_argument("--generation-n", type=int, default=16)
    p.add_argument("--max-new", type=int, default=12)
    a = p.parse_args()
    prefix, splits, _ = plan(a.data, a.demos, a.out)
    chosen = sorted(splits["validation"], key=lambda r: key("generation-and-cost", r["id"]))[:a.generation_n+1]
    config = {
        "warmup_id": chosen[0]["id"], "timed_ids": [r["id"] for r in chosen[1:]], "budget_new_tokens": a.max_new,
        "scoring_batch": "All unique candidate token sequences for each question, scored as one right-padded batch.",
        "generation": "Unconstrained full-vocabulary exact greedy selection. Synchronized wall-clock times "
                      "for prefill and per generated step at batch size 1. One fresh cache per prompt/method.",
        "exact": "Unique canonical full first response line or entire displayed letter, with optional punctuation. "
                 "Full-answer outputs that only emit a bare letter do not match content. No semantic grading.",
        "implementation_sha256": digest(Path(__file__).read_bytes()),
    }
    dump(a.out/"GENERATION_COST_CONFIG.json", config)
    model, tok, _ = load_runtime(a.model)
    all_metrics = {}
    for protocol in "ABCDEF":
        all_metrics[protocol] = {}
        for method in sorted(NATIVE | LOOPCD, key=lambda n: (n[-1], n)):
            path = a.out/f"generation.{protocol}.{method}.json"
            if path.exists():
                records = json.loads(path.read_bytes())
                assert [r["id"] for r in records] == config["timed_ids"]
            else:
                records = run(model, tok, prefix, chosen, protocol, method, a.max_new)
                dump(path, records)
            decodes = [v*1000 for row in records for v in row["decode_seconds"]]
            all_metrics[protocol][method] = {
                "rows": len(records), "exact_correct": sum(r["exact"] == r["gold"] for r in records),
                "exact_parseable": sum(r["exact"] is not None for r in records),
                "bare_letters": sum(r["bare_letter"] for r in records),
                "truncated": sum(r["stop"] == "token_limit" for r in records),
                "median_candidate_ms": 1000*statistics.median(r["candidate_seconds"] for r in records),
                "median_prefill_ms": 1000*statistics.median(r["prefill_seconds"] for r in records),
                "median_decode_ms": statistics.median(decodes) if decodes else None,
                "median_total_generated_ms": 1000*statistics.median(
                    r["prefill_seconds"] + sum(r["decode_seconds"]) for r in records),
                "candidate_scored_tokens": sum(r["distinct_input_tokens"] for r in records),
                "padded_candidate_tokens": sum(r["padded_input_tokens"] for r in records),
                "records_sha256": digest(path.read_bytes()),
            }
            print(protocol, method, len(records), all_metrics[protocol][method]["exact_correct"], flush=True)
    dump(a.out/"GENERATION_COST_METRICS.json", all_metrics)


if __name__ == "__main__":
    main()
