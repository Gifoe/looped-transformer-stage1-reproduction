"""Keep 25-shot input length and all query positions while masking the demonstrations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from protocol_arc import (
    DEMOS, NATIVE, TEMPLATE, batches, build_jobs, digest, dump, key, load_runtime,
    plan, question, score_raw, tokenize,
)


def masked_jobs(tok, pre, cond, rows):
    demo, shown, _ = TEMPLATE[cond]
    requests, facts = build_jobs(tok, pre, cond, rows, False)
    demo_tokens = None
    for row, rec in zip(rows, facts):
        q = question(row["question"], row["choices"]["text"] if shown else None)
        full_ids, query_ids = tokenize(tok, pre[demo] + q), tokenize(tok, q)
        assert full_ids[-len(query_ids):] == query_ids
        demo_n = len(full_ids)-len(query_ids)
        assert demo_n > 0
        actual_demo_tokens = full_ids[:demo_n]
        if demo_tokens is None:
            demo_tokens = actual_demo_tokens
        assert demo_tokens == actual_demo_tokens
        assert rec["context_ids_sha256"] == digest(json.dumps(full_ids).encode())
        rec["masked_positions"] = [0,demo_n]
    result = []
    for ids, no_mask, targets in requests:
        assert no_mask is None and list(ids[:demo_n]) == demo_tokens
        result.append(((0,) * demo_n + ids[demo_n:], (0,demo_n), targets))
    return result, facts, demo_n


def summary(a, b, m):
    def prediction(row, scores):
        denominator = row["characters"] if m in ("B","D") else [1] * len(scores)
        return max(range(len(scores)), key=lambda idx: scores[idx]/denominator[idx])
    return {
        "rows": len(a),
        "masked_correct": [sum(prediction(r,r["scores"][k]) == r["gold"] for r in a)
                           for k in sorted(NATIVE)],
        "same_length_unmasked_correct": [sum(prediction(r,r["scores"][k]) == r["gold"] for r in b)
                                        for k in sorted(NATIVE)],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--demos", type=Path, default=DEMOS)
    a = p.parse_args()
    pre, split, selection = plan(a.data, a.demos, a.out)
    ids = set(selection["validation"]["controls"])
    rows = sorted((r for r in split["validation"] if r["id"] in ids),
                  key=lambda r: key("demonstration-mask-subset",r["id"]))[:32]
    config = {
        "dataset": "ARC-Challenge Validation",
        "ids": [r["id"] for r in rows],
        "protocols": ["B","D","F"],
        "context_control": "Mask all demonstration keys and replace token IDs with 0. The same token count, "
                           "all query position IDs and answer scoring target are preserved. The effective "
                           "visible context is shortened by the number of masked demonstration tokens.",
        "implementation_sha256": digest(Path(__file__).read_bytes()),
    }
    dump(a.out/"DEMONSTRATION_MASK_CONFIG.json", config)
    model, tok, _ = load_runtime(a.model)
    report = {}
    for condition in config["protocols"]:
        jobs, facts, n_demo = masked_jobs(tok, pre, condition, rows)
        original = {r["id"]: r for r in (json.loads(x) for x in (
            a.out/f"native.validation.{condition}.jsonl").read_text().splitlines())}
        baseline = [original[r["id"]] for r in facts]
        assert all(a["context_ids_sha256"]==b["context_ids_sha256"] and
                   a["context_tokens"]==b["context_tokens"] and a["gold"]==b["gold"]
                   for a,b in zip(facts,baseline))
        torch.cuda.synchronize()
        start = time.perf_counter()
        for items in batches(jobs, 4, 8192):
            members, scored = score_raw(model, items, 4, NATIVE)
            for name, scores in scored.items():
                assert np.isfinite(scores).all()
                for (q,c), score in zip(members,scores):
                    dst = facts[q]
                    values = dst["scores"].setdefault(name, [None]*len(dst["labels"]))
                    values[c] = float(score)
        torch.cuda.synchronize()
        path = a.out/f"native.validation.{condition}_mask_demonstrations.jsonl"
        path.write_text("".join(json.dumps(m)+"\n" for m in facts), encoding="utf-8")
        report[condition] = {**summary(facts,baseline,condition),
                             "masked_prefix_tokens": n_demo,
                             "wall_seconds": time.perf_counter()-start,
                             "sha256": digest(path.read_bytes())}
        print(condition, report[condition], flush=True)
    dump(a.out/"DEMONSTRATION_MASK_METRICS.json", report)


if __name__ == "__main__":
    main()
