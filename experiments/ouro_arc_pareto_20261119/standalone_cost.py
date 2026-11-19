"""Measure real full-split execution for each native depth and frozen references."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze_arc import load
from run_arc import dump, encode, load_runtime, pack, read_data, scores, states, variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    model, tokenizer, _ = load_runtime(args.model)
    rows = read_data(args.data, "test")
    shared = load(args.out / "test_predictions.jsonl")
    assert [row["id"] for row in rows] == [row["id"] for row in shared]
    frozen = json.loads((args.out / "freeze.json").read_text())
    references = set(frozen["strong_baseline_by_depth"].values())
    methods = sorted(references | {f"native_k{k}" for k in range(1, 5)})
    candidates, _ = encode(tokenizer, rows)
    candidates.sort(key=lambda x: len(x[0]))
    result = {}

    for name in methods:
        depth = int(name[-1])
        needs_raw = name.startswith("loopcd_hidden")

        def process(items):
            inp, group, position, targets = pack(items)
            h, raw, _ = states(model, inp, depth, capture_raw=needs_raw)
            logits = dict(variants(model, [v[group, position] for v in h],
                                   [v[group, position] for v in raw], {name}))[name]
            return scores(logits, group.cpu().numpy(), targets, len(items))

        process(candidates[:16])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        max_error, total = 0.0, 0
        for i in range(0, len(candidates), 16):
            items = candidates[i:i+16]
            evaluated = process(items)
            recorded = [shared[q]["scores"][name][a] for _, _, q, a in items]
            max_error = max(max_error, float(np.max(np.abs(evaluated - recorded))))
            total += len(items)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        result[name] = dict(items=len(rows), candidates=total, executed_loops=depth, batch_size=16,
                            wall_s=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            max_abs_score_diff_vs_shared_run=max_error)
        print(name, result[name], flush=True)
        assert max_error == 0, (name, max_error)
    dump(args.out / "standalone_cost.json", result)


if __name__ == "__main__":
    main()
