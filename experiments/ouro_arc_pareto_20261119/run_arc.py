import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ouro_cross_task_20261118"))
from run import load_runtime

REVISION = "fdb98c79578d61bdf502e2ade3253c71154dd11f"
SPLIT_COUNTS = {"train": 1119, "validation": 299, "test": 1172}
SEED = 20261119


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_data(path, split):
    rows = pq.read_table(path / f"{split}.parquet").to_pylist()
    assert len(rows) == len({row["id"] for row in rows}) == SPLIT_COUNTS[split]
    return rows


def encode(tokenizer, rows):
    candidates, result = [], []
    for question, row in enumerate(rows):
        context = f'Question: {row["question"]}\nAnswer:'
        labels = row["choices"]["label"]
        lengths, tokens, work = [], [], []
        prefix = tokenizer.encode(context, add_special_tokens=False)
        for choice, text in enumerate(row["choices"]["text"]):
            joint = tokenizer.encode(context + " " + text, add_special_tokens=False)
            assert joint[:len(prefix)] == prefix
            n = len(joint) - len(prefix)
            assert n > 0 and len(text) > 0
            candidates.append((joint, n, question, choice))
            lengths.append(len(text))
            tokens.append(n)
            work.append(len(joint) - 1)
        result.append(dict(id=row["id"], labels=labels, gold=labels.index(row["answerKey"]),
                           characters=lengths, answer_tokens=tokens, scored_input_tokens=work,
                           scores={}))
    return candidates, result


def pack(items):
    width = max(len(ids) - 1 for ids, *_ in items)
    r, p, target, inp = [], [], [], []
    for i, (ids, n, *_) in enumerate(items):
        r.extend([i] * n)
        p.extend(range(len(ids) - n - 1, len(ids) - 1))
        target.extend(ids[-n:])
        inp.append(ids[:-1] + [0] * (width + 1 - len(ids)))
    return tuple(torch.tensor(a, device="cuda") for a in (inp, r, p, target))


def states(model, inp, depth, capture_raw=False, use_cache=False, cache=None):
    core = model.model
    core.total_ut_steps = depth
    if use_cache:
        model.config.total_ut_steps = depth
    raw = []
    hook = core.norm.register_forward_pre_hook(lambda _, args: raw.append(args[0])) if capture_raw else None
    try:
        out, hidden, _ = core(input_ids=inp, use_cache=use_cache, past_key_values=cache)
    finally:
        if hook:
            hook.remove()
    assert len(hidden) == depth
    return hidden, raw, out.past_key_values


def hidden_mix(model, a, b):
    return model.model.norm((1.5 * b.float() - 0.5 * a.float()).to(b.dtype))


def secant(model, a, b, c, direct=False):
    # Estimate local contraction in the latest observed direction. Clip the
    # fixed-point displacement to at most one last step without using labels.
    a, b, end = a.float(), b.float(), c.float()
    previous, latest = b - a, end - b
    ratio = (previous * latest).sum(-1, keepdim=True) / previous.square().sum(-1, keepdim=True).clamp_min(1e-8)
    gain = torch.ones_like(ratio) if direct else ratio.clamp(0.0, 0.5) / (1 - ratio.clamp(0.0, 0.5))
    return model.model.norm((end + gain * latest).to(c.dtype))


def variants(model, h, raw, names=None):
    """Yield one tensor at a time to keep vocabulary buffers bounded."""
    def needed(name):
        return names is None or name in names

    z1 = None
    for depth, hidden in enumerate(h, start=1):
        native = f"native_k{depth}"
        fixed = f"loopcd_fixed_k{depth}"
        adaptive = f"loopcd_adaptive_k{depth}"
        guided_hidden = f"loopcd_hidden_k{depth}"
        if (needed(native) or (depth > 1 and (needed(fixed) or needed(adaptive)))):
            logits = model.lm_head(hidden).float()
            if needed(native):
                yield native, logits
        if depth > 1 and (needed(fixed) or needed(adaptive)):
            if z1 is None:
                z1 = model.lm_head(h[0]).float()
            delta = logits - z1
            if needed(fixed):
                yield fixed, logits + 0.5 * delta
            if needed(adaptive):
                top = logits.topk(2, dim=-1).values
                top = (top - logits.logsumexp(dim=-1, keepdim=True)).exp()
                gain = 1 - (top[:, :1] - top[:, 1:])
                yield adaptive, logits + gain * delta
        if depth > 1 and needed(guided_hidden):
            yield guided_hidden, model.lm_head(hidden_mix(model, raw[0], raw[depth - 1])).float()
        if depth >= 3:
            for tag, direct in (("secant", False), ("laststep", True)):
                name = f"{tag}_k{depth}"
                if needed(name):
                    mixed = secant(model, *raw[depth-3:depth], direct=direct)
                    yield name, model.lm_head(mixed).float()


def scores(logits, group, target, batch_size):
    selected = logits.log_softmax(dim=-1).gather(1, target[:, None]).squeeze(1)
    # Ordered CPU float64 sums avoid scatter_add atomic accumulation order.
    return np.bincount(group, weights=selected.cpu().numpy(), minlength=batch_size)


def record(data, items, outputs):
    for name, values in outputs.items():
        for (_, _, q, a), value in zip(items, values):
            table = data[q]["scores"].setdefault(name, [None] * len(data[q]["labels"]))
            table[a] = float(value)


def write_predictions(path, rows):
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            for values in row["scores"].values():
                assert np.isfinite(values).all()
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def check_native(model, packed):
    inp, r, p, _ = packed
    h, _, _ = states(model, inp, 4)
    differences = {}
    for depth in range(1, 5):
        model.model.total_ut_steps = depth
        indices = torch.unique(p, sorted=True)
        direct = model(input_ids=inp, use_cache=False, logits_to_keep=indices,
                       exit_at_step=depth - 1).logits[r, torch.searchsorted(indices, p)].float()
        direct_dense = torch.cat([model.lm_head(h[depth - 1][r, p]).float()])
        differences[f"k{depth}"] = {
            "max_abs_logit": (direct - direct_dense).abs().max().item(),
            "top1_equal": (direct.argmax(-1) == direct_dense.argmax(-1)).sum().item(),
            "tokens": len(r),
        }
    return differences


def score_split(model, tokenizer, rows, batch_size, out, split):
    candidates, result = encode(tokenizer, rows)
    candidates.sort(key=lambda x: len(x[0]))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for begin in range(0, len(candidates), batch_size):
        items = candidates[begin:begin+batch_size]
        inp, group, position, target = pack(items)
        if begin == 0 and split == "validation":
            dump(out / "native_parity.json", check_native(model, (inp, group, position, target)))
        h, raw, _ = states(model, inp, 4, capture_raw=True)
        h = [x[group, position] for x in h]
        raw = [x[group, position] for x in raw]
        index = group.cpu().numpy()
        output = {name: scores(z, index, target, len(items)) for name, z in variants(model, h, raw)}
        record(result, items, output)
        if begin % (batch_size * 24) == 0:
            print(split, begin + len(items), "/", len(candidates), flush=True)
    write_predictions(out / f"{split}_predictions.jsonl", result)
    dump(out / f"{split}_execution.json", dict(
        candidates=len(candidates), items=len(rows), batches=(len(candidates)+batch_size-1)//batch_size,
        evaluated_stacks_per_token=4, jointly_evaluated_variants=sorted(result[0]["scores"]),
        wall_s=time.perf_counter() - start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), batch_limit=batch_size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=["validation", "test", "oracle", "latency"], required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model, tokenizer, runtime = load_runtime(args.model)
    if args.stage == "validation":
        runtime.update(dataset_revision=REVISION, seed=SEED,
                       validation_sha256=hashlib.sha256((args.data / "validation.parquet").read_bytes()).hexdigest(),
                       model_config_layers=len(model.model.layers),
                       vocabulary_size=model.lm_head.out_features, hidden_size=model.config.hidden_size,
                       block_linear_weights=sum(p.numel() for m in model.model.layers.modules()
                                                if isinstance(m, torch.nn.Linear)
                                                for p in [m.weight]))
        dump(args.out / "runtime.json", runtime)
    if args.stage in ("validation", "test"):
        if args.stage == "test":
            freeze = json.loads((args.out / "freeze.json").read_text())
            for name, digest in freeze["config_sha256"].items():
                assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
            dump(args.out / "test_input.json", dict(sha256=hashlib.sha256(
                (args.data / "test.parquet").read_bytes()).hexdigest(), dataset_revision=REVISION))
        score_split(model, tokenizer, read_data(args.data, args.stage),
                    args.batch_size, args.out, args.stage)
    else:
        from diagnostics import latency, oracle
        if args.stage == "oracle":
            oracle(model, tokenizer, read_data(args.data, "validation"), args.out, args.batch_size)
        else:
            latency(model, tokenizer, read_data(args.data, "validation"), args.out)


if __name__ == "__main__":
    main()
