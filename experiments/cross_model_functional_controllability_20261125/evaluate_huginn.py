"""Evaluate Huginn Native and Rank-1 on the fixed eight-benchmark population."""

import argparse
from collections import Counter
from itertools import groupby
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

from evaluate_benchmarks import candidate_row, mmlu_metrics
from evaluate_core import finish, verified
from evaluate_transfer import ARM_NAMES, TASKS, pair, timestamp_cost
from huginn_control import CAP, REVISION, SEED, RankOne, forward, load_model, sha256
import minimal


DEPTH = 32
SCORE = "huginn"
CACHE_TOKEN_CAP = 2304


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_shard(stage, model, tok, rows, config, controller, seed):
    encoder = stage.HarnessEncoder(tok, config["max_length"])
    work = stage.compact_prefixes(stage.encode_rows(rows, encoder))
    output = stage.render(rows)
    cost = Counter()
    batches = stage.batch_work(work, config["max_batch"], config["token_cap"], config["attention_cap"])
    for step, batch in enumerate(batches):
        packed, refs, gather, token_refs, counts = stage.pack(batch, model.device)
        index = (packed[1], packed[2])
        states = forward(model, packed[0], index, None, DEPTH, seed + step, controller)
        scores = stage.score_head(model, [states], (DEPTH,), config["head_rows"], gather, token_refs, counts,
                                  cost, apple=False)
        stage.record(output, refs, {SCORE: scores["minimal"]})
        cost.update(forward_calls=1, padded_input_tokens=packed[0].numel())
    result = [candidate_row(row, SCORE) for row in output]
    for prediction, source in zip(result, rows, strict=True):
        if (prediction["id"], prediction["choices"], prediction["gold"]) != (
                source["id"], source["choices"], source["target"]):
            raise ValueError(f"Huginn candidate metadata mismatch at {source['id']}")
    cost["dense_loop_token_calls"] = DEPTH * cost["padded_input_tokens"]
    return result, dict(cost)


def exact_length_batches(items, max_new, max_batch):
    ordered = sorted(items, key=lambda item: (len(item.ids), item.key))
    for length, group in groupby(ordered, key=lambda item: len(item.ids)):
        rows = list(group)
        size = max(1, min(max_batch, CACHE_TOKEN_CAP // (length + max_new)))
        for lo in range(0, len(rows), size):
            yield rows[lo:lo + size]


def make_cache(model, batch, length, kind):
    official = importlib.import_module(type(model).__module__)
    if kind == "dynamic":
        return official.HuginnDynamicCache(lookup_strategy="full")
    c = model.config
    layers = c.n_layers_in_prelude + DEPTH * c.n_layers_in_recurrent_block + c.n_layers_in_coda
    return official.HuginnStaticCache(length, layers, c.num_key_value_heads, c.head_dim, batch,
                                      lookup_strategy="full", device=model.device, dtype=model.dtype)


def generation_shard(decoder, model, tok, items, max_new, controller, seed, cache_kind="static"):
    incoming = torch.tensor([r.ids for r in items], dtype=torch.long, device=model.device)
    batch, length = incoming.shape
    if length + max_new > model.config.block_size:
        raise ValueError("MMLU-Pro prompt exceeds Huginn's native context capacity")
    cache = make_cache(model, batch, length + max_new, cache_kind)
    row = torch.arange(batch, device=model.device)
    positions = torch.arange(length, device=model.device)
    generated = [[] for _ in items]
    stopped = [None] * batch
    for step in range(max_new):
        index = (row, torch.full_like(row, incoming.shape[1] - 1))
        logits = forward(model, incoming, index, index, DEPTH, seed + step, controller, cache, positions)
        token = logits.argmax(-1)
        for j, value in enumerate(token.tolist()):
            if stopped[j] is not None:
                continue
            generated[j].append(value)
            text = tok.decode(generated[j], skip_special_tokens=True)
            if value == tok.eos_token_id:
                stopped[j] = "eos"
            elif decoder.extract("mmlu_pro", text, items[j].choices):
                stopped[j] = "letter"
        if all(stopped):
            break
        incoming = token[:, None]
        positions = torch.tensor([length + step], device=model.device)
    summary = []
    for item, ids, reason in zip(items, generated, stopped, strict=True):
        text = tok.decode(ids, skip_special_tokens=True)
        summary.append(dict(id=item.key, gold=item.gold, generated=text, tokens=ids, stop=reason or "limit",
                            predicted=decoder.extract("mmlu_pro", text, item.choices)))
    return mmlu_metrics(summary, items, decoder), dict(calls=step+1, generated_tokens=sum(map(len, generated)))


def score(a, task, arm, plan, model, tok, controller, runtime):
    import generation_followup_run as decoder
    import stage2_score as stage

    target = a.out / f"{task}__{arm}.jsonl"
    shards = a.out / "shards" / target.stem
    shards.mkdir(parents=True, exist_ok=True)
    spec = dict(task=task, arm=arm, protocol_sha256=plan["protocol_sha256"], depth=DEPTH,
                checkpoint=plan["controller_sha256"] if arm == "t1" else None, seed=SEED)
    generation = task == "mmlu_pro"
    if generation:
        source = a.followup / "data/mmlu_pro.parquet"
        if (sha256(source) != plan["original"]["mmlu_pro_sha256"]
                or sha256(Path(decoder.__file__)) != plan["original"]["mmlu_pro_code_sha256"]):
            raise ValueError("The original MMLU-Pro protocol artifacts changed")
        rows = decoder.prompt_items(source, task, tok)
        n, batch_size = decoder.TASKS[task]
        chunks = list(exact_length_batches(rows, n, batch_size))
        ids = [[item.key for item in chunk] for chunk in chunks]
        spec.update(max_new=n, exact_length_batches=True, cache="official_full_static", cache_token_cap=CACHE_TOKEN_CAP)
    else:
        source = a.stage2 / "inputs" / f"{task}.jsonl"
        if (sha256(source) != plan["original"]["seven_requests_sha256"][task]
                or sha256(Path(stage.__file__)) != plan["original"]["stage2_sha256"]):
            raise ValueError(f"The original Stage2 artifacts changed for {task}")
        stage.APPLE_BENCHMARKS = dict.fromkeys(stage.APPLE_BENCHMARKS, 0)
        rows, _ = stage.read_export(SimpleNamespace(input=source, manifest=source.with_suffix(".jsonl.manifest.json"),
                                                   plan=False, benchmarks=(task,)))
        archive_cfg = load_json(a.stage2 / "scores" / task / "LOCK.json")
        config = {name: archive_cfg[name] for name in
                  ("max_length", "max_batch", "token_cap", "attention_cap", "head_rows", "shard_size")}
        config.update(max_batch=min(config["max_batch"], 32), max_length=min(config["max_length"], model.config.block_size),
                      token_cap=min(config["token_cap"], 1984))
        spec["config"] = config
        stride = config["shard_size"]
        chunks = [rows[start:start+stride] for start in range(0, len(rows), stride)]
        ids = [[item["id"] for item in chunk] for chunk in chunks]
    all_ids = [key for group in ids for key in group]
    if len(all_ids) != TASKS[task][0]:
        raise ValueError(f"Incomplete Huginn population {task}")
    if previous := verified(target, spec, all_ids):
        return previous
    total, complete = Counter(), []
    if generation:
        audit = a.out / f"CACHE_PARITY_{arm}.json"
        if not audit.is_file():
            sample = chunks[len(chunks) // 2][:2]
            direct, _ = generation_shard(decoder, model, tok, sample, n, controller, SEED, "dynamic")
            static, _ = generation_shard(decoder, model, tok, sample, n, controller, SEED)
            if direct != static:
                raise RuntimeError(f"Official Huginn static and dynamic cache generation disagree for {arm}")
            minimal.dump(audit, dict(ids=[r["id"] for r in static], depth=DEPTH, runs_match=True, seed=SEED))
    for i, (chunk, keys) in enumerate(zip(chunks, ids, strict=True)):
        path = shards / f"{i:05d}.jsonl"
        shard = dict(spec, batch_index=i)
        previous = verified(path, shard, keys)
        if previous:
            outputs, cost = previous
        else:
            seed = SEED + 10_000*i
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            if generation:
                outputs, cost = generation_shard(decoder, model, tok, chunk, n, controller, seed)
            else:
                outputs, cost = candidate_shard(stage, model, tok, chunk, config, controller, seed)
            torch.cuda.synchronize()
            cost.update(seconds=time.perf_counter()-begin, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
            finish(path, outputs, shard, cost, runtime)
        complete.extend(outputs)
        timestamp_cost(total, cost)
    finish(target, complete, spec, total, runtime)
    return complete, total


@torch.inference_mode()
def run(a):
    sys.path[:0] = [str(a.stage2 / "code"), str(a.followup)]
    torch.backends.cuda.matmul.allow_tf32 = False
    a.out.mkdir(parents=True, exist_ok=True)
    fit = load_json(a.controller.parent / "FIT.json")
    original = load_json(a.reference)
    if original["expected"] != {task: n for task, (n, _) in TASKS.items()} or original["depth"] != 4:
        raise ValueError("Unexpected original benchmark panel")
    if (fit["steps"], fit["examples"], fit["trainable_params"], fit["checkpoint_sha256"]) != (
            280, 1119, 10560, sha256(a.controller)):
        raise ValueError("Huginn controller differs from the complete frozen training receipt")
    plan = dict(original=original, controller_sha256=fit["checkpoint_sha256"], revision=REVISION,
                model_sha256=sha256(a.model / "MODEL_LOCK.json"), seed=SEED, depth=DEPTH,
                cap=CAP, write="Before final recurrent block at causal predictor positions",
                evaluator_sha256=sha256(Path(__file__)), receiver_sha256=sha256(Path(__file__).with_name("huginn_control.py")))
    lock = a.out / "FREEZE.json"
    if lock.is_file() and load_json(lock) != plan:
        raise ValueError("Existing official Huginn population was scored with a different plan")
    minimal.dump(lock, plan)
    plan["protocol_sha256"] = sha256(lock)
    model, tok = load_model(a.model)
    net = RankOne().to(model.device)
    net.load_state_dict(torch.load(a.controller, weights_only=True, map_location="cpu")["state"])
    runtime = dict(name=REVISION, gpu=torch.cuda.get_device_name(), dtype=str(model.dtype), torch=torch.__version__,
                   width=model.config.n_embd, seed=SEED, bos=tok.bos_token_id, eos=tok.eos_token_id, pad=tok.pad_token_id)
    path = a.out / "RESULTS.json"
    outcomes = load_json(path) if path.is_file() else {}
    for task in (TASKS if a.task == "all" else (a.task,)):
        arms = {name: score(a, task, name, plan, model, tok, net if name == "t1" else None, runtime)
                for name in ARM_NAMES}
        native, edited = [arms[key][0] for key in ARM_NAMES]
        if [(r["id"], r["gold"]) for r in native] != [(r["id"], r["gold"]) for r in edited]:
            raise ValueError(f"Divergent Huginn paired population {task}")
        primary = TASKS[task][1]
        outcomes[task] = dict(primary=primary, metrics={key: pair(native, edited, key) for key in native[0]["metrics"]},
                              cost={arm: dict(arms[arm][1]) for arm in ARM_NAMES},
                              predictions={arm: sha256(a.out / f"{task}__{arm}.jsonl") for arm in ARM_NAMES})
        minimal.dump(path, outcomes)
        print("HUGINN PAIRED", task, outcomes[task]["metrics"][primary], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "controller", "stage2", "followup", "reference", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--task", choices=("all", *TASKS), default="all")
    run(parser.parse_args())
