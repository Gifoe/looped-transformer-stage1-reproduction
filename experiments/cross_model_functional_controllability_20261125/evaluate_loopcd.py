"""Fill missing zero-shot LoopCD comparisons while reusing verified published cells."""

import argparse
from collections import Counter
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

from evaluate_benchmarks import candidate_row, mmlu_metrics
from evaluate_core import finish, verified
from evaluate_transfer import TASKS, load_runtime, pair, timestamp_cost
import minimal


ARMS = ("fixed", "adaptive")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def guidance(last, early, arm, fixed, cap):
    if arm == "fixed":
        strength = fixed
    else:
        probabilities = (last.topk(2, dim=-1).values - last.logsumexp(-1, keepdim=True)).exp()
        strength = cap * (1 - (probabilities[:, :1] - probabilities[:, 1:]))
    return last + strength * (last - early)


def candidate(a, task, model, tok, plan, runtime):
    import stage2_score as stage

    source = a.stage2 / "inputs" / f"{task}.jsonl"
    if (minimal.sha(source) != plan["original"]["seven_requests_sha256"][task]
            or minimal.sha(Path(stage.__file__)) != plan["original"]["stage2_sha256"]):
        raise ValueError(f"Archived candidate requests differ for {task}")
    stage.APPLE_BENCHMARKS = dict.fromkeys(stage.APPLE_BENCHMARKS, 0)
    rows, _ = stage.read_export(SimpleNamespace(input=source, manifest=source.with_suffix(".jsonl.manifest.json"),
                                               plan=False, benchmarks=(task,)))
    ids = [row["id"] for row in rows]
    if len(ids) != TASKS[task][0]:
        raise ValueError(f"Candidate population is incomplete for {task}")
    lock = read(a.stage2 / "scores" / task / "LOCK.json")
    config = {k: lock[k] for k in ("max_length", "max_batch", "token_cap", "attention_cap", "head_rows", "shard_size")}
    protocol = dict(depth=4, task=task, protocol_sha256=plan["sha"], config=config, joint_variants=True,
                    source_hash=minimal.sha(source), fixed=.5, adaptive_cap=1.)
    paths = {arm: a.out / f"{task}__{arm}.jsonl" for arm in ARMS}
    full = {arm: verified(p, dict(protocol, arm=arm), ids) for arm, p in paths.items()}
    if all(full.values()):
        return full
    enc = stage.HarnessEncoder(tok, config["max_length"])
    completed = {arm: [] for arm in ARMS}
    total = Counter()
    stride = config["shard_size"]
    for start in range(0, len(rows), stride):
        chunk = rows[start:start+stride]
        shard_ids = ids[start:start+stride]
        specs = {arm: dict(protocol, start=start, arm=arm) for arm in ARMS}
        files = {arm: a.out / "shards" / f"{task}__{arm}" / f"{start:05d}.jsonl" for arm in ARMS}
        for path in files.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        previous = {arm: verified(p, specs[arm], shard_ids) for arm, p in files.items()}
        if all(previous.values()):
            cost = previous["fixed"][1]
            for arm in ARMS:
                completed[arm].extend(previous[arm][0])
        else:
            rendered = stage.render(chunk)
            work = stage.compact_prefixes(stage.encode_rows(chunk, enc))
            batches = stage.batch_work(work, config["max_batch"], config["token_cap"], config["attention_cap"])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            cost = Counter(head_calls=0)
            for batch in batches:
                packed, refs, gather, token_refs, counts = stage.pack(batch, model.device)
                states, _ = stage.native_trunk(model, packed, (4,), shared=False)
                scores = stage.score_head(model, states, (4,), config["head_rows"], gather, token_refs, counts, cost,
                                          apple=True)
                stage.record(rendered, refs, {arm: scores[f"loopcd_{arm}_k4"] for arm in ARMS})
                cost.update(forward_calls=1, dense_loop_token_calls=4*packed[0].numel())
            torch.cuda.synchronize()
            cost.update(elapsed_seconds=time.perf_counter()-begin, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
            for arm in ARMS:
                predictions = [candidate_row(row, arm) for row in rendered]
                if [(row["id"], row["choices"], row["gold"]) for row in predictions] != [
                        (row["id"], row["choices"], row["target"]) for row in chunk]:
                    raise ValueError(f"Wrong prediction pairing for {task}/{arm}")
                finish(files[arm], predictions, specs[arm], cost, runtime)
                completed[arm].extend(predictions)
        timestamp_cost(total, cost)
    for arm, path in paths.items():
        finish(path, completed[arm], dict(protocol, arm=arm), total, runtime)
    return {arm: (completed[arm], total) for arm in ARMS}


def ouro_generation_forward(cap):
    def logits(model, tokens, valid, indices, arm, net, dose, cache, eligible):
        positions = (valid.long().cumsum(-1) - 1).clamp_min_(0)[:, indices]
        mask = valid[:, None, None, :] & (torch.arange(valid.shape[1], device=tokens.device)[None, :]
                                         <= indices[:, None])[None, None, :, :]
        _, states, _ = model.model(input_ids=tokens, attention_mask={"full_attention": mask}, position_ids=positions,
                                   cache_position=indices, past_key_values=cache, use_cache=cache is not None)
        last = model.lm_head(states[-1][:, -1]).float()
        early = model.lm_head(states[0][:, -1]).float()
        return guidance(last, early, arm, .3, cap)
    return logits


def huginn_generation_forward(arm):
    def contrast(model, tokens, predictors, head_index, depth, seed, controller, cache, positions):
        original = model.core_block_forward
        first = None

        def capture(state, embedded, freqs, mask, past, layer, step):
            nonlocal first
            result, block = original(state, embedded, freqs, mask, past, layer, step)
            if int(step) == 0:
                first = result
            return result, block

        if not hasattr(cache, "loopcd_first_coda"):
            cls = importlib.import_module(type(model).__module__).HuginnStaticCache
            c = model.config
            cache.loopcd_first_coda = cls(cache.max_length, c.n_layers_in_coda, c.num_key_value_heads, c.head_dim,
                                         tokens.shape[0], device=model.device, dtype=model.dtype)
        early_cache = cache.loopcd_first_coda
        hook = model.lm_head.register_forward_pre_hook(lambda _, args: (args[0][head_index],))
        try:
            model.core_block_forward = capture
            try:
                with torch.random.fork_rng(devices=[tokens.device.index]):
                    torch.cuda.manual_seed(seed)
                    details = dict(return_logits=True, return_latents=False, return_head=False, return_stats=False)
                    last = model(input_ids=tokens, num_steps=(depth, 0), output_details=details,
                                 use_cache=True, past_key_values=cache, cache_position=positions).logits
            finally:
                del model.core_block_forward
            if first is None:
                raise RuntimeError("Failed to capture Huginn's first recurrent output")
            early_cache._seen_tokens = cache.get_seq_length()
            early = model.predict_from_latents(first, cache_position=positions, past_key_values=early_cache).logits
        finally:
            hook.remove()
        return guidance(last, early, arm, .2, .25)
    return contrast


@contextmanager
def replace_forward(module, implementation):
    original = module.forward
    module.forward = implementation
    try:
        yield
    finally:
        module.forward = original


def generation(a, model, tok, plan, runtime):
    import generation_followup_run as decoder

    task = "mmlu_pro"
    source = a.followup / "data/mmlu_pro.parquet"
    if (minimal.sha(source) != plan["original"]["mmlu_pro_sha256"]
            or minimal.sha(Path(decoder.__file__)) != plan["original"]["mmlu_pro_code_sha256"]):
        raise ValueError("Archived zero-shot MMLU-Pro generation requests differ")
    items = decoder.prompt_items(source, task, tok)
    if len(items) != TASKS[task][0]:
        raise ValueError("Full MMLU-Pro population required")
    max_new, batch_size = decoder.TASKS[task]
    if a.name == "huginn-0125":
        import evaluate_huginn as huginn

        tasks = list(huginn.exact_length_batches(items, max_new, batch_size))
        module = huginn
        common = dict(depth=32, fixed=.2, cap=.25, capacity_tokens=huginn.CACHE_TOKEN_CAP, separate_coda_cache=True)
    else:
        layers = len(model.model.layers)
        capacity = (14500 * 24) // layers
        tasks = list(decoder.batches(items, max_new, batch_size, capacity))
        bounded = decoder.cache_type(model)
        module = decoder
        common = dict(depth=4, fixed=.3, cap=.5 if a.name.startswith("ouro-1.4b") else 1., capacity_tokens=capacity)

        def cache(n):
            result = bounded(n)
            result.slots = [None] * (4*layers)
            result.lengths = [0] * (4*layers)
            return result

    populated = {}
    ids = [item.key for batch in tasks for item in batch]
    for arm in ARMS:
        spec = dict(task=task, arm=arm, protocol_sha256=plan["sha"], max_new=max_new, **common)
        target = a.out / f"{task}__{arm}.jsonl"
        if prior := verified(target, spec, ids):
            populated[arm] = prior
            continue
        parts = a.out / "shards" / target.stem
        parts.mkdir(parents=True, exist_ok=True)
        results, total = [], Counter()
        implementation = huginn_generation_forward(arm) if a.name == "huginn-0125" else ouro_generation_forward(common["cap"])
        with replace_forward(module, implementation):
            for i, batch in enumerate(tasks):
                path = parts / f"{i:05d}.jsonl"
                block = dict(spec, batch_index=i)
                previous = verified(path, block, [item.key for item in batch])
                if previous:
                    out, cost = previous
                else:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    begin = time.perf_counter()
                    if a.name == "huginn-0125":
                        out, cost = huginn.generation_shard(decoder, model, tok, batch, max_new, None, huginn.SEED+10_000*i)
                    else:
                        raw, _ = decoder.greedy(task, model, tok, batch, arm, None, 1., max_new, cache)
                        out = mmlu_metrics(raw, batch, decoder)
                        cost = dict(calls=max(len(row["tokens"]) for row in raw),
                                    generated_tokens=sum(len(row["tokens"]) for row in raw))
                    torch.cuda.synchronize()
                    cost.update(elapsed_seconds=time.perf_counter()-begin,
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    finish(path, out, block, cost, runtime)
                results.extend(out)
                timestamp_cost(total, cost)
        finish(target, results, spec, total, runtime)
        populated[arm] = (results, total)
    return populated


def matched_predictions(output, others, metric):
    control_tables = {arm: {r["id"]: r for r in rows} for arm, rows in others.items()}
    keys = {r["id"] for r in output}
    for arm, table in control_tables.items():
        if table.keys() != keys or any(table[r["id"]]["gold"] != r["gold"] for r in output):
            raise ValueError(f"Mismatch with frozen {arm} prediction alignment")
    return dict(versus_native=pair([control_tables["native"][r["id"]] for r in output], output, metric),
                rank1_versus_loopcd=pair(output, [control_tables["t1"][r["id"]] for r in output], metric))


def read_predictions(path, sha):
    if minimal.sha(path) != sha:
        raise ValueError(f"Changed archived paired predictions at {path}")
    return minimal.load_jsonl(path)


@torch.inference_mode()
def run(a):
    sys.path[:0] = [str(a.stage2 / "code"), str(a.followup)]
    a.out.mkdir(parents=True, exist_ok=True)
    papers = read(a.paper)["published_zero_shot"].get(a.name, {})
    historical = read(a.transfer / "RESULTS.json")
    if set(historical) != TASKS.keys():
        raise ValueError("Complete Native and Rank-1 populations are required")
    components = [Path(__file__), a.reference, a.paper, a.transfer / "RESULTS.json",
                  a.transfer / "FREEZE.json", a.model / "MODEL_LOCK.json"]
    if a.name == "huginn-0125":
        import evaluate_huginn
        import huginn_control
        components.extend(Path(e.__file__) for e in (evaluate_huginn, huginn_control))
    plan = dict(original=read(a.reference), files={str(path): minimal.sha(path) for path in components},
                source="Liu et al., 2610.02185v1, LoopCD-Logits Fixed / Adaptive", model=a.name)
    lock = a.out / "FREEZE.json"
    if lock.is_file() and read(lock) != plan:
        raise ValueError("Existing LoopCD protocol cannot change after evaluation starts")
    minimal.dump(lock, plan)
    plan["sha"] = minimal.sha(lock)
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.name == "huginn-0125":
        from huginn_control import load_model
        model, tok = load_model(a.model)
        runtime = dict(gpu=torch.cuda.get_device_name(), depth=32, dtype=str(model.dtype))
    else:
        model, tok, _, runtime = load_runtime(a)
    path = a.out / "RESULTS.json"
    results = read(path) if path.exists() else {}
    for task, (n, metric) in TASKS.items():
        if task in papers:
            native, fixed, adaptive = papers[task]
            results[task] = dict(source="paper_aggregate_only", paper_native=native, fixed=fixed, adaptive=adaptive,
                                 primary=metric)
        else:
            outputs = (generation(a, model, tok, plan, runtime) if task == "mmlu_pro"
                       else candidate(a, task, model, tok, plan, runtime))
            predictions = {arm: read_predictions(a.transfer / f"{task}__{arm}.jsonl",
                                                  historical[task]["predictions"][arm]) for arm in ("native", "t1")}
            arms = {}
            for arm, (rows, costs) in outputs.items():
                if len(rows) != n:
                    raise ValueError(f"Incomplete scored results at {task}/{arm}")
                arms[arm] = dict(metrics={key: matched_predictions(rows, predictions, key)
                                          for key in rows[0]["metrics"]}, cost=dict(costs),
                                 rows_sha256=minimal.sha(a.out / f"{task}__{arm}.jsonl"))
            results[task] = dict(source="current_matched", joint_candidate_time=task != "mmlu_pro",
                                 primary=metric, arms=arms)
        minimal.dump(path, results)
        print("LOOPCD", a.name, task, results[task]["source"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "model", "t1", "reference", "paper", "transfer", "stage2", "followup", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--name", required=True)
    run(p.parse_args())
