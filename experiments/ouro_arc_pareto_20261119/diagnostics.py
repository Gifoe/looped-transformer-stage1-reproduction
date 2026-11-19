"""Development-only causal interventions and actual, cached latency probes."""
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from run_arc import SEED, dump, encode, pack, record, scores, states, variants, write_predictions

# Zero-based shared block indices. One predeclared contiguous window.
SITES = tuple(range(8, 12))
TAGS = ("q", "k", "v")
ARMS = ("teacher", "random", "previous_position")


def oracle(model, tokenizer, rows, out, batch_size):
    rows = sorted(rows, key=lambda row: hashlib.sha256(
        f'{SEED}:oracle:{row["id"]}'.encode()).digest())[:128]
    items, result = encode(tokenizer, rows)
    items.sort(key=lambda x: len(x[0]))
    core = model.model
    interface = core.layers[SITES[0]].self_attn.forward.__func__.__globals__["ALL_ATTENTION_FUNCTIONS"]
    native = interface["sdpa"]
    sites = {id(core.layers[i].self_attn): i for i in SITES}
    condition, capture, seen, generator = ["teacher_run"], {}, {}, torch.Generator(device="cuda")
    generator.manual_seed(SEED)

    def attention(module, q, k, v, mask, **kwargs):
        index = sites.get(id(module))
        tensors = [q, k, v]
        if index is not None:
            visit = seen.get(index, 0) + 1
            seen[index] = visit
            if condition[0] == "teacher_run" and visit == 4:
                capture[index] = tuple(t.detach() for t in tensors)
            elif visit == 2 and condition[0] not in ("teacher_run", "noop"):
                field, arm = condition[0].split(":", 1)
                component = TAGS.index(field)
                current, late = tensors[component], capture[index][component]
                assert current.shape == late.shape
                if arm == "teacher":
                    tensors[component] = late
                else:
                    delta = late.float() - current.float()
                    norm = delta.norm(dim=-1, keepdim=True)
                    noise = torch.randn(current.shape, device=current.device, generator=generator)
                    direction = noise
                    if arm == "previous_position":
                        direction = torch.cat((noise[:, :, :1], delta[:, :, :-1]), dim=2)
                        direction = torch.where(direction.norm(dim=-1, keepdim=True) > 1e-8, direction, noise)
                    matched = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8) * norm
                    tensors[component] = (current.float() + matched).to(current.dtype)
        return native(module, *tensors, mask, **kwargs)

    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        for begin in range(0, len(items), batch_size):
            batch = items[begin:begin + batch_size]
            inp, group, position, target = pack(batch)
            batch_map = group.cpu().numpy()

            def likelihood(hidden):
                return scores(model.lm_head(hidden[group, position]).float(), batch_map, target, len(batch))

            interface["sdpa"] = native
            base = states(model, inp, 2)[0][-1]
            output = {"native_k2": likelihood(base)}
            # Keep the model's SDPA mask selection unchanged when installing the hook.
            interface["sdpa"] = attention
            condition[0] = "teacher_run"
            capture.clear()
            seen.clear()
            deep = states(model, inp, 4)[0][-1]
            assert sorted(capture) == list(SITES) and set(seen.values()) == {4}
            output["native_k4"] = likelihood(deep)
            if begin == 0:
                condition[0] = "noop"
                seen.clear()
                untouched = states(model, inp, 2)[0][-1]
                assert torch.equal(base, untouched), (base - untouched).abs().max().item()
            for field in TAGS:
                for arm in ARMS:
                    condition[0] = f"{field}:{arm}"
                    seen.clear()
                    early = states(model, inp, 2)[0][-1]
                    assert set(seen.values()) == {2}
                    output[f"{field}_{arm}"] = likelihood(early)
            record(result, batch, output)
            if begin % (batch_size * 8) == 0:
                print("oracle", begin + len(batch), "/", len(items), flush=True)
    finally:
        interface["sdpa"] = native
    write_predictions(out / "oracle_predictions.jsonl", result)
    dump(out / "oracle_execution.json", {
        "rows": len(rows), "candidates": len(items), "selection": "first 128 validation IDs ordered by seeded SHA256",
        "zero_based_layers": SITES, "early_loop": 2, "teacher_loop": 4, "seed": SEED,
        "base_attention": "SDPA", "no_op_bitwise_equal": True,
        "control_dose": "float32 L2 of late minus current, separately per token and head, before BF16 round",
        "measured_wall_s": time.perf_counter() - start, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    })


def latency(model, tokenizer, rows, out):
    rows = sorted(rows, key=lambda row: hashlib.sha256(
        f'{SEED}:latency:{row["id"]}'.encode()).digest())[:24]
    raw_records, parity = [], {}
    samples = []
    for row in rows:
        ids = tokenizer.encode(f'Question: {row["question"]}\nAnswer:', add_special_tokens=False)
        continuation = tokenizer.encode(" " + row["choices"]["text"][0], add_special_tokens=False)
        prompt = torch.tensor([ids], device="cuda")
        first = torch.tensor([[continuation[0]]], device="cuda")
        samples.append((row["id"], prompt, first))

    def read_logits(inp, depth, name, cache=None):
        if name.startswith("native_"):
            model.model.total_ut_steps = depth
            model.config.total_ut_steps = depth
            output = model(input_ids=inp, use_cache=True, past_key_values=cache,
                           logits_to_keep=1, exit_at_step=depth - 1)
            return output.logits[:, -1, :], output.past_key_values
        needs_raw = not name.startswith(("loopcd_fixed_", "loopcd_adaptive_"))
        h, raw, cache = states(model, inp, depth, capture_raw=needs_raw, use_cache=True, cache=cache)
        result = dict(variants(model, [x[:, -1] for x in h], [x[:, -1] for x in raw], {name}))
        return result[name], cache

    for depth in range(1, 5):
        names = [f"native_k{depth}"]
        if depth > 1:
            names.extend(f"{x}_k{depth}" for x in ("loopcd_fixed", "loopcd_adaptive", "loopcd_hidden"))
        if depth >= 3:
            names.extend(f"{x}_k{depth}" for x in ("secant", "laststep"))
        for name in names:
            for iteration, (row_id, prompt, token) in enumerate(samples[:3] + samples):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                _, cache = read_logits(prompt, depth, name)
                torch.cuda.synchronize()
                prefill_end = time.perf_counter()
                if cache is None or cache.get_seq_length() != prompt.shape[1]:
                    raise RuntimeError("Ouro cache failed prefill sequence-length check")
                next_logits, cache = read_logits(token, depth, name, cache)
                torch.cuda.synchronize()
                decode_end = time.perf_counter()
                assert cache.get_seq_length() == prompt.shape[1] + 1
                if iteration == 3 and name.startswith("native_"):
                    joined = torch.cat((prompt, token), dim=1)
                    full = model(input_ids=joined, use_cache=False, logits_to_keep=1,
                                 exit_at_step=depth - 1).logits[:, -1, :]
                    parity[name] = dict(max_abs_logit=(next_logits.float() - full.float()).abs().max().item(),
                                        top1_equal=bool(next_logits.argmax(-1).eq(full.argmax(-1)).all().item()))
                if iteration >= 3:
                    raw_records.append(dict(
                        id=row_id, method=name, prefix_tokens=prompt.shape[1], loops=depth,
                        prefill_ms=1000 * (prefill_end - start), decode_ms=1000 * (decode_end - prefill_end),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            print("latency", name, flush=True)
    dump(out / "latency.json", dict(
        protocol="Ouro native KV cache; batch 1; new cache per prompt; one forced token after prefill; wall clock with CUDA synchronization",
        warmup_per_method=3, samples_per_method=len(samples), parity=parity, observations=raw_records))
