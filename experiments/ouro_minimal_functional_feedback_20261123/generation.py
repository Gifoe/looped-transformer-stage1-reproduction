"""Full ARC Validation and OpenBookQA Test option-letter greedy generation."""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

import minimal


DEPTHS = (1, 2, 3, 4)
SPLIT_SIZES = {"arc_validation": 299, "openbookqa_test": 500}
LEGACY_ARMS = ("native", "original_feedback", "original_joint")
OPTION = re.compile(r"^\s*(?:\(|\[)?([A-E])(?:\b|[.)\]])")
SINGLE_OPTION = re.compile(r"^\s*(?:[A-E][.)]?|\([A-E]\)|\[[A-E]\])\s*$")


@dataclass
class Arm:
    kind: str
    net: object = None
    dose: float = 0.0
    readout: object = None
    readout_dose: float = 0.0


def classify(text, options):
    match = OPTION.match(text)
    if match:
        if match[1] in options:
            return match[1], None, bool(SINGLE_OPTION.fullmatch(text))
        return None, "unavailable_option", False
    value = text.strip()
    if not value:
        return None, "empty_or_special_only", False
    if re.match(r"^(?:\(|\[)?[a-e](?:\b|[.)\]])", value):
        return None, "lowercase_option", False
    if re.match(r"^(?:\(|\[)?[F-Z](?:\b|[.)\]])", value):
        return None, "out_of_range_option", False
    if re.match(r"^(?:\(|\[)?[0-9]", value):
        return None, "numeric_instead_of_option", False
    return None, "no_leading_option", False


def pack_generation(prompts, generated, active, pad_id, device):
    lengths = [len(prompts[i]) + len(generated[i]) for i in active]
    edge = max(lengths)
    sequences, group, pos, last = [], [], [], []
    for j, (i, length) in enumerate(zip(active, lengths, strict=True)):
        sequences.append(prompts[i] + generated[i] + [pad_id] * (edge - length))
        positions = range(len(prompts[i]) - 1, length)
        pos.extend(positions)
        group.extend([j] * len(positions))
        last.append(len(pos) - 1)
    packed = tuple(torch.tensor(x, dtype=torch.long, device=device)
                   for x in (sequences, group, pos, [0] * len(group)))
    return packed, torch.tensor(last, dtype=torch.long, device=device), edge


def model_next(model, feedback, arm, depth, packed, last):
    if arm.kind == "native":
        inp, group, pos, _ = packed
        core = model.model
        core.total_ut_steps = depth
        _, states, _ = core(input_ids=inp, use_cache=False)
        final = states[-1][group[last], pos[last]]
    elif arm.kind == "minimal":
        states, _ = minimal.forward(model, packed, depth, arm.net, arm.dose)
        final = states[last]
    else:
        states, edited, _ = feedback.forward(model, packed, depth, arm.net, arm.dose,
                                             arm.readout, arm.readout_dose)
        final = (edited if edited is not None else states)[last]
    return model.lm_head(final).argmax(dim=-1).tolist()


def generate_batch(tokenizer, prompts, next_tokens, max_new, device):
    generated = [[] for _ in prompts]
    stops = ["token_limit"] * len(prompts)
    active = list(range(len(prompts)))
    padded_tokens = 0
    calls = 0
    for _ in range(max_new):
        packed, last, width = pack_generation(prompts, generated, active, 0, device)
        predicted = next_tokens(packed, last)
        padded_tokens += len(active) * width
        calls += 1
        retained = []
        for i, token in zip(active, predicted, strict=True):
            generated[i].append(int(token))
            if token == tokenizer.eos_token_id:
                stops[i] = "eos"
            elif "\n" in tokenizer.decode(generated[i], skip_special_tokens=True).strip("\n "):
                stops[i] = "internal_newline"
            else:
                retained.append(i)
        active = retained
        if not active:
            break
    return generated, stops, calls, padded_tokens


def write_jsonl(path, rows):
    temp = path.with_suffix(".partial")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def bootstrap_pair(rescue, damage, n):
    rng = np.random.default_rng(261123439)
    samples = rng.multinomial(n, [rescue / n, damage / n, (n - rescue - damage) / n], 10_000)
    values = 100 * (samples[:, 0] - samples[:, 1]) / n
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def summarize(rows, native):
    peers = {row["id"]: row for row in native}
    if peers.keys() != {row["id"] for row in rows} or len(rows) != len(peers):
        raise RuntimeError("Paired comparison has mismatched IDs")
    rescue = sum(row["correct"] and not peers[row["id"]]["correct"] for row in rows)
    damage = sum(not row["correct"] and peers[row["id"]]["correct"] for row in rows)
    n = len(rows)
    valid = sum(x["valid"] for x in rows)
    correct = sum(x["correct"] for x in rows)
    counts = [len(x["tokens"]) for x in rows]
    return dict(n=n, valid=valid, valid_rate=valid/n, correct=correct, accuracy=correct/n,
                strict_format=sum(x["strict_format"] for x in rows),
                invalid_categories=dict(Counter(x["invalid_category"] for x in rows if not x["valid"])),
                stop_reasons=dict(Counter(x["stop"] for x in rows)), total_tokens=sum(counts),
                mean_tokens=statistics.mean(counts), median_tokens=statistics.median(counts),
                rescue_vs_native=rescue, damage_vs_native=damage,
                accuracy_delta_pp_vs_native=100*(rescue-damage)/n,
                paired_95ci_pp_vs_native=bootstrap_pair(rescue, damage, n))


def read_datasets(old, tokenizer, corpus, external):
    import pyarrow.parquet as pq

    arc_file = corpus / "validation.parquet"
    sources = {"arc_validation": arc_file, "openbookqa_test": external}
    raw = {"arc_validation": old.read_data(corpus, "validation"),
           "openbookqa_test": pq.read_table(external).to_pylist()}
    meta = json.loads(external.with_suffix(".json").read_text(encoding="utf-8"))
    if (meta.get("dataset"), meta.get("split"), meta.get("n"), meta.get("canonical_sha256")) != (
        "allenai/openbookqa", "main/test", 500, minimal.sha(external)
    ):
        raise RuntimeError("OpenBookQA input does not match the archived canonical test receipt")
    prepared = {}
    for split, questions in raw.items():
        if len(questions) != SPLIT_SIZES[split] or len({x["id"] for x in questions}) != len(questions):
            raise RuntimeError(f"{split} is missing rows or has duplicated IDs")
        prepared[split] = []
        for row in questions:
            prompt, options, gold = minimal.label_prompt(row)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not ids or not 2 <= len(options) <= 5 or gold not in options:
                raise RuntimeError(f"Malformed prompt or answer for {row['id']}")
            prepared[split].append(dict(id=row["id"], prompt=prompt, prompt_ids=ids, options=options,
                                        gold=gold, official_labels=row["choices"]["label"]))
        prepared[split].sort(key=lambda row: (len(row["prompt_ids"]), row["id"]))
    metadata = {split: {"n": SPLIT_SIZES[split], "sha256": minimal.sha(path)}
                for split, path in sources.items()}
    metadata["openbookqa_test"]["archived_data_receipt"] = meta
    return prepared, metadata


def run_slice(split, name, depth, dataset, batch_size, max_new, model, tokenizer, feedback, arm):
    batches, predictions = [], []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident_allocated = torch.cuda.memory_allocated()
    resident_reserved = torch.cuda.memory_reserved()
    started = time.perf_counter()
    for lo in range(0, len(dataset), batch_size):
        batch = dataset[lo:lo+batch_size]
        prompts = [row["prompt_ids"] for row in batch]
        tick = time.perf_counter()
        tokens, stops, calls, padded = generate_batch(
            tokenizer, prompts, lambda packed, last: model_next(model, feedback, arm, depth, packed, last),
            max_new, "cuda")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        batches.append(dict(n=len(batch), wall_ms=1000*elapsed, calls=calls,
                            sequence_tokens_with_padding=padded, generated_tokens=sum(map(len, tokens))))
        for row, seq, stop in zip(batch, tokens, stops, strict=True):
            decoded = tokenizer.decode(seq, skip_special_tokens=True)
            choice, category, strict = classify(decoded, row["options"])
            predictions.append(dict(split=split, method=name, depth=depth, id=row["id"],
                                    prompt=row["prompt"], prompt_token_count=len(row["prompt_ids"]),
                                    options=row["options"], official_labels=row["official_labels"], gold=row["gold"],
                                    text=decoded, tokens=seq, stop=stop, option=choice, valid=choice is not None,
                                    strict_format=strict, invalid_category=category, correct=choice == row["gold"],
                                    predictor_calls=len(seq), batch_index=len(batches)-1))
    elapsed = time.perf_counter() - started
    calls = sum(b["calls"] for b in batches)
    latencies = [b["wall_ms"] for b in batches]
    cost = dict(questions=len(dataset), max_batch_size=batch_size, batches=batches, wall_s=elapsed,
                mean_question_ms=1000*elapsed/len(dataset), median_batch_ms=statistics.median(latencies),
                p95_batch_ms=float(np.quantile(latencies, .95)), model_forward_calls=calls, lm_head_calls=calls,
                shared_loop_passes=depth*calls, sequence_tokens_with_padding=sum(b["sequence_tokens_with_padding"] for b in batches),
                generated_tokens=sum(b["generated_tokens"] for b in batches),
                resident_allocated_bytes=resident_allocated, resident_reserved_bytes=resident_reserved,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return predictions, cost


def named_values(values, kind):
    output = {}
    for entry in values:
        name, sep, value = entry.partition("=")
        if not sep or not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", name) or not value or name in output:
            raise ValueError(f"Expected unique {kind} NAME=VALUE with lowercase alphanumeric name")
        output[name] = value
    return output


@torch.inference_mode()
def run(a):
    minimal_paths = {n: Path(p).resolve() for n, p in named_values(a.minimal, "checkpoint").items()}
    doses = {n: float(d) for n, d in named_values(a.dose, "dose").items()}
    if (set(minimal_paths) & set(LEGACY_ARMS) or not set(doses) <= set(minimal_paths)
            or any(not math.isfinite(d) or d <= 0 for d in doses.values())):
        raise ValueError("New arm names may not reuse baseline names, and doses must be positive and finite")
    legacy = a.legacy.resolve()
    sys.path.insert(0, str(legacy))
    import feedback

    legacy_args = SimpleNamespace(prior=a.prior, archive=a.archive, out=a.legacy_freeze)
    old = feedback.configuration(legacy_args)
    model, tokenizer, runtime = old.load_runtime(a.model)
    model.eval().requires_grad_(False)
    legacy_net, legacy_dose, readout, readout_dose = feedback.frozen(legacy_args)
    if readout is None:
        raise RuntimeError("The original joint checkpoint is missing from the legacy freeze")
    arms = {"native": Arm("native"), "original_feedback": Arm("legacy", legacy_net, legacy_dose),
            "original_joint": Arm("legacy", legacy_net, legacy_dose, readout, readout_dose)}
    for name, path in minimal_paths.items():
        arms[name] = Arm("minimal", minimal.load_net(path), doses.get(name, 1.0))
    datasets, sources = read_datasets(old, tokenizer, a.corpus, a.external)
    freeze_file = a.legacy_freeze / "FREEZE.json"
    new_arms = {n: {"checkpoint": str(p), "sha256": minimal.sha(p), "dose": arms[n].dose,
                    "parameters": sum(w.numel() for w in arms[n].net.parameters())} for n,p in minimal_paths.items()}
    loss_scope = "Canonical option-letter prompt. Unrestricted vocabulary greedy decoding, no cache, compact live batch."
    lock = dict(datasets=sources, original_freeze_sha256=minimal.sha(freeze_file),
                original_freeze=json.loads(freeze_file.read_text(encoding="utf-8")),
                checkpoints=new_arms, code_sha256=minimal.sha(Path(__file__)),
                minimal_source_sha256=minimal.sha(Path(minimal.__file__)), model=str(a.model.resolve()),
                runtime=runtime, batch_size=a.batch_size, max_new=a.max_new,
                native_model_parameters=sum(w.numel() for w in model.parameters()),
                legacy_module_parameters={"original_feedback": sum(w.numel() for w in legacy_net.parameters()),
                                          "original_joint": sum(w.numel() for w in legacy_net.parameters())
                                          + sum(w.numel() for w in readout.parameters())},
                scope=loss_scope, gpu=torch.cuda.get_device_name())
    a.out.mkdir(parents=True, exist_ok=True)
    fragments = a.out / "fragments"
    fragments.mkdir(exist_ok=True)
    lock_path = a.out / "GENERATION_LOCK.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != json.loads(json.dumps(lock)):
            raise RuntimeError("Generation lock differs from existing experiment")
    else:
        minimal.dump(lock_path, lock)
    summary, costs = {}, {}
    all_rows = []
    for split, dataset in datasets.items():
        for depth in DEPTHS:
            native = None
            for name, arm in arms.items():
                key = f"{split}__{name}__k{depth}"
                raw_path = fragments / f"{key}.jsonl"
                cost_path = fragments / f"{key}.cost.json"
                if cost_path.exists():
                    rows = minimal.load_jsonl(raw_path)
                    receipt = json.loads(cost_path.read_text(encoding="utf-8"))
                    if (receipt.pop("raw_sha256") != minimal.sha(raw_path)
                            or [r["id"] for r in rows] != [r["id"] for r in dataset]):
                        raise RuntimeError(f"Fragment integrity mismatch: {key}")
                else:
                    # Warm up one representative full batch outside all measured intervals.
                    warm = [row["prompt_ids"] for row in dataset[:a.batch_size]]
                    generate_batch(tokenizer, warm, lambda packed, last: model_next(model, feedback, arm, depth, packed, last),
                                   1, "cuda")
                    rows, receipt = run_slice(split, name, depth, dataset, a.batch_size, a.max_new,
                                              model, tokenizer, feedback, arm)
                    write_jsonl(raw_path, rows)
                    minimal.dump(cost_path, {**receipt, "raw_sha256": minimal.sha(raw_path)})
                if name == "native":
                    native = rows
                summary[key] = summarize(rows, native)
                costs[key] = receipt
                all_rows.extend(rows)
                print(key, summary[key]["correct"], summary[key]["valid"], round(receipt["wall_s"], 2), flush=True)
                minimal.dump(a.out / "GENERATION_SUMMARY.json", {"runs": summary, "scope": loss_scope})
                minimal.dump(a.out / "GENERATION_COST.json", {"runs": costs, "warmup_calls_excluded_per_run": 1})
    write_jsonl(a.out / "GENERATION_RESULTS.jsonl", all_rows)
    print("DONE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "archive", "corpus", "prior", "legacy", "legacy-freeze", "external", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--minimal", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--dose", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new", type=int, default=6)
    a = p.parse_args()
    if a.batch_size < 1 or a.max_new < 1:
        p.error("--batch-size and --max-new must be positive")
    run(a)


if __name__ == "__main__":
    main()
