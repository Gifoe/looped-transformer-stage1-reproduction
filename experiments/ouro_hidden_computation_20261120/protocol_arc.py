"""Matched six-way Ouro ARC scoring and frozen candidate/order controls."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ouro_arc_pareto_20261119"))
from run_arc import REVISION, dump, load_runtime, read_data, states, variants

DEMOS = (
    HERE.parent / "ouro_latent_state_mechanism_v1" /
    "research_plans/latent_dynamics_continuation_20261113/E7_natural_reencoding/"
    "source_and_protocol/REENCODING_DEMONSTRATIONS.json"
)
TEMPLATE = {
    # code -> demonstration format, candidates in query, output format.
    "A": ("none", False, "text"),
    "B": ("open_text", False, "text"),
    "C": ("none", True, "text"),
    "D": ("shown_text", True, "text"),
    "E": ("none", True, "letter"),
    "F": ("shown_letter", True, "letter"),
    "D_letter_demo": ("shown_letter", True, "text"),
    "F_text_demo": ("shown_text", True, "letter"),
    "C_mask_options": ("none", True, "text"),
    "D_mask_options": ("shown_text", True, "text"),
}
CONTROLS = set(TEMPLATE) - set("ABCDEF")
NATIVE = {f"native_k{k}" for k in range(1, 5)}
LOOPCD = {f"loopcd_{m}_k{k}" for m in ("fixed", "adaptive", "hidden")
          for k in range(2, 5)}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def key(seed, identifier):
    return digest(f"20261120:{seed}:{identifier}".encode())


def canonical(value):
    return re.sub(r"\W+", " ", value.lower()).strip()


def question(stem, choices=None):
    start = f"Question: {stem}"
    options = "".join(f"\n{chr(65+j)}. {text}" for j, text in enumerate(choices or []))
    return start + options + "\nAnswer:"


def prefixes(demos):
    def display(d, show, letter):
        choice_index = ord(d["answer_text"].strip()) - 65
        assert 0 <= choice_index < len(d["choices"])
        target = chr(65 + choice_index) if letter else d["choices"][choice_index]
        return question(d["stem"], d["choices"] if show else None) + " " + target

    assert len(demos) == len({canonical(d["stem"]) for d in demos}) == 25
    return {
        "none": "",
        "open_text": "\n\n".join(display(d, False, False) for d in demos) + "\n\n",
        "shown_text": "\n\n".join(display(d, True, False) for d in demos) + "\n\n",
        "shown_letter": "\n\n".join(display(d, True, True) for d in demos) + "\n\n",
    }


def populations(data, demos):
    train, valid = [read_data(data, split) for split in ("train", "validation")]
    valid_ids = {r["id"] for r in valid}
    demo_stems = {canonical(d["stem"]) for d in demos}
    valid_stems = {canonical(r["question"]) for r in valid}
    assert valid_ids.isdisjoint(r["id"] for r in train)
    assert demo_stems.isdisjoint(valid_stems)
    eligible = [r for r in train if canonical(r["question"]) not in demo_stems | valid_stems]
    assert len(eligible) >= 64
    dev = sorted(eligible, key=lambda r: key("development", r["id"]))[:64]
    return {"train": dev, "validation": valid}


def tokenize(tok, text):
    return tok.encode(text, add_special_tokens=False)


def build_jobs(tokenizer, pre, protocol, rows, rotated):
    demotype, display, output = TEMPLATE[protocol]
    jobs, facts = [], []
    for index, row in enumerate(rows):
        c = row["choices"]["text"]
        labels = row["choices"]["label"]
        gold = labels.index(row["answerKey"])
        n = len(c)
        shift = 0
        if rotated:
            shift = 1 + (int(key("rotation-shift", row["id"]), 16) % (n - 1))
        order = [(j + shift) % n for j in range(n)]
        prompt = pre[demotype] + question(row["question"], [c[j] for j in order] if display else None)
        prefix = tokenize(tokenizer, prompt)
        masked = None
        if protocol.endswith("_mask_options"):
            before = tokenize(tokenizer, pre[demotype] + "Question: " + row["question"])
            after = tokenize(tokenizer, "\nAnswer:")
            assert prefix[:len(before)] == before and prefix[-len(after):] == after, row["id"]
            assert len(before) < len(prefix) - len(after)
            masked = (len(before), len(prefix) - len(after))
        lengths = [len(s) for s in c]
        content_tokens = [len(tokenize(tokenizer, " " + s)) for s in c]
        continuation_lens = [None] * n
        assert all(lengths) and all(content_tokens)
        distinct_inputs = defaultdict(list)
        for displayed, original in enumerate(order):
            target_text = c[original] if output == "text" else chr(65+displayed)
            complete = tokenize(tokenizer, prompt + " " + target_text)
            assert complete[:len(prefix)] == prefix, row["id"]
            targets = complete[len(prefix):]
            assert targets and len(complete) <= 32768
            continuation_lens[original] = len(targets)
            if output == "letter":
                assert len(targets) == 1
            inputs = complete[:-1]
            if masked:
                lo, hi = masked
                inputs[lo:hi] = [0] * (hi - lo)
            distinct_inputs[tuple(inputs)].append((index, original, tuple(targets)))
        for ids, requests in distinct_inputs.items():
            jobs.append((ids, masked, requests))
        facts.append({
            "id": row["id"], "gold": gold, "labels": labels, "order": order,
            "context_tokens": len(prefix), "characters": lengths, "content_tokens": content_tokens,
            "continuation_tokens": continuation_lens,
            "prompt_sha256": digest(prompt.encode()),
            "context_ids_sha256": digest(json.dumps(prefix).encode()),
            "masked_positions": masked,
            "distinct_forward_requests": len(distinct_inputs), "scores": {},
        })
    jobs.sort(key=lambda job: len(job[0]))
    return jobs, facts


def batches(jobs, batch, max_tokens):
    start = 0
    for end, job in enumerate(jobs, 1):
        if end > start + 1 and ((end-start) > batch or (end-start) * len(job[0]) > max_tokens):
            yield jobs[start:end-1]
            start = end - 1
    if start < len(jobs):
        yield jobs[start:]


def assemble(items):
    width = max(len(ids) for ids, *_ in items)
    mask, inp, owner, irow, positions, targets = [], [], [], [], [], []
    members = []
    for i, (ids, masked, requests) in enumerate(items):
        inp.append(ids + (0,) * (width - len(ids)))
        enabled = [1] * width
        if masked is not None:
            lo, hi = masked
            enabled[lo:hi] = [0] * (hi - lo)
        mask.append(enabled)
        for q, a, target in requests:
            members.append((q, a))
            owner.extend([len(members)-1] * len(target))
            irow.extend([i] * len(target))
            positions.extend(range(len(ids) - len(target) + 1, len(ids) + 1))
            targets.extend(target)
    positions = [p - 1 for p in positions]
    tensors = [torch.tensor(x, device="cuda") for x in (inp, irow, positions, targets)]
    attention = torch.tensor(mask, device="cuda") if any(x[1] is not None for x in items) else None
    return (*tensors, attention, np.asarray(owner), members)


def trajectory(model, inp, depth, raw, attention):
    if attention is None:
        return states(model, inp, depth, capture_raw=raw)[:2]
    model.model.total_ut_steps = depth
    captured = []
    handle = model.model.norm.register_forward_pre_hook(
        lambda _, args: captured.append(args[0])) if raw else None
    try:
        _, hidden, _ = model.model(input_ids=inp, use_cache=False, attention_mask=attention)
    finally:
        if handle is not None:
            handle.remove()
    return hidden, captured


def score_raw(model, items, depth, names):
    inp, group, positions, targets, attention, owners, members = assemble(items)
    raw_needed = any(n.startswith("loopcd_hidden") for n in names)
    h, raw = trajectory(model, inp, depth, raw_needed, attention)
    result = {}
    features = [v[group, positions] for v in h]
    raw_features = [v[group, positions] for v in raw]
    for method, z in variants(model, features, raw_features, names):
        selected = z.log_softmax(dim=-1).gather(1, targets[:, None]).squeeze(1)
        result[method] = np.bincount(owners, weights=selected.cpu().numpy(), minlength=len(members))
    return members, result


def parity(model, items):
    items = items[:2]
    inp, group, positions, _, attention, _, _ = assemble(items)
    assert attention is None
    hidden, _ = trajectory(model, inp, 4, False, attention)
    refs, errors = [], {}
    indices = positions.unique(sorted=True)
    for k, h in enumerate(hidden, 1):
        model.model.total_ut_steps = k
        direct = model(input_ids=inp, use_cache=False, logits_to_keep=indices, exit_at_step=k-1)
        z0 = direct.logits[group, torch.searchsorted(indices, positions)].float()
        z1 = model.lm_head(h[group, positions]).float()
        # Distinct GEMM shapes can be numerically different in BF16.
        errors[k] = {"max_abs_logit": (z0-z1).abs().max().item(),
                     "top1_equal": (z0.argmax(-1) == z1.argmax(-1)).sum().item(),
                     "rows": len(positions)}
        refs.append(z0.argmax(-1))
    return errors


def grid(model, tokenizer, rows, pre, phase, split, condition, rotated, out, batch, max_tokens):
    tag = f"{phase}.{split}.{condition}" + (".rotated" if rotated else "")
    path = out / f"{tag}.jsonl"
    if path.exists() and (out / f"{tag}.execution.json").exists():
        print("verified output exists", tag, flush=True)
        return
    jobs, facts = build_jobs(tokenizer, pre, condition, rows, rotated)
    if phase == "native" and split == "train" and condition in "ABCDEF" and not rotated:
        dump(out / f"{tag}.parity.json", parity(model, jobs))
    methods = NATIVE if phase == "native" else LOOPCD
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    total_batches, padded_tokens, distinct_tokens = 0, 0, 0
    for items in batches(jobs, batch, max_tokens):
        total_batches += 1
        padded_tokens += len(items) * max(len(x[0]) for x in items)
        distinct_tokens += sum(len(x[0]) for x in items)
        members, scored = score_raw(model, items, 4, methods)
        for method, output in scored.items():
            assert np.isfinite(output).all()
            for (row, candidate), v in zip(members, output):
                dst = facts[row]
                scores = dst["scores"].setdefault(method, [None] * len(dst["labels"]))
                assert scores[candidate] is None
                scores[candidate] = float(v)
        if total_batches % 64 == 0:
            print(tag, total_batches, "batches", round(time.perf_counter() - start, 1), "s", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert all(set(r["scores"]) == methods and
               all(all(v is not None for v in a) for a in r["scores"].values()) for r in facts)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for record in facts:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)
    dump(out / f"{tag}.execution.json", {
        "items": len(rows), "candidate_scores": sum(len(r["labels"]) for r in facts),
        "unique_forwards": len(jobs), "batches": total_batches, "loop_stacks": 4,
        "distinct_input_tokens": distinct_tokens, "padded_input_tokens": padded_tokens,
        "wall_seconds": elapsed, "scores_sha256": digest(path.read_bytes()),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "batch_limit": batch, "padded_token_batch_limit": max_tokens,
        "timing_scope": "joint trajectory and all predeclared readouts for this phase",
    })
    print(tag, len(rows), len(jobs), round(elapsed, 1), "s", flush=True)


def plan(data, demos_path, out):
    pre = prefixes(json.loads(demos_path.read_bytes()))
    populations_by_split = populations(data, json.loads(demos_path.read_bytes()))
    selection = {}
    for split, rows in populations_by_split.items():
        control_n = 32 if split == "train" else 64
        controls = sorted(rows, key=lambda r: key("controls", r["id"]))[:control_n]
        selection[split] = {"main": [x["id"] for x in rows],
                            "controls": [x["id"] for x in controls]}
    freeze = {
        "dataset_revision": REVISION,
        "dataset_sha256": {s: digest((data/f"{s}.parquet").read_bytes())
                           for s in populations_by_split},
        "demo_sha256": digest(demos_path.read_bytes()),
        "prefix_sha256": {n: digest(p.encode()) for n, p in pre.items()},
        "implementation_sha256": digest(Path(__file__).read_bytes()),
        "selection": selection, "condition_mapping": TEMPLATE,
        "control": "Fixed final position and length with masked option positions for C and D. "
                   "Single nonidentity hash-assigned cyclic order in C-F. "
                   "Crossed letter/full demonstration answers in 25-shot shown conditions.",
        "metric": "Target sums; divide by original target characters excluding continuation space "
                  "for normalized scores. Letter targets have denominator one. Token normalization secondary.",
        "tie": "First displayed option, then map to original content coordinates.",
        "scope": "Fixed development64 from ARC Train after excluding historical demonstrations and "
                 "normalized validation stem matches. All official validation299 once frozen. "
                 "No Test in the new score path.",
    }
    path = out / "PROTOCOL_FREEZE.json"
    if path.exists():
        assert json.loads(path.read_bytes()) == json.loads(json.dumps(freeze)), path
    else:
        dump(path, freeze)
    return pre, populations_by_split, selection


def load_archived_a(source, tokenizer, pre, rows, out, phase):
    p = source / "validation_predictions.jsonl"
    ref = [json.loads(line) for line in p.read_text().splitlines()]
    jobs, records = build_jobs(tokenizer, pre, "A", rows, False)
    assert [r["id"] for r in records] == [r["id"] for r in ref]
    for r, saved in zip(records, ref):
        assert saved["gold"] == r["gold"] and saved["labels"] == r["labels"]
        assert saved["characters"] == r["characters"] and saved["answer_tokens"] == r["continuation_tokens"]
        assert saved["scored_input_tokens"] == [r["context_tokens"] + n - 1 for n in r["continuation_tokens"]]
        r["scores"] = {m: a for m, a in saved["scores"].items() if m in (NATIVE if phase == "native" else LOOPCD)}
    dest = out / f"{phase}.validation.A.jsonl"
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    dump(out / f"{phase}.validation.A.execution.json", {
        "archived_source_sha256": digest(p.read_bytes()), "scores_sha256": digest(dest.read_bytes()),
        "source": "5557699:experiments/ouro_arc_pareto_20261119/results/validation_predictions.jsonl",
        "source_execution": json.loads((source/"validation_execution.json").read_bytes()),
        "items": len(records), "unique_forward_requests_under_new_dedup": len(jobs),
        "timing_scope": "Archived timing includes all original variants. Native/LoopCD subset reused.",
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--demos", type=Path, default=DEMOS)
    p.add_argument("--phase", choices=("plan", "native", "loopcd"), required=True)
    p.add_argument("--split", choices=("both", "train", "validation"), default="both")
    p.add_argument("--conditions", nargs="+", choices=sorted(TEMPLATE), default=list(TEMPLATE))
    p.add_argument("--archive", type=Path)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--run-rotation", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pre, by_split, selection = plan(a.data, a.demos, a.out)
    if a.phase == "plan":
        return
    model, tokenizer, runtime = load_runtime(a.model)
    dump(a.out / f"{a.phase}.runtime.json", runtime)
    for split, rows in by_split.items():
        if a.split not in ("both", split):
            continue
        controls = [r for r in rows if r["id"] in set(selection[split]["controls"])]
        for condition in a.conditions:
            if condition == "A" and split == "validation" and a.archive:
                load_archived_a(a.archive, tokenizer, pre, rows, a.out, a.phase)
                continue
            current = controls if condition in CONTROLS else rows
            grid(model, tokenizer, current, pre, a.phase, split, condition, False, a.out, a.batch, a.max_tokens)
            if a.run_rotation and condition in "CDEF":
                grid(model, tokenizer, controls, pre, a.phase, split, condition, True, a.out, a.batch, a.max_tokens)


if __name__ == "__main__":
    main()
