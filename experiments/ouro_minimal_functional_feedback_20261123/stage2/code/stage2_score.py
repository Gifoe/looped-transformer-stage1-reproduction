"""Score locked few-shot Harness likelihood requests with native Ouro and fixed feedback."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import minimal


APPLE_BENCHMARKS = {
    "arc_challenge": 25, "arc_easy": 8, "sciq": 0, "mmlu": 5,
    "hellaswag": 10, "winogrande": 5, "piqa": 0,
}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1 << 20), b""):
            sha.update(part)
    return sha.hexdigest()


@dataclass
class Request:
    row: int
    choice: int
    inp: tuple
    targets: tuple
    start: int
    dropped: int
    retokenized_boundary: bool


@dataclass
class Work:
    inp: tuple
    start: int
    requests: list = field(default_factory=list)


class HarnessEncoder:
    """Use HFLM causal scoring: split the jointly encoded text at context length."""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_tokens = lru_cache(maxsize=2048)(self.encode)

    def encode(self, text):
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def request(self, row, choice, context, continuation):
        if context:
            stripped = context.rstrip()
            continuation = context[len(stripped):] + continuation
            ctx_tokens = self.context_tokens(stripped)
            joint = self.encode(stripped + continuation)
            targets = joint[len(ctx_tokens):]
            retokenized = joint[:len(ctx_tokens)] != ctx_tokens
        else:
            prefix = self.tokenizer.bos_token_id
            if prefix is None:
                prefix = self.tokenizer.eos_token_id
            targets = self.encode(continuation)
            if targets and targets[0] == prefix:
                targets = targets[1:]
            ctx_tokens, retokenized = (prefix,), False
        if not ctx_tokens or not targets or len(targets) > self.max_length:
            raise ValueError(f"Invalid Harness continuation or context for row {row} choice {choice}")
        total = len(ctx_tokens) + len(targets)
        dropped = max(0, total - (self.max_length + 1))
        inp = (ctx_tokens + targets)[dropped:-1]
        return Request(row, choice, inp, targets, len(inp)-len(targets), dropped, retokenized)


def encode_rows(rows, encoder):
    requests = []
    for number, row in enumerate(rows):
        ir = sorted(row["requests"], key=lambda x:x["idx"])
        choices = row["choices"]
        if (len(ir) != len(choices) or not choices or not all(isinstance(c, str) and c for c in choices)
                or [x["idx"] for x in ir] != list(range(len(choices)))
                or not isinstance(row["target"], int) or not 0 <= row["target"] < len(choices)):
            raise ValueError(f"Unexpected Harness choices or target for {row['id']}")
        for request in ir:
            if request["type"] != "loglikelihood" or len(request["args"]) != 2:
                raise ValueError(f"Unexpected Harness request type for {row['id']}")
            requests.append(encoder.request(number, request["idx"], *request["args"]))
    return requests


def compact_prefixes(requests):
    """Reuse exact starts and common scored causal prefixes without KV approximation."""
    families = defaultdict(list)
    for req in requests:
        families[req.inp[:req.start+1]].append(req)
    result = []
    for group in families.values():
        distinct = {}
        for req in sorted(group, key=lambda r:(-len(r.inp), r.row, r.choice)):
            if req.inp in distinct:
                distinct[req.inp].requests.append(req)
                continue
            covering = next((item for inp,item in distinct.items() if inp[:len(req.inp)] == req.inp), None)
            if covering is not None:
                covering.requests.append(req)
            else:
                distinct[req.inp] = Work(req.inp, req.start, [req])
        result.extend(distinct.values())
    return sorted(result, key=lambda w:(len(w.inp), len(w.inp)-w.start, w.requests[0].row, w.requests[0].choice))


def batch_work(work, max_batch, token_cap, attention_cap):
    batch = []
    for item in work:
        n, longest = len(batch) + 1, len(item.inp)
        if batch and (n > max_batch or n*longest > token_cap or n*longest*longest > attention_cap):
            yield batch
            batch = []
        batch.append(item)
    if batch:
        yield batch


def pack(work, device):
    width = max(len(w.inp) for w in work)
    inp, group, pos, requests = [], [], [], []
    for number, item in enumerate(work):
        inp.append(item.inp + (0,) * (width-len(item.inp)))
        offset = len(group)
        group.extend([number] * (len(item.inp)-item.start))
        pos.extend(range(item.start, len(item.inp)))
        requests.extend((r, offset) for r in item.requests)
    packed = tuple(torch.tensor(x, device=device, dtype=torch.long) for x in (inp, group, pos, [0]*len(group)))
    queries, targets, refs, counts = [], [], [], []
    for rid, (req, offset) in enumerate(requests):
        queries.extend(range(offset, offset+len(req.targets)))
        targets.extend(req.targets)
        refs.extend([rid] * len(req.targets))
        counts.append(len(req.targets))
    gather = tuple(torch.tensor(x, device=device, dtype=torch.long) for x in (queries, targets))
    return packed, [r for r,_ in requests], gather, np.asarray(refs), np.asarray(counts)


def score_head(model, states, depths, max_rows, gather, refs, counts, cost, apple):
    """Compute requested log-likelihoods without retaining dense vocabulary buffers."""
    query, target = gather
    native_depths = sorted({1,*depths})
    names = ([f"native_k{k}" for k in native_depths] +
             [f"loopcd_{variant}_k{k}" for k in depths if k > 1 for variant in ("fixed", "adaptive")]
             if apple else ["minimal"])
    probabilities = [torch.empty(len(query), device=query.device) for _ in names]
    agreements = [torch.empty(len(query), device=query.device, dtype=torch.bool) for _ in names]
    index = {name:i for i,name in enumerate(names)}
    for lo in range(0, states[0].shape[0], max_rows):
        hi = min(lo+max_rows, states[0].shape[0])
        selected = (query >= lo) & (query < hi)
        rows, tok = query[selected]-lo, target[selected]

        def retain(name, logits):
            i = index[name]
            probabilities[i][selected] = logits.log_softmax(dim=-1)[rows, tok]
            agreements[i][selected] = logits.argmax(-1)[rows] == tok

        if not apple:
            retain("minimal", model.lm_head(states[0][lo:hi]).float())
            cost["head_calls"] += 1
            continue
        first = model.lm_head(states[0][lo:hi]).float()
        cost["head_calls"] += 1
        for k in native_depths:
            logits = first if k == 1 else model.lm_head(states[k-1][lo:hi]).float()
            cost["head_calls"] += int(k != 1)
            retain(f"native_k{k}", logits)
            if k == 1:
                continue
            delta = logits - first
            retain(f"loopcd_fixed_k{k}", logits + .5 * delta)
            top = (logits.topk(2, dim=-1).values - logits.logsumexp(dim=-1, keepdim=True)).exp()
            retain(f"loopcd_adaptive_k{k}", logits + (1 - (top[:, :1]-top[:, 1:])) * delta)
    result = {}
    for name, lp, agreement in zip(names, probabilities, agreements, strict=True):
        sums = np.bincount(refs, weights=lp.cpu().numpy(), minlength=len(counts))
        hits = np.bincount(refs, weights=agreement.cpu().numpy(), minlength=len(counts))
        if not np.isfinite(sums).all():
            raise RuntimeError(f"Nonfinite likelihood at {name}")
        result[name] = (sums.tolist(), (hits==counts).tolist())
    return result


def native_trunk(model, packed, depths, shared):
    inp, group, pos, _ = packed
    core = model.model
    core.total_ut_steps = max(depths)
    required = {k for d in depths if d > 1 for k in (d-1, d-2) if k > 0} if shared else set()
    raw = {}
    seen = 0

    def capture(_, args):
        nonlocal seen
        seen += 1
        if seen in required:
            raw[seen] = args[0]

    handle = core.norm.register_forward_pre_hook(capture) if required else None
    try:
        _, states, _ = core(input_ids=inp, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()
    if len(states) != max(depths) or (required and (set(raw) != required or seen != max(depths))):
        raise RuntimeError("Unexpected Ouro native depth or norm invocation pattern")
    return [h[group,pos] for h in states], raw


def branch_hidden(model, packed, depth, net, dose, raw):
    """Write the frozen update to the pre-norm anchor, then run one full native loop."""
    _, group, pos, _ = packed
    tensor = raw[depth-1]
    anchor = tensor[group,pos].float()
    previous = raw[depth-2][group,pos].float() if net.kind == "history" and depth >= 3 else None
    value = anchor + dose * net.delta(anchor, previous)
    core = model.model
    edited = tensor.index_put((group,pos), value.to(tensor.dtype))
    normalized = core.norm(edited)
    core.total_ut_steps = 1
    # In the pinned Ouro core current_ut only affects KV slot indexing. Uncached
    # input_ids and inputs_embeds entry points use the same positions and mask.
    _, states, _ = core(inputs_embeds=normalized, use_cache=False)
    if len(states) != 1:
        raise RuntimeError("A shared-prefix branch must run exactly one Ouro loop")
    return states[0][group,pos], anchor


def loop_work(depths, shared):
    return max(depths) + (len(depths) if shared else sum(depths))


def score_batch(model, net, work, depths, dose, head_chunk, cost, device="cuda", shared=False):
    packed, refs, gather, token_refs, counts = pack(work, device)
    inp, group, _, _ = packed
    phase_started = time.perf_counter()
    states, raw = native_trunk(model, packed, depths, shared)
    cost.update(native_calls=1, minimal_calls=len(depths), loop_passes=loop_work(depths, shared),
                shared_prefix_branches=sum(k > 1 for k in depths) if shared else 0,
                max_width=inp.shape[1], padded_tokens=inp.numel(), scored_positions=len(group), head_calls=0)
    out = score_head(model, states, depths, head_chunk, gather, token_refs, counts, cost, apple=True)
    cost["native_and_loopcd_seconds"] = time.perf_counter() - phase_started
    del states
    for k in depths:
        phase_started = time.perf_counter()
        if shared and k > 1:
            hidden, _ = branch_hidden(model, packed, k, net, dose, raw)
        else:
            hidden, _ = minimal.forward(model, packed, k, net, dose)
        outputs = score_head(model, [hidden], (k,), head_chunk, gather, token_refs, counts, cost, apple=False)
        out[f"minimal_k{k}"] = outputs["minimal"]
        cost[f"minimal_k{k}_seconds"] = time.perf_counter() - phase_started
    return refs, out


def verify_branches(model, net, rows, encoder, depths, dose, args, device="cuda"):
    """Compare exact tensors and the scored results on identical packed inputs."""
    targets = list(range(2, max(depths)+1))
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["benchmark"]].append(row)
    report = {"status": "running", "depths": targets, "cases": [], "device": device,
              "selection": "Per benchmark, short/median/long prompts by maximum request text length and ID"}
    # One case incorporates all options of the selected question. Length bands
    # are separate batches so padding cannot erase the short-context check.
    for task, candidates in sorted(by_task.items()):
        ordered = sorted(candidates, key=lambda r:(max(sum(map(len,q['args'])) for q in r['requests']), r['id']))
        picks = dict(short=ordered[0], median=ordered[len(ordered)//2], long=ordered[-1])
        seen = set()
        for band, row in picks.items():
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            work = compact_prefixes(encode_rows([row], encoder))
            for batch in batch_work(work, args.max_batch, args.token_cap, args.attention_cap):
                packed, refs, gather, token_refs, counts = pack(batch, device)
                _, raw = native_trunk(model, packed, targets, True)
                for k in targets:
                    new, new_anchor = branch_hidden(model, packed, k, net, dose, raw)
                    old, old_anchor = minimal.forward(model, packed, k, net, dose)
                    a = score_head(model, [new], (k,), args.head_rows, gather, token_refs, counts,
                                   {"head_calls":0}, apple=False)
                    b = score_head(model, [old], (k,), args.head_rows, gather, token_refs, counts,
                                   {"head_calls":0}, apple=False)
                    hidden_ok = torch.equal(new, old)
                    anchor_ok = torch.equal(new_anchor, old_anchor)
                    likelihood_ok = a == b
                    delta = max(abs(x-y) for x,y in zip(a["minimal"][0],b["minimal"][0],strict=True))
                    report["cases"].append(dict(task=task, band=band, id=row["id"], depth=k,
                         shape=list(packed[0].shape), candidates=len(refs), hidden_bitwise=hidden_ok,
                         anchor_bitwise=anchor_ok, likelihood_and_greedy_exact=likelihood_ok,
                         hidden_max_abs=(new.float()-old.float()).abs().max().item(), likelihood_max_abs=delta))
                    if not (hidden_ok and anchor_ok and likelihood_ok):
                        report["status"] = "failed"
                        return report
    report["status"] = "passed" if report["cases"] else "not_applicable"
    return report


def render(rows):
    return [dict(id=r["id"], task=r["task"], benchmark=r["benchmark"], doc_id=r["doc_id"], gold=r["target"],
                 choices=r["choices"], denominators=[len(c) for c in r["choices"]],
                 requests=[None]*len(r["choices"]), scores={}, greedy={}) for r in rows]


def record(output, requests, tables):
    for i, req in enumerate(requests):
        dest = output[req.row]
        dest["requests"][req.choice] = dict(idx=req.choice, input_tokens=len(req.inp), target_tokens=len(req.targets),
                                            truncated_context_tokens=req.dropped,
                                            retokenized_boundary=req.retokenized_boundary)
        for tag, (scores, greedy) in tables.items():
            dest["scores"].setdefault(tag, [None]*len(dest["choices"]))[req.choice] = scores[i]
            dest["greedy"].setdefault(tag, [None]*len(dest["choices"]))[req.choice] = greedy[i]


def write_jsonl(path, rows):
    temp = path.with_suffix(".partial")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def summarize(rows):
    counts = defaultdict(Counter)
    for r in rows:
        cohorts = set((r["benchmark"], r["task"]))
        denominators = np.asarray(r["denominators"])
        outcomes = {}
        for arm, scores in r["scores"].items():
            vec = np.asarray(scores)
            outcomes[arm] = (int(vec.argmax()) == r["gold"], int((vec/denominators).argmax()) == r["gold"])
        for cohort in cohorts:
            counts[cohort]["n"] += 1
            for name, (acc, norm) in outcomes.items():
                counts[cohort][name, "acc"] += acc
                counts[cohort][name, "acc_norm_char"] += norm
    return {label: {"n": vals["n"], "arms": {arm: {m: {"correct": vals[arm,m],
            "fraction": vals[arm,m]/vals["n"]} for m in ("acc", "acc_norm_char")}
            for arm in sorted({key[0] for key in vals if isinstance(key, tuple)})}}
            for label,vals in sorted(counts.items())}


def read_export(args):
    meta = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (meta.get("schema_version") != 1 or meta.get("harness_version") != "0.4.13"
            or meta.get("jsonl_sha256") != digest(args.input) or meta.get("chat_template") is not False):
        raise ValueError("Input is not the matching pinned base-model Harness export")
    if not args.plan and meta.get("limit_per_task") is not None:
        raise ValueError("Scoring requires a complete Harness export; use shard boundaries for a pilot")
    source = minimal.load_jsonl(args.input)
    keys = [r["id"] for r in source]
    if len(set(keys)) != len(keys) or not source:
        raise ValueError("Harness export is empty or has duplicated IDs")
    if set(args.benchmarks) - {r["benchmark"] for r in source}:
        raise ValueError("Requested benchmark is absent from Harness export")
    expected = {}
    for t in meta["tasks"]:
        if t["shots"] != APPLE_BENCHMARKS[t["benchmark"]] or t["task"] in expected:
            raise ValueError("Export has a shot mismatch or a duplicate task receipt")
        expected[t["task"]] = (t["benchmark"], t["documents"], t["requests"])
    actual_docs = Counter(r["task"] for r in source)
    actual_requests = Counter()
    for row in source:
        if row["benchmark"] != expected[row["task"]][0] or row["id"] != f"{row['task']}:{row['doc_id']}":
            raise ValueError("Export benchmark, task, or ID mismatch")
        actual_requests[row["task"]] += len(row["requests"])
    if actual_docs != {k:v[1] for k,v in expected.items()} or actual_requests != {k:v[2] for k,v in expected.items()}:
        raise ValueError("Harness request counts disagree with export receipt")
    selected = [r for r in source if r["benchmark"] in args.benchmarks]
    return selected, meta


def output_lock(args, model_info, source_meta, max_length):
    sourced = (args.input, args.manifest, Path(__file__), Path(minimal.__file__))
    return dict(files={p.name:digest(p) for p in sourced}, runtime=model_info,
                freeze_sha256=digest(args.freeze), arm=args.arm, benchmarks=args.benchmarks,
                harness_manifest=source_meta, depths=args.depths, max_length=max_length,
                shard_size=args.shard_size, max_batch=args.max_batch, head_rows=args.head_rows,
                token_cap=args.token_cap, attention_cap=args.attention_cap,
                protocol="HFLM causal scoring, terminal target excluded from forward tokens, model-limit-only truncation",
                predictor_prefix_reuse="Exact predictor-prefix sharing within shard, causal gather all targets; no KV cache",
                recurrent_prefix_reuse=args.shared_prefix,
                dense_passes_per_batch=loop_work(args.depths,args.shared_prefix),
                loopcd="Apple 2610.02185 Table A1: K1 reference, fixed 0.5/adaptive margin cap 1.0, no hidden variant",
                normalization="acc_norm_char divides by the original Harness doc_to_choice character counts")


@torch.inference_mode()
def run(args):
    rows, source_meta = read_export(args)
    if args.plan:
        from transformers import AutoTokenizer
        model_config = json.loads((args.model/"config.json").read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    else:
        freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
        arm = freeze["arms"][args.arm]
        checkpoint = args.freeze.parent / arm["file"]
        if digest(checkpoint) != arm["sha256"]:
            raise RuntimeError("Frozen checkpoint digest mismatch")
        if "minimal.py" in freeze.get("source", {}) and digest(Path(minimal.__file__)) != freeze["source"]["minimal.py"]:
            raise RuntimeError("Minimal implementation differs from the freeze")
        _, model, tokenizer, model_info = minimal.runtime(args.archive, args.previous, args.model)
        model_config = model.config.to_dict()
        net = minimal.load_net(checkpoint)
    max_length = next(model_config[k] for k in ("n_positions", "max_position_embeddings", "n_ctx") if k in model_config)
    encoder = HarnessEncoder(tokenizer, max_length)
    slices = [(i,rows[lo:lo+args.shard_size]) for i,lo in enumerate(range(0,len(rows),args.shard_size))]
    args.out.mkdir(exist_ok=True, parents=True)
    if args.plan:
        plan = defaultdict(Counter)
        for _, shard in slices:
            split_rows = defaultdict(list)
            for row in shard:
                split_rows[row["benchmark"]].append(row)
            for label, selection in split_rows.items():
                reqs = encode_rows(selection, encoder)
                work = compact_prefixes(reqs)
                matrix = list(batch_work(work, args.max_batch, args.token_cap, args.attention_cap))
                plan[label].update(questions=len(selection), requests=len(reqs), unique_forwards=len(work),
                                   unshared_input_tokens=sum(len(r.inp) for r in reqs),
                                   unique_input_tokens=sum(len(w.inp) for w in work),
                                   padded_input_tokens=sum(len(x)*len(x[-1].inp) for x in matrix),
                                   batches=len(matrix), truncated_requests=sum(r.dropped>0 for r in reqs),
                                   boundary_mismatches=sum(r.retokenized_boundary for r in reqs))
        minimal.dump(args.out/"PLAN.json", dict(benchmarks=plan, model_max_length=max_length,
                     standalone_dense_loop_multiplier=loop_work(args.depths,False),
                     shared_dense_loop_multiplier_if_parity_passes=loop_work(args.depths,True)))
        return
    lock = output_lock(args, model_info, source_meta, max_length)
    if args.shared_prefix or args.parity_check:
        report = verify_branches(model, net, rows, encoder, args.depths, arm["dose"], args)
        report.update(runtime=model_info, freeze_sha256=lock["freeze_sha256"], arm=args.arm,
                      checkpoint_sha256=arm["sha256"], actual_dose=arm["dose"], files=lock["files"],
                      requested_depths=args.depths, native_plus_standalone_dense_passes=loop_work(args.depths,False),
                      native_plus_shared_dense_passes=loop_work(args.depths,True))
        minimal.dump(args.out/"PARITY.json", report)
        if report["status"] != "passed":
            raise RuntimeError(f"K2–K4 shared-prefix parity {report['status']}; inspect PARITY.json")
        print("PARITY PASSED", len(report["cases"]), flush=True)
        if args.parity_check:
            return
    lock_file = args.out/"LOCK.json"
    if lock_file.exists():
        if json.loads(lock_file.read_text(encoding="utf-8")) != json.loads(json.dumps(lock)):
            raise RuntimeError("Existing stage-2 lock differs from current protocol")
    else:
        minimal.dump(lock_file, lock)
    for i, shard in slices[args.first_shard:args.end_shard]:
        result_path = args.out/f"shard_{i:05d}.jsonl"
        cost_path = args.out/f"shard_{i:05d}.cost.json"
        if cost_path.exists():
            previous = json.loads(cost_path.read_text(encoding="utf-8"))
            if previous["sha256"] != digest(result_path) or previous["ids"] != [r["id"] for r in shard]:
                raise RuntimeError(f"Completed shard {i} disagrees with its receipt")
            continue
        started = time.perf_counter()
        work = compact_prefixes(encode_rows(shard, encoder))
        output = render(shard)
        active = list(batch_work(work, args.max_batch, args.token_cap, args.attention_cap))[::-1]
        costs = []
        oom_retries = 0
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        while active:
            batch = active.pop()
            cost = {}
            tick = time.perf_counter()
            try:
                refs, scores = score_batch(model, net, batch, args.depths, arm["dose"], args.head_rows, cost,
                                           shared=args.shared_prefix)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                if len(batch) < 2:
                    raise
                oom = True
            else:
                oom = False
            if oom:
                middle = len(batch)//2
                active.extend((batch[middle:], batch[:middle]))
                oom_retries += 1
                torch.cuda.empty_cache()
                continue
            record(output, refs, scores)
            cost["wall_seconds"] = time.perf_counter()-tick
            costs.append(cost)
        if any(any(x is None for x in r["requests"]) or any(any(v is None for v in s) for s in r["scores"].values())
               for r in output):
            raise RuntimeError(f"Incomplete per-candidate likelihoods in shard {i}")
        write_jsonl(result_path, output)
        elapsed = time.perf_counter()-started
        minimal.dump(cost_path, dict(ids=[r["id"] for r in output], sha256=digest(result_path),
                     wall_seconds=elapsed, batches=costs, oom_retries=oom_retries,
                     dense_loop_tokens_padded=sum(b["loop_passes"]*b["padded_tokens"] for b in costs),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        print(i, len(shard), round(elapsed,2), flush=True)
    fragments = [args.out/f"shard_{i:05d}.jsonl" for i,_ in slices]
    have = [args.out/f"shard_{i:05d}.cost.json" for i,_ in slices]
    if all(f.exists() for f in have):
        results = []
        for raw, receipt_path, (_, shard) in zip(fragments, have, slices, strict=True):
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            chunk = minimal.load_jsonl(raw)
            if receipt["sha256"] != digest(raw) or [r["id"] for r in chunk] != [r["id"] for r in shard]:
                raise RuntimeError(f"Mismatched receipt {receipt_path.name}")
            results.extend(chunk)
        write_jsonl(args.out/"RESULTS.jsonl", results)
        minimal.dump(args.out/"SUMMARY.json", summarize(results))
        print("DONE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "input", "manifest", "out"):
        p.add_argument("--"+name, required=True, type=Path)
    for name in ("archive", "previous", "freeze"):
        p.add_argument("--"+name, type=Path)
    p.add_argument("--arm")
    p.add_argument("--depths", nargs="+", choices=(1,2,3,4), type=int, default=[4])
    p.add_argument("--benchmarks", nargs="+", choices=APPLE_BENCHMARKS, default=list(APPLE_BENCHMARKS))
    for name, value in (("shard-size",256), ("max-batch",32), ("head-rows",128), ("token-cap",4096),
                        ("attention-cap",8_388_608), ("first-shard",0), ("end-shard",None)):
        p.add_argument("--"+name, type=int, default=value)
    p.add_argument("--plan", action="store_true", help="Estimate exact model work with CPU tokenization only")
    p.add_argument("--parity-check", action="store_true", help="Verify frozen branch parity on GPU and exit before scoring")
    p.add_argument("--shared-prefix", action="store_true", help="Reuse native recurrence after repeating the GPU parity check")
    args = p.parse_args()
    if not args.plan and not all((args.archive,args.previous,args.freeze,args.arm)):
        p.error("Scoring requires --archive, --previous, --freeze and --arm")
    if (any(x<1 for x in (args.shard_size,args.max_batch,args.head_rows,args.token_cap,args.attention_cap))
            or args.first_shard < 0 or (args.end_shard is not None and args.end_shard <= args.first_shard)):
        p.error("Positive work caps and a valid shard interval are required")
    args.depths = sorted(set(args.depths))
    if args.plan and (args.parity_check or args.shared_prefix):
        p.error("--plan is a CPU-only work estimate")
    if (args.parity_check or args.shared_prefix) and max(args.depths) == 1:
        p.error("Shared-prefix parity requires at least one requested depth above K1")
    run(args)


if __name__ == "__main__":
    main()
