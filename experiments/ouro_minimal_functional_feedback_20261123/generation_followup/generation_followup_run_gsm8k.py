"""Run zero-shot greedy generation using locked Ouro and three matched K4 baselines."""
import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import re
import time

import pyarrow.parquet as pq
import torch

from minimal import dump, load_net, runtime, sha

TASKS = {
    "mmlu_pro": (8, 96),
    "gsm8k": (256, 48),
    "humaneval": (256, 48),
    "mbpp": (256, 48),
}
ARMS = ("native", "fixed", "adaptive", "feedback")
FINAL = re.compile(r"####[ \t]*\$?([+-]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+)?)(?=[ \t]*(?:[. \t]*\n|$))")
FIRST_CHOICE = re.compile(r"^\s*(?:\(\s*)?([A-J])(?=\W|$)")


@dataclass
class Item:
    key: str
    prompt: str
    gold: str | None
    ids: list
    choices: int = 0


def prompt_items(path, task, tokenizer):
    rows = pq.read_table(path).to_pylist()
    items = []
    for index, row in enumerate(rows):
        choices = 0
        if task == "gsm8k":
            prompt = ("Solve the following math problem. Show your work and end with a line of the form "
                      "#### followed by the final number.\n\nQuestion: " + row["question"] + "\nSolution:")
            gold = row["answer"].rsplit("####", 1)[1].strip()
            key = str(index)
        elif task == "mmlu_pro":
            choices = len(row["options"])
            text = "\n".join(f"{chr(65 + a)}. {x}" for a, x in enumerate(row["options"]))
            prompt = ("Choose the correct option. Return its letter only.\n\nQuestion: " + row["question"]
                      + "\n" + text + "\nAnswer:")
            gold = str(row["answer"])
            assert gold == chr(65 + int(row["answer_index"])) and 0 < choices <= 10
            key = str(row["question_id"])
        elif task == "humaneval":
            prompt = row["prompt"].expandtabs(4)
            gold = None
            key = str(row["task_id"])
        else:
            prompt = ("Write a Python solution for the following task. Return only Python code.\n\n"
                      + row["prompt"] + "\n\nPython solution:\n```python\n")
            gold = None
            key = str(row["task_id"])
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert ids
        items.append(Item(key, prompt, gold, ids, choices))
    assert len({item.key for item in items}) == len(items)
    return items


def extract(task, text, choices):
    if task == "mmlu_pro":
        if match := FIRST_CHOICE.match(text):
            value = match.group(1)
            return value if ord(value) - 65 < choices else None
    if task == "gsm8k":
        matches = FINAL.findall(text)
        return matches[-1] if matches else None
    return None


def batches(items, max_new, max_batch, capacity_tokens):
    remaining = sorted(items, key=lambda x: (len(x.ids), x.key))
    at = 0
    while at < len(remaining):
        end = at
        while end < len(remaining):
            capacity = len(remaining[end].ids) + max_new
            if end > at and ((end - at + 1) * capacity > capacity_tokens or end - at + 1 > max_batch):
                break
            end += 1
        assert end > at
        yield remaining[at:end]
        at = end


def cache_type(model):
    cls = importlib.import_module(type(model.model).__module__).UniversalTransformerCache

    class BoundedCache(cls):
        """Store each Ouro layer and loop separately using in-place append."""

        def __init__(self, capacity):
            self.layers = []
            self.capacity = capacity
            self.slots = [None] * 96
            self.lengths = [0] * 96

        def update(self, key, value, layer_idx, cache_kwargs=None):
            start = self.lengths[layer_idx]
            end = start + key.shape[2]
            if end > self.capacity:
                raise RuntimeError("KV cache exceeded the declared capacity")
            if self.slots[layer_idx] is None:
                size = (key.shape[0], key.shape[1], self.capacity, key.shape[-1])
                self.slots[layer_idx] = (key.new_empty(size), value.new_empty(size))
            a, b = self.slots[layer_idx]
            a[:, :, start:end].copy_(key)
            b[:, :, start:end].copy_(value)
            self.lengths[layer_idx] = end
            return a[:, :, :end], b[:, :, :end]

        def get_seq_length(self, layer_idx=0):
            return self.lengths[layer_idx or 0]

    return BoundedCache


def forward(model, tokens, valid, indices, arm, net, dose, cache, eligible):
    core = model.model
    positions = (valid.long().cumsum(-1) - 1).clamp_min_(0)[:, indices]
    mask = valid[:, None, None, :] & (torch.arange(valid.shape[1], device=tokens.device)[None, :]
                                          <= indices[:, None])[None, None, :, :]
    if arm == "feedback":
        loop = 0

        def hook(_, args):
            nonlocal loop
            loop += 1
            if loop != 3:
                return None
            h = args[0]
            z = h[:, eligible].float()
            update = (z + dose * net.delta(z)).to(h.dtype)
            h = h.clone()
            h[:, eligible] = update
            return (h, *args[1:])

        handle = core.norm.register_forward_pre_hook(hook)
    else:
        handle = None
    try:
        _, states, _ = core(input_ids=tokens, attention_mask={"full_attention": mask}, position_ids=positions,
                            cache_position=indices, past_key_values=cache, use_cache=cache is not None)
    finally:
        if handle is not None:
            handle.remove()
    logits = model.lm_head(states[-1][:, -1]).float()
    if arm in ("fixed", "adaptive"):
        early = model.lm_head(states[0][:, -1]).float()
        if arm == "fixed":
            weight = 0.3
        else:
            p = (logits.topk(2, dim=-1).values - logits.logsumexp(-1, keepdim=True)).exp()
            weight = 0.5 * (1 - (p[:, :1] - p[:, 1:]))
        logits = logits + weight * (logits - early)
    return logits


def prepare(items, eos):
    width = max(len(item.ids) for item in items)
    padded, masks = [], []
    for item in items:
        delta = width - len(item.ids)
        padded.append([eos] * delta + item.ids)
        masks.append([False] * delta + [True] * len(item.ids))
    return torch.tensor(padded, device="cuda"), torch.tensor(masks, device="cuda")


@torch.inference_mode()
def greedy(task, model, tokenizer, items, arm, net, dose, max_new, create_cache, *, parity=False):
    eos = tokenizer.eos_token_id
    tokens, valid = prepare(items, eos)
    b, width = tokens.shape
    base = torch.arange(width, device=tokens.device)
    cache = create_cache(width + max_new)
    generated = [[] for _ in items]
    stopped = [None] * b
    complete, maximum = 0, 0.
    for step in range(max_new):
        if step == 0:
            indices = base
            ready = slice(width - 1, None)
            incoming = tokens
        else:
            indices = torch.tensor([tokens.shape[1] - 1], device="cuda")
            ready = slice(None)
            incoming = tokens[:, -1:]
        logits = forward(model, incoming, valid, indices, arm, net, dose, cache, ready)
        chosen = logits.argmax(-1)
        if parity:
            direct = forward(model, tokens, valid, torch.arange(tokens.shape[1], device="cuda"),
                             arm, net, dose, None, slice(width - 1, None))
            maximum = max(maximum, float((direct - logits).abs().max()))
            reference = direct.argmax(-1)
            if not torch.equal(chosen, reference):
                for j in (chosen != reference).nonzero().flatten().tolist():
                    a, c = int(chosen[j]), int(reference[j])
                    raise RuntimeError(("KV cache greedy parity", task, arm, step, j, a, c, maximum,
                                        float(direct[j, c] - direct[j, a])))
        selected = chosen.cpu().tolist()
        for row, value in enumerate(selected):
            if stopped[row] is not None:
                continue
            generated[row].append(value)
            reason = "eos" if value == eos else None
            if reason is None and (task == "mmlu_pro" or step % 4 == 3):
                text = tokenizer.decode(generated[row], skip_special_tokens=True)
                if task == "mmlu_pro" and extract(task, text, items[row].choices):
                    reason = "letter"
                elif task == "gsm8k" and "\n" in text.rsplit("####", 1)[-1] and extract(task, text, 0):
                    reason = "number_line"
                elif task in ("humaneval", "mbpp") and "\n```" in text:
                    reason = "code_fence"
                elif task in ("humaneval", "mbpp") and re.search(r"\n(?:# (?:Test|Example)|if __name__)", text):
                    reason = "end_of_solution"
            if reason is not None:
                stopped[row] = reason
                complete += 1
        if complete == b:
            break
        tokens = torch.cat((tokens, chosen[:, None]), dim=1)
        valid = torch.cat((valid, valid.new_ones((b, 1))), dim=1)
    result = []
    for item, ids, reason in zip(items, generated, stopped, strict=True):
        tail = tokenizer.decode(ids, skip_special_tokens=True)
        result.append(dict(id=item.key, gold=item.gold, generated=tail, tokens=ids,
                           stop=reason or "limit", predicted=extract(task, tail, item.choices)))
    return result, maximum


def save_jsonl(path, records):
    tmp = path.with_suffix(".partial")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def run(args):
    settings = json.loads(args.freeze.read_text(encoding="utf-8"))
    arm = settings["arms"][settings["best"]]
    weight = args.freeze.parent / arm["file"]
    assert sha(weight) == arm["sha256"]
    _, model, tokenizer, model_info = runtime(args.archive, args.previous, args.model)
    model.model.total_ut_steps = 4
    model.eval()
    net = load_net(weight)
    create = cache_type(model)
    source = args.data / f"{args.task}.parquet"
    receipt = json.loads(source.with_suffix(".json").read_text())
    assert sha(source) == receipt["sha256"]
    items = prompt_items(source, args.task, tokenizer)
    assert len(items) == receipt["rows"]
    max_tokens, batch_size = TASKS[args.task]
    location = args.out / args.task
    location.mkdir(parents=True, exist_ok=True)
    lock_path = location / "LOCK.json"
    lock = dict(arms=list(ARMS), depth=4, max_new_tokens=max_tokens, batch_size=batch_size,
                capacity_tokens=14500, seed=0, decoding="greedy", shots=0, dataset=receipt,
                freeze_sha256=sha(args.freeze), weight_sha256=arm["sha256"], code_sha256=sha(Path(__file__)),
                sources=args.sources or "Ouro native plus LoopCD .3 fixed/.5 adaptive using K1 reference.",
                model=model_info)
    if lock_path.exists() and lock != json.loads(lock_path.read_text(encoding="utf-8")):
        raise RuntimeError("The existing generation lock differs")
    dump(lock_path, lock)
    order = sorted(items, key=lambda x: (len(x.ids), x.key))
    band = (order[len(order)//6], order[len(order)//2], order[-1])
    parity = {}
    for method in ARMS:
        _, difference = greedy(args.task, model, tokenizer, band, method, net, arm["dose"],
                               min(6, max_tokens), create, parity=True)
        parity[method] = difference
    dump(location / "CACHE_PARITY.json", dict(decoding="uncached versus cached greedy next-token exact",
                                               ids=[item.key for item in band], max_logit_diff=parity))
    print("CACHE_PARITY", args.task, parity, flush=True)
    if args.parity_only:
        return
    for method in ARMS:
        found = location / f"{method}.jsonl"
        if found.exists():
            continue
        folder = location / method
        folder.mkdir(exist_ok=True)
        complete = []
        for index, chunk in enumerate(batches(items, max_tokens, batch_size, 14500)):
            path = folder / f"{index:04}.jsonl"
            cost = path.with_suffix(".cost.json")
            if cost.exists():
                record = json.loads(cost.read_text())
                assert record["sha256"] == sha(path) and record["ids"] == [x.key for x in chunk]
            else:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                responses, _ = greedy(args.task, model, tokenizer, chunk, method, net, arm["dose"], max_tokens, create)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                save_jsonl(path, responses)
                dump(cost, dict(seconds=elapsed, peak_bytes=torch.cuda.max_memory_allocated(), sha256=sha(path),
                                ids=[x.key for x in chunk], new_tokens=sum(len(x["tokens"]) for x in responses)))
                if index % 4 == 0:
                    print(args.task, method, index, len(chunk), round(elapsed, 3), flush=True)
            complete.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        assert len(complete) == len({r["id"] for r in complete}) == len(items)
        save_jsonl(found, complete)
        print("COMPLETE", args.task, method, len(complete), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("archive", "previous", "model", "data", "out", "freeze"):
        parser.add_argument(f"--{path}", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--sources")
    parser.add_argument("--parity-only", action="store_true")
    run(parser.parse_args())
