"""Common K4 decoder hooks and candidate scoring for experiments C and D."""

import hashlib
import json
from pathlib import Path
import sys
from collections import defaultdict

import numpy as np
import torch

SEED = 20261123
BRANCHES = {"attention": "input_layernorm_2", "mlp": "post_attention_layernorm_2"}
METRICS = ("acc", "acc_norm", "margin", "margin_norm", "utility", "utility_norm")


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def hash_key(namespace, value):
    return hashlib.sha256(f"{SEED}/{namespace}/{value}".encode()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rms(tensor):
    return tensor.float().square().mean(dim=-1).sqrt()


def cosine(left, right):
    scale = rms(left) * rms(right)
    return ((left.float() * right.float()).mean(dim=-1) / scale.clamp_min(1e-20)).clamp(-1, 1)


def load_direction(path, device):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        obj = obj["state"]["out.weight"]
    vec = obj.float().flatten()
    if not torch.isfinite(vec).all() or rms(vec) == 0:
        raise ValueError("Invalid shared direction")
    return (vec / rms(vec)).to(device)


def load_runtime(args):
    sys.path.insert(0, str(args.archive.resolve()))
    import run_arc as archive
    model, tokenizer, runtime = archive.load_runtime(args.model)
    model.requires_grad_(False)
    device = next(model.parameters()).device
    axis = load_direction(args.direction, device)
    if axis.numel() != model.config.hidden_size or len(model.model.layers) != 24:
        raise ValueError("Expected the 24-layer Ouro checkpoint and matching direction")
    return archive, model, tokenizer, runtime, axis


def arguments(parser):
    for name in ("archive", "model", "corpus", "direction", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)


def population(archive, tokenizer, corpus, split, train_subset=None):
    rows = archive.read_data(corpus, split)
    if train_subset is not None:
        if split != "train":
            raise ValueError("Only the Train population can be subsampled")
        rows = sorted(rows, key=lambda row: hash_key("C-location-train", row["id"]))[:train_subset]
    items, info = archive.encode(tokenizer, rows)
    items.sort(key=lambda item: len(item[0]))
    return items, info


def make_manifest(out, value):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "MANIFEST.json"
    if path.exists():
        if load(path) != value:
            raise RuntimeError(f"Resume manifest differs in {out}")
    else:
        save(path, value)


def forward(model, packed, anchor=None, update=None, branch_layers=(), reference=None,
            clamp_layers=(), trace_turns=(), attention_probe=None):
    """Keep the direct edit and replace additive branch writes at the same sequence positions."""
    inp, group, positions, _ = packed
    core = model.model
    core.total_ut_steps = 4
    turn = 1
    raw, normalized, branches = {}, {}, {}
    changes = defaultdict(list)
    edits = None
    hooks = []
    branch_layers = set(branch_layers)
    clamp_layers = set(clamp_layers)

    def selected(tensor):
        return tensor[group, positions].float()

    def before_norm(_, args):
        nonlocal turn, edits
        tensor = args[0]
        before = selected(tensor)
        raw[turn] = before
        if turn == anchor and update is not None:
            after = (before + update).to(tensor.dtype)
            edits = after.float() - before
            tensor = tensor.index_put((group, positions), after)
        turn += 1
        return (tensor, *args[1:])

    def branch_hook(branch, layer):
        def capture(_, args, tensor):
            key = (turn, layer, branch)
            if key not in branch_layers:
                return None
            if reference is None:
                branches[key] = tensor.detach()
                return None
            native = reference["branches"][key]
            active = native if key in clamp_layers else tensor
            if turn in trace_turns:
                changes[(turn, branch)].append(rms(selected(active) - selected(native)))
            return native if key in clamp_layers else None
        return capture

    hooks.append(core.norm.register_forward_pre_hook(before_norm))
    for layer, branch in sorted({(layer, branch) for _, layer, branch in branch_layers}):
        attribute = BRANCHES[branch]
        hooks.append(getattr(core.layers[layer], attribute).register_forward_hook(branch_hook(branch, layer)))
    if attention_probe is not None:
        hooks.extend(attention_probe.attach(core, lambda: turn, group, positions))
    try:
        _, states, _ = core(input_ids=inp, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    if turn != 5 or len(states) != 4:
        raise RuntimeError("The decoder did not execute exactly four passes")
    normalized.update((index, selected(state)) for index, state in enumerate(states, start=1))
    return states[-1][group, positions], dict(raw=raw, normalized=normalized, edits=edits,
        branches=branches, changes={key: torch.stack(value).mean(0) for key, value in changes.items()})


def score(archive, model, packed, items, hidden):
    return archive.scores(model.lm_head(hidden).float(), packed[1].cpu().numpy(),
                          packed[-1], len(items)).tolist()


def candidate_trace(packed, count, metrics):
    group = packed[1].cpu().numpy()
    sizes = np.bincount(group, minlength=count)
    matrix = torch.stack(list(metrics.values())).cpu().numpy()
    return {name: (np.bincount(group, weights=values, minlength=count) / sizes).tolist()
            for name, values in zip(metrics, matrix, strict=True)}


def replay_batches(archive, items, info, batch_size, out, run_batch):
    chunks = out / "chunks"
    chunks.mkdir(exist_ok=True)
    for index, start in enumerate(range(0, len(items), batch_size)):
        subset = items[start:start + batch_size]
        identity = [[info[question]["id"], choice] for _, _, question, choice in subset]
        path = chunks / f"{index:04d}.json"
        if path.exists():
            rec = load(path)
            if rec["identity"] != identity:
                raise RuntimeError(f"Mismatched chunk identity in {path}")
        else:
            rec = run_batch(archive.pack(subset), subset)
            rec["identity"] = identity
            save(path, rec)
            if index % 8 == 0:
                print(f"{out.name} completed batch {index + 1}/{(len(items) + batch_size - 1)//batch_size}",
                      flush=True)
        archive.record(info, subset, rec["scores"])
        for tag, metrics in rec.get("traces", {}).items():
            for name, values in metrics.items():
                for (_, _, question, choice), value in zip(subset, values, strict=True):
                    row = info[question]
                    target = row.setdefault("trace", {}).setdefault(tag, {}).setdefault(
                        name, [None] * len(row["labels"]))
                    target[choice] = value
    output = out / "PREDICTIONS.jsonl"
    temporary = output.with_suffix(".partial")
    archive.write_predictions(temporary, info)
    temporary.replace(output)


def profile(info, tag, metric):
    values = []
    for row in info:
        scores = np.asarray(row["scores"][tag], dtype=np.float64)
        if metric.endswith("_norm"):
            scores /= row["characters"]
        gold = row["gold"]
        if metric.startswith("acc"):
            values.append(scores.argmax() == gold)
        else:
            wrong = np.delete(scores, gold)
            competitor = np.logaddexp.reduce(wrong) if metric.startswith("utility") else wrong.max()
            values.append(scores[gold] - competitor)
    return np.asarray(values, dtype=np.float64)


def question_trace(info, tag, key):
    return np.array([np.mean(row["trace"][tag][key]) for row in info])


def bootstrap(info):
    rng = np.random.default_rng(SEED)
    return rng.integers(len(info), size=(6000, len(info)), dtype=np.int32)


def paired(values, sample, alpha=.05):
    low, high = np.quantile(values[sample].mean(axis=1), (alpha / 2, 1 - alpha / 2))
    return dict(mean=float(values.mean()), lower=float(low), upper=float(high))
