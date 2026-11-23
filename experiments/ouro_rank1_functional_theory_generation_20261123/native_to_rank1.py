"""Compare natural Ouro transitions, task sensitivity, and frozen Rank-1 control."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from a_native_stats import summarize, save, CELLS, DOSE, SMALL, rms, cosine, unit


SEED = 26112371
PHYSICAL = "physical21"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_torch(path, value):
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_jsonl(path, rows):
    temporary = path.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def complete(path, result):
    receipt = path.with_suffix(".receipt.json")
    if not receipt.exists():
        return False
    values = read(receipt)
    if values["sha256"] != sha(path):
        raise RuntimeError(f"Corrupt {path}")
    result.append(values)
    return True


def encode(archive, tokenizer, raw, batch_size):
    items, info = archive.encode(tokenizer, raw)
    options = defaultdict(list)
    for item in items:
        options[item[2]].append(item)
    own = sorted(options, key=lambda q: (max(len(x[0]) for x in options[q]), info[q]["id"]))
    batches = [[item for q in own[i:i + batch_size] for item in options[q]]
               for i in range(0, len(own), batch_size)]
    return info, batches


def native_forward(model, packed, lead, depth, *, anchor=None, edit=None, gradient=False, retain=False):
    """Capture the residual before output normalization and edit only first answer predictors."""
    inp, groups, positions, _ = packed
    row, at = groups[lead], positions[lead]
    core = model.model
    core.total_ut_steps = depth
    turn = 0
    raw, captures = {}, {}

    def visit(tensor, key):
        if key != anchor and not retain:
            return tensor
        state = tensor[row, at].float()
        if retain and key != PHYSICAL:
            raw[key] = state.detach()
        if key != anchor and key != PHYSICAL:
            return tensor
        captures[key] = state.detach()
        if key != anchor:
            return tensor
        if gradient:
            leaf = state.detach().requires_grad_(True)
            captures["leaf"] = leaf
            return tensor.index_put((row, at), leaf.to(tensor.dtype))
        if edit is not None:
            adjusted = (state + edit).to(tensor.dtype)
            captures["actual"] = adjusted.float() - state
            return tensor.index_put((row, at), adjusted)
        return tensor

    def layer(_, __, tensor):
        return visit(tensor, PHYSICAL) if turn == 0 else None

    def norm(_, args):
        nonlocal turn
        turn += 1
        value = visit(args[0], turn)
        return (value, *args[1:]) if value is not args[0] else None

    hooks = [core.norm.register_forward_pre_hook(norm)]
    if retain or anchor == PHYSICAL:
        hooks.append(core.layers[20].register_forward_hook(layer))
    try:
        _, states, _ = core(input_ids=inp, use_cache=False)
    finally:
        for handle in hooks:
            handle.remove()
    if turn != depth or len(states) != depth or (anchor is not None and anchor not in captures):
        raise RuntimeError(f"Invalid {anchor=} or {depth=} in native forward")
    return states, raw, captures


def lead_indices(items, device):
    starts = np.cumsum([0, *(n for _, n, _, _ in items[:-1])])
    return torch.as_tensor(starts, dtype=torch.long, device=device)


def candidate_score(archive, model, h, packed, count):
    return archive.scores(model.lm_head(h).float(), packed[1].cpu().numpy(), packed[-1], count)


def gradient_utility(model, h, packed, items, info):
    logits = model.lm_head(h).float()
    target = logits.log_softmax(dim=-1).gather(1, packed[-1][:, None]).squeeze(-1)
    score = target.new_zeros(len(items)).index_add(0, packed[1], target)
    counts = []
    questions = []
    for _, _, q, a in items:
        if a == 0:
            questions.append(q)
            counts.append(len(info[q]["characters"]))
    chunks = score.split(counts)
    utility = []
    for scores, q in zip(chunks, questions, strict=True):
        row = info[q]
        gold = row["gold"]
        z = scores / scores.new_tensor(row["characters"])
        utility.append(z[gold] - torch.cat((z[:gold], z[gold + 1:])).logsumexp(0))
    return torch.stack(utility), counts


def raw_for(raw, captures, anchor):
    return captures[PHYSICAL] if anchor == PHYSICAL else raw[anchor]


def next_native(raw, captures, anchor):
    return raw[1 if anchor == PHYSICAL else anchor + 1] - raw_for(raw, captures, anchor)


def prior_progress(raw, anchor):
    return raw[anchor] - raw[anchor - 1] if anchor in (2, 3) else None


def geometry_state(raw, captures, items, net):
    candidates = [i for i, (_, _, _, choice) in enumerate(items) if choice == 0]
    q = [items[i][2] for i in candidates]
    states = {str(k): v[candidates].cpu() for k, v in raw.items()}
    states[PHYSICAL] = captures[PHYSICAL][candidates].cpu()
    z = {str(k): net.encode(v[candidates]).squeeze(-1).cpu() for k, v in raw.items() if k < 4}
    z[PHYSICAL] = net.encode(captures[PHYSICAL][candidates]).squeeze(-1).cpu()
    return q, states, z


class Study:
    def __init__(self, args):
        sys.path.insert(0, str(args.minimal.resolve()))
        import minimal

        self.a = args
        self.out = args.out
        self.old, self.model, self.tokenizer, runtime = minimal.runtime(args.archive, args.previous, args.model)
        self.model.eval().requires_grad_(False)
        self.device = next(self.model.parameters()).device
        self.net = minimal.Minimal("linear", 1, .45).to(self.device).eval().requires_grad_(False)
        checkpoint = torch.load(args.rank1, map_location="cpu", weights_only=True)
        self.net.load_state_dict(checkpoint["state"])
        if checkpoint["form"] != "rank1" or self.net.out.weight.shape[1] != 1:
            raise RuntimeError("Expected the locked learned Rank-1 checkpoint")
        freeze = read(args.rank1_freeze)
        item = freeze["weights"]["rank1_t1"]
        if freeze["primary"] != "rank1_t1" or item["sha256"] != sha(args.rank1):
            raise RuntimeError("Rank-1 differs from the previous study's main checkpoint")
        self.cohorts = {}
        for split, batch in (("train", args.train_batch), ("validation", args.batch)):
            rows = self.old.read_data(args.corpus, split)
            if split == "train":
                rows = sorted(rows, key=lambda r: hashlib.sha256(f"{SEED}/native-A/{r['id']}".encode()).digest())[:args.train_n]
            self.cohorts[split] = encode(self.old, self.tokenizer, rows, batch)
        train_info, val_info = (self.cohorts[x][0] for x in ("train", "validation"))
        if {x["id"] for x in train_info} & {x["id"] for x in val_info}:
            raise RuntimeError("Train and held-out partitions overlap")
        lock = dict(runtime=runtime, code_sha256={name: sha(Path(__file__).with_name(name))
                        for name in ("native_to_rank1.py", "a_native_stats.py")},
                    previous_minimal_sha256=sha(Path(minimal.__file__)),
                    rank1_sha256=item["sha256"], previous_freeze_sha256=sha(args.rank1_freeze),
                    hyperparameters=dict(dose=DOSE, slope_step=SMALL, train_batch=args.train_batch, batch=args.batch, seed=SEED),
                    train_sha256=sha(args.corpus / "train.parquet"), validation_sha256=sha(args.corpus / "validation.parquet"),
                    train_ids=[r["id"] for r in train_info], validation_ids=[r["id"] for r in val_info],
                    cells={key: [anchor, depth] for key, (anchor, depth) in CELLS.items()})
        manifest = self.out / "MANIFEST.json"
        if manifest.exists():
            if json.loads(json.dumps(lock)) != read(manifest):
                raise RuntimeError("Existing Experiment A manifest differs")
        else:
            save(manifest, lock)

    @torch.no_grad()
    def native(self, split):
        path = self.out / f"NATIVE_{split}.pt"
        receipts = []
        if complete(path, receipts):
            return torch.load(path, map_location="cpu", weights_only=True)
        info, batches = self.cohorts[split]
        records = {k: [None] * len(info) for k in ("states", "encoder", "scores")}
        start = time.monotonic()
        for items in batches:
            packed = self.old.pack(items)
            leads = lead_indices(items, self.device)
            states, raw, captures = native_forward(self.model, packed, leads, 4, retain=True)
            qs, h, z = geometry_state(raw, captures, items, self.net)
            for pos, q in enumerate(qs):
                records["states"][q] = {k: v[pos] for k, v in h.items()}
                records["encoder"][q] = {k: v[pos] for k, v in z.items()}
            for k, state in enumerate(states, start=1):
                scores = candidate_score(self.old, self.model, state[packed[1], packed[2]], packed, len(items))
                self.old.record(info, items, {f"k{k}": scores})
        records["scores"] = [row["scores"] for row in info]
        for record in info:
            record["scores"] = {}
        artifact = dict(ids=[r["id"] for r in info], rows=records, runtime_seconds=time.monotonic() - start)
        atomic_torch(path, artifact)
        save(path.with_suffix(".receipt.json"), {"sha256": sha(path), "n": len(info), "batches": len(batches)})
        print("native", split, len(info), round(artifact["runtime_seconds"], 2), flush=True)
        return artifact

    @torch.enable_grad()
    def fit(self, native_train):
        path = self.out / "A_DIRECTIONS.pt"
        if complete(path, []):
            return torch.load(path, map_location="cpu", weights_only=True)
        info, batches = self.cohorts["train"]
        records = {}
        raw = torch.stack([x[str(k)] for x in native_train["rows"]["encoder"] for k in (1, 2, 3)])
        orientation = 1. if float(raw.median()) >= 0 else -1.
        learned = unit(self.net.out.weight[:, 0].detach().cpu()) * orientation
        rng = torch.Generator().manual_seed(SEED)
        rotation = torch.randn(learned.shape, generator=rng)
        rotation = unit(rotation - (rotation * learned).mean() * learned)
        for key, (anchor, depth) in CELLS.items():
            gradient_path = self.out / f"SENSITIVITY_{key}.pt"
            if complete(gradient_path, []):
                records[key] = torch.load(gradient_path, map_location="cpu", weights_only=True)["summary"]
                continue
            gradients = [None] * len(info)
            start = time.monotonic()
            for items in batches:
                packed = self.old.pack(items)
                leads = lead_indices(items, self.device)
                states, _, trace = native_forward(self.model, packed, leads, depth, anchor=anchor, gradient=True)
                u, counts = gradient_utility(self.model, states[-1][packed[1], packed[2]], packed, items, info)
                tips = torch.autograd.grad(u.sum(), trace["leaf"])[0].float()
                scaled = tips * rms(trace[anchor])[:, None]
                offset = 0
                for n, qg in zip(counts, scaled.split(counts), strict=True):
                    gradients[items[offset][2]] = qg.detach().sum(0).cpu()
                    offset += n
            grads = torch.stack(gradients)
            mat = grads.double()
            evals, evecs = torch.linalg.eigh(mat @ mat.T)
            top = torch.arange(len(evals) - 1, len(evals) - 9, -1)
            lam = evals[top].clamp_min(1e-30)
            basis = ((mat.T @ evecs[:, top]) / lam.sqrt()).float().T.contiguous()
            if not bool(torch.isclose(basis @ basis.T, torch.eye(8), atol=1e-4).all()):
                raise RuntimeError("Train utility-gradient basis has fewer than eight stable directions")
            axis = unit(grads.mean(0))
            records[key] = dict(axis=axis, basis=basis,
                                eigenvalue_energy_fraction=(lam / evals.clamp_min(0).sum()).float(),
                                mean_gradient_rms=float(rms(grads.mean(0))), seconds=time.monotonic()-start)
            atomic_torch(gradient_path, dict(ids=[r["id"] for r in info], gradients=grads, summary=records[key]))
            save(gradient_path.with_suffix(".receipt.json"), dict(sha256=sha(gradient_path), n=len(info), batches=len(batches)))
            print("train sensitivity", key, round(records[key]["seconds"], 2), flush=True)
        artifact = dict(learned=learned, rotation=rotation, orientation=orientation, cells=records)
        atomic_torch(path, artifact)
        save(path.with_suffix(".receipt.json"), dict(sha256=sha(path), n=len(info)))
        return artifact

    @torch.no_grad()
    def evaluate(self, split, native, directions):
        info, batches = self.cohorts[split]
        axis = directions["learned"].to(self.device)
        rotate = directions["rotation"].to(self.device)
        for key, (anchor, depth) in CELLS.items():
            path = self.out / f"PROBES_{split}_{key}.jsonl"
            if complete(path, []):
                continue
            start = time.monotonic()
            record = [dict(x, scores={"native": native["rows"]["scores"][q][f"k{depth}"]})
                      for q, x in enumerate(info)]
            for row in record:
                row["treatments"] = {}
            func = directions["cells"][key]["axis"].to(self.device)
            basis = directions["cells"][key]["basis"].to(self.device)
            calls = padded = 0
            for items in batches:
                packed = self.old.pack(items)
                leads = lead_indices(items, self.device)
                states, raw, captures = native_forward(self.model, packed, leads, 4, retain=True)
                base = raw_for(raw, captures, anchor)
                natural = next_native(raw, captures, anchor)
                incoming = prior_progress(raw, anchor)
                h = unit(base)
                scale = rms(base)
                outgoing = unit(natural)
                native_end_shift = raw[depth] - (captures[PHYSICAL] if depth == 1 else raw[depth - 1])
                features = torch.column_stack((scale, (h * axis).mean(-1), (h * func).mean(-1),
                    self.net.encode(base).squeeze(-1) / scale, (h @ basis[:4].T) / (h.shape[-1]**.5)))
                descriptions = torch.column_stack((rms(natural) / scale, cosine(natural, axis), cosine(natural, func),
                    cosine(outgoing, rotate), (outgoing @ basis.T).square().sum(-1) / h.shape[-1],
                    (h @ basis.T).square().sum(-1) / h.shape[-1]))
                selected = [i for i, (_, _, _, choice) in enumerate(items) if choice == 0]
                for i in selected:
                    q = items[i][2]
                    record[q]["features"] = features[i].tolist()
                    record[q]["geometry"] = descriptions[i].tolist()
                    if incoming is not None:
                        record[q]["incoming"] = [float(rms(incoming[i]) / scale[i]), float(cosine(incoming[i], axis)),
                                                   float(cosine(incoming[i], natural[i]))]
                    if anchor != PHYSICAL:
                        record[q]["native_anchor_scores"] = native["rows"]["scores"][q][f"k{anchor}"]
                actions = {"learned": axis, "learned_small_plus": axis, "learned_small_minus": axis}
                if split == "validation":
                    actions.update(learned_law=None, future_native=outgoing, functional=func, rotation=rotate)
                    if incoming is not None:
                        actions["past_native"] = unit(incoming)
                del states
                calls += 1
                padded += packed[0].numel() * 4
                for name, v in actions.items():
                    if name == "learned_law":
                        requested = self.net.delta(base)
                    else:
                        dose = SMALL if name.endswith("plus") else -SMALL if name.endswith("minus") else DOSE
                        requested = scale[:, None] * dose * v
                    result, end_raw, active = native_forward(self.model, packed, leads, depth, anchor=anchor,
                                                              edit=requested, retain=True)
                    if not torch.equal(active[anchor], base):
                        raise RuntimeError(f"Pre-edit state differs for {key}, {name}")
                    scores = candidate_score(self.old, self.model, result[-1][packed[1], packed[2]], packed, len(items))
                    self.old.record(record, items, {name: scores})
                    applied = active["actual"]
                    terminal = end_raw[depth] - raw[depth]
                    diagnostics = torch.column_stack((rms(applied) / scale, rms(terminal) / rms(raw[depth]),
                        cosine(terminal, native_end_shift), cosine(terminal, applied), cosine(applied, requested),
                        (applied * axis).mean(-1) / scale))
                    for i in selected:
                        record[items[i][2]]["treatments"][name] = diagnostics[i].tolist()
                    calls += 1
                    padded += packed[0].numel() * depth
            atomic_jsonl(path, record)
            save(path.with_suffix(".receipt.json"), dict(sha256=sha(path), n=len(record), batch=len(batches),
                                                        model_calls=calls, padded_token_loops=padded,
                                                        wall_seconds=time.monotonic()-start))
            print("finite probes", split, key, len(record), round(time.monotonic()-start, 2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("archive", "previous", "minimal", "model", "corpus", "rank1", "rank1-freeze", "out"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--train-n", type=int, default=128)
    parser.add_argument("--train-batch", type=int, default=6)
    parser.add_argument("--batch", type=int, default=12)
    args = parser.parse_args()
    if not 16 <= args.train_n <= 1119 or min(args.train_batch, args.batch) < 1:
        parser.error("Use 16–1119 training questions and positive batch sizes")
    args.out.mkdir(parents=True, exist_ok=True)
    model = Study(args)
    data = {split: model.native(split) for split in model.cohorts}
    directions = model.fit(data["train"])
    for split, native in data.items():
        model.evaluate(split, native, directions)
    summarize(args.out, directions)


if __name__ == "__main__":
    main()
