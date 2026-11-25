"""Measure Huginn task sensitivity, recurrent mediation, and finite-dose response."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from arc_features import archive as load_archive
from collect_identification import batches, pack, source_rows, torch_save
from finite_identification import DOSES
from freeze_forecast import top
from huginn_control import SEED, WIDTH, RankOne, forward, load_model, sha256
import minimal
from task_data import hash_order


DEPTH = 32


def score(model, state, batch):
    _, rows, positions, targets = batch.packed
    index = (rows, positions)
    hook = model.lm_head.register_forward_pre_hook(lambda _, args: (args[0][index],))
    try:
        logits = model.predict_from_latents(state).logits
    finally:
        hook.remove()
    logp = -F.cross_entropy(logits, targets, reduction="none")
    sums = logp.new_zeros(len(batch.example)).index_add_(0, rows, logp)
    sizes = [len(item["characters"]) for item in batch.metadata]
    denominator = sums.new_tensor([x for item in batch.metadata for x in item["characters"]])
    margins = []
    for item, option_scores in zip(batch.metadata, (16*sums/denominator).split(sizes), strict=True):
        gold = item["gold"]
        rivals = torch.cat((option_scores[:gold], option_scores[gold+1:]))
        margins.append(option_scores[gold] - rivals.logsumexp(0) + rivals.new_tensor(rivals.numel()).log())
    return torch.stack(margins), logits


def prefix(model, tokens, seed):
    with torch.no_grad(), torch.random.fork_rng(devices=[tokens.device.index]):
        torch.cuda.manual_seed(seed)
        embedded, layer = model.embed_inputs(tokens, use_cache=False)
        freqs = model.freqs_cis[:, :tokens.shape[1]]
        state = model.initialize_state(embedded)
        for step in range(DEPTH - 1):
            state, layer = model.core_block_forward(state, embedded, freqs, None, None, layer, step)
    return state, embedded, freqs, layer


def outputs(model, batch, leaf, embedded, freqs, layer):
    loop, _ = model.core_block_forward(leaf, embedded, freqs, None, None, layer, DEPTH-1)
    return score(model, leaf, batch), score(model, loop, batch)


def verify_native(model, batch, seed, direct, recurrent):
    tokens, rows, pos, _ = batch.packed
    index = (rows, pos)
    with torch.inference_mode():
        for depth, want in ((DEPTH-1, direct), (DEPTH, recurrent)):
            native = forward(model, tokens, index, index, depth, seed)
            if not torch.equal(native, want):
                raise ValueError(f"The extracted Huginn suffix differs from the native K{depth} computation")


def collect(model, archive, tok, samples, a):
    all_results = []
    for fold in range(4):
        rows = ordered([r for r in samples if int(hash_order(r["sample_id"], "huginn/fold"), 16) % 4 == fold])
        path = a.out / f"fold_{fold}.pt"
        checkpoint = torch.load(path, weights_only=True, map_location="cpu") if path.exists() else {}
        saved = checkpoint.get("ids", [])
        if saved != [r["sample_id"] for r in rows[:len(saved)]]:
            raise ValueError("Huginn fold resume does not match the locked question prefix")
        if len(saved) == len(rows):
            all_results.append(checkpoint)
            continue
        matrices = [checkpoint.get(name, torch.zeros(WIDTH, WIDTH)).to(model.device)
                    for name in ("direct", "recurrent")]
        for i, segment in enumerate(batches(rows, len(saved), 4, 1600, "candidate")):
            data = pack(archive, tok, segment, "candidate")
            state, x, rot, layer = prefix(model, data.packed[0], SEED+fold*10_000+len(saved))
            h = state.detach().float().requires_grad_(True)
            (direct, p), (recurrent, q) = outputs(model, data, h.to(state.dtype), x, rot, layer)
            if not saved and fold == 0:
                verify_native(model, data, SEED, p, q)
            gd, = torch.autograd.grad(direct.sum(), h, retain_graph=True)
            gr, = torch.autograd.grad(recurrent.sum(), h)
            ix = (data.packed[1], data.packed[2])
            with torch.no_grad():
                for matrix, gradient in zip(matrices, (gd, gr), strict=True):
                    matrix.addmm_(gradient[ix].T, h[ix])
            saved.extend(data.ids)
            if (i+1) % 12 == 0 or len(saved) == len(rows):
                checkpoint = dict(ids=saved, direct=matrices[0].cpu(), recurrent=matrices[1].cpu())
                torch_save(path, checkpoint)
                print("HUGINN MOMENT", fold, len(saved), len(rows), flush=True)
        all_results.append(checkpoint)
    if len({row for item in all_results for row in item["ids"]}) != 1119:
        raise ValueError("Incomplete Huginn ARC moment population")
    return all_results


def ordered(rows):
    return sorted(rows, key=lambda r: (r.get("input_tokens", len(r.get("prompt", ""))), r["sample_id"]))


@torch.no_grad()
def finite(model, net, archive, tok, samples):
    selected = sorted(samples, key=lambda r: hash_order(r["sample_id"], "loop-identification/finite"))[:24]
    records = []
    for start in range(0, len(selected), 4):
        data = pack(archive, tok, selected[start:start+4], "candidate")
        state, x, rot, layer = prefix(model, data.packed[0], SEED+start)
        ix = (data.packed[1], data.packed[2])
        perturb = net(state[ix].float())
        baseline = torch.stack([q for q, _ in outputs(model, data, state, x, rot, layer)], dim=1)
        for dose in DOSES:
            for sign in (1, -1):
                updated = state.index_put(ix, (state[ix].float()+sign*dose*perturb).to(state.dtype))
                effect = torch.stack([q for q, _ in outputs(model, data, updated, x, rot, layer)], dim=1)-baseline
                for key, (direct, recurrent) in zip(data.ids, effect.tolist(), strict=True):
                    records.append(dict(id=key, dose=dose, sign=sign, direct=direct, recurrent=recurrent,
                                        contrast=recurrent-direct))
    return records


def summaries(records, folds, net):
    spectra = {}
    for name in ("ordinary", "contrast"):
        components = [part["recurrent"] if name == "ordinary" else part["recurrent"]-part["direct"] for part in folds]
        pooled = sum(components)/1119
        _, s, _ = top(pooled, 16, SEED)
        directions = [top(m, 8, SEED+1) for m in components]
        stability = [float(((u.T @ v).square().sum() + (w.T @ x).square().sum())/16)
                     for i, (u, _, w) in enumerate(directions) for v, _, x in directions[i+1:]]
        slope = float(net.write.weight.detach().cpu().flatten() @ pooled @ net.read.weight.detach().cpu().flatten())
        spectra[name] = dict(singular_values=s.tolist(), stable_rank=float(pooled.square().sum()/s[0].square()),
                             fold_top8_overlap=stability, rank1_unclipped_slope=slope)
    response = []
    for dose in DOSES:
        for sign in (1, -1):
            changes = torch.tensor([[r[k] for k in ("direct", "recurrent", "contrast")] for r in records
                                    if (r["dose"], r["sign"]) == (dose, sign)], dtype=torch.float64)
            sample = torch.randint(len(changes), (4000, len(changes)), generator=torch.Generator().manual_seed(SEED))
            q = torch.quantile(changes[sample].mean(1), torch.tensor([.025, .975], dtype=changes.dtype), dim=0)
            response.append(dict(dose=dose, sign=sign, means=changes.mean(0).tolist(), ci95=q.T.tolist()))
    return dict(n_train=1119, n_validation=24, spectra=spectra, finite=response,
                dose_fields=["direct", "recurrent", "contrast"])


def run(a):
    a.out.mkdir(parents=True, exist_ok=True)
    lock = dict(depth=DEPTH, train=sha256(a.train), validation=sha256(a.validation), controller=sha256(a.weights),
                model=sha256(a.model/"MODEL_LOCK.json"), evaluator=sha256(Path(__file__)))
    path = a.out / "LOCK.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != lock:
        raise ValueError("Huginn mechanism inputs changed")
    minimal.dump(path, lock)
    model, tok = load_model(a.model)
    net = RankOne().to(model.device)
    net.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True)["state"])
    net.requires_grad_(False).eval()
    arc = load_archive(a.archive)
    train = list(source_rows(a.train, "arc_challenge", "candidate"))
    dev = list(source_rows(a.validation, "arc_challenge", "candidate"))
    if len(train) != 1119 or len(dev) != 299:
        raise ValueError("Unexpected Huginn mechanism cohorts")
    folds = collect(model, arc, tok, train, a)
    records = finite(model, net, arc, tok, dev)
    minimal.dump(a.out / "FINITE.json", records)
    torch.set_num_threads(4)
    minimal.dump(a.out / "MECHANISM.json", summaries(records, folds, net))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "weights", "archive", "train", "validation", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    run(p.parse_args())
