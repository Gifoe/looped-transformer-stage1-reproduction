"""Fit one frozen-backbone controller using grouped functional advantages or matched baselines."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import minimal
from advantages import MULTIPLE_CHOICE, choice_batch, forward, token_batch, utilities
from collect_moments import DATASETS
from on_policy import choose_batch, score_prefixes
from task_data import load
from train_generative import normalized_nll, ranking


OBJECTIVES = ("ranking", "generation", "mixed", "mean_advantage", "worst_advantage")


def load_fit(folder):
    lock = json.loads((folder / "DATA_LOCK.json").read_text())
    data = {}
    for ds in DATASETS:
        file = folder / f"{ds}_fit.jsonl"
        if minimal.sha(file) != lock["files"][file.name]["sha256"]:
            raise RuntimeError(f"Frozen training rows differ {file}")
        data[ds] = load(file)
    return data


def schedule(data, size, seed):
    rng = np.random.default_rng(seed)
    ordered = {}
    for domain, rows in data.items():
        items = [rows[i] for i in rng.permutation(len(rows))]
        for start in range(0, len(items), 96):
            items[start:start + 96] = sorted(items[start:start + 96], key=lambda row: row["input_tokens"])
        ordered[domain] = [items[i:i + size] for i in range(0, len(items), size)]
    for batch_id, groups in enumerate(zip(*ordered.values(), strict=True)):
        for position, group in enumerate(groups):
            depth = 2 + (batch_id + position) % 3
            yield depth, group


def scale_manifest(folder):
    scales = {}
    for path in sorted(folder.glob("*.json")):
        dataset, stage, k, _ = path.stem.split("__")
        value = json.loads(path.read_text())["sigma"]
        scales[(f"{dataset}/{stage}", int(k[1:]))] = value
    if len(scales) != 27 or not all(x > 0 for x in scales.values()):
        raise ValueError("Expected nine normalized decision groups at each of K2–K4")
    return scales


class GroupWeights:
    def __init__(self, scales, temperature, eta):
        self.scales = scales
        self.temperature = temperature
        self.eta = eta
        self.values = {key: 0. for key in scales}

    def coefficients(self):
        logits = -np.asarray(list(self.values.values())) / self.temperature
        weight = np.exp(logits - logits.max())
        return dict(zip(self.values, weight / weight.sum(), strict=True))

    def update(self, normalized):
        for key, value in normalized.items():
            self.values[key] = (1. - self.eta) * self.values[key] + self.eta * value


def scores_for_baseline(network, model, choice, token, depth, objective):
    if objective == "ranking":
        logits, _ = forward(model, choice, depth, network)
        packed = choice.packed
        ll = -F.cross_entropy(logits, packed[-1], reduction="none")
        loss, _ = ranking(ll, packed[1], choice.metadata, list(range(len(choice.metadata))))
        return loss
    logits, _ = forward(model, token, depth, network)
    gen = normalized_nll(logits, token.packed, len(token.packed[0]))
    if objective == "mixed" and choice is not None:
        return gen + .25 * scores_for_baseline(network, model, choice, None, depth, "ranking")
    return gen


def gain_pass(model, network, packets, depth, tokenizer, groups, robust, rows, policy):
    active = {}
    observations = {}
    selected = choose_batch(rows, policy)
    replacements = {}
    policy_work = 0
    if selected:
        texts = [row for _, row, _ in selected]
        prefixes = [point["prefix"] for _, _, point in selected]
        with torch.no_grad():
            natural, work = score_prefixes(model, None, tokenizer, texts, prefixes, depth)
        moved, _ = score_prefixes(model, network, tokenizer, texts, prefixes, depth)
        policy_work = 2 * work
        for j, (i, _, point) in enumerate(selected):
            replacements.setdefault(point["stage"], []).append((i, moved[j] - natural[j]))
    for packet in packets:
        with torch.no_grad():
            natural, _ = forward(model, packet, depth)
            reference = utilities(natural, packet, tokenizer.eos_token_id)
        logits, _ = forward(model, packet, depth, network)
        prediction = utilities(logits, packet, tokenizer.eos_token_id)
        for group, result in prediction.items():
            key = (group, depth)
            pair_gain = result - reference[group]
            if group in replacements:
                indexes, values = zip(*replacements[group], strict=True)
                index = torch.tensor(indexes, device=pair_gain.device)
                combined = .5 * (pair_gain[index] + torch.stack(values))
                pair_gain = pair_gain.index_copy(0, index, combined)
            gain = pair_gain.mean() / groups.scales[key]
            active[key] = gain
            observations[key] = float(gain.detach())
    if robust:
        weights = groups.coefficients()
    else:
        weights = dict.fromkeys(groups.scales, 1. / len(groups.scales))
    # Sampling selects one of 4 domains at one of 3 budgets. Its contribution
    # estimates the full dual-weighted objective using the inclusion probability.
    loss = -12 * sum(weights[key] * gain for key, gain in active.items())
    groups.update(observations)
    return loss, {group: value for (group, _), value in observations.items()}, policy_work, len(selected)


def checkpoint(path, controller, details):
    minimal.save_net(path, controller, details)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "previous", "model", "data", "moments", "out"):
        p.add_argument(f"--{arg}", type=Path, required=True)
    p.add_argument("--objective", required=True, choices=OBJECTIVES)
    p.add_argument("--initial", type=Path)
    p.add_argument("--on-policy", type=Path)
    p.add_argument("--steps", type=int, default=840)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=261124173)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--temperature", type=float, default=.15)
    a = p.parse_args()
    if a.temperature <= 0 or a.steps < 1:
        raise ValueError("Require positive temperature and a nonempty common schedule")
    data = load_fit(a.data)
    policy = {}
    if a.on_policy:
        candidates = load(a.on_policy)
        allowed = {r["sample_id"] for r in data["gsm8k"]}
        policy = {row["id"]: row for row in candidates}
        if a.objective != "worst_advantage" or len(candidates) != len(policy) or set(policy) - allowed:
            raise ValueError("Free prefixes must come only from the locked GSM8K fit set")
    old, model, tok, runtime = minimal.runtime(a.archive, a.previous, a.model)
    torch.manual_seed(a.seed)
    controller = minimal.Minimal("linear", 1, .45).cuda()
    if a.initial:
        controller.load_state_dict(torch.load(a.initial, map_location="cpu", weights_only=True)["state"])
    parameters = list(controller.parameters())
    if sum(x.numel() for x in parameters) != 4096 or any(x.dtype != torch.float32 for x in parameters):
        raise RuntimeError("Controller must retain the 4,096 FP32 parameter structure")
    optimizer = torch.optim.AdamW(parameters, lr=a.lr, weight_decay=.03)
    scales = scale_manifest(a.moments)
    weights = GroupWeights(scales, a.temperature, .1)
    a.out.mkdir(parents=True, exist_ok=True)
    lock = dict(objective=a.objective, initial_sha256=minimal.sha(a.initial) if a.initial else None,
                on_policy_sha256=minimal.sha(a.on_policy) if a.on_policy else None, script_sha256=minimal.sha(__file__),
                data_sha256=minimal.sha(a.data/"DATA_LOCK.json"), planned_steps=a.steps, seed=a.seed,
                batch=a.batch, temperature=a.temperature, scale={f"{k[0]}/k{k[1]}": v for k, v in scales.items()},
                learning_rate=a.lr, model_sha256=runtime["model_sha256"])
    path = a.out / "TRAIN_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != lock:
        raise RuntimeError("Existing training receipt differs")
    minimal.dump(path, lock)
    logs = []
    clock = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step, (depth, rows) in enumerate(schedule(data, a.batch, a.seed), 1):
        if step > a.steps:
            break
        ds = rows[0]["dataset"]
        if a.objective == "ranking" and ds not in MULTIPLE_CHOICE:
            continue
        choice = choice_batch(old, tok, rows) if ds in MULTIPLE_CHOICE and a.objective != "generation" else None
        token = token_batch(old, tok, rows) if a.objective != "ranking" else None
        optimizer.zero_grad(set_to_none=True)
        used_policy = 0
        with torch.enable_grad():
            if a.objective in ("mean_advantage", "worst_advantage"):
                packets = [x for x in (choice, token) if x is not None]
                loss, gains, extra_work, used_policy = gain_pass(model, controller, packets, depth, tok, weights,
                    a.objective == "worst_advantage", rows, policy if ds == "gsm8k" else {})
                compute = 2 * sum(x.packed[0].numel() * depth for x in packets) + extra_work
            else:
                loss = scores_for_baseline(controller, model, choice, token, depth, a.objective)
                gains = {}
                compute = sum(x.packed[0].numel() * depth for x in (choice, token) if x is not None)
            loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(gradient):
            raise FloatingPointError(f"Invalid controller gradient at {step}")
        optimizer.step()
        row = dict(step=step, depth=depth, dataset=ds, ids=[x["sample_id"] for x in rows],
                   group_gain=gains, loss=float(loss.detach()), gradient=float(gradient), padded_token_loops=compute,
                   off_policy_prefixes=used_policy)
        logs.append(row)
        if step == 1 or step % 48 == 0:
            print(step, ds, depth, round(row["loss"], 4), round(row["gradient"], 4), gains, flush=True)
    torch.cuda.synchronize()
    model_out = a.out / "controller.pt"
    checkpoint(model_out, controller, {"objective": a.objective, "train_lock_sha256": minimal.sha(path)})
    final_weights = {f"{k[0]}/k{k[1]}": v for k, v in weights.coefficients().items()}
    minimal.dump(a.out / "TRAIN.json", dict(logs=logs, seconds=time.perf_counter()-clock, number_updates=len(logs),
        cuda_peak=torch.cuda.max_memory_allocated(), group_dual_weights=final_weights,
        padded_token_loops=sum(row["padded_token_loops"] for row in logs), checkpoint_sha256=minimal.sha(model_out)))
    print("DONE", minimal.sha(model_out), flush=True)


if __name__ == "__main__":
    main()
