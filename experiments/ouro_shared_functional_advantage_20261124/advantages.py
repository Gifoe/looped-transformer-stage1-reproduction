"""Correct-versus-competing decisions at the controller's deployed write boundary."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

import minimal
from task_data import encode_target
from train_generative import pack_entries


MULTIPLE_CHOICE = {"arc_challenge", "csqa"}
LONG_GROUPS = ("early", "middle", "final", "end")


@dataclass
class Decisions:
    packed: tuple
    groups: dict
    metadata: list | None = None


def choice_batch(archive, tok, rows):
    items, metadata = archive.encode(tok, [r["candidate"] for r in rows])
    return Decisions(archive.pack(items), {f"{rows[0]['dataset']}/candidate": None}, metadata)


def rationale_boundary(tok, row, prompt, whole):
    reason = row["response"].rsplit("####", 1)[0]
    prefix = tok.encode(row["prompt"] + " " + reason, add_special_tokens=False)
    token_count = len(prefix) - len(prompt)
    # BPE can merge the separator immediately before the marker into the marker.
    if whole[:len(prefix)] != prefix:
        common = next((j for j, (a, b) in enumerate(zip(prefix, whole)) if a != b), len(prefix))
        token_count = common - len(prompt)
    if not 3 <= token_count < len(whole) - len(prompt) - 1:
        raise ValueError(f"Unreliable arithmetic stage boundary for {row['sample_id']}")
    return token_count


def token_batch(archive, tok, rows):
    ds = rows[0]["dataset"]
    encoded = [encode_target(tok, row) for row in rows]
    packed = pack_entries(archive, encoded)
    if ds != "gsm8k":
        role = "span" if ds == "squad" else "content"
        groups = {f"{ds}/{role}": torch.ones_like(packed[-1], dtype=torch.bool)}
    else:
        stages = []
        for row, (prompt, whole, n) in zip(rows, encoded, strict=True):
            at = rationale_boundary(tok, row, prompt, whole)
            half = (at + 1) // 2
            stages.extend([0] * half + [1] * (at - half) + [2] * (n - at - 1) + [3])
        index = torch.tensor(stages, device=packed[-1].device)
        if len(index) != len(packed[-1]):
            raise RuntimeError("Token stage and packed token coordinates differ")
        groups = {f"{ds}/{group}": index == i for i, group in enumerate(LONG_GROUPS)}
    return Decisions(packed, groups)


def sequence_mean(values, example_index, n):
    denominator = values.new_zeros(n).index_add_(0, example_index, torch.ones_like(values))
    if not bool((denominator > 0).all()):
        raise ValueError("Each example must contribute one or more decisions per group")
    return values.new_zeros(n).index_add_(0, example_index, values) / denominator


def token_margin(logits, target, eos):
    positives = logits.gather(1, target[:, None]).squeeze(1)
    top, ids = logits.topk(3, dim=-1)
    rivals = top.masked_fill((ids == target[:, None]) | (ids == eos), -torch.inf).amax(-1)
    # A premature EOS is an explicit rival for every continuation. A correct
    # final EOS is itself the positive decision and faces the best non-EOS rival.
    competing_eos = logits[:, eos].masked_fill(target == eos, -torch.inf)
    return positives - torch.logaddexp(rivals, competing_eos)


def candidate_margin(logits, batch):
    inputs, idx, _, target = batch.packed
    per_token = -F.cross_entropy(logits, target, reduction="none")
    sums = per_token.new_zeros(len(inputs)).index_add_(0, idx, per_token)
    choices = [len(row["characters"]) for row in batch.metadata]
    lengths = sums.new_tensor([n for row in batch.metadata for n in row["characters"]])
    normalized = (16 * sums / lengths).split(choices)
    values = []
    for row, scores in zip(batch.metadata, normalized, strict=True):
        gold = row["gold"]
        other = torch.cat((scores[:gold], scores[gold + 1:]))
        values.append(scores[gold] - other.logsumexp(0))
    return torch.stack(values)


def utilities(logits, batch, eos):
    if batch.metadata is not None:
        return {next(iter(batch.groups)): candidate_margin(logits, batch)}
    inputs, examples, _, target = batch.packed
    value = token_margin(logits, target, eos)
    return {group: sequence_mean(value[mask], examples[mask], len(inputs)) for group, mask in batch.groups.items()}


def forward(model, batch, depth, controller=None, leaf=False):
    hidden, anchor = minimal.forward(model, batch.packed, depth, controller, leaf=leaf)
    return model.lm_head(hidden).float(), anchor
