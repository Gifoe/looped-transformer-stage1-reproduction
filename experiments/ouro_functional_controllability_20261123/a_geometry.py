"""Low-rank geometry and reproducible question-disjoint amplitude permutations."""

from collections import defaultdict
import hashlib
from itertools import combinations

import numpy as np
import torch


SEED = 261123903
EPS = 1e-20


def rms(x):
    return x.square().mean(dim=-1, keepdim=True).sqrt()


def unit_rms(x):
    return x / rms(x).clamp_min(EPS)


def capture(model, packed, depths):
    """Capture native, pre-edit anchors for multiple depths in one causal pass."""
    inp, group, positions, _ = packed
    core = model.model
    core.total_ut_steps = max(1, max(depths) - 1)
    anchors, hooks = {}, []
    turn = 0

    def at_layer(_, __, tensor):
        if turn == 0:
            anchors[1] = tensor[group, positions]

    def at_norm(_, args):
        nonlocal turn
        turn += 1
        if turn + 1 in depths:
            anchors[turn + 1] = args[0][group, positions]

    if 1 in depths:
        hooks.append(core.layers[20].register_forward_hook(at_layer))
    hooks.append(core.norm.register_forward_pre_hook(at_norm))
    try:
        _, states, _ = core(input_ids=inp, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    if turn != len(states) or turn != core.total_ut_steps or anchors.keys() != set(depths):
        raise RuntimeError(f"Missing an Ouro compute anchor: {anchors.keys()}, {depths}, {turn}")
    return anchors


class Geometry:
    """Keep orthonormal U coordinates and exact out-of-subspace energy per token."""

    def __init__(self, net, axis, dose):
        self.net = net
        self.axis = axis
        self.dose = dose
        self.basis = torch.linalg.qr(net.out.weight.detach().double(), mode="reduced")[0]

    def measure(self, native):
        h = native.float()
        raw = self.net.out(self.net.encode(h))
        bounded = self.net.delta(h) * self.dose
        actual = (h + bounded).to(native.dtype).float() - h
        scale = rms(h).clamp_min(EPS)
        dose_columns = [rms(raw) / scale, rms(bounded) / scale, rms(actual) / scale]
        amplitude_columns = [(edit @ self.axis)[:, None] / h.shape[-1] / scale
                             for edit in (raw, bounded, actual)]
        scalars = torch.cat([scale, rms(raw), rms(bounded), rms(actual), *dose_columns,
                             *amplitude_columns], dim=-1)
        edits = torch.stack([raw, bounded, actual]).double()
        coords = edits @ self.basis
        off = edits[2] - coords[2] @ self.basis.T
        # This retains only the actual norm and one vector per batch for centered
        # spectral bounds. The unrounded edits are mathematically confined to U.
        return dict(coords=coords.cpu().numpy().transpose(1, 0, 2),
                    scalars=scalars.cpu().numpy(), cast_off=off.square().sum(-1).cpu().numpy(),
                    cast_off_vector=off.float().cpu().numpy())


def buckets_seed(tag):
    return int.from_bytes(hashlib.sha256(f"{SEED}/{tag}".encode()).digest()[:8], "little")


def shuffled_ratios(keys, values, tag):
    """Derange donor slots within answer offset, preserving each marginal exactly."""
    by_slot = defaultdict(lambda: defaultdict(list))
    for i, (question, choice, offset) in enumerate(keys):
        by_slot[int(offset)][int(question)].append(i)
    rng = np.random.default_rng(buckets_seed(tag))
    assigned = np.arange(len(keys))
    for offset, groups in sorted(by_slot.items()):
        # The last prompt token occurs before the choices diverge. Use one donor
        # per question at that position and write that same edit for all choices.
        cluster = list(groups.values())
        ordered = [cluster[i] for i in rng.permutation(len(cluster))]
        if offset == 0:
            for recipient, donor in zip(ordered, ordered[1:] + ordered[:1], strict=True):
                assigned[recipient] = donor[0]
            continue
        flat = np.concatenate(ordered)
        largest = max(map(len, ordered))
        if 2 * largest <= len(flat):
            shift = rng.integers(largest, len(flat) - largest + 1)
        else:
            # An imbalanced stratum cannot derange every slot. This rotation
            # minimizes same-question donors while preserving its marginals.
            shift = largest
        assigned[flat] = np.roll(flat, int(shift))
    return values[assigned], assigned


def stats(x):
    if not len(x):
        return None
    return dict(mean=float(np.mean(x)), quantiles=np.quantile(x, [0, .05, .25, .5, .75, .95, 1]).tolist())


def spectrum(coords, centered, extra=0.):
    matrix = coords - coords.mean(axis=0) if centered else coords
    eigenvalues, eigvecs = np.linalg.eigh(matrix.T @ matrix)
    eigenvalues = np.maximum(eigenvalues[::-1], 0)
    energy = float(eigenvalues.sum() + extra)
    if energy <= EPS:
        return dict(energy=energy, effective_rank=0, energy_fraction=[], axis=None)
    ratios = eigenvalues / energy
    within = eigenvalues / eigenvalues.sum().clip(EPS)
    nonzero = within > 0
    result = dict(energy=energy, effective_rank=float(np.exp(-np.sum(within[nonzero] * np.log(within[nonzero])))),
                  energy_fraction=ratios.tolist(), axis=eigvecs[:, -1].tolist())
    if extra:
        result.update(outside_subspace_energy_fraction=float(extra / energy),
                      full_top1_fraction_interval=[float(ratios[0]), float(min(1., ratios[0] + extra / energy))])
    return result


def cosines(coords, keys):
    """Exact pairwise mean cosines from grouped sums, excluding zero edits."""
    norms = np.linalg.norm(coords, axis=1)
    valid = norms > EPS
    unit = coords[valid] / norms[valid, None]
    keys = keys[valid]
    if len(keys) < 2:
        return {}

    def pair_energy(groups):
        _, inverse, counts = np.unique(groups, axis=0, return_inverse=True, return_counts=True)
        pooled = np.zeros((len(counts), unit.shape[1]))
        np.add.at(pooled, inverse, unit)
        return float(np.square(pooled).sum()), int(np.square(counts.astype(np.int64)).sum())

    across_q = pair_energy(keys[:, :1])
    same_slot = pair_energy(keys[:, [0, 2]])
    n = len(unit)
    differences = [("cross_question", (float(np.square(unit.sum(axis=0)).sum()), n*n), across_q),
                   ("cross_token_within_question", across_q, same_slot)]
    return {name: dict(mean_cosine=float(np.clip((large[0]-small[0]) / (large[1]-small[1]), -1, 1)),
                       pairs=(large[1]-small[1]) // 2)
            for name, large, small in differences if large[1] > small[1]}


def geometry_summary(values, mask, bound):
    coord = values["coords"][mask]
    if not len(coord):
        return dict(n=0)
    scalars = values["scalars"][mask]
    keys = values["keys"][mask]
    result = dict(n=len(coord), questions=len(np.unique(keys[:, 0])),
                  rms_native_state=stats(scalars[:, 0]),
                  clip_fraction=float(np.mean(scalars[:, 4] > bound)), updates={})
    for i, name in enumerate(("raw", "clipped_before_cast", "actual_after_cast")):
        data = coord[:, i]
        norms = np.linalg.norm(data, axis=1)
        off = values["cast_off"][mask] if i == 2 else np.zeros(len(data))
        norm_full = np.sqrt(norms**2 + off)
        active = norm_full > EPS
        denom = np.maximum(norm_full, EPS)
        unit = data / denom[:, None]
        signed = scalars[:, 7+i]
        cosine = signed[active] / scalars[active, 4+i]
        spectra = {}
        for normalized, coords_here in ((False, data), (True, unit)):
            group = "unit_direction" if normalized else "edit_vector"
            extra = off / denom**2 if normalized else off
            offset = values["cast_off_vector"][mask] if i == 2 else None
            chart = {}
            for centered in (False, True):
                extra_energy = float(extra.sum())
                if centered and offset is not None:
                    center = np.mean(offset / denom[:, None], axis=0) if normalized else offset.mean(axis=0, dtype=np.float64)
                    extra_energy = max(0., extra_energy - len(data) * float(np.square(center).sum()))
                chart["centered" if centered else "uncentered"] = spectrum(coords_here, centered, extra_energy)
            spectra[group] = chart
        result["updates"][name] = dict(nonzero=int(active.sum()), rms_update=stats(scalars[:, 1+i]),
                                        rms_relative_to_state=stats(scalars[:, 4+i]), signed_amplitude_ratio=stats(signed),
                                        sign_positive=int((signed > 0).sum()), sign_negative=int((signed < 0).sum()),
                                        sign_zero=int((signed == 0).sum()), cosine_to_train_axis=stats(cosine), spectra=spectra,
                                        clipped_space_pair_cosines=cosines(data, keys))
    return result


def across_depth(geometries, category):
    summaries = {}
    for ka, kb in combinations(sorted(geometries), 2):
        a, b = geometries[ka], geometries[kb]
        ma, mb = position_masks(a["keys"], category), position_masks(b["keys"], category)
        ai = {(tuple(row), digest): i for i, (row, digest) in enumerate(zip(a["keys"], a["prefix_hash"], strict=True)) if ma[i]}
        pairs = [(ai[(tuple(row), digest)], j) for j, (row, digest) in enumerate(
            zip(b["keys"], b["prefix_hash"], strict=True)) if mb[j] and (tuple(row), digest) in ai]
        if not pairs:
            continue
        ix, iy = np.asarray(pairs).T
        x, y = a["coords"][ix, 1], b["coords"][iy, 1]
        denom = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
        valid = denom > EPS
        dot = (x[valid] * y[valid]).sum(axis=-1) / denom[valid]
        summaries[f"k{ka}_k{kb}"] = dict(pairs=int(valid.sum()), cosine=stats(dot.clip(-1, 1)))
    return summaries


def position_masks(keys, category):
    offset = keys[:, 2]
    if category == "last_prompt":
        return (keys[:, 1] == 0) & (offset == 0)
    if category == "continuation":
        return offset > 0
    if category == "all":
        return np.ones(len(keys), dtype=bool)
    return offset == int(category.removeprefix("offset_"))
