"""Intervene on one causal source-to-answer readout in native Ouro SDPA."""

from collections.abc import Sequence
import sys

import torch


def rms(value):
    return value.float().square().mean(dim=-1).sqrt()


def pull_by_position(tensor, rows, positions):
    return tensor.transpose(1, 2)[rows, positions]


def read_edges(attention, projected, kwargs, query, sources):
    """Return actual pre-RoPE model projections and reconstructed per-source contributions."""
    if attention.sliding_window is not None or kwargs.get("past_key_value") is not None:
        raise RuntimeError("The source-edge test requires uncached full attention")
    module = sys.modules[type(attention).__module__]
    q, k, v = (projected[tag].unflatten(-1, (-1, attention.head_dim)).transpose(1, 2)
               for tag in ("q", "k", "v"))
    q, k = module.apply_rotary_pos_emb(q, k, *kwargs["position_embeddings"])
    k = module.repeat_kv(k, attention.num_key_value_groups)
    v = module.repeat_kv(v, attention.num_key_value_groups)
    row = torch.arange(len(query), device=query.device)
    sel = pull_by_position(q, row, query).float()
    sequence = torch.arange(k.shape[-2], device=k.device)
    weights = (sel.unsqueeze(-2) @ k.float().transpose(-1, -2)).squeeze(-2) * attention.scaling
    weights = weights.masked_fill(sequence[None, None, :] > query[:, None, None], -torch.inf)
    mask = kwargs.get("attention_mask")
    if mask is not None:
        mask = mask.expand(q.shape[0], q.shape[1], q.shape[-2], k.shape[-2])
        active = pull_by_position(mask, row, query)
        weights = weights.masked_fill(~active, -torch.inf) if active.dtype == torch.bool else weights + active
    weights = weights.softmax(dim=-1)
    actual = (weights.unsqueeze(-2) @ v.float()).squeeze(-2).flatten(1)
    site_weights = weights.transpose(1, 2)[row[:, None, None], sources]
    site_values = v.transpose(1, 2)[row[:, None, None], sources].float()
    return dict(predicted_readout=actual, weights=site_weights, values=site_values)


def contribution(weights, values):
    return (weights.unsqueeze(-1) * values).sum(dim=2).flatten(2)


def matched_deltas(deltas: Sequence[torch.Tensor]):
    """Scale all signed readout patches to the smallest per-recipient RMS."""
    sizes = torch.stack([rms(delta) for delta in deltas])
    common = sizes.min(dim=0).values
    return tuple(delta * (common / size.clamp_min(1e-20)).unsqueeze(-1)
                 for size, delta in zip(sizes, deltas, strict=True))


class EdgeProbe:
    def __init__(self, layer, query, sources, patch=None, record=False):
        self.layer = layer
        self.query = query
        self.sources = sources
        self.patch = patch
        self.record = record
        self.projected = {}
        self.trace = None
        self.applied = None
        self.args = None

    def attach(self, core, turn, group, positions):
        module = core.layers[self.layer].self_attn
        row = torch.arange(len(self.query), device=self.query.device)
        hooks = []

        def module_args(_, __, kwargs):
            if turn() == 4:
                self.args = kwargs

        def projection_hook(tag):
            def capture(_, __, output):
                if turn() == 4:
                    self.projected[tag] = output
            return capture

        def readout(_, args):
            if turn() != 4:
                return None
            tensor = args[0]
            before = tensor[row, self.query].float()
            if self.record:
                self.trace = read_edges(module, self.projected, self.args, self.query, self.sources)
                recovered = self.trace.pop("predicted_readout")
                self.trace["relative_reconstruction_rms"] = rms(recovered - before) / rms(before).clamp_min(1e-20)
                if not torch.isfinite(recovered).all() or bool((self.trace["relative_reconstruction_rms"] > .02).any()):
                    raise RuntimeError("The reconstructed attention readout exceeded 2% native SDPA RMS error")
            if self.patch is not None:
                updated = (before + self.patch).to(tensor.dtype)
                self.applied = rms(updated.float() - before)
                return (tensor.index_put((row, self.query), updated), *args[1:])
            return None

        if self.record:
            hooks.append(module.register_forward_pre_hook(module_args, with_kwargs=True))
            for name in ("q", "k", "v"):
                hooks.append(getattr(module, f"{name}_proj").register_forward_hook(projection_hook(name)))
        hooks.append(module.o_proj.register_forward_pre_hook(readout))
        return hooks

    def audit(self):
        if (self.record and self.trace is None) or (self.patch is not None and self.applied is None):
            raise RuntimeError("The specified source edit or readout probe did not execute")


def path_patches(native, rank, donor, screening):
    # Donor values at the two source spans come from the same matched counterfactual prompt.
    rn = native["values"]
    rr = rank["values"]
    wn, wr = native["weights"], rank["weights"]
    if not torch.equal(rn, rr):
        raise RuntimeError("Rank-1 changed prefix-source values before the first direct edit")
    swap_before = contribution(wn, rn[donor] - rn)
    swap_after = contribution(wr, rn[donor] - rr)
    matched = matched_deltas([swap_before[:, 0], swap_before[:, 1], swap_after[:, 0], swap_after[:, 1]])
    result = dict(zip(("native_target_swap", "native_control_swap",
                       "rank1_target_swap", "rank1_control_swap"), matched, strict=True))
    if screening:
        return result
    restoration = contribution(wn, rn) - contribution(wr, rr)
    target, control = restoration.unbind(dim=1)
    result["rank1_target_restore"] = target
    result["rank1_control_restore"] = control
    normalized = matched_deltas((target, control))
    result["rank1_target_restore_matched"], result["rank1_control_restore_matched"] = normalized
    return result
