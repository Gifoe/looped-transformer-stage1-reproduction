"""Native Q/K and V replacement with verified routing diagnostics for Experiment C."""

from pathlib import Path
import sys

import numpy as np
import torch

from c_common import (
    METRICS, bootstrap, candidate_trace, digest, forward, hash_key, load, make_manifest,
    paired, population, profile, replay_batches, rms, save, score,
)
from experiment_c import BANDS, PRIMARY, band_keys, condition_table, ensure_matched, grouped_traces, trace, updates

PROJECTIONS = ("q", "k", "v")
PATHS = {"qk": ("q", "k"), "v": ("v",)}
ROUTING_TOLERANCE = .02


def reading(attention, activations, kwargs, group, position):
    """Reconstruct causal SDPA routing in float32 at the audited predictor rows."""
    head = attention.head_dim
    module = sys.modules[type(attention).__module__]
    q, k, v = [activations[name].unflatten(-1, (-1, head)).transpose(1, 2) for name in PROJECTIONS]
    cos, sin = kwargs["position_embeddings"]
    q, k = module.apply_rotary_pos_emb(q, k, cos, sin)
    k = module.repeat_kv(k, attention.num_key_value_groups)
    v_rep = module.repeat_kv(v, attention.num_key_value_groups)
    if attention.sliding_window is not None or kwargs.get("past_key_value") is not None:
        raise RuntimeError("This routing reconstruction requires uncached full attention")
    length = k.shape[-2]
    columns = torch.arange(length, device=position.device)
    pre = ((q.float() @ k.float().transpose(-1, -2)) * attention.scaling).transpose(1, 2)
    actual = pre[group, position].masked_fill(columns[None, None, :] > position[:, None, None], -torch.inf)
    mask = kwargs.get("attention_mask")
    if mask is not None:
        if mask.ndim != 4:
            raise RuntimeError("Expected an SDPA attention mask with four dimensions")
        expanded = mask.expand(q.shape[0], q.shape[1], q.shape[-2], length)
        selected = expanded.transpose(1, 2)[group, position]
        actual = actual.masked_fill(~selected, -torch.inf) if selected.dtype == torch.bool else actual + selected
    weights = actual.softmax(dim=-1)
    # Keep the value reduction within the original candidate batch and avoid a repeated KV cache.
    padded = torch.zeros_like(pre)
    padded[group, position] = weights
    predicted = (padded.transpose(1, 2) @ v_rep.float()).transpose(1, 2)[group, position].flatten(1)
    return dict(q=q.transpose(1, 2)[group, position].flatten(1),
                k=k.transpose(1, 2)[group, position].flatten(1),
                v=v.transpose(1, 2)[group, position].flatten(1), weights=weights, arm=predicted)


class AttentionProbe:
    def __init__(self, layers, diagnostic_layer, bank=None, clamp=()):
        self.layers = layers
        self.diagnostic_layer = diagnostic_layer
        self.exports = {}
        self.bank = bank
        self.clamp = set(clamp)
        self.applied = set()
        self.input = {}
        self.trace = None

    def projection_hook(self, layer, name, turn):
        def hook(_, __, tensor):
            if turn() != 4:
                return None
            key = (layer, name)
            if key in self.clamp:
                replacement = self.bank[key]
                self.applied.add(key)
                self.exports[key] = replacement
                return replacement
            self.exports[key] = tensor
            return None
        return hook

    def attach(self, core, turn, group, position):
        handles = []
        for layer in self.layers:
            attention = core.layers[layer].self_attn
            for name in PROJECTIONS:
                handles.append(getattr(attention, f"{name}_proj").register_forward_hook(
                    self.projection_hook(layer, name, turn)))
        if self.diagnostic_layer is None:
            return handles
        layer = self.diagnostic_layer
        attention = core.layers[layer].self_attn

        def capture(_, __, kwargs):
            if turn() == 4:
                self.input = {key: kwargs.get(key) for key in (
                    "position_embeddings", "attention_mask", "past_key_value")}

        def compare(_, inputs):
            if turn() != 4:
                return
            active = {name: self.exports[(layer, name)] for name in PROJECTIONS}
            derived = reading(attention, active, self.input, group, position)
            derived["exact_readout"] = inputs[0][group, position].float()
            derived["reconstruction_relative_rms"] = (
                rms(derived.pop("arm") - derived["exact_readout"]) /
                rms(derived["exact_readout"]).clamp_min(1e-20))
            self.trace = derived

        handles.append(attention.register_forward_pre_hook(capture, with_kwargs=True))
        handles.append(attention.o_proj.register_forward_pre_hook(compare))
        return handles

    def audit(self):
        if self.clamp != self.applied:
            raise RuntimeError("A selected path replacement did not execute")
        if self.diagnostic_layer is not None and self.trace is None:
            raise RuntimeError("Missing target attention trace")


def routing_metrics(current, native, packed):
    if current is None or native is None:
        raise RuntimeError("Routing microscope did not record its prespecified layer")
    _, group, position, _ = packed
    columns = torch.arange(current["weights"].shape[-1], device=position.device)
    first = torch.full((packed[0].shape[0],), len(columns), device=position.device, dtype=position.dtype)
    first.scatter_reduce_(0, group, position, reduce="amin", include_self=True)
    before_edit = columns[None, :] < first[group, None]
    weights = current["weights"]
    original = native["weights"]
    change = weights - original
    shift = change.mean(dim=1)
    actual_delta = current["exact_readout"] - native["exact_readout"]
    self_mask = columns[None, :] == position[:, None]
    checked_error = torch.maximum(current["reconstruction_relative_rms"], native["reconstruction_relative_rms"])
    return dict(
        actual_q_postrope_delta_rms=rms(current["q"].float() - native["q"].float()),
        actual_k_postrope_delta_rms=rms(current["k"].float() - native["k"].float()),
        actual_v_delta_rms=rms(current["v"].float() - native["v"].float()),
        actual_pre_o_projection_readout_delta_rms=rms(actual_delta),
        routing_reconstruction_relative_rms=checked_error,
        routing_reconstruction_pass=(checked_error <= ROUTING_TOLERANCE).float(),
        routing_total_variation=change.abs().sum(dim=-1).mean(dim=-1) * .5,
        top_read_position_switched_fraction=(weights.argmax(dim=-1) != original.argmax(dim=-1)).float().mean(dim=-1),
        routing_prefix_mass_shift=(shift * before_edit).sum(-1),
        routing_self_mass_shift=(shift * self_mask).sum(-1),
        routing_expected_read_distance_shift=(shift * (position[:, None] - columns)).sum(-1),
    )


def native_manipulation_parity(model, packed, edit, base, native, attention_band, layers, current):
    all_qkv = AttentionProbe(layers, None, native.exports,
                             {(layer, name) for layer in layers for name in PROJECTIONS})
    replaced, active = forward(model, packed, 3, edit, attention_probe=all_qkv)
    all_qkv.audit()
    ensure_matched(active, base, None)
    expected, _ = forward(model, packed, 3, edit, branch_layers=attention_band, reference=base,
                          clamp_layers=attention_band)
    if not torch.equal(expected, replaced):
        raise RuntimeError("Joint native QKV replacement differed from additive attention clamping")
    restore = AttentionProbe(layers, None, current.exports,
                             {(layer, name) for layer in layers for name in PROJECTIONS})
    restored, _ = forward(model, packed, 3, edit, attention_probe=restore)
    restore.audit()
    return restored


def observe_batch(archive, model, axis, rotated, args, band, mlp_band, packed, subset, audit):
    start, stop = BANDS[band]
    layers = range(start, stop)
    diagnostic = (start + stop - 1) // 2
    attention_band = band_keys("attention", band)
    branches = attention_band | band_keys("mlp", mlp_band)
    native = AttentionProbe(layers, diagnostic)
    hidden, base = forward(model, packed, branch_layers=branches, attention_probe=native)
    native.audit()
    scored = {"native": score(archive, model, packed, subset, hidden)}
    traces = {}
    for direction, edit in updates(base, axis, rotated, args.dose).items():
        observed = AttentionProbe(layers, diagnostic)
        hidden, state = forward(model, packed, 3, edit, branch_layers=branches, reference=base,
                                trace_turns=(4,), attention_probe=observed)
        observed.audit()
        ensure_matched(state, base, None)
        name = f"{direction}_live"
        scored[name] = score(archive, model, packed, subset, hidden)
        traces[name] = candidate_trace(packed, len(subset), dict(
            **trace(state, base), **routing_metrics(observed.trace, native.trace, packed)))
        if audit and not torch.equal(hidden, native_manipulation_parity(
                model, packed, edit, base, native, attention_band, layers, observed)):
            raise RuntimeError("Joint exact live QKV restoration differed from the untouched live run")
        for path, selected in PATHS.items():
            probe = AttentionProbe(layers, diagnostic, native.exports,
                                   {(layer, name) for layer in layers for name in selected})
            hidden, patched = forward(model, packed, 3, edit, branch_layers=branches, reference=base,
                                      trace_turns=(4,), attention_probe=probe)
            probe.audit()
            ensure_matched(patched, base, state)
            name = f"{direction}_{path}_native"
            scored[name] = score(archive, model, packed, subset, hidden)
            traces[name] = candidate_trace(packed, len(subset), dict(
                **trace(patched, base), **routing_metrics(probe.trace, native.trace, packed)))
    return dict(scores=scored, traces=traces)


def compare_paths(info, sample):
    result = {}
    for path in PATHS:
        metrics = {}
        for metric in METRICS:
            a, b = (profile(info, name, metric) for name in ("learned_live", f"learned_{path}_native"))
            c, d = (profile(info, name, metric) for name in ("rotated_live", f"rotated_{path}_native"))
            difference = (a - b) - (c - d)
            metrics[metric] = dict(difference=paired(difference, sample),
                                   adjusted=paired(difference, sample, .05 / (len(PATHS) * len(METRICS))),
                                   learned_live_minus_clamp=paired(a - b, sample),
                                   rotated_live_minus_clamp=paired(c - d, sample))
        result[path] = metrics
    return result


def run_paths(args, archive, model, tokenizer, axis, rotated, metadata):
    lock_path = args.out / "LOCKED_LOCATIONS.json"
    confirmation_path = args.out / "validation" / "C_SUMMARY.json"
    lock = load(lock_path)
    confirm = load(confirmation_path)
    if lock["direction_sha256"] != metadata["direction_sha256"] or lock["dose"] != args.dose:
        raise RuntimeError("QKV path implementation requires the factorial experiment's locked direction and dose")
    band = lock["locations"]["attention"]
    summary = confirm["interaction"][f"attention/{band}"][PRIMARY]
    support = summary["adjusted"]["lower"] > 0
    items, info = population(archive, tokenizer, args.corpus, "validation")
    if len(info) != 299:
        raise RuntimeError("The projection path experiment requires all 299 ARC Validation questions")
    out = args.out / "paths"
    start, stop = BANDS[band]
    make_manifest(out, dict(**metadata, locked_locations_sha256=digest(lock_path),
                            validation_summary_sha256=digest(confirmation_path), attention_band=band,
                            confirmatory_branch_support=support, qkv_code_sha256=digest(__file__),
                            question_id_sha256=hash_key("questions", "/".join(row["id"] for row in info)),
                            parquet_sha256=digest(args.corpus / "validation.parquet"),
                            diagnostic_layer_zero_based=(start + stop - 1) // 2,
                            path_scope="Replace selected native pre-RoPE projections across the locked band, all sequence positions",
                            routing_tolerance=ROUTING_TOLERANCE))
    first = items[0]
    replay_batches(archive, items, info, args.batch_size, out,
                   lambda packed, subset: observe_batch(archive, model, axis, rotated, args, band,
                                                        lock["locations"]["mlp"], packed, subset,
                                                        subset[0] is first))
    sample = bootstrap(info)
    tags = list(info[0]["scores"])
    coverage = {tag: float(np.mean([np.mean(row["trace"][tag]["routing_reconstruction_pass"])
                                    for row in info])) for tag in tags if tag != "native"}
    save(out / "C_PATHS_SUMMARY.json", dict(questions=len(info), candidates=len(items),
        attention_band=band, diagnostic_layer=(start + stop - 1) // 2,
        confirmatory_branch_support=support, conditions=condition_table(info, sample, tags),
        path_interaction=compare_paths(info, sample), grouped_responses=grouped_traces(info, tags),
        reconstruction_coverage=coverage, reconstruction_tolerance=ROUTING_TOLERANCE,
        reconstruction_measurements_are_conditional_on_exact_sdpa_readout_parity=all(
            rate == 1. for rate in coverage.values()),
        interpretation="Controlled replacement effects of all affected projections in one fixed band. "
            "The routing microscope samples a fixed middle layer and tested predictor positions. "
            "Interactions and branch sign effects need not add and do not identify a mediated fraction or a semantic head. "
            "If routing parity coverage is below 100%, metrics from the offline reconstruction are descriptive only."))
    print("C projection-path complete", f"band={band}", f"branch_support={support}", flush=True)
