"""Measure or edit a frozen Ouro recurrence through its native model entry point."""

from collections.abc import Callable

import torch


def scope_indices(packed, scope):
    tokens, groups, positions, _ = packed
    if scope == "predictors":
        return groups, positions
    if scope == "first_predictor":
        first = torch.ones_like(groups, dtype=torch.bool)
        first[1:] = groups[1:] != groups[:-1]
        return groups[first], positions[first]
    if scope == "all_valid":
        last = positions.new_full((tokens.shape[0],), -1)
        last.scatter_reduce_(0, groups, positions, reduce="amax")
        if bool((last < 0).any()):
            raise ValueError("Every packed candidate must have a scored predictor")
        return (torch.arange(tokens.shape[1], device=tokens.device)[None] <= last[:, None]).nonzero(as_tuple=True)
    raise ValueError(f"Unknown token scope {scope!r}")


def forward_control(
    model, packed, depth, *, anchor_loop=None, anchor_layer=None,
    controller=None, token_scope="predictors", dose=1., after_norm=False,
    edit: Callable | None = None, leaf=False, capture=(), mechanisms=(),
):
    """Use one-based layer/loop anchors. Capture full padded tensors; score the original predictors."""
    inp, groups, positions, _ = packed
    core = model.model
    if depth < 1 or (anchor_loop is not None and anchor_layer is not None):
        raise ValueError("Specify one anchor and a positive total depth")
    if anchor_loop is None and anchor_layer is None:
        if depth == 1:
            anchor_layer = 21
        else:
            anchor_loop = depth - 1
    if anchor_loop is not None and not 1 <= anchor_loop <= depth:
        raise ValueError("The recipient boundary must be inside the executed loops")
    if anchor_layer is not None and (not 1 <= anchor_layer <= core.config.num_hidden_layers or after_norm):
        raise ValueError("A physical-layer anchor is in the first loop and precedes loop normalization")
    if sum((controller is not None, edit is not None, bool(leaf))) > 1:
        raise ValueError("Use one intervention or a gradient leaf")
    row, position = scope_indices(packed, token_scope)
    captures = set(capture)
    paths = {loop: set(names) for loop, names in mechanisms}
    trace = {"pre": {}, "post": {}, "mechanisms": {}, "indices": (row, position)}
    turn = 0

    def apply(tensor):
        selected = tensor[row, position].float()
        trace["anchor"] = selected.detach()
        if controller is not None:
            value = selected + dose * controller.delta(selected)
        elif edit is not None:
            delta = edit(selected)
            if delta.shape != selected.shape:
                raise ValueError(f"Intervention returns shape {delta.shape}, expected {selected.shape}")
            value = selected + delta
        elif leaf:
            value = selected.detach().requires_grad_(True)
            trace["leaf"] = value
        else:
            return tensor
        replacement = value.to(tensor.dtype)
        trace["delta"] = replacement.detach().float() - selected.detach()
        return tensor.index_put((row, position), replacement)

    def norm_pre(_, inputs):
        nonlocal turn
        turn += 1
        h = inputs[0]
        if turn == anchor_loop and not after_norm:
            h = apply(h)
        if turn in captures:
            trace["pre"][turn] = h.detach()
        if h is not inputs[0]:
            return (h, *inputs[1:])

    def norm_post(_, inputs, output):
        h = apply(output) if turn == anchor_loop and after_norm else output
        if turn in captures:
            trace["post"][turn] = h.detach()
        if h is not output:
            return h

    def physical(_, inputs, output):
        if turn == 0:
            return apply(output)

    def record_path(name, layer):
        def hook(_, inputs, output):
            loop = turn + 1
            if name in paths.get(loop, ()):
                value = output[0] if isinstance(output, tuple) else output
                trace["mechanisms"][(loop, layer, name)] = value.detach()
        return hook

    handles = [core.norm.register_forward_pre_hook(norm_pre), core.norm.register_forward_hook(norm_post)]
    if anchor_layer is not None:
        handles.append(core.layers[anchor_layer - 1].register_forward_hook(physical))
    if paths:
        for index, layer in enumerate(core.layers[:core.config.num_hidden_layers], 1):
            handles.append(layer.self_attn.register_forward_hook(record_path("attention", index)))
            handles.append(layer.mlp.register_forward_hook(record_path("mlp", index)))
    original_depth = core.total_ut_steps
    try:
        core.total_ut_steps = depth
        _, states, _ = core(input_ids=inp, use_cache=False)
    finally:
        core.total_ut_steps = original_depth
        for handle in handles:
            handle.remove()
    if turn != depth or len(states) != depth or "anchor" not in trace:
        raise RuntimeError("Native execution did not reach the requested anchor")
    return states[-1][groups, positions], trace
