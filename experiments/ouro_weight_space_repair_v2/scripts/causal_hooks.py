"""Forward-only causal interventions for pinned Ouro; no data/files are read.

Physical layer indices are ZERO based; recurrent loops are ONE based. A loop-only
activation lesion is NOT a shared weight edit. Materialized edits affect ALL loops.
After registration/removal/parameter changes call scorer.invalidate() and rebuild
every cached prefix. Changing capture positions alone does not change computation.
Hooks are token-local and work on every token in the current chunk. Capture indices
are local to that chunk, NOT absolute sequence positions. Empty indices disable
recording (use this during demo cache construction). Not thread-safe.
"""
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class BranchIntervention:
    layer: int
    branch: str                         # 'attention' or 'mlp'
    mode: str = 'project'                # 'project', 'damp', or 'record'
    alpha: float = 0.0                   # removal amount; damp: (1-alpha)*post
    loops: Optional[tuple] = None        # None -> ALL invocations (weight analogue)
    U: Optional[torch.Tensor] = None     # [output_dim, rank], orthonormal columns
    columns: Optional[tuple] = None      # sparse INPUT column support
    preserve_post_norm: bool = False     # match untreated POST-branch RMS norm


@dataclass(frozen=True)
class BoundaryDamping:
    loop: int = 5
    step: float = 1.0                    # entry + step*(native_normalized-entry)


def _validate_basis(U, width):
    if U is None or U.ndim != 2 or U.shape[0] != width or U.shape[1] < 1:
        raise ValueError('U must have shape [output_dim, positive rank]')
    if not torch.isfinite(U).all():
        raise ValueError('Nonfinite basis')
    gram = U.detach().float().T @ U.detach().float()
    if not torch.allclose(gram, torch.eye(U.shape[1], device=gram.device), atol=2e-3, rtol=2e-3):
        raise ValueError('Projector basis must be orthonormal; no implicit fitting')


def _columns(columns, width):
    if columns is None:
        return None
    result = tuple(int(c) for c in columns)
    if not result or len(set(result)) != len(result) or any(c < 0 or c >= width for c in result):
        raise ValueError('Columns must be unique valid input-column indices')
    return result


@torch.no_grad()
def materialize_projection_edit(weight, U, alpha, columns=None):
    """Return new BF16 W and ACTUAL FP32 delta; source tensor is never modified.

    W'=BF16(FP32(W)-alpha*U*(U.T*FP32(W))*D_columns).
    Rank refers to intended FP32 delta; BF16 rounding can increase realized rank.
    Manifest denominator must be supplied separately as total UNIQUE model params.
    """
    if weight.ndim != 2:
        raise ValueError('Expected linear [output,input] weight')
    if weight.dtype != torch.bfloat16:
        raise ValueError('Materialize from the evaluated BF16 source weight; no implicit source recast')
    _validate_basis(U, weight.shape[0])
    columns = _columns(columns, weight.shape[1])
    if not torch.isfinite(torch.tensor(float(alpha))):
        raise ValueError('Nonfinite alpha')
    base = weight.detach().float()
    basis = U.detach().to(device=weight.device, dtype=torch.float32)
    result = base.clone()
    if float(alpha) != 0:
        if columns is None:
            result -= float(alpha) * (basis @ (basis.T @ base))
        else:
            idx = torch.tensor(columns, device=weight.device, dtype=torch.long)
            selected = base.index_select(1, idx)
            result[:, idx] = selected - float(alpha) * (basis @ (basis.T @ selected))
    edited = result.to(torch.bfloat16)
    actual_delta = edited.float() - base
    stats = weight_edit_stats(weight, edited, intended_rank=U.shape[1], columns=columns)
    stats.update(alpha=float(alpha), arithmetic='FP32 subtract then BF16 cast',
                 source_dtype=str(weight.dtype))
    return edited, actual_delta, stats


@torch.no_grad()
def weight_edit_stats(original, edited, total_unique_parameters=None,
                      intended_rank=None, columns=None):
    if original.shape != edited.shape:
        raise ValueError('Weight shape mismatch')
    columns = _columns(columns, original.shape[1])
    delta = edited.detach().float() - original.detach().float()
    changed = int(torch.count_nonzero(delta))
    norm = float(torch.linalg.vector_norm(delta))
    base_norm = float(torch.linalg.vector_norm(original.detach().float()))
    support = original.numel() if columns is None else original.shape[0] * len(columns)
    if total_unique_parameters is not None and total_unique_parameters < original.numel():
        raise ValueError('Invalid model parameter denominator')
    return dict(shape=list(original.shape), intended_rank=intended_rank,
                intended_support_coordinates=support, changed_coordinates=changed,
                unchanged_in_support=support-changed, input_columns=None if columns is None else list(columns),
                matrix_parameters=original.numel(), changed_fraction_of_matrix=changed/original.numel(),
                total_unique_parameters=total_unique_parameters,
                changed_fraction_of_model=None if total_unique_parameters is None else changed/total_unique_parameters,
                delta_frobenius=norm, relative_delta_frobenius=norm/max(base_norm, 1e-30),
                maximum_abs_delta=float(delta.abs().max()), evaluated_dtype=str(edited.dtype))


def _native_rmsnorm(norm, x):
    """Pinned OuroRMSNorm math; no module invocation, hence no recursive hooks."""
    original_dtype = x.dtype
    value = x.float()
    value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    return norm.weight * value.to(original_dtype)


class OuroCausalHooks:
    """Context-managed reversible activation edits and selected-position recorder.

    Example: with OuroCausalHooks(model, [BranchIntervention(3,'mlp',
        mode='project', alpha=.5, U=basis, loops=(5,))]) as hooks:
        scorer.invalidate()  # registration changed model computation
        hooks.set_capture_positions([])  # build edited demo cache
        ...
        hooks.set_capture_positions([0, 2])  # current forward local positions
        ...
        records = hooks.records  # reset at next model.model forward
    scorer.invalidate()  # removal changed model computation

    Set positions separately for each scorer chunk if recording; the scorer's
    internal chunk calls do not automatically translate these local indices.
    """
    def __init__(self, model, branches=(), boundary=None):
        self.model = model
        self.branches = tuple(branches)
        self.boundary = boundary
        self.positions = ()
        self.records = []
        self.current_loop = None
        self._entry = None
        self._pending = {}
        self._stack = None
        self._modules = []
        seen = set()
        for spec in self.branches:
            key = (spec.layer, spec.branch)
            if key in seen:
                raise ValueError('Only one intervention per physical layer/branch')
            seen.add(key)
            if spec.branch not in ('attention', 'mlp') or spec.mode not in ('project', 'damp', 'record'):
                raise ValueError('Invalid branch or intervention mode')
            if not 0 <= spec.layer < len(model.model.layers):
                raise ValueError('Invalid physical layer')
            if not torch.isfinite(torch.tensor(float(spec.alpha))):
                raise ValueError('Nonfinite alpha')
            if spec.loops is not None and (not spec.loops or any(k < 1 for k in spec.loops)):
                raise ValueError('Loops are positive ONE-based indices')
            layer = model.model.layers[spec.layer]
            projection = layer.self_attn.o_proj if spec.branch == 'attention' else layer.mlp.down_proj
            norm = layer.input_layernorm_2 if spec.branch == 'attention' else layer.post_attention_layernorm_2
            _columns(spec.columns, projection.weight.shape[1])
            if spec.mode == 'project':
                _validate_basis(spec.U, projection.weight.shape[0])
            elif spec.columns is not None:
                raise ValueError('Sparse columns only apply to projection interventions')
            self._modules.append((spec, projection, norm))
        if boundary is not None and (boundary.loop < 1 or not torch.isfinite(torch.tensor(float(boundary.step)))):
            raise ValueError('Invalid boundary damping')

    def set_capture_positions(self, positions):
        indices = tuple(int(i) for i in positions)
        if any(i < 0 for i in indices) or len(set(indices)) != len(indices):
            raise ValueError('Capture indices must be unique nonnegative local positions')
        self.positions = indices

    def reset(self):
        """Explicit reset; also automatically called at each model.model forward."""
        self.records = []
        self.current_loop = None
        self._entry = None
        self._pending.clear()

    def _sample(self, x):
        if not self.positions:
            return None
        if max(self.positions) >= x.shape[1]:
            raise ValueError('Capture position outside current chunk; reset positions for each forward')
        idx = torch.tensor(self.positions, device=x.device, dtype=torch.long)
        return x.detach().index_select(1, idx).to(device='cpu', dtype=torch.bfloat16).clone()

    def _active(self, spec):
        return spec.loops is None or self.current_loop in spec.loops

    @torch.no_grad()
    def _before_layer(self, index, module, args, kwargs):
        if 'current_ut' not in kwargs:
            raise RuntimeError('Pinned Ouro current_ut keyword is required')
        self.current_loop = int(kwargs['current_ut']) + 1
        if index == 0:
            entry = args[0] if args else kwargs['hidden_states']
            if self.boundary is not None and self.current_loop == self.boundary.loop and self.boundary.step != 1:
                self._entry = entry  # no clone: pinned layers are out-of-place
            if self.positions:
                self.records.append(dict(kind='loop_entry', loop=self.current_loop,
                    positions=list(self.positions), normalized_entry=self._sample(entry)))

    @torch.no_grad()
    def _raw_hook(self, spec, module, args, output):
        active = self._active(spec)
        changed = active and spec.mode == 'project' and spec.alpha != 0
        treated = output
        if changed:
            basis = spec.U.detach().to(device=output.device, dtype=torch.float32)
            if spec.columns is None:
                component_source = output.float()
            else:
                idx = torch.tensor(spec.columns, device=output.device, dtype=torch.long)
                component_source = args[0].float().index_select(-1, idx) @ module.weight.detach().float().index_select(1, idx).T
            treated = (output.float() - float(spec.alpha) * ((component_source @ basis) @ basis.T)).to(output.dtype)
        # Keep original tensor reference only until following branch norm. No clones
        # of full-prefix outputs are retained; selected snapshots alone go to CPU.
        if self.positions or (active and spec.preserve_post_norm and spec.alpha != 0 and spec.mode != 'record'):
            self._pending[(spec.layer, spec.branch)] = (output,
                self._sample(output), self._sample(treated))
        return treated if changed else None  # EXACT no-op: no casts/reconstruction

    @torch.no_grad()
    def _post_hook(self, spec, module, args, output):
        active = self._active(spec)
        changed = active and spec.mode != 'record' and spec.alpha != 0
        treated = output
        if changed and spec.mode == 'damp':
            treated = (output.float() * (1-float(spec.alpha))).to(output.dtype)
        pending = self._pending.pop((spec.layer, spec.branch), None)
        original = output
        if pending is not None and active and spec.mode == 'project' and spec.alpha != 0:
            original = _native_rmsnorm(module, pending[0])
        if changed and spec.preserve_post_norm:
            target_norm = original.float().norm(dim=-1, keepdim=True)
            current_norm = treated.float().norm(dim=-1, keepdim=True)
            if torch.any((current_norm == 0) & (target_norm != 0)):
                raise RuntimeError('Cannot norm-match a zero branch; choose weaker intervention')
            scale = torch.where(current_norm > 0, target_norm / current_norm.clamp_min(1e-30), torch.ones_like(current_norm))
            treated = (treated.float() * scale).to(output.dtype)
        if self.positions:
            self.records.append(dict(kind='branch', layer=spec.layer, branch=spec.branch,
                loop=self.current_loop, active=active, mode=spec.mode, alpha=float(spec.alpha),
                preserve_post_norm=spec.preserve_post_norm, positions=list(self.positions),
                raw_original=pending[1], raw_treated=pending[2],
                post_original=self._sample(original), post_treated=self._sample(treated)))
        return treated if changed else None

    @torch.no_grad()
    def _boundary_hook(self, module, args, output):
        treated = output
        changed = self.boundary is not None and self.current_loop == self.boundary.loop and self.boundary.step != 1
        if changed:
            if self._entry is None or self._entry.shape != output.shape:
                raise RuntimeError('Missing actual same-chunk loop-entry state')
            if self.boundary.step == 0:
                treated = self._entry  # exact prior boundary, no needless rounding
            else:
                treated = (self._entry.float() + float(self.boundary.step) * (output.float()-self._entry.float())).to(output.dtype)
        if self.positions:
            self.records.append(dict(kind='boundary', loop=self.current_loop,
                positions=list(self.positions), raw=self._sample(args[0]),
                normalized_native=self._sample(output), normalized_treated=self._sample(treated)))
        self._entry = None
        return treated if changed else None

    def __enter__(self):
        if self._stack is not None:
            raise RuntimeError('Context already active')
        if any(m.training for m in self.model.modules()):
            raise RuntimeError('Require model.eval() before registering interventions')
        stack = ExitStack()
        try:
            stack.enter_context(torch.no_grad())
            def reset_hook(module, args, kwargs):
                self.reset()
            handle = self.model.model.register_forward_pre_hook(reset_hook, with_kwargs=True)
            stack.callback(handle.remove)
            for index, layer in enumerate(self.model.model.layers):
                def pre(module, args, kwargs, index=index):
                    return self._before_layer(index, module, args, kwargs)
                handle = layer.register_forward_pre_hook(pre, with_kwargs=True)
                stack.callback(handle.remove)
            for spec, projection, norm in self._modules:
                def raw(module, args, output, spec=spec):
                    return self._raw_hook(spec, module, args, output)
                def post(module, args, output, spec=spec):
                    return self._post_hook(spec, module, args, output)
                stack.callback(projection.register_forward_hook(raw).remove)
                stack.callback(norm.register_forward_hook(post).remove)
            stack.callback(self.model.model.norm.register_forward_hook(self._boundary_hook).remove)
            self._stack = stack
            return self
        except BaseException:
            stack.close()
            self.reset()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        stack, self._stack = self._stack, None
        try:
            return stack.__exit__(exc_type, exc_value, traceback)
        finally:
            self._pending.clear()
            self._entry = None
