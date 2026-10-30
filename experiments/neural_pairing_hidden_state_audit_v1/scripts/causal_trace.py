"""Teacher-forced, same-checkpoint instrumentation of upstream Pair.

Never enables Pair.record: original fused attention remains the forward path.
Use one context per joint pass. Donor tensors must have identical shapes/timing;
the caller is responsible for checking token/tick alignment before supplying them.
This module does not implement or train a new communication architecture.
"""
import math
import types

import torch


class CausalTrace:
    def __init__(self, pair, positions, *, keep_receivers=None, keep_blocks=None,
                 scale=1.0, sender_donors=None, post_donors=None, capture_full=False, capture=True,
                 sender_kv_donors=None, sender_gate_donors=None, drop_writes=None, write_norms=None):
        self.pair, self.positions = pair, positions
        self.keep_receivers = set(keep_receivers) if keep_receivers is not None else set(pair.models)
        self.keep_blocks = set(keep_blocks) if keep_blocks is not None else set(pair.xblocks)
        self.scale = float(scale)
        self.sender_donors, self.post_donors = sender_donors or {}, post_donors or {}
        self.sender_kv_donors, self.sender_gate_donors = sender_kv_donors or {}, sender_gate_donors or {}
        self.drop_writes, self.write_norms = set(drop_writes or []), write_norms or {}
        self.capture_full = capture_full
        self.capture = capture
        self.states, self.full_pre, self.full_post, self.stats, self.full_write_norms = {}, {}, {}, [], {}
        self._restore, self._hooks = [], []

    def select(self, x, value):
        p = self.positions[x].to(value.device)
        assert p.ndim == 2 and p.shape[0] == value.shape[0]
        return value[torch.arange(value.shape[0], device=value.device)[:, None], p]

    def save(self, x, layer, kind, value):
        if not self.capture:
            return
        self.states[(x, layer, kind)] = self.select(x, value).detach().float().cpu()

    def replace(self, obj, name, method):
        # Restore the original object attribute, including absence of an instance override.
        had = name in obj.__dict__
        old = obj.__dict__.get(name)
        setattr(obj, name, types.MethodType(method, obj))
        self._restore.append((obj, name, had, old))

    def __enter__(self):
        import link
        assert self.pair.record is None, 'Pair.record would replace fused attention'
        assert not hasattr(self.pair, '_audit_trace_active'), 'nested trace contexts are unsupported'
        self.pair._audit_trace_active = True
        self._original_exchange = self.pair._exchange
        tracer = self

        def exchange(pair, k, h, qmask, mode, gen, write_mode=None, allow=None, ages=None, cache=None, tick=None):
            if cache is not None:
                raise ValueError('this tracer is teacher-forced only; generation requires a separate aligned cache audit')
            layer = pair.xblocks[k]
            for x, value in h.items():
                tracer.save(x, layer, 'pre', value)
                if tracer.capture_full:
                    tracer.full_pre[(x, layer)] = value.detach().cpu().clone()
            out, gates = tracer._original_exchange(k, h, qmask, mode, gen, write_mode, allow, ages, cache, tick)
            for x in out:
                donor = tracer.post_donors.get((x, layer))
                if donor is not None:
                    assert donor.shape == out[x].shape, 'post-state donor shape mismatch'
                    # Do not patch inactive/padded query positions.
                    out[x] = torch.where(qmask[x][..., None].bool(), donor.to(out[x]), out[x])
                tracer.save(x, layer, 'post', out[x])
                tracer.save(x, layer, 'effective_write', out[x].float() - h[x].float())
                if tracer.capture_full:
                    tracer.full_post[(x, layer)] = out[x].detach().cpu().clone()
            return out, gates

        self.replace(self.pair, '_exchange', exchange)
        for x, side in self.pair.bridge.items():
            original_show, original_read = side.show, side.read

            def show(side, k, h, src=None, rec=None, *, _x=x, _original=original_show):
                layer = tracer.pair.xblocks[k]
                donor = tracer.sender_donors.get((_x, layer))
                if donor is not None:
                    assert src is None, 'sender patch and randomized sender states are separate interventions'
                    assert donor.shape == h.shape, 'sender donor shape mismatch'
                    h = donor.to(h)
                kv_donor = tracer.sender_kv_donors.get((_x, layer))
                if kv_donor is not None:
                    assert donor is None and src is None
                    assert kv_donor.shape == h.shape
                    src = kv_donor.to(h)
                keys, vals, gates = _original(k, h, src, rec)
                gate_donor = tracer.sender_gate_donors.get((_x, layer))
                if gate_donor is not None:
                    assert donor is None and gate_donor.shape == h.shape
                    gh = gate_donor.to(h).float()
                    if side.gate_norm:
                        gh = torch.nn.functional.rms_norm(gh, (gh.shape[-1],), eps=1e-6)
                    gates = torch.sigmoid(side.gate[k](gh).squeeze(-1))
                    z = side.kv_norm[k]((h if src is None else src).float())
                    vals = side.split(side.v[k](z) * gates[..., None])
                tracer.save(_x, layer, 'shown_k', keys.transpose(1, 2).flatten(-2))
                tracer.save(_x, layer, 'shown_v', vals.transpose(1, 2).flatten(-2))
                tracer.save(_x, layer, 'sender_gate', gates[..., None])
                return keys, vals, gates

            def read(side, k, h, keys, vals, allow, ages, qmask, rec=None, *, _x=x, _original=original_read):
                layer = tracer.pair.xblocks[k]
                raw = _original(k, h, keys, vals, allow, ages, qmask, rec)
                if tracer.capture_full:
                    tracer.full_write_norms[(_x, layer)] = raw.detach().float().norm(dim=-1, keepdim=True).cpu()
                tracer.save(_x, layer, 'raw_write_fp32', raw)
                enabled = _x in tracer.keep_receivers and layer in tracer.keep_blocks and (_x, layer) not in tracer.drop_writes
                actual = raw if enabled and tracer.scale == 1 else raw * (tracer.scale if enabled else 0)
                norm = tracer.write_norms.get((_x, layer))
                if norm is not None:
                    actual = actual * (norm.to(actual) / actual.norm(dim=-1, keepdim=True).clamp_min(1e-12))
                tracer.save(_x, layer, 'applied_write_fp32', actual)
                if not tracer.capture:
                    return actual
                # Read-only attention diagnostic, selected query positions, never substituted into actual.
                q = side.split(side.q[k](side.q_norm[k](tracer.select(_x, h).float())))
                p = tracer.positions[_x].to(h.device)
                rows = torch.arange(h.shape[0], device=h.device)[:, None]
                ag, a = ages[rows, p], allow[rows, p]
                nk = side.null[k][None, :, None, :].expand(h.shape[0], -1, 1, -1)
                all_keys = torch.cat([nk, keys], 2)
                bias = side.age[k][:, ag].transpose(0, 1)
                bias = torch.cat([bias.new_zeros(*bias.shape[:-1], 1), bias], -1)
                a = torch.cat([a.new_ones(*a.shape[:-1], 1), a], -1)[:, None]
                prob = (q @ all_keys.transpose(-1, -2) / math.sqrt(side.head_dim)
                        + bias.masked_fill(~a, float('-inf'))).softmax(-1)
                tracer.states[(_x, layer, 'attention')] = prob.detach().cpu()
                active = tracer.select(_x, qmask[..., None]).bool().squeeze(-1)
                wr = tracer.select(_x, actual).float().norm(dim=-1)
                hr = tracer.select(_x, h).float().norm(dim=-1)
                for i, j in active.nonzero().tolist():
                    tracer.stats.append(dict(receiver=_x, layer=layer, row=i, selected_position=j,
                        token_position=int(p[i, j]), alpha=float(torch.tanh(side.alpha[k]).detach()),
                        write_l2=float(wr[i, j]), hidden_l2=float(hr[i, j]),
                        write_hidden_ratio=float(wr[i, j] / hr[i, j].clamp_min(1e-12)),
                        null_attention_mass=float(prob[i, :, j, 0].mean()),
                        retained=enabled, scale=tracer.scale))
                return actual

            self.replace(side, 'show', show)
            self.replace(side, 'read', read)
        for x, model in (self.pair.models.items() if self.capture else []):
            for layer, block in enumerate(link.find_layers(model)):
                def hook(module, args, out, *, _x=x, _layer=layer):
                    tracer.save(_x, _layer, 'native_block_out', out[0] if isinstance(out, tuple) else out)
                self._hooks.append(block.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for hook in self._hooks:
            hook.remove()
        for obj, name, had, value in reversed(self._restore):
            if had:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
        delattr(self.pair, '_audit_trace_active')
        return False
