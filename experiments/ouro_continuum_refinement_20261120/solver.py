"""Integrate two distinct vector fields defined by the pinned Ouro operators."""
from contextlib import contextmanager
import hashlib
import sys
from pathlib import Path
import time

import torch

PREVIOUS = Path(__file__).resolve().parent.parent / "ouro_continuous_latent_20261120"
sys.path.insert(0, str(PREVIOUS))
from dynamics import NativeBatch


class Operators(NativeBatch):
    def __init__(self, model, items, depth=2):
        super().__init__(model, items)
        self.dtype = next(model.parameters()).dtype
        self.valid = torch.arange(self.ids.shape[1], device=self.ids.device)[None] < self.lengths[:, None]
        self.positions = torch.as_tensor([w["read"] for w in items], device=self.ids.device)
        self.candidates = torch.as_tensor([w["content"] for w in items], device=self.ids.device)
        self.core.total_ut_steps = depth
        self.counts["native_loops"] += depth
        self.counts["attention_calls"] += 24 * depth
        self.counts["mlp_calls"] += 24 * depth
        seen = {}
        actual = {}
        handles = []

        def layer_pre(i, args, kwargs):
            seen[i] = kwargs["current_ut"]
            if seen[i] == depth - 1:
                assert kwargs["past_key_value"] is None and not kwargs["use_cache"]
                self.kwargs[i] = kwargs

        def mlp_pre(i, args):
            if seen[i] == depth - 1:
                actual[2 * i + 1] = args[0]

        def layer_post(i, output):
            if seen[i] == depth - 1:
                actual[2 * i + 2] = output

        for idx, layer in enumerate(self.layers):
            handles.extend((
                layer.register_forward_pre_hook(lambda _, a, k, i=idx: layer_pre(i, a, k), with_kwargs=True),
                layer.post_attention_layernorm.register_forward_pre_hook(lambda _, a, i=idx: mlp_pre(i, a)),
                layer.register_forward_hook(lambda _, _a, o, i=idx: layer_post(i, o))))
        try:
            output, states, _ = self.core(input_ids=self.ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        assert output.past_key_values is None
        self.k1, self.k2 = states[-2:]
        self.actual = [self.k1] + [actual[i] for i in range(1, 49)]

    def field(self, state, terminal_norm=True):
        self.counts["loop_field_calls"] += 1
        self.counts["attention_calls"] += 24
        self.counts["mlp_calls"] += 24
        h = state.to(self.dtype)
        for i, layer in enumerate(self.layers):
            h = layer(h, **self.kwargs[i])
        return self.norm(h) if terminal_norm else h

    def branch(self, index, state):
        layer = self.layers[index // 2]
        x = state.to(self.dtype)
        if index % 2 == 0:
            self.counts["attention_calls"] += 1
            y, _ = layer.self_attn(hidden_states=layer.input_layernorm(x), **self.kwargs[index // 2])
            return layer.input_layernorm_2(y)
        self.counts["mlp_calls"] += 1
        return layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(x)))

    def norm(self, state):
        return self.core.norm(state.to(self.dtype))

    def scores(self, h):
        return self.model.lm_head(h[self.rows, self.positions].to(self.dtype)).float().gather(-1, self.candidates)

    def parity_receipt(self):
        branches, endpoint = integrate_b(self,1,self.k1,local_grid=1)
        for i, native in enumerate(self.actual):
            assert torch.equal(branches[i], native), i
        assert torch.equal(endpoint, self.k2)
        assert torch.equal(integrate_a(self,1,self.k1)[32], self.k2)
        self.counts["native_loops"] += self.core.total_ut_steps
        self.counts["attention_calls"] += 24 * self.core.total_ut_steps
        self.counts["mlp_calls"] += 24 * self.core.total_ut_steps
        direct = self.model(input_ids=self.ids, use_cache=False, exit_at_step=self.core.total_ut_steps - 1)
        dense = self.model.lm_head(self.k2)
        assert torch.equal(dense, direct.logits) and direct.past_key_values is None
        capture = self.kwargs[0]
        for i in range(24):
            for name in ("attention_mask", "position_ids", "cache_position"):
                a,b = capture[name],self.kwargs[i][name]
                assert (a is None and b is None) or (a is not None and b is not None and torch.equal(a,b))
            assert all(torch.equal(a,b) for a,b in zip(capture["position_embeddings"],
                                                       self.kwargs[i]["position_embeddings"], strict=True))
        return {"all_48_native_branch_boundaries_bitwise":True,"terminal_native_bitwise":True,
                "full_loop_m1_bitwise":True,"full_vocab_values_bitwise":dense.numel(),
                "native_mask_rope_positions":True,"use_cache":False}


def step(field, h, dt, rk4):
    a = field(h).float()
    if not rk4:
        return h + dt * a
    b = field(h + (dt / 2) * a).float()
    c = field(h + (dt / 2) * b).float()
    d = field(h + dt * c).float()
    return h + (dt / 6) * (a + 2*b + 2*c + d)


def integrate_a(op, m, start, *, rk4=False, round_state=False, terminal_norm=True, horizon=1, grid=32):
    if m==1 and not rk4 and not round_state:
        h=start.to(op.dtype)
        states={0:h}
        for j in range(1,horizon+1):
            h=op.field(h,terminal_norm=terminal_norm)
            states[j*grid]=h
        return states
    h = start.float()
    states = {0:h}
    g = lambda x: op.field(x, terminal_norm=terminal_norm).float() - x
    for j in range(1,horizon*m+1):
        h = step(g,h,1/m,rk4)
        if round_state:
            h = h.bfloat16().float()
        if j*grid % m == 0:
            states[j*grid//m] = h
    return states


def integrate_b(op, m, start, *, rk4=False, local_grid=4, start_branch=0, end_branch=48,
                round_state=False):
    native_step=m==1 and not rk4
    h = start.to(op.dtype) if native_step else start.float()
    states = {start_branch*local_grid:h}
    for i in range(start_branch,end_branch):
        g = lambda x: op.branch(i,x)
        for j in range(1,m+1):
            h = h + g(h) if native_step else step(g,h,1/m,rk4)
            if round_state:
                h = h.bfloat16().float()
            if j*local_grid % m == 0:
                states[i*local_grid+j*local_grid//m] = h
    return states, op.norm(h) if end_branch == 48 else None


@contextmanager
def timer(op, label, log):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    counts = dict(op.counts)
    yield
    torch.cuda.synchronize()
    entry = dict(run=label,wall_s=time.perf_counter()-start,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(), batch=list(op.ids.shape),
                 precision=str(op.dtype),**{k:v-counts[k] for k,v in op.counts.items()})
    log.append(entry)
    print(label,round(entry["wall_s"],2),flush=True)


def weight_hash(model, dtype):
    checksum=hashlib.sha256()
    for name,param in sorted(model.named_parameters()):
        checksum.update(name.encode("utf-8")+b"\0")
        checksum.update(param.detach().reshape(-1).to(dtype).view(torch.uint8).cpu().numpy().tobytes())
    return checksum.hexdigest()
