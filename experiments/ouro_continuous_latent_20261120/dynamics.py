"""Instrument the pinned Ouro implementation without changing its parameters."""
from contextlib import contextmanager
import torch


def relative(a, b, lengths):
    a, b = a.float(), b.float()
    mask = torch.arange(a.shape[1], device=a.device)[None] < lengths[:, None]
    delta = ((a - b).square().sum(-1) * mask).sum(-1).sqrt()
    norm = (b.square().sum(-1) * mask).sum(-1).sqrt()
    return (delta / norm.clamp_min(1e-12)).cpu().tolist()


def step_cosine(a, b, c, d, lengths):
    dx, dy = b.float() - a.float(), d.float() - c.float()
    mask = (torch.arange(dx.shape[1], device=dx.device)[None] < lengths[:, None]).unsqueeze(-1)
    return torch.nn.functional.cosine_similarity((dx * mask).flatten(1), (dy * mask).flatten(1)).cpu().tolist()


class NativeBatch:
    def __init__(self, model, encodings, trace=False, parity=False):
        self.model = model
        self.core = model.model
        self.encodings = encodings
        self.layers = list(self.core.layers)
        assert len(self.layers) == 24
        length = max(len(e["ids"]) for e in encodings)
        self.ids = torch.tensor([e["ids"] + [0] * (length - len(e["ids"])) for e in encodings], device="cuda")
        self.lengths = torch.tensor([len(e["ids"]) for e in encodings], device="cuda")
        self.rows = torch.arange(len(encodings), device="cuda")
        self.kwargs = {}
        self.counts = {"native_loops": 2, "loop_field_calls": 0, "attention_calls": 48, "mlp_calls": 48}
        handles = []
        layer_inputs = []

        def capture(i):
            def save(_, args, kwargs):
                if kwargs["current_ut"] == 1:
                    assert kwargs["past_key_value"] is None and not kwargs["use_cache"]
                    self.kwargs[i] = kwargs
                    layer_inputs.append(args[0] if args else kwargs["hidden_states"])
            return save

        try:
            for i, layer in enumerate(self.layers):
                handles.append(layer.register_forward_pre_hook(capture(i), with_kwargs=True))
            self.core.total_ut_steps = 2
            output, hidden, _ = self.core(input_ids=self.ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        assert output.past_key_values is None and len(self.kwargs) == 24
        self.k1, self.k2 = hidden
        assert torch.equal(layer_inputs[0], self.k1)
        self.native = None
        if trace or parity:
            self.native = self.branch_path(1, capture_every=True)
            assert torch.equal(self.native[1.0], self.k2)
        self.parity = self.check_parity() if parity else None

    def check_parity(self):
        checks = {}
        for depth, h in enumerate((self.k1, self.k2), 1):
            self.core.total_ut_steps = depth
            direct = self.model(input_ids=self.ids, use_cache=False, exit_at_step=depth - 1)
            _, states, _ = self.core(input_ids=self.ids, use_cache=False)
            assert direct.past_key_values is None
            assert torch.equal(h, states[-1])
            z = self.model.lm_head(h)
            assert torch.equal(z, direct.logits)
            checks[f"K{depth}"] = {"hidden_bitwise": True, "full_logits_bitwise": True,
                                    "full_logit_values": z.numel()}
        for name, h in (("A", self.euler(1)[1.0]), ("B", self.native[1.0])):
            assert torch.equal(h, self.k2)
            assert torch.equal(self.model.lm_head(h), self.model.lm_head(self.k2))
            checks[name + "_m1"] = {"hidden_bitwise": True, "full_logits_bitwise": True}
        for kwargs in self.kwargs.values():
            assert torch.equal(kwargs["position_ids"], self.kwargs[0]["position_ids"])
            assert torch.equal(kwargs["cache_position"], self.kwargs[0]["cache_position"])
            assert all(torch.equal(x, y) for x, y in zip(kwargs["position_embeddings"],
                                                        self.kwargs[0]["position_embeddings"], strict=True))
        return checks

    def field(self, h):
        self.counts["loop_field_calls"] += 1
        self.counts["attention_calls"] += 24
        self.counts["mlp_calls"] += 24
        h = h.to(torch.bfloat16)
        for i, layer in enumerate(self.layers):
            h = layer(h, **self.kwargs[i])
        return self.core.norm(h)

    def euler(self, m, start=None, from_index=0, rk4=False, bf16_accum=False):
        h = self.k1 if start is None else start
        out = {from_index / m: h}
        if m == 1 and not rk4:
            out[1.] = self.field(h)
            return out
        h = h.float()
        for j in range(from_index, m):
            dt = 1 / m
            a = self.field(h).float() - h
            if rk4:
                v = h + dt * a / 2
                b = self.field(v).float() - v
                v = h + dt * b / 2
                c = self.field(v).float() - v
                v = h + dt * c
                d = self.field(v).float() - v
                h = h + dt * (a + 2*b + 2*c + d) / 6
            else:
                h = h + dt * a
            if bf16_accum:
                h = h.to(torch.bfloat16).float()
            out[(j + 1)/m] = h
        return out

    def update(self, i, branch, h):
        layer = self.layers[i]
        h = h.to(torch.bfloat16)
        if branch == 0:
            self.counts["attention_calls"] += 1
            h, _ = layer.self_attn(hidden_states=layer.input_layernorm(h), **self.kwargs[i])
            return layer.input_layernorm_2(h)
        self.counts["mlp_calls"] += 1
        return layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(h)))

    def branch_path(self, m, start=None, from_branch=0, capture_every=False):
        h = self.k1 if start is None else start
        out = {from_branch / 48: h}
        if m > 1:
            h = h.float()
        for index in range(from_branch, 48):
            for _ in range(m):
                v = self.update(index // 2, index % 2, h)
                h = h + (v if m == 1 else v.float() / m)
            if capture_every or (index + 1) % 6 == 0:
                out[(index + 1) / 48] = h
        out[1.] = self.core.norm(h.to(torch.bfloat16))
        if capture_every:
            self.pre_boundary_norm = h
        return out

    def head(self, h, full=False, role=None):
        positions = [e["read"] if role is None else e["roles"].get(role, [e["read"]])[-1]
                     for e in self.encodings]
        x = h[self.rows, positions].to(torch.bfloat16)
        if full:
            return self.model.lm_head(x).float()
        candidates = torch.tensor([e["content"] for e in self.encodings], device="cuda")
        return self.model.lm_head(x).float().gather(-1, candidates)


def interpolation(a, b, times, match_norm=False):
    a, b = a.float(), b.float()
    output = {}
    for t in times:
        h = a.lerp(b, t)
        if match_norm and 0 < t < 1:
            target = a.norm(dim=-1, keepdim=True).lerp(b.norm(dim=-1, keepdim=True), t)
            h = h * (target / h.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        output[t] = h
    return output


def random_walk(path, generator):
    out = {}
    past = None
    for t, state in path.items():
        if past is None:
            h = state.float()
        else:
            direction = torch.randn(h.shape, device=h.device, generator=generator)
            dose = (state.float() - past.float()).norm(dim=-1, keepdim=True)
            h = h + direction * dose / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        out[t] = h
        past = state
    return out
