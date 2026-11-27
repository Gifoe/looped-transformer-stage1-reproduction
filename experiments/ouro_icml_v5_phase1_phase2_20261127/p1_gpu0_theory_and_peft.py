"""ICML Phase 1 — GPU 0:
Part 1: Centered vs Uncentered Sigma_h Geometry Audit (all 16 E0 Rank-only + v4 winners).
Part 2: P0 Theory-Derived Fixed-Read-Axis w (2,048 params) + Pseudo-Label Mechanistic Dissection
        (Correct vs Wrong pseudo-labels) + Language-Fidelity NLL Pareto Frontier.
Part 3: Track B — Direct Training-Free Readout Calibration (PMI / Rotated MC) +
        6 Canonical PEFT & Activation Steering Baselines across 3 sources
        (S1: self_arc_hardpseudo, S2: CFIT Top-6 supervised, S3: MedMCQA raw poison stress test),
        with exact parameter counts, wall-time, effective loop writes, and inference latency.
"""
import contextlib
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/ouro-v4-cfit-20261126")
import minimal
import v4_common
from v4_common import (
    DEPTHS,
    EXT_DEV_DIR,
    RECIPE,
    UNLABELED_AUDIT,
    UNLABELED_FIT,
    V3_ROOT,
    FastUnifiedDevEvaluator,
    cloze_option_logits_differentiable,
    evaluate_unlabeled_manifold,
    extract_q_opts_gold,
    forward_with_hook_all_tokens,
    load_base_runtime,
    load_controller_weights,
    load_jsonl,
    load_minimal_controller,
    make_candidate_row,
    save_json,
    save_rank1_controller,
    score_cloze_probs,
    score_mc_letter_rotation_probs,
)

V4_ROOT = Path("/root/autodl-tmp/ouro-v4-cfit-20261126")
P1_ROOT = Path("/root/autodl-tmp/ouro-v5-icml-phase1-20261127")
OUT_DIR = P1_ROOT / "track_b_and_theory"
SEEDS = (26112701, 26112702)
STEPS = 840


# ---------------------------------------------------------------------------
# Helper: Custom Controller & PEFT / Steering Modules
# ---------------------------------------------------------------------------
class FixedWController(nn.Module):
    """2,048 trainable params: frozen read axis w, trainable write axis u."""
    def __init__(self, w_fixed: torch.Tensor, bound: float = 0.45):
        super().__init__()
        self.kind = "fixed_w"
        self.rank = 1
        self.bound = bound
        self.fixed = True
        w_vec = w_fixed.detach().float().flatten()
        w_vec = w_vec / w_vec.norm().clamp_min(1e-12)
        self.register_buffer("w_fixed", w_vec)
        self.u = nn.Parameter(torch.zeros_like(w_vec))

    def delta(self, h, previous=None):
        scale = minimal.rms(h)
        proj = torch.einsum("...d,d->...", h, self.w_fixed)
        x = proj.unsqueeze(-1) * self.u
        return x if self.bound is None else x * (self.bound * scale / minimal.rms(x)).clamp_max(1.0)


class StaticSteeringController(nn.Module):
    """Training-free CAA/RepE static vector: delta(h) = alpha * rms(h) * u_unit."""
    def __init__(self, u_unit: torch.Tensor, alpha: float):
        super().__init__()
        self.kind = "static_caa"
        self.rank = 1
        self.bound = alpha
        self.fixed = True
        u = u_unit.detach().float().flatten()
        self.register_buffer("u_unit", u / u.norm().clamp_min(1e-12))
        self.alpha = float(alpha)

    def delta(self, h, previous=None):
        scale = minimal.rms(h)
        return self.alpha * scale * self.u_unit.expand_as(h)


class HookAdapterBase(nn.Module):
    """Base for hook-based PEFT adapters compatible with both training and FastUnifiedDevEvaluator."""
    @contextlib.contextmanager
    def attach(self, model):
        raise NotImplementedError


class PostLoopLMHeadRank1(HookAdapterBase):
    """Exact 4,096-param Rank-1 applied right before lm_head AFTER all K recurrent loops finish."""
    def __init__(self, d=2048, bound=0.45):
        super().__init__()
        self.net = minimal.Minimal("linear", 1, bound)

    @contextlib.contextmanager
    def attach(self, model):
        def pre_lm_head(_, args):
            h = args[0]
            hf = h.float()
            dh = self.net.delta(hf)
            return ((hf + dh).to(h.dtype), *args[1:])
        handle = model.lm_head.register_forward_pre_hook(pre_lm_head)
        try:
            yield
        finally:
            handle.remove()


class LoRAAdapter(HookAdapterBase):
    """Standard LoRA (r=1) on selected Linear layers, exact identity at init (B=0)."""
    def __init__(self, target_specs, rank=1, lora_alpha=1.0):
        super().__init__()
        # target_specs: list of (name, in_f, out_f)
        self.names = [n for n, _, _ in target_specs]
        self.scale = lora_alpha / rank
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        for n, inf, outf in target_specs:
            key = n.replace(".", "__")
            a = nn.Parameter(torch.empty(rank, inf))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            b = nn.Parameter(torch.zeros(outf, rank))
            self.A[key] = a
            self.B[key] = b

    @contextlib.contextmanager
    def attach(self, model):
        mod_map = dict(model.named_modules())
        handles = []
        for n in self.names:
            key = n.replace(".", "__")
            A = self.A[key]
            B = self.B[key]
            scale = self.scale
            def make_hook(a_p, b_p, sc):
                def fwd_hook(_, inp, out):
                    x = inp[0].float()
                    delta = F.linear(F.linear(x, a_p), b_p) * sc
                    return out + delta.to(out.dtype)
                return fwd_hook
            handles.append(mod_map[n].register_forward_hook(make_hook(A, B, scale)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()


class IA3Adapter(HookAdapterBase):
    """Standard IA3 (Liu et al. 2022 T-Few): scales k_proj output, v_proj output, and down_proj INPUT."""
    def __init__(self, layer_indices, d=2048, d_ff=5632):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.lk = nn.ParameterDict()
        self.lv = nn.ParameterDict()
        self.lff = nn.ParameterDict()
        for idx in self.layer_indices:
            self.lk[str(idx)] = nn.Parameter(torch.ones(d))
            self.lv[str(idx)] = nn.Parameter(torch.ones(d))
            self.lff[str(idx)] = nn.Parameter(torch.ones(d_ff))

    @contextlib.contextmanager
    def attach(self, model):
        handles = []
        for idx in self.layer_indices:
            layer = model.model.layers[idx]
            pk, pv, pff = self.lk[str(idx)], self.lv[str(idx)], self.lff[str(idx)]
            def make_post(p):
                def h(_, _inp, out):
                    return (out.float() * p).to(out.dtype)
                return h
            def make_pre(p):
                def h(_, args):
                    x = args[0]
                    return ((x.float() * p).to(x.dtype), *args[1:])
                return h
            handles.append(layer.self_attn.k_proj.register_forward_hook(make_post(pk)))
            handles.append(layer.self_attn.v_proj.register_forward_hook(make_post(pv)))
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_pre(pff)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()


class SoftPromptAdapter(HookAdapterBase):
    """Exact 4,096-param Soft Prompt (2 virtual prefix tokens = 2 x 2048), initialized from real embeddings."""
    def __init__(self, init_embeds: torch.Tensor):
        super().__init__()
        # init_embeds: [P, d]
        self.n_prompt = init_embeds.shape[0]
        self.soft_prompt = nn.Parameter(init_embeds.detach().float().clone())

    @contextlib.contextmanager
    def attach(self, model):
        core = model.model
        orig_forward = core.forward
        n_p = self.n_prompt
        sp_param = self.soft_prompt

        def wrapped_forward(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            cache_position=None,
            **kwargs,
        ):
            is_prefill = (past_key_values is None) or (past_key_values.get_seq_length() == 0)
            if is_prefill:
                if inputs_embeds is None:
                    inputs_embeds = core.embed_tokens(input_ids)
                    input_ids = None
                B = inputs_embeds.shape[0]
                sp = sp_param.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device).unsqueeze(0).expand(B, -1, -1)
                inputs_embeds = torch.cat([sp, inputs_embeds], dim=1)
                if attention_mask is not None and not isinstance(attention_mask, dict):
                    ones = torch.ones((B, n_p), dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([ones, attention_mask], dim=1)
                if position_ids is not None:
                    p_pre = torch.arange(n_p, device=position_ids.device, dtype=position_ids.dtype).unsqueeze(0).expand(B, -1)
                    position_ids = torch.cat([p_pre, position_ids + n_p], dim=1)
                if cache_position is not None:
                    c_pre = torch.arange(n_p, device=cache_position.device, dtype=cache_position.dtype)
                    cache_position = torch.cat([c_pre, cache_position + n_p], dim=0)
                out = orig_forward(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )
                # Strip prefix positions from returned hidden states so downstream callers see exact [B, T, d]
                base_out, h_list, g_list = out
                base_out.last_hidden_state = base_out.last_hidden_state[:, n_p:, :]
                h_list = [h[:, n_p:, :] for h in h_list]
                g_list = [g[:, n_p:, :] for g in g_list]
                return (base_out, h_list, g_list)
            else:
                # Decode step with cached past_key_values: cache already includes n_p tokens,
                # so attention_mask (if provided for original len) must be extended by n_p ones if shorter!
                if attention_mask is not None and not isinstance(attention_mask, dict):
                    past_len = past_key_values.get_seq_length()
                    cur_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
                    if attention_mask.shape[1] == past_len + cur_len - n_p:
                        ones = torch.ones((attention_mask.shape[0], n_p), dtype=attention_mask.dtype, device=attention_mask.device)
                        attention_mask = torch.cat([ones, attention_mask], dim=1)
                if position_ids is not None:
                    position_ids = position_ids + n_p
                if cache_position is not None:
                    cache_position = cache_position + n_p
                return orig_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

        core.forward = wrapped_forward
        try:
            yield
        finally:
            core.forward = orig_forward


@torch.inference_mode()
def evaluate_unlabeled_manifold_adapter(model, tokenizer, adapter, corpus_path=UNLABELED_AUDIT, max_docs=64, depth=4, batch_size=4):
    """Exact natural-text NLL / KL / EOS audit comparing Native (unattached) vs HookAdapterBase (attached)."""
    rows = load_jsonl(corpus_path)[:max_docs]
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    nll_nat_sum = nll_ctrl_sum = kl_sum = eos_abs_sum = eos_signed_sum = 0.0
    total_toks = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        enc = [tokenizer.encode(r["text"], add_special_tokens=False)[:256] for r in chunk]
        mx = max(len(x) for x in enc)
        ids = torch.full((len(enc), mx), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(enc), mx), dtype=torch.long, device=device)
        for b, seq in enumerate(enc):
            ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            attn[b, :len(seq)] = 1
        valid_mask = attn[:, 1:].bool()
        targets = ids[:, 1:][valid_mask]
        h_nat = forward_with_hook_all_tokens(model, ids, attn, depth=depth, net=None)
        z_nat = model.lm_head(h_nat[:, :-1][valid_mask]).float()
        with adapter.attach(model):
            h_ctl = forward_with_hook_all_tokens(model, ids, attn, depth=depth, net=None)
            z_ctl = model.lm_head(h_ctl[:, :-1][valid_mask]).float()
        lp_nat = F.log_softmax(z_nat, dim=-1)
        lp_ctl = F.log_softmax(z_ctl, dim=-1)
        p_nat = lp_nat.exp()
        p_ctl = lp_ctl.exp()
        nll_nat_sum += float(-lp_nat.gather(1, targets[:, None]).sum())
        nll_ctrl_sum += float(-lp_ctl.gather(1, targets[:, None]).sum())
        kl_sum += float((p_nat * (lp_nat - lp_ctl)).sum())
        if eos_id is not None:
            de = p_ctl[:, eos_id] - p_nat[:, eos_id]
            eos_abs_sum += float(de.abs().sum())
            eos_signed_sum += float(de.sum())
        total_toks += int(targets.numel())
    nll_nat = nll_nat_sum / total_toks
    nll_ctrl = nll_ctrl_sum / total_toks
    return {
        "n_docs": len(rows), "n_tokens": total_toks,
        "nll_nat": nll_nat, "nll_ctrl": nll_ctrl, "delta_nll": nll_ctrl - nll_nat,
        "kl_nat_to_ctrl": kl_sum / total_toks,
        "eos_abs_shift": eos_abs_sum / total_toks,
        "eos_signed_shift": eos_signed_sum / total_toks,
    }


def evaluate_any(evaluator, model, tokenizer, controller_or_adapter, include_mmlu_dev158=True):
    """Evaluate either a Minimal-like controller (passed as net) OR a HookAdapterBase (attached, net=None)."""
    t0 = time.perf_counter()
    if controller_or_adapter is None:
        ev = evaluator.evaluate(None, include_mmlu_dev158=include_mmlu_dev158)
        unlab = evaluate_unlabeled_manifold(model, tokenizer, None, UNLABELED_AUDIT, max_docs=64)
    elif isinstance(controller_or_adapter, HookAdapterBase):
        controller_or_adapter.eval()
        with controller_or_adapter.attach(model):
            ev = evaluator.evaluate(None, include_mmlu_dev158=include_mmlu_dev158)
        unlab = evaluate_unlabeled_manifold_adapter(model, tokenizer, controller_or_adapter, UNLABELED_AUDIT, max_docs=64)
    else:
        controller_or_adapter.eval()
        ev = evaluator.evaluate(controller_or_adapter, include_mmlu_dev158=include_mmlu_dev158)
        unlab = evaluate_unlabeled_manifold(model, tokenizer, controller_or_adapter, UNLABELED_AUDIT, max_docs=64)
    return ev, unlab, time.perf_counter() - t0


def measure_cloze_latency_ms(archive, model, tokenizer, sample_rows, controller_or_adapter):
    """Measure exact per-question Cloze forward latency (ms) on 48 questions at K=4."""
    rows = sample_rows[:48]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if isinstance(controller_or_adapter, HookAdapterBase):
        controller_or_adapter.eval()
        with controller_or_adapter.attach(model):
            score_cloze_probs(archive, model, tokenizer, rows, depth=4, net=None, batch_size=8)
    else:
        score_cloze_probs(archive, model, tokenizer, rows, depth=4, net=controller_or_adapter, batch_size=8)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / len(rows)


def prepare_trust_seqs(tokenizer, max_docs=128):
    rows = load_jsonl(UNLABELED_FIT)[:max_docs]
    seqs = [tokenizer.encode(r["text"], add_special_tokens=False)[:192] for r in rows]
    return [s for s in seqs if len(s) >= 64]


def onesided_nll_pen(model, tokenizer, seqs, net_or_adapter, depth, nat_cache, margin=0.02):
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    mx = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), mx), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), mx), dtype=torch.long, device=device)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        attn[b, :len(s)] = 1
    valid = attn[:, 1:].bool()
    targets = ids[:, 1:][valid]
    key = (depth, tuple(len(s) for s in seqs), int(ids[0, 0]), int(ids[-1, -1]))
    if key not in nat_cache:
        with torch.no_grad():
            h0 = forward_with_hook_all_tokens(model, ids, attn, depth=depth, net=None)
            z0 = model.lm_head(h0[:, :-1][valid]).float()
            nat_cache[key] = float(F.cross_entropy(z0, targets).detach())
    if isinstance(net_or_adapter, HookAdapterBase):
        with net_or_adapter.attach(model):
            h1 = forward_with_hook_all_tokens(model, ids, attn, depth=depth, net=None)
            z1 = model.lm_head(h1[:, :-1][valid]).float()
    else:
        h1 = forward_with_hook_all_tokens(model, ids, attn, depth=depth, net=net_or_adapter)
        z1 = model.lm_head(h1[:, :-1][valid]).float()
    nll1 = F.cross_entropy(z1, targets)
    return F.relu(nll1 - nat_cache[key] - margin)


def train_generic(
    archive, model, tokenizer, bank, module, seed, lr=1e-3, steps=STEPS,
    target_key="pseudo_label", weight_mode="conf_floor005",
    trust_seqs=None, trust_lam=0.0, tag=""
):
    """Generic trainer supporting BOTH Minimal-like controllers AND HookAdapterBase PEFT modules."""
    torch.manual_seed(seed)
    module = module.to(model.device)
    trainable = [p for p in module.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=RECIPE["weight_decay"])
    rng = random.Random(f"p1/{tag}/{seed}")
    order = list(range(len(bank)))
    rng.shuffle(order)
    ptr = 0
    nat_cache = {}
    is_hook = isinstance(module, HookAdapterBase)

    def next_mb(bs=4):
        nonlocal ptr
        out = []
        for _ in range(bs):
            if ptr >= len(order):
                rng.shuffle(order)
                ptr = 0
            out.append(bank[order[ptr]])
            ptr += 1
        return out

    t0 = time.perf_counter()
    losses = []
    module.train()
    for step in range(1, steps + 1):
        depth = DEPTHS[(step - 1) % len(DEPTHS)]
        mb = next_mb(RECIPE["effective_batch"])
        # Adaptive microbatch for long sequences (e.g. CFIT Top-6 RACE passages)
        max_len = max(len(it["question"]) + max(len(o) for o in it["options"]) for it in mb)
        max_opt = max(len(it["options"]) for it in mb)
        micro = 1 if (max_len > 1400 or max_len * max_opt > 5200) else (2 if (max_len > 650 or max_len * max_opt > 2600) else len(mb))

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        # Compute item weights across the full minibatch
        raw_w = []
        for it in mb:
            if weight_mode == "uniform":
                raw_w.append(1.0)
            elif weight_mode == "conf_nofloor":
                raw_w.append(max(1e-4, float(it["conf"])))
            elif weight_mode == "conf_floor005":
                raw_w.append(max(0.05, float(it["conf"])))
            else:
                raise ValueError(weight_mode)
        w_sum = sum(raw_w)

        with torch.enable_grad():
            for st in range(0, len(mb), micro):
                sub = mb[st:st + micro]
                sub_w = raw_w[st:st + micro]
                crows = [make_candidate_row(it["id"], it["question"], it["options"], 0) for it in sub]
                if is_hook:
                    with module.attach(model):
                        opt_logits, _, _ = cloze_option_logits_differentiable(archive, model, tokenizer, crows, depth, None)
                else:
                    opt_logits, _, _ = cloze_option_logits_differentiable(archive, model, tokenizer, crows, depth, module)
                sub_losses = []
                for lg, it, wi in zip(opt_logits, sub, sub_w):
                    log_p = F.log_softmax(lg, dim=0)
                    l_i = -log_p[int(it[target_key])]
                    sub_losses.append(l_i * (wi / w_sum))
                l_mb = sum(sub_losses)
                l_mb.backward()
                step_loss += float(l_mb.detach())

            if trust_lam > 0.0 and trust_seqs and step % 2 == 0:
                t_mb = [trust_seqs[(step + j) % len(trust_seqs)] for j in range(2)]
                pen = trust_lam * onesided_nll_pen(model, tokenizer, t_mb, module, depth, nat_cache, margin=0.02)
                pen.backward()
                step_loss += float(pen.detach())

        grad = torch.nn.utils.clip_grad_norm_(trainable, RECIPE["clip"])
        assert grad.isfinite().item(), f"Non-finite grad in {tag} step={step}"
        opt.step()
        losses.append(step_loss)
        if step % 280 == 0 or step == steps:
            print(f"  [{tag} s={seed}] step={step}/{steps} loss={sum(losses[-70:])/min(len(losses),70):.4f} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return module, {
        "n_params": n_params,
        "lr": lr,
        "steps": steps,
        "seconds": time.perf_counter() - t0,
        "final_loss_avg70": sum(losses[-70:]) / min(len(losses), 70),
    }


# ---------------------------------------------------------------------------
# PART 1: Centered vs Uncentered Sigma_h Audit
# ---------------------------------------------------------------------------
def run_part1_centered_sigma_audit(cfg, archive, model, tokenizer):
    out_path = OUT_DIR / "PART1_CENTERED_SIGMA_AUDIT.json"
    if out_path.exists():
        print("PART 1 already cached, loading...", flush=True)
        return json.loads(out_path.read_text(encoding="utf-8"))

    print("\n================ PART 1: CENTERED vs UNCENTERED SIGMA_H AUDIT ================", flush=True)
    sd = torch.load(V4_ROOT / "SIGMA_H.pt", map_location="cpu", weights_only=False)
    S_u = sd["sigma_unlab"].double()
    mu_u = sd["mu_unlab"].double()
    S_c = S_u - torch.outer(mu_u, mu_u)

    def spec_stats(M):
        ev, evec = torch.linalg.eigh((M + M.T) * 0.5)
        ev = ev.flip(0).clamp_min(0.0)
        evec = evec.flip(1)
        tr = float(ev.sum())
        return {
            "trace": tr,
            "lam1": float(ev[0]),
            "top1_frac": float(ev[0] / tr),
            "top8_frac": float(ev[:8].sum() / tr),
            "top16_frac": float(ev[:16].sum() / tr),
            "top64_frac": float(ev[:64].sum() / tr),
            "stable_rank": float(tr / ev[0].clamp_min(1e-12)),
        }, evec[:, 0].clone()

    st_u, top1_u = spec_stats(S_u)
    st_c, top1_c = spec_stats(S_c)
    mu_frac = float((mu_u @ mu_u) / S_u.trace())
    cos_top1_mu = float((top1_u @ (mu_u / mu_u.norm())).abs())
    cos_top1_uc = float((top1_u @ top1_c).abs())
    print(f"  Uncentered: sr={st_u['stable_rank']:.3f} top1={st_u['top1_frac']*100:.2f}% | Centered: sr={st_c['stable_rank']:.3f} top1={st_c['top1_frac']*100:.2f}% | ||mu||^2/tr={mu_frac*100:.2f}% | |cos(v1_u, mu)|={cos_top1_mu:.3f}", flush=True)

    # Audit all 16 E0 rank-only arms + key v4 controllers under BOTH S_u and S_c
    u_t1, w_t1 = load_controller_weights(Path(cfg["historical_t1"]))

    def metric_cos(a, b, M):
        na = float(torch.clamp(a @ (M @ a), min=1e-12).sqrt())
        nb = float(torch.clamp(b @ (M @ b), min=1e-12).sqrt())
        return float((a @ (M @ b)) / (na * nb))

    arm_paths = {}
    e0_dir = V3_ROOT / "stage1_e0" / "runs"
    if e0_dir.exists():
        for pt in sorted(e0_dir.glob("*.pt")):
            if ".rank." in pt.name and ".nll_rank." not in pt.name:
                arm_paths[pt.stem] = pt
    # Also include key v4 winners
    for label, p in [
        ("Best_ZeroLabel_4096", V4_ROOT / "exp_t2a_self_distill" / "self_arc_hardpseudo.s26112681.pt"),
        ("Best_NonARC_4096", V4_ROOT / "exp_t5_mixing" / "T5_CFIT_NonARC_4096.s26112801.pt"),
        ("Best_Rank1_4096", V4_ROOT / "exp_t5_mixing" / "T5_CFIT_Top6_4096.s26112801.pt"),
    ]:
        if p.exists():
            arm_paths[label] = p

    arms_audit = {}
    e0_cw_u, e0_cw_c, e0_op_u, e0_op_c, e0_cu = [], [], [], [], []
    for name, p in arm_paths.items():
        u, w = load_controller_weights(p)
        cu = float((u @ u_t1) / (u.norm() * u_t1.norm()).clamp_min(1e-12))
        cw_euc = float((w @ w_t1) / (w.norm() * w_t1.norm()).clamp_min(1e-12))
        cw_u = metric_cos(w, w_t1, S_u)
        cw_c = metric_cos(w, w_t1, S_c)
        w_mu_frac = float(((w @ mu_u) ** 2) / torch.clamp(w @ (S_u @ w), min=1e-12))
        arms_audit[name] = {
            "cos_u_t1": cu,
            "cos_w_euc_t1": cw_euc,
            "cos_w_uncent_t1": cw_u,
            "cos_w_cent_t1": cw_c,
            "cos_op_uncent_t1": cu * cw_u,
            "cos_op_cent_t1": cu * cw_c,
            "w_mean_energy_frac": w_mu_frac,
        }
        if name not in ("Best_ZeroLabel_4096", "Best_NonARC_4096", "Best_Rank1_4096"):
            e0_cw_u.append(abs(cw_u))
            e0_cw_c.append(abs(cw_c))
            e0_op_u.append(abs(cu * cw_u))
            e0_op_c.append(abs(cu * cw_c))
            e0_cu.append(abs(cu))

    res = {
        "uncentered_unlab": st_u,
        "centered_unlab": st_c,
        "mean_energy_frac": mu_frac,
        "abs_cos_top1_uncent_vs_mu": cos_top1_mu,
        "abs_cos_top1_uncent_vs_cent": cos_top1_uc,
        "n_e0_rank_only_arms": len(e0_cw_u),
        "e0_summary": {
            "abs_cos_w_uncent_range": [min(e0_cw_u), max(e0_cw_u)] if e0_cw_u else None,
            "abs_cos_w_cent_range": [min(e0_cw_c), max(e0_cw_c)] if e0_cw_c else None,
            "abs_cos_u_range": [min(e0_cu), max(e0_cu)] if e0_cu else None,
            "abs_cos_op_uncent_range": [min(e0_op_u), max(e0_op_u)] if e0_op_u else None,
            "abs_cos_op_cent_range": [min(e0_op_c), max(e0_op_c)] if e0_op_c else None,
        },
        "arms": arms_audit,
    }
    save_json(out_path, res)
    # Save top eigenvectors for Part 2 Fixed-w experiments
    torch.save({"mu": mu_u.float(), "top1_uncent": top1_u.float(), "top1_cent": top1_c.float()}, OUT_DIR / "SIGMA_EIGENVECTORS.pt")
    print("  E0 rank-only summary:", json.dumps(res["e0_summary"], indent=2), flush=True)
    return res


# ---------------------------------------------------------------------------
# Build ARC bank (with BOTH gold and frozen MC teacher) + CFIT Top-6 + MedMCQA
# ---------------------------------------------------------------------------
def build_arc_bank_with_mc(cfg, archive, model, tokenizer):
    cache = OUT_DIR / "ARC_BANK_S1.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    raw = load_jsonl(Path(cfg["pools"]["arc_challenge"]["path"]))
    items = []
    for r in raw:
        q, opts, g = extract_q_opts_gold(r)
        if 2 <= len(opts) <= 5:
            items.append({"id": r["sample_id"], "question": q, "options": opts, "gold": g})
    q_opts = [(it["question"], it["options"]) for it in items]
    mc_probs = score_mc_letter_rotation_probs(model, tokenizer, q_opts, depth=4, batch_size=8)
    for it, mp in zip(items, mc_probs):
        m = len(it["options"])
        it["q_mc"] = mp
        it["conf"] = max(0.0, (max(mp) - 1.0 / m) / (1.0 - 1.0 / m))
        it["pseudo_label"] = int(torch.tensor(mp).argmax())
        it["mc_correct"] = int(it["pseudo_label"] == it["gold"])
    save_json(cache, items)
    return items


def build_cfit_top6_bank():
    """Load exact CFIT Top-6 mixed source pool from V4 T5_TRAINING_LOCK.json (top-6 by CFIT proxy)."""
    lock = json.loads((V4_ROOT / "exp_t5_mixing" / "T5_TRAINING_LOCK.json").read_text(encoding="utf-8"))
    ranked = sorted(lock["sources"].items(), key=lambda kv: kv[1]["proxy"], reverse=True)[:6]
    tot_p = sum(v["proxy"] for _, v in ranked)
    weights = {k: v["proxy"] / tot_p for k, v in ranked}
    rng = random.Random(26112701)
    out = []
    for src, w in sorted(weights.items()):
        cands = [
            Path("/root/ouro-loop-aware-functional-identification-20261125/data/pools_complete") / f"{src}.jsonl",
            V3_ROOT / "stage0_env" / "PROSPECTIVE_TRAIN_DRAFT" / f"{src}.jsonl",
            V3_ROOT / "stage0_env" / "PROSPECTIVE_TRAIN_DRAFT" / f"{src}_train.jsonl",
        ]
        p = next((x for x in cands if x.exists()), None)
        assert p is not None, f"Missing pool for {src}: {cands}"
        rows = load_jsonl(p)
        parsed = []
        for r in rows:
            q, opts, g = extract_q_opts_gold(r)
            if 2 <= len(opts) <= 8:
                parsed.append({"id": r["sample_id"], "question": q, "options": opts, "gold": g, "conf": 1.0, "source": src})
        rng.shuffle(parsed)
        take = max(40, int(round(1200 * w)))
        out.extend(parsed[:take])
    rng.shuffle(out)
    print(f"Built CFIT Top-6 bank: n={len(out)}, weights={weights}", flush=True)
    return out


def build_medmcqa_poison_bank():
    cands = [
        V3_ROOT / "stage0_env" / "PROSPECTIVE_TRAIN_DRAFT" / "medmcqa_train.jsonl",
        V3_ROOT / "stage0_env" / "PROSPECTIVE_TRAIN_DRAFT" / "medmcqa.jsonl",
        Path("/root/ouro-loop-aware-functional-identification-20261125/data/pools_complete/medmcqa.jsonl"),
    ]
    p = next((x for x in cands if x.exists()), None)
    assert p is not None, f"Missing medmcqa pool: {cands}"
    rows = load_jsonl(p)
    rng = random.Random(26112701)
    parsed = []
    for r in rows:
        q, opts, g = extract_q_opts_gold(r)
        if 2 <= len(opts) <= 5:
            parsed.append({"id": r["sample_id"], "question": q, "options": opts, "gold": g, "conf": 1.0, "source": "medmcqa"})
    rng.shuffle(parsed)
    return parsed[:512]


# ---------------------------------------------------------------------------
# PART 2: Fixed-Read-Axis w (2,048) + Pseudo-Label Dissection + NLL Pareto
# ---------------------------------------------------------------------------
def run_part2(cfg, archive, model, tokenizer, evaluator, native_ev, arc_bank, trust_seqs):
    out_path = OUT_DIR / "PART2_FIXED_W_AND_MECHANISM.json"
    res = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {
        "native": native_ev, "fixed_w": {}, "pseudo_mechanism": {}, "nll_pareto": {}
    }

    arc_tr, arc_va = arc_bank[:860], arc_bank[860:]
    # Log exact pseudo-label accuracy & confidence breakdown!
    corr = [it for it in arc_tr if it["mc_correct"] == 1]
    wrong = [it for it in arc_tr if it["mc_correct"] == 0]
    res["arc_tr_pseudo_stats"] = {
        "n_total": len(arc_tr),
        "n_correct": len(corr),
        "n_wrong": len(wrong),
        "mc_acc": len(corr) / len(arc_tr),
        "mean_conf_correct": sum(x["conf"] for x in corr) / max(len(corr), 1),
        "mean_conf_wrong": sum(x["conf"] for x in wrong) / max(len(wrong), 1),
        "n_wrong_conf_ge_05": sum(1 for x in wrong if x["conf"] >= 0.5),
        "n_wrong_conf_ge_08": sum(1 for x in wrong if x["conf"] >= 0.8),
    }
    print("\n=== ARC Train Pseudo-Label Stats ===", json.dumps(res["arc_tr_pseudo_stats"], indent=2), flush=True)

    eig = torch.load(OUT_DIR / "SIGMA_EIGENVECTORS.pt", map_location="cpu", weights_only=False)
    anc = torch.load(V4_ROOT / "V4_SELF_ANCHOR.pt", map_location="cpu", weights_only=False)
    torch.manual_seed(99991)
    w_rand = torch.randn(2048)

    # --- 2A: P0 Fixed-Read-Axis w (2,048 params) ---
    print("\n================ PART 2A: P0 FIXED-READ-AXIS w (2,048 PARAMS) ================", flush=True)
    fixed_w_specs = [
        ("FixW_MeanMu_2048", eig["mu"]),
        ("FixW_Top1Uncent_2048", eig["top1_uncent"]),
        ("FixW_Top1Cent_2048", eig["top1_cent"]),
        ("FixW_SelfAnchor_2048", anc["w_self"]),
        ("FixW_Random_2048", w_rand),
        ("Full_Rank1_4096", None),
    ]
    for src_tag, tkey, wmode in [("S1_ZeroLabel", "pseudo_label", "conf_floor005"), ("S0_HumanGold", "gold", "uniform")]:
        for name, w_vec in fixed_w_specs:
            for seed in SEEDS:
                arm_id = f"{src_tag}__{name}.s{seed}"
                if arm_id in res["fixed_w"]:
                    continue
                torch.manual_seed(seed)
                mod = minimal.Minimal("linear", 1, 0.45) if w_vec is None else FixedWController(w_vec, 0.45)
                mod, meta = train_generic(archive, model, tokenizer, arc_tr, mod, seed, lr=1e-3, steps=STEPS, target_key=tkey, weight_mode=wmode, tag=arm_id)
                ev, unlab, ev_s = evaluate_any(evaluator, model, tokenizer, mod, include_mmlu_dev158=True)
                d_m5 = (ev["macro5"] - native_ev["macro5"]) * 100
                d_c3 = (ev["cloze3"] - native_ev["cloze3"]) * 100
                ckpt = OUT_DIR / "weights" / f"{arm_id}.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state": {k: v.cpu() for k, v in mod.state_dict().items()}, "meta": meta}, ckpt)
                res["fixed_w"][arm_id] = {
                    "source": src_tag, "variant": name, "seed": seed, "train": meta,
                    "eval": ev, "unlabeled": unlab, "delta_macro5_pp": d_m5, "delta_cloze3_pp": d_c3,
                }
                print(f"  => {arm_id:42s} params={meta['n_params']:5d} dMac5={d_m5:+6.2f} dClz3={d_c3:+6.2f} dNLL={unlab['delta_nll']:+.3f}", flush=True)
                save_json(out_path, res)

    # --- 2B: Correct vs Wrong Pseudo-Label Mechanistic Dissection ---
    print("\n================ PART 2B: PSEUDO-LABEL MECHANISM DISSECTION ================", flush=True)
    mech_specs = [
        ("ARC_HumanGold_Unif", arc_tr, "gold", "uniform"),
        ("ARC_HardPseudo_Unif", arc_tr, "pseudo_label", "uniform"),
        ("ARC_HardPseudo_ConfNoFloor", arc_tr, "pseudo_label", "conf_nofloor"),
        ("ARC_HardPseudo_ConfFloor005", arc_tr, "pseudo_label", "conf_floor005"),
        ("ARC_HardPseudo_CorrectOnly", corr, "pseudo_label", "conf_floor005"),
        ("ARC_HardPseudo_WrongOnly", wrong, "pseudo_label", "conf_floor005"),
        ("ARC_HumanGold_OnMCCorrect", corr, "gold", "uniform"),
    ]
    for name, sub_bank, tkey, wmode in mech_specs:
        for seed in SEEDS:
            arm_id = f"{name}.s{seed}"
            if arm_id in res["pseudo_mechanism"]:
                continue
            torch.manual_seed(seed)
            mod = minimal.Minimal("linear", 1, 0.45)
            mod, meta = train_generic(archive, model, tokenizer, sub_bank, mod, seed, lr=1e-3, steps=STEPS, target_key=tkey, weight_mode=wmode, tag=arm_id)
            ev, unlab, _ = evaluate_any(evaluator, model, tokenizer, mod, include_mmlu_dev158=True)
            d_m5 = (ev["macro5"] - native_ev["macro5"]) * 100
            d_c3 = (ev["cloze3"] - native_ev["cloze3"]) * 100
            res["pseudo_mechanism"][arm_id] = {
                "variant": name, "n_items": len(sub_bank), "seed": seed, "train": meta,
                "eval": ev, "unlabeled": unlab, "delta_macro5_pp": d_m5, "delta_cloze3_pp": d_c3,
            }
            print(f"  => {arm_id:38s} n={len(sub_bank):4d} dMac5={d_m5:+6.2f} dClz3={d_c3:+6.2f} MMLU={ev['mmlu_val']['acc']*100:.2f}% dNLL={unlab['delta_nll']:+.3f}", flush=True)
            save_json(out_path, res)

    # --- 2C: NLL-Fidelity Pareto Frontier on self_arc_hardpseudo ---
    print("\n================ PART 2C: NLL-FIDELITY PARETO FRONTIER (ZERO-LABEL) ================", flush=True)
    for lam in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        for seed in SEEDS:
            arm_id = f"pareto_lam{lam:.2f}.s{seed}"
            if arm_id in res["nll_pareto"]:
                continue
            torch.manual_seed(seed)
            mod = minimal.Minimal("linear", 1, 0.45)
            mod, meta = train_generic(
                archive, model, tokenizer, arc_tr, mod, seed, lr=1e-3, steps=STEPS,
                target_key="pseudo_label", weight_mode="conf_floor005",
                trust_seqs=trust_seqs, trust_lam=lam, tag=arm_id,
            )
            ev, unlab, _ = evaluate_any(evaluator, model, tokenizer, mod, include_mmlu_dev158=True)
            d_m5 = (ev["macro5"] - native_ev["macro5"]) * 100
            d_c3 = (ev["cloze3"] - native_ev["cloze3"]) * 100
            ppl_ratio = math.exp(unlab["delta_nll"])
            ckpt = OUT_DIR / "weights" / f"{arm_id}.pt"
            save_rank1_controller(ckpt, mod, {"arm": arm_id, "trust_lam": lam, "seed": seed})
            res["nll_pareto"][arm_id] = {
                "trust_lam": lam, "seed": seed, "train": meta, "eval": ev, "unlabeled": unlab,
                "delta_macro5_pp": d_m5, "delta_cloze3_pp": d_c3, "ppl_ratio": ppl_ratio,
            }
            print(f"  => {arm_id:24s} dMac5={d_m5:+6.2f} dClz3={d_c3:+6.2f} MMLU={ev['mmlu_val']['acc']*100:.2f}% dNLL={unlab['delta_nll']:+.3f} (PPLx{ppl_ratio:.2f}) eos={unlab['eos_abs_shift']:.4f}", flush=True)
            save_json(out_path, res)
    return res


# ---------------------------------------------------------------------------
# PART 3: Direct Readout Calibration + 6 Canonical PEFT & Steering Baselines
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_direct_readout_calibration(archive, model, tokenizer, evaluator):
    """Measure training-free Native Cloze, PMI calibration (log p(c|x)-log p(c)), and Rotated MC teacher on Cloze3."""
    print("\n================ PART 3A: TRAINING-FREE DIRECT READOUT CALIBRATION ================", flush=True)
    out = {}
    for task in ("sciq_val", "arc_easy_val", "cosmosqa_val"):
        raw = load_jsonl(EXT_DEV_DIR / f"{task}.jsonl")
        items = []
        for r in raw:
            q, opts, g = extract_q_opts_gold(r)
            items.append((r["sample_id"], q, opts, g))
        # 1. Native Cloze (1x forward)
        crows = [make_candidate_row(sid, q, opts, g) for sid, q, opts, g in items]
        t0 = time.perf_counter()
        raw_norm, _ = score_cloze_probs(archive, model, tokenizer, crows, depth=4, net=None, batch_size=8)
        t_clz = time.perf_counter() - t0
        acc_clz = sum(int(torch.tensor(s).argmax()) == g for s, (_, _, _, g) in zip(raw_norm, items)) / len(items)

        # 2. PMI Calibration: log p(c|x)/len - log p(c|"Answer:")/len (2x forward)
        empty_rows = [make_candidate_row(sid, "", opts, g) for sid, q, opts, g in items]
        t0 = time.perf_counter()
        prior_norm, _ = score_cloze_probs(archive, model, tokenizer, empty_rows, depth=4, net=None, batch_size=8)
        t_pmi = t_clz + (time.perf_counter() - t0)
        acc_pmi = sum(
            int(torch.tensor([a - b for a, b in zip(sc, pr)]).argmax()) == g
            for sc, pr, (_, _, _, g) in zip(raw_norm, prior_norm, items)
        ) / len(items)

        # 3. Rotated MC (4x forward teacher)
        t0 = time.perf_counter()
        mc_probs = score_mc_letter_rotation_probs(model, tokenizer, [(q, opts) for _, q, opts, _ in items], depth=4, batch_size=8)
        t_mc = time.perf_counter() - t0
        acc_mc = sum(int(torch.tensor(p).argmax()) == g for p, (_, _, _, g) in zip(mc_probs, items)) / len(items)

        out[task] = {
            "n": len(items),
            "native_cloze_acc_norm": acc_clz, "cloze_ms_per_q": t_clz * 1000.0 / len(items),
            "pmi_calib_acc": acc_pmi, "pmi_ms_per_q": t_pmi * 1000.0 / len(items),
            "rotated_mc_acc": acc_mc, "rot_mc_ms_per_q": t_mc * 1000.0 / len(items),
        }
        print(f"  {task:14s}: Cloze={acc_clz*100:.2f}% ({out[task]['cloze_ms_per_q']:.1f}ms) | PMI={acc_pmi*100:.2f}% ({out[task]['pmi_ms_per_q']:.1f}ms) | RotMC={acc_mc*100:.2f}% ({out[task]['rot_mc_ms_per_q']:.1f}ms)", flush=True)
    out["cloze3_mean"] = {
        "native_cloze": sum(out[t]["native_cloze_acc_norm"] for t in ("sciq_val", "arc_easy_val", "cosmosqa_val")) / 3.0,
        "pmi_calib": sum(out[t]["pmi_calib_acc"] for t in ("sciq_val", "arc_easy_val", "cosmosqa_val")) / 3.0,
        "rotated_mc": sum(out[t]["rotated_mc_acc"] for t in ("sciq_val", "arc_easy_val", "cosmosqa_val")) / 3.0,
    }
    return out


@torch.inference_mode()
def fit_caa_repe_vectors(model, tokenizer, bank, target_key="pseudo_label", max_items=256):
    """Extract training-free CAA (mean-diff) and RepE (top-PC of pos-neg diffs) at loop boundary k=3."""
    diffs = []
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for it in bank[:max_items]:
        pos_idx = int(it[target_key])
        neg_idxs = [j for j in range(len(it["options"])) if j != pos_idx]
        if not neg_idxs:
            continue
        neg_idx = neg_idxs[0]
        p_pos = f"Question: {it['question']}\nAnswer: {it['options'][pos_idx]}"
        p_neg = f"Question: {it['question']}\nAnswer: {it['options'][neg_idx]}"
        enc = [tokenizer.encode(p_pos, add_special_tokens=False)[-512:], tokenizer.encode(p_neg, add_special_tokens=False)[-512:]]
        mx = max(len(x) for x in enc)
        ids = torch.full((2, mx), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((2, mx), dtype=torch.long, device=device)
        for b, seq in enumerate(enc):
            ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[b, :len(seq)] = 1
        h = v4_common.capture_boundary_states(model, ids, mask, depth=4)
        h_pos = h[0, len(enc[0]) - 1].double().cpu()
        h_neg = h[1, len(enc[1]) - 1].double().cpu()
        diffs.append(h_pos - h_neg)
    D = torch.stack(diffs, dim=0)  # [N, d]
    u_caa = D.mean(dim=0)
    u_caa = u_caa / u_caa.norm().clamp_min(1e-12)
    _, _, Vt = torch.linalg.svd(D, full_matrices=False)
    u_repe = Vt[0]
    if float(u_repe @ u_caa) < 0:
        u_repe = -u_repe
    return u_caa.float(), u_repe.float()


def run_part3_peft_baselines(cfg, archive, model, tokenizer, evaluator, native_ev, arc_bank):
    out_path = OUT_DIR / "PART3_DIRECT_CALIB_AND_PEFT_BASELINES.json"
    res = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {
        "native": native_ev, "direct_calibration": None, "caa_repe": {}, "peft_arms": {}
    }

    if res["direct_calibration"] is None:
        res["direct_calibration"] = run_direct_readout_calibration(archive, model, tokenizer, evaluator)
        save_json(out_path, res)

    arc_tr, arc_va = arc_bank[:860], arc_bank[860:]
    cfit_top6 = build_cfit_top6_bank()
    med_poison = build_medmcqa_poison_bank()
    lat_rows = [make_candidate_row(it["id"], it["question"], it["options"], 0) for it in arc_va[:48]]
    res["native_cloze_ms_per_q"] = measure_cloze_latency_ms(archive, model, tokenizer, lat_rows, None)

    sources = [
        ("S1_ZeroLabel_ARC", arc_tr, "pseudo_label", "conf_floor005"),
        ("S2_Superv_CFIT_Top6", cfit_top6, "gold", "uniform"),
        ("S3_Poison_MedMCQA", med_poison, "gold", "uniform"),
    ]

    # --- 3B-1: Training-Free CAA / RepE Static Steering ---
    print("\n================ PART 3B-1: TRAINING-FREE CAA & RepE STEERING ================", flush=True)
    for src_name, bank, tkey, _ in sources:
        if src_name in res["caa_repe"]:
            continue
        t0 = time.perf_counter()
        u_caa, u_repe = fit_caa_repe_vectors(model, tokenizer, bank, target_key=tkey, max_items=256)
        fit_s = time.perf_counter() - t0
        # Select best alpha on held-out arc_va unlabeled MC-Cloze agreement (for S1) or evaluate all alphas cleanly
        src_out = {"fit_seconds": fit_s, "variants": {}}
        for mname, uvec in [("M1a_CAA_MeanDiff_2048", u_caa), ("M1b_RepE_TopPC_2048", u_repe)]:
            best_alpha, best_m5 = None, -1e9
            for alpha in (0.05, 0.15, 0.30, 0.45):
                ctl = StaticSteeringController(uvec, alpha).to(model.device)
                ev, unlab, _ = evaluate_any(evaluator, model, tokenizer, ctl, include_mmlu_dev158=True)
                d_m5 = (ev["macro5"] - native_ev["macro5"]) * 100
                d_c3 = (ev["cloze3"] - native_ev["cloze3"]) * 100
                key = f"{mname}_a{alpha:.2f}"
                src_out["variants"][key] = {
                    "n_params": 2048, "alpha": alpha, "eval": ev, "unlabeled": unlab,
                    "delta_macro5_pp": d_m5, "delta_cloze3_pp": d_c3,
                }
                print(f"  => {src_name} | {key:28s}: dMac5={d_m5:+6.2f} dClz3={d_c3:+6.2f} dNLL={unlab['delta_nll']:+.3f}", flush=True)
        res["caa_repe"][src_name] = src_out
        save_json(out_path, res)

    # Prepare SoftPrompt initial embeddings from 2 neutral real tokens ("Question", ":")
    with torch.no_grad():
        init_tok_ids = tokenizer.encode("Question:", add_special_tokens=False)[:2]
        if len(init_tok_ids) < 2:
            init_tok_ids = [tokenizer.eos_token_id or 1, tokenizer.eos_token_id or 1]
        sp_init = model.model.embed_tokens(torch.tensor(init_tok_ids, device=model.device)).detach().cpu()

    # Build factory for all 7 trainable methods (exact parameter counts & loop write counts!)
    def build_method(mname, seed):
        torch.manual_seed(seed)
        if mname == "M0_InLoop_Rank1_4096":
            return minimal.Minimal("linear", 1, 0.45), 1e-3, 1
        elif mname == "M2_Learned_Const_2048":
            return minimal.Minimal("constant", 1, 0.45), 1e-3, 1
        elif mname == "M3_SoftPrompt_4096":
            return SoftPromptAdapter(sp_init), 3e-3, 4
        elif mname == "M4_PostLoop_LMHead_Rank1_4096":
            return PostLoopLMHeadRank1(2048, 0.45), 1e-3, 1
        elif mname == "M5a_IA3_Last1_9728":
            return IA3Adapter([23], d=2048, d_ff=5632), 3e-3, 4
        elif mname == "M5b_IA3_All24_233k":
            return IA3Adapter(list(range(24)), d=2048, d_ff=5632), 1e-3, 4
        elif mname == "M6a_LoRA_r1_LastO_4096":
            return LoRAAdapter([("model.layers.23.self_attn.o_proj", 2048, 2048)], rank=1, lora_alpha=4.0), 1e-3, 4
        elif mname == "M6b_LoRA_r1_AllQV_196k":
            specs = []
            for i in range(24):
                specs.append((f"model.layers.{i}.self_attn.q_proj", 2048, 2048))
                specs.append((f"model.layers.{i}.self_attn.v_proj", 2048, 2048))
            return LoRAAdapter(specs, rank=1, lora_alpha=4.0), 3e-4, 4
        else:
            raise ValueError(mname)

    method_order = [
        "M0_InLoop_Rank1_4096",
        "M2_Learned_Const_2048",
        "M3_SoftPrompt_4096",
        "M4_PostLoop_LMHead_Rank1_4096",
        "M5a_IA3_Last1_9728",
        "M5b_IA3_All24_233k",
        "M6a_LoRA_r1_LastO_4096",
        "M6b_LoRA_r1_AllQV_196k",
    ]

    print("\n================ PART 3B-2: 6 CANONICAL PEFT & STEERING BASELINES ================", flush=True)
    for src_name, bank, tkey, wmode in sources:
        for mname in method_order:
            for seed in SEEDS:
                arm_id = f"{src_name}__{mname}.s{seed}"
                if arm_id in res["peft_arms"]:
                    continue
                mod, lr, loop_writes = build_method(mname, seed)
                mod, meta = train_generic(
                    archive, model, tokenizer, bank, mod, seed,
                    lr=lr, steps=STEPS, target_key=tkey, weight_mode=wmode, tag=arm_id
                )
                ev, unlab, ev_s = evaluate_any(evaluator, model, tokenizer, mod, include_mmlu_dev158=True)
                lat_ms = measure_cloze_latency_ms(archive, model, tokenizer, lat_rows, mod)
                d_m5 = (ev["macro5"] - native_ev["macro5"]) * 100
                d_c3 = (ev["cloze3"] - native_ev["cloze3"]) * 100
                ckpt = OUT_DIR / "weights" / f"{arm_id}.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"method": mname, "source": src_name, "seed": seed, "state": {k: v.cpu() for k, v in mod.state_dict().items()}, "meta": meta}, ckpt)
                res["peft_arms"][arm_id] = {
                    "source": src_name, "method": mname, "seed": seed,
                    "n_params": meta["n_params"], "effective_loop_writes": loop_writes,
                    "cloze_ms_per_q": lat_ms, "train": meta, "eval": ev, "unlabeled": unlab,
                    "delta_macro5_pp": d_m5, "delta_cloze3_pp": d_c3,
                }
                print(
                    f"  => {arm_id:50s} P={meta['n_params']:6d} W={loop_writes}x "
                    f"dMac5={d_m5:+6.2f} dClz3={d_c3:+6.2f} MMLU={ev['mmlu_val']['acc']*100:.2f}% "
                    f"dNLL={unlab['delta_nll']:+.3f} t={meta['seconds']:.0f}s lat={lat_ms:.1f}ms",
                    flush=True,
                )
                save_json(out_path, res)
                del mod
                torch.cuda.empty_cache()

    print("\nALL_PHASE1_GPU0_THEORY_AND_PEFT_DONE", flush=True)
    return res


def main():
    torch.set_num_threads(4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg, archive, model, tokenizer = load_base_runtime()
    evaluator = FastUnifiedDevEvaluator(cfg, archive, model, tokenizer)
    native_ev = evaluator.evaluate(None, include_mmlu_dev158=True)
    print(f"Native UNIFIED_EXT_DEV: macro5={native_ev['macro5']*100:.2f}% cloze3={native_ev['cloze3']*100:.2f}% mmlu={native_ev['mmlu_val']['acc']*100:.2f}%", flush=True)

    trust_seqs = prepare_trust_seqs(tokenizer, max_docs=128)
    run_part1_centered_sigma_audit(cfg, archive, model, tokenizer)
    arc_bank = build_arc_bank_with_mc(cfg, archive, model, tokenizer)
    run_part2(cfg, archive, model, tokenizer, evaluator, native_ev, arc_bank, trust_seqs)
    run_part3_peft_baselines(cfg, archive, model, tokenizer, evaluator, native_ev, arc_bank)


if __name__ == "__main__":
    main()
