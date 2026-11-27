"""ICML Phase 2 — GPU 1: Cross-Architecture, Scaling, Depth, & Model-Conditioned CFIT
1. Module 1 (Qwen2.5-1.5B, Non-Recurrent 28L Transformer, d=1536, 3,072 / 1,536 params):
   - Centered vs Uncentered Sigma_h across depths L in {14, 21, 24, 26, PostNorm}
   - Native CFIT Regime-III profile on 8 external sources
   - Controlled arms: S0 HumanGold, S1 ZeroLabel, S1 FixW_MeanMu (1,536 params!), S1 FixW_Random,
     S1 PostNorm LMHead control, S1 Pareto lam=1.0, S3 MedMCQA HumanGold vs ZeroLabel, + Depth Sweep!
2. Module 2 (Ouro-2.6B, 2x Scale 48L Looped Transformer, d=2048, K in {2, 4, 8}):
   - Recurrence Depth Scaling K in {2, 4, 8}: Sigma_h collapse, Native MC-Cloze gap, Rank-1 ZeroLabel gain
   - Cross-Model CFIT Data Value V(D; Ouro-2.6B) across 8 sources (profile + 8 native trained controllers)
   - Core Mechanism Replication: S0 HumanGold vs S1 ZeroLabel vs FixW_MeanMu (2,048) vs Pareto lam=1.0
     vs S3 MedMCQA ZeroLabel vs Zero-Shot Transfer of Ouro-1.4B Best_ZeroLabel_4096!
3. Module 3 (Huginn-0125, Different Looped Arch, d=5280, K=32):
   - Centered vs Uncentered Sigma_h at pre-step-32 boundary
   - Cross-Model CFIT Regime-III profile on 8 sources (verifying Regime-II shift on Huginn!)
   - Falsifiable Cure for Huginn's -3.19 pp MMLU poison: S0 HumanGold vs S1 ZeroLabel vs Pareto lam=1.0
     vs FixW_MeanMu (5,280 params!) vs S3 MedMCQA HumanGold & ZeroLabel!
4. Module 4 (Ouro-1.4B Recurrence Depth Scaling K in {2, 4, 8}):
   - Exact companion K={2, 4, 8} table matching Ouro-2.6B!
"""

import contextlib
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
transformers.utils.import_utils._torchvision_available = False
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/autodl-tmp/ouro-v4-cfit-20261126")
sys.path.insert(0, "/root/ouro-cross-model-functional-controllability-20261125")

from v4_common import (
    E0_CONFIG_PATH,
    EXT_DEV_DIR,
    UNLABELED_AUDIT,
    UNLABELED_FIT,
    V3_ROOT,
    load_jsonl,
    save_json,
)

V4_ROOT = Path("/root/autodl-tmp/ouro-v4-cfit-20261126")
P2_OUT = Path("/root/autodl-tmp/ouro-v5-icml-phase1-20261127/phase2_cross_model")
MODELS_DIR = Path("/dev/shm/ouro-cross-model-20261125")
POOLS_DIR = Path("/root/ouro-loop-aware-functional-identification-20261125/data/pools_complete")
DRAFT_DIR = V3_ROOT / "stage0_env" / "PROSPECTIVE_TRAIN_DRAFT"

EIGHT_SOURCES = [
    "arc_challenge",
    "obqa",
    "qasc",
    "scienceqa",
    "race",
    "csqa",
    "boolq",
    "medmcqa",
]


# ---------------------------------------------------------------------------
# Universal Rank-1 & Fixed-w Controller
# ---------------------------------------------------------------------------
class UniversalRank1(nn.Module):
    def __init__(self, d: int, bound: float = 0.45, fixed_w: torch.Tensor = None):
        super().__init__()
        self.d = int(d)
        self.bound = float(bound)
        self.is_fixed_w = fixed_w is not None
        if fixed_w is not None:
            w = fixed_w.detach().float().flatten()
            w = w / w.norm().clamp_min(1e-12)
            self.register_buffer("w_fixed", w)
            self.u = nn.Parameter(torch.zeros(self.d, dtype=torch.float32))
        else:
            self.encode = nn.Linear(self.d, 1, bias=False)
            self.out = nn.Linear(1, self.d, bias=False)
            nn.init.zeros_(self.out.weight)

    def delta(self, h: torch.Tensor) -> torch.Tensor:
        hf = h.float()
        rms_h = hf.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        if self.is_fixed_w:
            proj = torch.einsum("...d,d->...", hf, self.w_fixed)
            x = proj.unsqueeze(-1) * self.u
        else:
            x = self.out(self.encode(hf))
        rms_x = x.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        return x * (self.bound * rms_h / rms_x).clamp_max(1.0)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Dataset Parsing (Universal across all Dev & External Pool schemas)
# ---------------------------------------------------------------------------
def parse_mc_row(row: dict):
    """Extract (sample_id, question, options_list, gold_idx) cleanly from any dev or pool schema."""
    sid = str(row.get("sample_id", row.get("id", "")))
    if "candidate" in row and isinstance(row["candidate"], dict):
        c = row["candidate"]
        q = str(c["question"]).strip()
        ch = c["choices"]
        opts = [str(x).strip() for x in ch["text"]]
        labels = [str(x).strip() for x in ch["label"]]
        ak = str(c["answerKey"]).strip()
        gold = labels.index(ak) if ak in labels else int(ak)
        return sid, q, opts, gold
    if "correct_answer" in row and "distractor1" in row:
        q = str(row["question"]).strip()
        opts = [
            str(row["correct_answer"]).strip(),
            str(row["distractor1"]).strip(),
            str(row["distractor2"]).strip(),
            str(row["distractor3"]).strip(),
        ]
        # Deterministic shuffle using sample_id so correct_answer isn't always slot 0
        rng = random.Random(f"sciq_shuf/{sid}")
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        return sid, q, [opts[k] for k in order], order.index(0)
    if "choices" in row and isinstance(row["choices"], list):
        q = str(row["question"]).strip()
        opts = [str(x).strip() for x in row["choices"]]
        ans = row["answer"]
        gold = int(ans) if isinstance(ans, int) or str(ans).isdigit() else (ord(str(ans).strip().upper()[0]) - 65)
        return sid, q, opts, gold
    if "choices" in row and isinstance(row["choices"], dict):
        q = str(row["question"]).strip()
        opts = [str(x).strip() for x in row["choices"]["text"]]
        labels = [str(x).strip() for x in row["choices"]["label"]]
        ak = str(row["answerKey"]).strip()
        gold = labels.index(ak) if ak in labels else int(ak)
        return sid, q, opts, gold
    if "answer0" in row and "answer1" in row:
        ctx = str(row.get("context", "")).strip()
        q0 = str(row.get("question", "")).strip()
        q = f"{ctx}\n{q0}".strip() if ctx else q0
        opts = [str(row[f"answer{i}"]).strip() for i in range(4) if f"answer{i}" in row]
        gold = int(row["label"])
        return sid, q, opts, gold
    if "options" in row and isinstance(row["options"], list):
        q = str(row.get("source_question") or row.get("context") or row.get("question") or "").strip()
        opts = [str(x).strip() for x in row["options"]]
        lbl = str(row.get("label", "0")).strip()
        gold = int(lbl) if lbl.isdigit() else (ord(lbl.upper()[0]) - 65)
        return sid, q, opts, gold
    raise KeyError(f"Cannot parse MC row keys: {list(row.keys())}")


def load_source_items(name: str, max_items: int = 800, seed: int = 26112799):
    p1 = POOLS_DIR / f"{name}.jsonl"
    p2 = DRAFT_DIR / f"{name}.jsonl"
    path = p1 if p1.exists() else p2
    assert path.exists(), f"Missing source pool for {name}"
    raw = load_jsonl(path)
    items = []
    for r in raw:
        try:
            sid, q, opts, gold = parse_mc_row(r)
            if 2 <= len(opts) <= 8 and 0 <= gold < len(opts) and q and all(opts):
                items.append({"id": sid, "question": q, "options": opts, "gold": gold})
        except Exception:
            continue
    rng = random.Random(f"src/{name}/{seed}")
    rng.shuffle(items)
    return items[:max_items]


def load_dev_suite():
    suite = {}
    for task in ("sciq_val", "arc_easy_val", "cosmosqa_val", "mmlu_val"):
        rows = load_jsonl(EXT_DEV_DIR / f"{task}.jsonl")
        suite[task] = [
            {"id": sid, "question": q, "options": opts, "gold": gold}
            for sid, q, opts, gold in (parse_mc_row(r) for r in rows)
        ]
    return suite


# ---------------------------------------------------------------------------
# Model Backends (Qwen2.5, Ouro-2.6B/1.4B, Huginn-0125)
# ---------------------------------------------------------------------------
class BaseBackend:
    name: str
    d: int
    default_depth: int

    def forward_logits(self, ids, attn, net=None, token_mask=None, depth=None, hook_site=None, select_positions=None, seed=261127):
        raise NotImplementedError

    @torch.inference_mode()
    def capture_boundary(self, ids, attn, depth=None, hook_site=None, seed=261127):
        raise NotImplementedError

    def cleanup(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


class QwenBackend(BaseBackend):
    def __init__(self, model_dir: Path, default_site: str = "layer_21"):
        self.name = model_dir.name
        self.default_site = default_site
        self.default_depth = 1
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=torch.bfloat16, device_map={"": 0}
        ).requires_grad_(False).eval()
        self.d = int(self.model.config.hidden_size)
        self.device = next(self.model.parameters()).device

    def _get_hook_module(self, hook_site):
        site = hook_site or self.default_site
        if site == "post_norm":
            return self.model.lm_head
        assert site.startswith("layer_"), f"Unknown Qwen site: {site}"
        idx = int(site.split("_")[1])
        return self.model.model.layers[idx]

    def forward_logits(self, ids, attn, net=None, token_mask=None, depth=None, hook_site=None, select_positions=None, seed=261127):
        handles = []
        if net is not None:
            mod = self._get_hook_module(hook_site)

            def pre_hook(_, args):
                h = args[0]
                hf = h.float()
                dh = net.delta(hf)
                if token_mask is not None:
                    dh = dh * token_mask.unsqueeze(-1).to(hf.dtype)
                return ((hf + dh).to(h.dtype), *args[1:])

            handles.append(mod.register_forward_pre_hook(pre_hook))

        if select_positions is not None:
            rows_ix, pos_ix = select_positions
            handles.append(self.model.lm_head.register_forward_pre_hook(lambda _, args: (args[0][rows_ix, pos_ix],)))

        try:
            out = self.model(input_ids=ids, attention_mask=attn, use_cache=False)
            return out.logits.float()
        finally:
            for hd in handles:
                hd.remove()

    @torch.inference_mode()
    def capture_boundary(self, ids, attn, depth=None, hook_site=None, seed=261127):
        mod = self._get_hook_module(hook_site)
        captured = {}

        def pre_hook(_, args):
            captured["h"] = args[0].detach()
            return None

        hd = mod.register_forward_pre_hook(pre_hook)
        try:
            self.model(input_ids=ids, attention_mask=attn, use_cache=False)
        finally:
            hd.remove()
        return captured["h"]


class OuroBackend(BaseBackend):
    def __init__(self, model_dir: Path, name: str, default_depth: int = 4):
        self.name = name
        self.default_depth = default_depth
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16, device_map={"": 0}
        ).requires_grad_(False).eval()
        self.d = int(self.model.config.hidden_size)
        self.device = next(self.model.parameters()).device

    def forward_logits(self, ids, attn, net=None, token_mask=None, depth=None, hook_site=None, select_positions=None, seed=261127):
        d_steps = int(depth if depth is not None else self.default_depth)
        core = self.model.model
        old_steps = getattr(core, "total_ut_steps", 4)
        core.total_ut_steps = d_steps
        counter = 0
        handles = []

        if net is not None:
            if hook_site == "post_norm":
                def pre_lm(_, args):
                    h = args[0]
                    hf = h.float()
                    dh = net.delta(hf)
                    if token_mask is not None:
                        dh = dh * token_mask.unsqueeze(-1).to(hf.dtype)
                    return ((hf + dh).to(h.dtype), *args[1:])
                handles.append(self.model.lm_head.register_forward_pre_hook(pre_lm))
            else:
                def pre_loop(_, args):
                    nonlocal counter
                    counter += 1
                    if counter == d_steps - 1:
                        h = args[0]
                        hf = h.float()
                        dh = net.delta(hf)
                        if token_mask is not None:
                            dh = dh * token_mask.unsqueeze(-1).to(hf.dtype)
                        return ((hf + dh).to(h.dtype), *args[1:])
                    return None
                handles.append(core.norm.register_forward_pre_hook(pre_loop))

        try:
            out = core(input_ids=ids, attention_mask=attn, use_cache=False)
            if isinstance(out, tuple) and len(out) >= 2 and isinstance(out[1], (list, tuple)) and len(out[1]) > 0:
                hidden = out[1][-1]
            else:
                hidden = out[0] if isinstance(out, tuple) else out
                if hasattr(hidden, "last_hidden_state"):
                    hidden = hidden.last_hidden_state
            if select_positions is not None:
                rows_ix, pos_ix = select_positions
                hidden = hidden[rows_ix, pos_ix]
            return self.model.lm_head(hidden).float()
        finally:
            for hd in handles:
                hd.remove()
            core.total_ut_steps = old_steps

    @torch.inference_mode()
    def capture_boundary(self, ids, attn, depth=None, hook_site=None, seed=261127):
        d_steps = int(depth if depth is not None else self.default_depth)
        core = self.model.model
        old_steps = getattr(core, "total_ut_steps", 4)
        core.total_ut_steps = d_steps
        counter = 0
        captured = {}

        def pre_hook(_, args):
            nonlocal counter
            counter += 1
            if counter == d_steps - 1:
                captured["h"] = args[0].detach()
            return None

        hd = core.norm.register_forward_pre_hook(pre_hook)
        try:
            core(input_ids=ids, attention_mask=attn, use_cache=False)
        finally:
            hd.remove()
            core.total_ut_steps = old_steps
        return captured["h"]


class HuginnBackend(BaseBackend):
    def __init__(self, model_dir: Path, default_depth: int = 32):
        import huginn_control
        self.name = "huginn-0125"
        self.default_depth = default_depth
        self.model, self.tokenizer = huginn_control.load_model(model_dir)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.d = int(self.model.config.n_embd)
        self.device = next(self.model.parameters()).device

    def forward_logits(self, ids, attn, net=None, token_mask=None, depth=None, hook_site=None, select_positions=None, seed=261127):
        d_steps = int(depth if depth is not None else self.default_depth)
        training = net is not None and torch.is_grad_enabled()
        steps = (d_steps - 1, 1) if training else (d_steps, 0)
        details = dict(return_logits=True, return_latents=False, return_head=False, return_stats=False)
        dev_idx = ids.device.index

        head_handle = None
        if select_positions is not None:
            rows_ix, pos_ix = select_positions
            head_handle = self.model.lm_head.register_forward_pre_hook(lambda _, args: (args[0][rows_ix, pos_ix],))

        orig_core = self.model.core_block_forward

        def wrapped_core(x, embedded, freqs, mask, cache, block_idx, current_step):
            if net is not None and int(current_step) == d_steps - 1:
                hf = x.float()
                dh = net.delta(hf)
                if token_mask is not None:
                    dh = dh * token_mask.unsqueeze(-1).to(hf.dtype)
                x = (hf + dh).to(x.dtype)
            return orig_core(x, embedded, freqs, mask, cache, block_idx, current_step)

        self.model.core_block_forward = wrapped_core
        try:
            with torch.random.fork_rng(devices=[dev_idx]):
                torch.cuda.manual_seed(seed)
                out = self.model(input_ids=ids, num_steps=steps, output_details=details, use_cache=False)
            return out.logits.float()
        finally:
            del self.model.core_block_forward
            if head_handle is not None:
                head_handle.remove()

    @torch.inference_mode()
    def capture_boundary(self, ids, attn, depth=None, hook_site=None, seed=261127):
        d_steps = int(depth if depth is not None else self.default_depth)
        dev_idx = ids.device.index
        with torch.random.fork_rng(devices=[dev_idx]):
            torch.cuda.manual_seed(seed)
            embedded, layer = self.model.embed_inputs(ids, use_cache=False)
            freqs = self.model.freqs_cis[:, :ids.shape[1]]
            state = self.model.initialize_state(embedded)
            for step in range(d_steps - 1):
                state, layer = self.model.core_block_forward(state, embedded, freqs, None, None, layer, step)
        return state.detach()


# ---------------------------------------------------------------------------
# Universal Geometry, Scoring, Eval, and Training Primitives
# ---------------------------------------------------------------------------
def spectrum_stats(S_u: torch.Tensor, mu: torch.Tensor):
    S_u = (S_u.double() + S_u.double().T) * 0.5
    mu = mu.double()
    S_c = S_u - torch.outer(mu, mu)
    S_c = (S_c + S_c.T) * 0.5

    ev_u, vec_u = torch.linalg.eigh(S_u)
    ev_u = ev_u.clamp_min(0.0)
    ev_u_desc = ev_u.flip(0)
    v1_u = vec_u[:, -1]

    ev_c, vec_c = torch.linalg.eigh(S_c)
    ev_c = ev_c.clamp_min(0.0)
    ev_c_desc = ev_c.flip(0)
    v1_c = vec_c[:, -1]

    tr_u = float(ev_u_desc.sum().clamp_min(1e-12))
    tr_c = float(ev_c_desc.sum().clamp_min(1e-12))
    mu_sq = float(mu.dot(mu))
    cos_v1_mu = float(abs(v1_u.dot(mu) / (v1_u.norm() * mu.norm()).clamp_min(1e-12)))
    return {
        "trace_uncent": tr_u,
        "trace_cent": tr_c,
        "mu_energy_frac": mu_sq / tr_u,
        "abs_cos_v1_unc_mu": cos_v1_mu,
        "top1_uncent": float(ev_u_desc[0] / tr_u),
        "top16_uncent": float(ev_u_desc[:16].sum() / tr_u),
        "stable_rank_uncent": tr_u / float(ev_u_desc[0].clamp_min(1e-12)),
        "top1_cent": float(ev_c_desc[0] / tr_c),
        "top16_cent": float(ev_c_desc[:16].sum() / tr_c),
        "stable_rank_cent": tr_c / float(ev_c_desc[0].clamp_min(1e-12)),
        "v1_uncent": v1_u.float(),
        "v1_cent": v1_c.float(),
        "mu": mu.float(),
    }


@torch.inference_mode()
def measure_sigma_h(backend: BaseBackend, max_docs: int = 96, depth=None, hook_site=None, batch_size: int = 4):
    rows = load_jsonl(UNLABELED_AUDIT)[:max_docs]
    d = backend.d
    moment = torch.zeros((d, d), dtype=torch.float64, device=backend.device)
    mean_sum = torch.zeros((d,), dtype=torch.float64, device=backend.device)
    total_toks = 0
    pad_id = backend.tokenizer.pad_token_id
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        enc = [backend.tokenizer.encode(r["text"], add_special_tokens=False)[:192] for r in chunk]
        mx = max(len(x) for x in enc)
        ids = torch.full((len(enc), mx), pad_id, dtype=torch.long, device=backend.device)
        attn = torch.zeros((len(enc), mx), dtype=torch.long, device=backend.device)
        for b, seq in enumerate(enc):
            ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=backend.device)
            attn[b, :len(seq)] = 1
        h = backend.capture_boundary(ids, attn, depth=depth, hook_site=hook_site, seed=261127 + i)
        valid = h[attn.bool()].double()
        moment.addmm_(valid.T, valid)
        mean_sum.add_(valid.sum(dim=0))
        total_toks += int(valid.shape[0])
    S_u = (moment / total_toks).cpu()
    mu = (mean_sum / total_toks).cpu()
    st = spectrum_stats(S_u, mu)
    st["n_tokens"] = total_toks
    return st


def encode_cloze_item(tokenizer, question: str, options: list):
    pref_str = f"Question: {question}\nAnswer:"
    p_ids = tokenizer.encode(pref_str, add_special_tokens=False)[-384:]
    seqs, spans, char_lens = [], [], []
    for opt in options:
        opt_clean = opt.strip()
        full_ids = tokenizer.encode(f"{pref_str} {opt_clean}", add_special_tokens=False)
        if len(full_ids) <= len(p_ids) or len(full_ids) > 480:
            c_ids = tokenizer.encode(" " + opt_clean, add_special_tokens=False)[:96]
            if not c_ids:
                c_ids = [tokenizer.eos_token_id]
            seq = p_ids + c_ids
            st = len(p_ids)
        else:
            seq = full_ids
            st = len(p_ids)
        seqs.append(seq)
        spans.append((st, len(seq)))
        char_lens.append(max(len(opt_clean), 1))
    return seqs, spans, char_lens


def cloze_logits_batch(backend: BaseBackend, items: list, net=None, depth=None, hook_site=None, seed=261127):
    """Differentiable or inference-mode Cloze option logits (16 * sum_logp / chars) and raw acc_norm scores."""
    tok = backend.tokenizer
    pad_id = tok.pad_token_id
    all_seqs, all_spans, all_chars, sizes = [], [], [], []
    for it in items:
        seqs, spans, clens = encode_cloze_item(tok, it["question"], it["options"])
        all_seqs.extend(seqs)
        all_spans.extend(spans)
        all_chars.extend(clens)
        sizes.append(len(seqs))

    B = len(all_seqs)
    mx = max(len(s) for s in all_seqs)
    ids = torch.full((B, mx), pad_id, dtype=torch.long, device=backend.device)
    attn = torch.zeros((B, mx), dtype=torch.long, device=backend.device)
    tmask = torch.zeros((B, mx), dtype=torch.bool, device=backend.device)
    rows_ix, pos_ix, targets = [], [], []

    for b, (seq, (st, en)) in enumerate(zip(all_seqs, all_spans)):
        ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=backend.device)
        attn[b, :len(seq)] = 1
        for p in range(st, en):
            tmask[b, p - 1] = True
            rows_ix.append(b)
            pos_ix.append(p - 1)
            targets.append(seq[p])

    r_t = torch.tensor(rows_ix, dtype=torch.long, device=backend.device)
    p_t = torch.tensor(pos_ix, dtype=torch.long, device=backend.device)
    y_t = torch.tensor(targets, dtype=torch.long, device=backend.device)

    logits = backend.forward_logits(
        ids, attn, net=net, token_mask=tmask, depth=depth, hook_site=hook_site,
        select_positions=(r_t, p_t), seed=seed
    )
    tok_lp = -F.cross_entropy(logits, y_t, reduction="none")
    sums = tok_lp.new_zeros(B).index_add_(0, r_t, tok_lp)
    chars_t = sums.new_tensor(all_chars)
    raw_norm = (sums / chars_t).split(sizes)
    scaled = (16.0 * sums / chars_t).split(sizes)
    return scaled, raw_norm


@torch.inference_mode()
def score_mc_rotated(backend: BaseBackend, items: list, depth=None, batch_size: int = 8):
    tok = backend.tokenizer
    pad_id = tok.pad_token_id
    letter_ids = [tok.encode(" " + chr(65 + i), add_special_tokens=False)[-1] for i in range(10)]
    prompts, meta = [], []
    for qi, it in enumerate(items):
        opts = it["options"]
        m = len(opts)
        shifts = list(range(m)) if m <= 5 else [0, 2, 4, 6]
        for sh in shifts:
            content = [(pos - sh) % m for pos in range(m)]
            lines = "\n".join(f"{chr(65 + pos)}. {opts[content[pos]]}" for pos in range(m))
            prompts.append(f"Question: {it['question']}\n{lines}\nAnswer:")
            meta.append((qi, m, content))

    prob_sums = [torch.zeros(len(it["options"]), dtype=torch.float64) for it in items]
    counts = [0 for _ in items]
    for st in range(0, len(prompts), batch_size):
        bp = prompts[st:st + batch_size]
        bm = meta[st:st + batch_size]
        enc = [tok.encode(p, add_special_tokens=False)[-512:] for p in bp]
        mx = max(len(x) for x in enc)
        ids = torch.full((len(enc), mx), pad_id, dtype=torch.long, device=backend.device)
        attn = torch.zeros((len(enc), mx), dtype=torch.long, device=backend.device)
        r_ix, p_ix = [], []
        for b, seq in enumerate(enc):
            ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=backend.device)
            attn[b, :len(seq)] = 1
            r_ix.append(b)
            p_ix.append(len(seq) - 1)
        logits = backend.forward_logits(
            ids, attn, net=None, depth=depth,
            select_positions=(torch.tensor(r_ix, device=backend.device), torch.tensor(p_ix, device=backend.device)),
            seed=261127 + st,
        )
        for b, (qi, m, content) in enumerate(bm):
            c_ids = [letter_ids[pos] for pos in range(m)]
            sp = torch.softmax(logits[b, c_ids], dim=0).double().cpu()
            for pos in range(m):
                prob_sums[qi][content[pos]] += sp[pos]
            counts[qi] += 1
    return [(s / c).tolist() for s, c in zip(prob_sums, counts)]


@torch.inference_mode()
def profile_source_cfit(backend: BaseBackend, items: list, depth=None, batch_size: int = 6):
    mc_p = score_mc_rotated(backend, items, depth=depth, batch_size=max(4, batch_size))
    clz_p = []
    for st in range(0, len(items), batch_size):
        sc, _ = cloze_logits_batch(backend, items[st:st + batch_size], net=None, depth=depth, seed=261127 + st)
        for lg in sc:
            clz_p.append(torch.softmax(lg.float(), dim=0).cpu().tolist())

    mc_hits, clz_hits, confs, kls, r3_hits, r2_hits = [], [], [], [], [], []
    for it, mp, cp in zip(items, mc_p, clz_p):
        g = it["gold"]
        m = len(mp)
        mc_ok = int(int(torch.tensor(mp).argmax()) == g)
        clz_ok = int(int(torch.tensor(cp).argmax()) == g)
        mc_hits.append(mc_ok)
        clz_hits.append(clz_ok)
        r3_hits.append(int(mc_ok == 1 and clz_ok == 0))
        r2_hits.append(int(mc_ok == 0))
        sp = sorted(mp, reverse=True)
        confs.append(max(0.0, (sp[0] - 1.0 / m) / (1.0 - 1.0 / m)))
        kl = sum(p * math.log(max(p, 1e-9) / max(q, 1e-9)) for p, q in zip(mp, cp))
        kls.append(kl)

    return {
        "n": len(items),
        "mc_acc": sum(mc_hits) / len(items),
        "cloze_acc": sum(clz_hits) / len(items),
        "mc_minus_cloze_pp": (sum(mc_hits) - sum(clz_hits)) / len(items) * 100.0,
        "regime3_frac": sum(r3_hits) / len(items),
        "regime2_frac": sum(r2_hits) / len(items),
        "mean_conf": sum(confs) / len(items),
        "mean_kl_mc_clz": sum(kls) / len(items),
    }, mc_p


@torch.inference_mode()
def evaluate_dev_suite(backend: BaseBackend, suite: dict, net=None, depth=None, hook_site=None, bs_cloze: int = 6, bs_mmlu: int = 8):
    if net is not None:
        net.eval()
    out = {}
    for task in ("sciq_val", "arc_easy_val", "cosmosqa_val"):
        items = suite[task]
        hits = 0
        for st in range(0, len(items), bs_cloze):
            chunk = items[st:st + bs_cloze]
            _, raw_norm = cloze_logits_batch(backend, chunk, net=net, depth=depth, hook_site=hook_site, seed=261127 + st)
            for it, rn in zip(chunk, raw_norm):
                if int(rn.argmax()) == it["gold"]:
                    hits += 1
        out[task] = hits / len(items)

    # MMLU-Val (658)
    tok = backend.tokenizer
    pad_id = tok.pad_token_id
    letter_ids = [tok.encode(" " + chr(65 + i), add_special_tokens=False)[-1] for i in range(4)]
    m_items = suite["mmlu_val"]
    m_hits = 0
    for st in range(0, len(m_items), bs_mmlu):
        chunk = m_items[st:st + bs_mmlu]
        prompts = [
            f"Question: {it['question']}\nA. {it['options'][0]}\nB. {it['options'][1]}\nC. {it['options'][2]}\nD. {it['options'][3]}\nAnswer:"
            for it in chunk
        ]
        enc = [tok.encode(p, add_special_tokens=False)[-512:] for p in prompts]
        mx = max(len(x) for x in enc)
        ids = torch.full((len(enc), mx), pad_id, dtype=torch.long, device=backend.device)
        attn = torch.zeros((len(enc), mx), dtype=torch.long, device=backend.device)
        tmask = torch.zeros((len(enc), mx), dtype=torch.bool, device=backend.device)
        r_ix, p_ix = [], []
        for b, seq in enumerate(enc):
            ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=backend.device)
            attn[b, :len(seq)] = 1
            tmask[b, len(seq) - 1] = True
            r_ix.append(b)
            p_ix.append(len(seq) - 1)
        logits = backend.forward_logits(
            ids, attn, net=net, token_mask=tmask, depth=depth, hook_site=hook_site,
            select_positions=(torch.tensor(r_ix, device=backend.device), torch.tensor(p_ix, device=backend.device)),
            seed=261127 + st,
        )
        for b, it in enumerate(chunk):
            pred = int(logits[b, letter_ids].argmax())
            if pred == it["gold"]:
                m_hits += 1
    out["mmlu_val"] = m_hits / len(m_items)
    out["cloze3"] = (out["sciq_val"] + out["arc_easy_val"] + out["cosmosqa_val"]) / 3.0
    out["macro4"] = (out["sciq_val"] + out["arc_easy_val"] + out["cosmosqa_val"] + out["mmlu_val"]) / 4.0
    return out


@torch.inference_mode()
def evaluate_wikitext_fidelity(backend: BaseBackend, net=None, depth=None, hook_site=None, max_docs: int = 48, batch_size: int = 2, nat_cache: dict = None):
    rows = load_jsonl(UNLABELED_AUDIT)[:max_docs]
    tok = backend.tokenizer
    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id
    nll_nat_s = nll_ctl_s = eos_abs_s = 0.0
    tot = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        enc = [tok.encode(r["text"], add_special_tokens=False)[:160] for r in chunk]
        mx = max(len(x) for x in enc)
        ids = torch.full((len(enc), mx), pad_id, dtype=torch.long, device=backend.device)
        attn = torch.zeros((len(enc), mx), dtype=torch.long, device=backend.device)
        for b, seq in enumerate(enc):
            ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=backend.device)
            attn[b, :len(seq)] = 1
        valid = attn[:, 1:].bool()
        rows_v, pos_v = valid.nonzero(as_tuple=True)
        targets = ids[rows_v, pos_v + 1]

        ckey = (depth, i)
        if nat_cache is not None and ckey in nat_cache:
            n_nll, n_eos = nat_cache[ckey]
        else:
            z0 = backend.forward_logits(ids, attn, net=None, depth=depth, select_positions=(rows_v, pos_v), seed=261127 + i)
            lp0 = F.log_softmax(z0, dim=-1)
            n_nll = float(-lp0.gather(1, targets[:, None]).sum())
            n_eos = lp0[:, eos_id].exp().cpu() if eos_id is not None else None
            if nat_cache is not None:
                nat_cache[ckey] = (n_nll, n_eos)

        if net is None:
            nll_nat_s += n_nll
            nll_ctl_s += n_nll
        else:
            z1 = backend.forward_logits(ids, attn, net=net, depth=depth, hook_site=hook_site, select_positions=(rows_v, pos_v), seed=261127 + i)
            lp1 = F.log_softmax(z1, dim=-1)
            nll_nat_s += n_nll
            nll_ctl_s += float(-lp1.gather(1, targets[:, None]).sum())
            if eos_id is not None and n_eos is not None:
                eos_abs_s += float((lp1[:, eos_id].exp().cpu() - n_eos).abs().sum())
        tot += int(targets.numel())
    nll_nat = nll_nat_s / tot
    nll_ctl = nll_ctl_s / tot
    dnll = nll_ctl - nll_nat
    return {
        "nll_nat": nll_nat,
        "nll_ctrl": nll_ctl,
        "delta_nll": dnll,
        "ppl_ratio": math.exp(min(dnll, 20.0)),
        "eos_abs_shift": eos_abs_s / tot,
    }


def train_universal_rank1(
    backend: BaseBackend,
    items: list,
    objective: str = "hard_pseudo",
    mc_probs: list = None,
    fixed_w: torch.Tensor = None,
    train_depths=None,
    hook_site=None,
    steps: int = 560,
    micro_bs: int = 2,
    grad_accum: int = 2,
    lr: float = 1e-3,
    wd: float = 0.03,
    trust_lam: float = 0.0,
    trust_seqs: list = None,
    seed: int = 26112701,
    tag: str = "",
):
    torch.manual_seed(seed)
    net = UniversalRank1(backend.d, bound=0.45, fixed_w=fixed_w).to(backend.device)
    trainable = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
    depths = list(train_depths) if train_depths else [backend.default_depth]
    rng = random.Random(f"p2_tr/{backend.name}/{tag}/{seed}")
    order = list(range(len(items)))
    rng.shuffle(order)
    ptr = 0
    pad_id = backend.tokenizer.pad_token_id
    trust_nat_cache = {}

    def next_indices(k):
        nonlocal ptr
        out = []
        for _ in range(k):
            if ptr >= len(order):
                rng.shuffle(order)
                ptr = 0
            out.append(order[ptr])
            ptr += 1
        return out

    def nll_pen(seq_list, d_step, s_seed):
        mx = max(len(s) for s in seq_list)
        ids = torch.full((len(seq_list), mx), pad_id, dtype=torch.long, device=backend.device)
        attn = torch.zeros((len(seq_list), mx), dtype=torch.long, device=backend.device)
        for b, s in enumerate(seq_list):
            ids[b, :len(s)] = torch.tensor(s, dtype=torch.long, device=backend.device)
            attn[b, :len(s)] = 1
        valid = attn[:, 1:].bool()
        rows_v, pos_v = valid.nonzero(as_tuple=True)
        targets = ids[rows_v, pos_v + 1]
        ckey = (d_step, tuple(len(s) for s in seq_list), int(ids[0, 0]))
        if ckey not in trust_nat_cache:
            with torch.no_grad():
                z0 = backend.forward_logits(ids, attn, net=None, depth=d_step, select_positions=(rows_v, pos_v), seed=s_seed)
                trust_nat_cache[ckey] = float(F.cross_entropy(z0, targets).detach())
        z1 = backend.forward_logits(ids, attn, net=net, depth=d_step, hook_site=hook_site, select_positions=(rows_v, pos_v), seed=s_seed)
        return F.relu(F.cross_entropy(z1, targets) - trust_nat_cache[ckey] - 0.02)

    t0 = time.perf_counter()
    losses = []
    for step in range(1, steps + 1):
        d_step = depths[(step - 1) % len(depths)]
        opt.zero_grad(set_to_none=True)
        step_loss_val = 0.0
        for ga in range(grad_accum):
            idxs = next_indices(micro_bs)
            mb = [items[idx] for idx in idxs]
            with torch.enable_grad():
                scaled, _ = cloze_logits_batch(
                    backend, mb, net=net, depth=d_step, hook_site=hook_site, seed=seed + step * 10 + ga
                )
                w_losses, w_sum = [], 0.0
                for lg, idx, it in zip(scaled, idxs, mb):
                    log_p = F.log_softmax(lg, dim=0)
                    if objective == "human_gold":
                        l_i = -log_p[it["gold"]]
                        w_i = 1.0
                    elif objective == "hard_pseudo":
                        mp = mc_probs[idx]
                        m = len(mp)
                        sp = sorted(mp, reverse=True)
                        conf = max(1e-4, (sp[0] - 1.0 / m) / (1.0 - 1.0 / m))
                        plab = int(torch.tensor(mp).argmax())
                        l_i = -log_p[plab]
                        w_i = conf
                    else:
                        raise ValueError(objective)
                    w_losses.append(l_i * w_i)
                    w_sum += w_i
                loss = (sum(w_losses) / max(w_sum, 1e-6)) / grad_accum
                loss.backward()
                step_loss_val += float(loss.detach())

        if trust_lam > 0.0 and trust_seqs and step % 2 == 0:
            t_mb = [trust_seqs[(step + j) % len(trust_seqs)] for j in range(2)]
            with torch.enable_grad():
                t_loss = trust_lam * nll_pen(t_mb, d_step, seed + step)
                t_loss.backward()
                step_loss_val += float(t_loss.detach())

        grad = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        assert grad.isfinite().item(), f"Non-finite grad in {backend.name}/{tag}"
        opt.step()
        losses.append(step_loss_val)
        if step % max(steps // 3, 1) == 0 or step == steps:
            avg_l = sum(losses[-60:]) / min(len(losses), 60)
            print(f"  [{backend.name} | {tag}] step={step}/{steps} loss={avg_l:.4f} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return net, {"n_params": net.n_params(), "steps": steps, "seconds": time.perf_counter() - t0}


def build_trust_seqs(tokenizer, max_docs=96):
    rows = load_jsonl(UNLABELED_FIT)[:max_docs]
    return [s for s in (tokenizer.encode(r["text"], add_special_tokens=False)[:144] for r in rows) if len(s) >= 48]


def spearman_rho(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        for rank, idx in enumerate(order):
            r[idx] = float(rank)
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Module 1: Qwen2.5-1.5B (Non-Recurrent Standard Transformer Control)
# ---------------------------------------------------------------------------
def run_module1_qwen(suite: dict):
    print("\n================ MODULE 1: QWEN2.5-1.5B (NON-RECURRENT CONTROL) ================", flush=True)
    out_path = P2_OUT / "MODULE1_QWEN25_1P5B.json"
    res = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {
        "model": "Qwen2.5-1.5B", "geometry": {}, "source_profiles": {}, "arms": {}
    }
    if res.get("done"):
        print("  Module 1 already complete, skipping.", flush=True)
        return res

    backend = QwenBackend(MODELS_DIR / "Qwen2.5-1.5B", default_site="layer_21")
    nat_fid_cache = {}
    if "native" not in res:
        res["native"] = evaluate_dev_suite(backend, suite, net=None, bs_cloze=8, bs_mmlu=12)
        res["native_fidelity"] = evaluate_wikitext_fidelity(backend, net=None, nat_cache=nat_fid_cache)
        print(f"  Qwen2.5-1.5B Native: Clz3={res['native']['cloze3']*100:.2f}% MMLU={res['native']['mmlu_val']*100:.2f}% Mac4={res['native']['macro4']*100:.2f}% NLL={res['native_fidelity']['nll_nat']:.3f}", flush=True)
        save_json(out_path, res)

    # 1. Centered vs Uncentered Sigma_h across depths
    mu_l21 = None
    for site in ("layer_7", "layer_14", "layer_21", "layer_24", "layer_26", "post_norm"):
        if site not in res["geometry"] or site == "layer_21":
            st = measure_sigma_h(backend, max_docs=96, hook_site=site, batch_size=4)
            if site == "layer_21":
                mu_l21 = st["mu"]
            res["geometry"][site] = {k: v for k, v in st.items() if not isinstance(v, torch.Tensor)}
            print(f"  [Qwen Sigma {site:9s}] top1_unc={st['top1_uncent']*100:5.1f}% sr_unc={st['stable_rank_uncent']:.2f} | top1_cent={st['top1_cent']*100:5.1f}% sr_cent={st['stable_rank_cent']:.2f} | mu_frac={st['mu_energy_frac']*100:5.1f}% | cos(v1,mu)={st['abs_cos_v1_unc_mu']:.3f}", flush=True)
            save_json(out_path, res)

    # 2. Native CFIT Regime-III profile on 8 sources
    arc_items = load_source_items("arc_challenge", max_items=800)
    med_items = load_source_items("medmcqa", max_items=800)
    for sname in EIGHT_SOURCES:
        if sname not in res["source_profiles"]:
            s_it = load_source_items(sname, max_items=200)
            prof, _ = profile_source_cfit(backend, s_it, batch_size=8)
            res["source_profiles"][sname] = prof
            print(f"  [Qwen CFIT {sname:14s}] MC={prof['mc_acc']*100:5.1f}% Clz={prof['cloze_acc']*100:5.1f}% gap={prof['mc_minus_cloze_pp']:+5.1f} R3={prof['regime3_frac']*100:4.1f}% R2={prof['regime2_frac']*100:4.1f}%", flush=True)
            save_json(out_path, res)

    arc_mc = score_mc_rotated(backend, arc_items, batch_size=12)
    med_mc = score_mc_rotated(backend, med_items, batch_size=12)
    trust_seqs = build_trust_seqs(backend.tokenizer)
    torch.manual_seed(99991)
    rand_w = torch.randn(backend.d)

    plan = [
        ("S0_ARC_HumanGold_L21_3072", arc_items, None, "human_gold", None, "layer_21", 0.0, 26112701),
        ("S0_ARC_HumanGold_L21_3072_s2", arc_items, None, "human_gold", None, "layer_21", 0.0, 26112702),
        ("S1_ARC_ZeroLabel_L21_3072", arc_items, arc_mc, "hard_pseudo", None, "layer_21", 0.0, 26112701),
        ("S1_ARC_ZeroLabel_L21_3072_s2", arc_items, arc_mc, "hard_pseudo", None, "layer_21", 0.0, 26112702),
        ("S1_ARC_FixW_MeanMu_L21_1536", arc_items, arc_mc, "hard_pseudo", mu_l21, "layer_21", 0.0, 26112701),
        ("S1_ARC_FixW_Random_L21_1536", arc_items, arc_mc, "hard_pseudo", rand_w, "layer_21", 0.0, 26112701),
        ("S1_ARC_ZeroLabel_Pareto_L21_lam100", arc_items, arc_mc, "hard_pseudo", None, "layer_21", 1.0, 26112701),
        ("S1_ARC_ZeroLabel_L07_3072", arc_items, arc_mc, "hard_pseudo", None, "layer_7", 0.0, 26112701),
        ("S1_ARC_ZeroLabel_L14_3072", arc_items, arc_mc, "hard_pseudo", None, "layer_14", 0.0, 26112701),
        ("S1_ARC_ZeroLabel_L26_3072", arc_items, arc_mc, "hard_pseudo", None, "layer_26", 0.0, 26112701),
        ("S1_ARC_ZeroLabel_PostNorm_LMHead_3072", arc_items, arc_mc, "hard_pseudo", None, "post_norm", 0.0, 26112701),
        ("S3_MedMCQA_HumanGold_L21_3072", med_items, None, "human_gold", None, "layer_21", 0.0, 26112701),
        ("S3_MedMCQA_ZeroLabel_L21_3072", med_items, med_mc, "hard_pseudo", None, "layer_21", 0.0, 26112701),
    ]

    for arm_name, tr_items, mcp, obj, fw, site, lam, sd in plan:
        if arm_name in res["arms"]:
            continue
        net, meta = train_universal_rank1(
            backend, tr_items, objective=obj, mc_probs=mcp, fixed_w=fw, hook_site=site,
            steps=560, micro_bs=2, grad_accum=2, trust_lam=lam, trust_seqs=trust_seqs, seed=sd, tag=arm_name
        )
        ev = evaluate_dev_suite(backend, suite, net=net, hook_site=site, bs_cloze=8, bs_mmlu=12)
        fid = evaluate_wikitext_fidelity(backend, net=net, hook_site=site, nat_cache=nat_fid_cache)
        d_m4 = (ev["macro4"] - res["native"]["macro4"]) * 100.0
        d_c3 = (ev["cloze3"] - res["native"]["cloze3"]) * 100.0
        d_mm = (ev["mmlu_val"] - res["native"]["mmlu_val"]) * 100.0
        res["arms"][arm_name] = {
            "site": site, "objective": obj, "trust_lam": lam, "seed": sd,
            "train": meta, "eval": ev, "fidelity": fid,
            "delta_macro4_pp": d_m4, "delta_cloze3_pp": d_c3, "delta_mmlu_pp": d_mm,
        }
        print(f"  => [Qwen] {arm_name:38s} P={meta['n_params']:4d} dMac4={d_m4:+6.2f} dClz3={d_c3:+6.2f} dMMLU={d_mm:+5.2f} dNLL={fid['delta_nll']:+.3f}", flush=True)
        save_json(out_path, res)

    res["done"] = True
    save_json(out_path, res)
    backend.cleanup()
    return res


# ---------------------------------------------------------------------------
# Module 2: Ouro-2.6B (Scaling + K={2,4,8} + Cross-Model CFIT 8 Sources)
# ---------------------------------------------------------------------------
def run_module2_ouro26b(suite: dict):
    print("\n================ MODULE 2: OURO-2.6B (SCALING + K={2,4,8} + CFIT 8 SOURCES) ================", flush=True)
    out_path = P2_OUT / "MODULE2_OURO_2P6B.json"
    res = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {
        "model": "Ouro-2.6B", "depth_scaling": {}, "source_profiles": {}, "source_supervised_arms": {}, "core_arms": {}
    }
    if res.get("done"):
        print("  Module 2 already complete, skipping.", flush=True)
        return res

    backend = OuroBackend(MODELS_DIR / "ouro-2.6b", name="Ouro-2.6B", default_depth=4)
    arc_items = load_source_items("arc_challenge", max_items=800)
    med_items = load_source_items("medmcqa", max_items=800)
    trust_seqs = build_trust_seqs(backend.tokenizer)
    nat_fid_cache = {}

    # 1. Recurrence Depth Scaling K in {2, 4, 8}
    mu_k4 = None
    for K in (2, 4, 8):
        kkey = f"K{K}"
        if kkey not in res["depth_scaling"] or K == 4:
            st = measure_sigma_h(backend, max_docs=64, depth=K, batch_size=2)
            if K == 4:
                mu_k4 = st["mu"]
            if kkey not in res["depth_scaling"]:
                ev_nat = evaluate_dev_suite(backend, suite, net=None, depth=K, bs_cloze=4, bs_mmlu=6)
                prof_arc, arc_mc_k = profile_source_cfit(backend, arc_items[:200], depth=K, batch_size=6)
                arc_mc_full = score_mc_rotated(backend, arc_items, depth=K, batch_size=6)
                net_k, meta_k = train_universal_rank1(
                    backend, arc_items, objective="hard_pseudo", mc_probs=arc_mc_full,
                    train_depths=[K], steps=420, micro_bs=2, grad_accum=2, seed=26112701, tag=f"ZeroLabel_{kkey}"
                )
                ev_ctl = evaluate_dev_suite(backend, suite, net=net_k, depth=K, bs_cloze=4, bs_mmlu=6)
                fid_ctl = evaluate_wikitext_fidelity(backend, net=net_k, depth=K, max_docs=32, batch_size=2, nat_cache=nat_fid_cache)
                res["depth_scaling"][kkey] = {
                    "K": K,
                    "geometry": {k: v for k, v in st.items() if not isinstance(v, torch.Tensor)},
                    "arc_profile_200": prof_arc,
                    "native": ev_nat,
                    "zero_label_ctrl": ev_ctl,
                    "fidelity": fid_ctl,
                    "train": meta_k,
                    "delta_macro4_pp": (ev_ctl["macro4"] - ev_nat["macro4"]) * 100.0,
                    "delta_cloze3_pp": (ev_ctl["cloze3"] - ev_nat["cloze3"]) * 100.0,
                    "delta_mmlu_pp": (ev_ctl["mmlu_val"] - ev_nat["mmlu_val"]) * 100.0,
                }
                print(
                    f"  [Ouro-2.6B {kkey}] sr_unc={st['stable_rank_uncent']:.2f} mu_f={st['mu_energy_frac']*100:.1f}% | "
                    f"ARC gap={prof_arc['mc_minus_cloze_pp']:+.1f} | Nat Mac4={ev_nat['macro4']*100:.2f}% -> "
                    f"dMac4={res['depth_scaling'][kkey]['delta_macro4_pp']:+.2f} dClz3={res['depth_scaling'][kkey]['delta_cloze3_pp']:+.2f} dMMLU={res['depth_scaling'][kkey]['delta_mmlu_pp']:+.2f}",
                    flush=True,
                )
                save_json(out_path, res)

    nat_k4 = res["depth_scaling"]["K4"]["native"]

    # 2. Core Phase 1 Replication on Ouro-2.6B (K=4, multi-depth (2,3,4) training matching 1.4B!)
    arc_mc_k4 = score_mc_rotated(backend, arc_items, depth=4, batch_size=6)
    med_mc_k4 = score_mc_rotated(backend, med_items, depth=4, batch_size=6)

    # Also test Zero-Shot Transfer of Ouro-1.4B Best_ZeroLabel_4096 onto Ouro-2.6B!
    if "Transfer_1p4B_ZeroLabel_4096" not in res["core_arms"]:
        z14_path = V4_ROOT / "exp_t2a_self_distill" / "self_arc_hardpseudo.s26112681.pt"
        if z14_path.exists():
            obj = torch.load(z14_path, map_location="cpu", weights_only=False)
            st_d = obj["state"]
            net_tr = UniversalRank1(2048, bound=0.45).to(backend.device)
            net_tr.encode.weight.data.copy_(st_d["encode.weight"].float().to(backend.device))
            net_tr.out.weight.data.copy_(st_d["out.weight"].float().to(backend.device))
            ev = evaluate_dev_suite(backend, suite, net=net_tr, depth=4, bs_cloze=4, bs_mmlu=6)
            fid = evaluate_wikitext_fidelity(backend, net=net_tr, depth=4, max_docs=32, batch_size=2, nat_cache=nat_fid_cache)
            res["core_arms"]["Transfer_1p4B_ZeroLabel_4096"] = {
                "eval": ev, "fidelity": fid,
                "delta_macro4_pp": (ev["macro4"] - nat_k4["macro4"]) * 100.0,
                "delta_cloze3_pp": (ev["cloze3"] - nat_k4["cloze3"]) * 100.0,
                "delta_mmlu_pp": (ev["mmlu_val"] - nat_k4["mmlu_val"]) * 100.0,
            }
            print(f"  => [Ouro-2.6B Transfer 1.4B] dMac4={res['core_arms']['Transfer_1p4B_ZeroLabel_4096']['delta_macro4_pp']:+.2f} dClz3={res['core_arms']['Transfer_1p4B_ZeroLabel_4096']['delta_cloze3_pp']:+.2f} dMMLU={res['core_arms']['Transfer_1p4B_ZeroLabel_4096']['delta_mmlu_pp']:+.2f}", flush=True)
            save_json(out_path, res)

    core_plan = [
        ("Ouro26B_S0_ARC_HumanGold_4096", arc_items, None, "human_gold", None, 0.0),
        ("Ouro26B_S1_ARC_ZeroLabel_4096", arc_items, arc_mc_k4, "hard_pseudo", None, 0.0),
        ("Ouro26B_S1_ARC_FixW_MeanMu_2048", arc_items, arc_mc_k4, "hard_pseudo", mu_k4, 0.0),
        ("Ouro26B_S1_ARC_Pareto_lam100_4096", arc_items, arc_mc_k4, "hard_pseudo", None, 1.0),
        ("Ouro26B_S3_MedMCQA_ZeroLabel_4096", med_items, med_mc_k4, "hard_pseudo", None, 0.0),
    ]
    for arm_name, tr_items, mcp, obj, fw, lam in core_plan:
        if arm_name in res["core_arms"]:
            continue
        net, meta = train_universal_rank1(
            backend, tr_items, objective=obj, mc_probs=mcp, fixed_w=fw,
            train_depths=[2, 3, 4], steps=560, micro_bs=2, grad_accum=2,
            trust_lam=lam, trust_seqs=trust_seqs, seed=26112701, tag=arm_name
        )
        ev = evaluate_dev_suite(backend, suite, net=net, depth=4, bs_cloze=4, bs_mmlu=6)
        fid = evaluate_wikitext_fidelity(backend, net=net, depth=4, max_docs=32, batch_size=2, nat_cache=nat_fid_cache)
        ckpt = P2_OUT / "weights" / f"{arm_name}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()}, "meta": meta}, ckpt)
        res["core_arms"][arm_name] = {
            "train": meta, "eval": ev, "fidelity": fid,
            "delta_macro4_pp": (ev["macro4"] - nat_k4["macro4"]) * 100.0,
            "delta_cloze3_pp": (ev["cloze3"] - nat_k4["cloze3"]) * 100.0,
            "delta_mmlu_pp": (ev["mmlu_val"] - nat_k4["mmlu_val"]) * 100.0,
        }
        print(f"  => [Ouro-2.6B] {arm_name:36s} P={meta['n_params']:4d} dMac4={res['core_arms'][arm_name]['delta_macro4_pp']:+6.2f} dClz3={res['core_arms'][arm_name]['delta_cloze3_pp']:+6.2f} dMMLU={res['core_arms'][arm_name]['delta_mmlu_pp']:+5.2f} dNLL={fid['delta_nll']:+.3f}", flush=True)
        save_json(out_path, res)

    # 3. Cross-Model CFIT Source Ranking V(D; Ouro-2.6B) across all 8 sources
    for sname in EIGHT_SOURCES:
        s_items = load_source_items(sname, max_items=800)
        if sname not in res["source_profiles"]:
            prof, _ = profile_source_cfit(backend, s_items[:200], depth=4, batch_size=6)
            res["source_profiles"][sname] = prof
            print(f"  [Ouro-2.6B CFIT {sname:14s}] MC={prof['mc_acc']*100:5.1f}% Clz={prof['cloze_acc']*100:5.1f}% gap={prof['mc_minus_cloze_pp']:+5.1f} R3={prof['regime3_frac']*100:4.1f}% R2={prof['regime2_frac']*100:4.1f}%", flush=True)
            save_json(out_path, res)
        if sname not in res["source_supervised_arms"]:
            net, meta = train_universal_rank1(
                backend, s_items, objective="human_gold", train_depths=[2, 3, 4],
                steps=480, micro_bs=2, grad_accum=2, seed=26112701, tag=f"Src_{sname}"
            )
            ev = evaluate_dev_suite(backend, suite, net=net, depth=4, bs_cloze=4, bs_mmlu=6)
            fid = evaluate_wikitext_fidelity(backend, net=net, depth=4, max_docs=32, batch_size=2, nat_cache=nat_fid_cache)
            res["source_supervised_arms"][sname] = {
                "train": meta, "eval": ev, "fidelity": fid,
                "delta_macro4_pp": (ev["macro4"] - nat_k4["macro4"]) * 100.0,
                "delta_cloze3_pp": (ev["cloze3"] - nat_k4["cloze3"]) * 100.0,
                "delta_mmlu_pp": (ev["mmlu_val"] - nat_k4["mmlu_val"]) * 100.0,
            }
            print(f"  => [Ouro-2.6B V({sname:13s})] dMac4={res['source_supervised_arms'][sname]['delta_macro4_pp']:+6.2f} dClz3={res['source_supervised_arms'][sname]['delta_cloze3_pp']:+6.2f} dMMLU={res['source_supervised_arms'][sname]['delta_mmlu_pp']:+5.2f} dNLL={fid['delta_nll']:+.3f}", flush=True)
            save_json(out_path, res)

    # Compute Spearman rho between Ouro-2.6B CFIT predictors and actual V(D; Ouro-2.6B)
    gains = [res["source_supervised_arms"][s]["delta_macro4_pp"] for s in EIGHT_SOURCES]
    r3s = [res["source_profiles"][s]["regime3_frac"] for s in EIGHT_SOURCES]
    gaps = [res["source_profiles"][s]["mc_minus_cloze_pp"] for s in EIGHT_SOURCES]
    r2_neg = [-res["source_profiles"][s]["regime2_frac"] for s in EIGHT_SOURCES]
    pres = [-res["source_supervised_arms"][s]["fidelity"]["delta_nll"] for s in EIGHT_SOURCES]
    res["cfit_2p6b_spearman"] = {
        "regime3_frac_vs_gain": spearman_rho(r3s, gains),
        "mc_minus_cloze_vs_gain": spearman_rho(gaps, gains),
        "neg_regime2_vs_gain": spearman_rho(r2_neg, gains),
        "neg_dnll_vs_gain": spearman_rho(pres, gains),
    }
    print("  Ouro-2.6B 8-Source CFIT Spearman:", json.dumps(res["cfit_2p6b_spearman"], indent=2), flush=True)
    res["done"] = True
    save_json(out_path, res)
    backend.cleanup()
    return res


# ---------------------------------------------------------------------------
# Module 3: Huginn-0125 (K=32, Different Looped Arch, d=5280)
# ---------------------------------------------------------------------------
def run_module3_huginn(suite: dict):
    print("\n================ MODULE 3: HUGINN-0125 (K=32, d=5280, CROSS-ARCH LOOPED) ================", flush=True)
    out_path = P2_OUT / "MODULE3_HUGINN_K32.json"
    res = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {
        "model": "Huginn-0125-K32", "geometry": {}, "source_profiles": {}, "arms": {}
    }
    if res.get("done"):
        print("  Module 3 already complete, skipping.", flush=True)
        return res

    backend = HuginnBackend(MODELS_DIR / "huginn-0125", default_depth=32)
    nat_fid_cache = {}
    if "native" not in res:
        res["native"] = evaluate_dev_suite(backend, suite, net=None, depth=32, bs_cloze=4, bs_mmlu=6)
        res["native_fidelity"] = evaluate_wikitext_fidelity(backend, net=None, depth=32, max_docs=32, batch_size=2, nat_cache=nat_fid_cache)
        print(f"  Huginn-0125 K32 Native: Clz3={res['native']['cloze3']*100:.2f}% MMLU={res['native']['mmlu_val']*100:.2f}% Mac4={res['native']['macro4']*100:.2f}% NLL={res['native_fidelity']['nll_nat']:.3f}", flush=True)
        save_json(out_path, res)

    st = measure_sigma_h(backend, max_docs=48, depth=32, batch_size=2)
    mu_hug = st["mu"]
    res["geometry"]["K32"] = {k: v for k, v in st.items() if not isinstance(v, torch.Tensor)}
    print(f"  [Huginn Sigma K32] top1_unc={st['top1_uncent']*100:5.1f}% sr_unc={st['stable_rank_uncent']:.2f} | top1_cent={st['top1_cent']*100:5.1f}% sr_cent={st['stable_rank_cent']:.2f} | mu_frac={st['mu_energy_frac']*100:5.1f}% | cos(v1,mu)={st['abs_cos_v1_unc_mu']:.3f}", flush=True)
    save_json(out_path, res)

    for sname in EIGHT_SOURCES:
        if sname not in res["source_profiles"]:
            s_it = load_source_items(sname, max_items=120)
            prof, _ = profile_source_cfit(backend, s_it, depth=32, batch_size=4)
            res["source_profiles"][sname] = prof
            print(f"  [Huginn CFIT {sname:14s}] MC={prof['mc_acc']*100:5.1f}% Clz={prof['cloze_acc']*100:5.1f}% gap={prof['mc_minus_cloze_pp']:+5.1f} R3={prof['regime3_frac']*100:4.1f}% R2={prof['regime2_frac']*100:4.1f}%", flush=True)
            save_json(out_path, res)

    arc_items = load_source_items("arc_challenge", max_items=600)
    med_items = load_source_items("medmcqa", max_items=600)
    arc_mc = score_mc_rotated(backend, arc_items, depth=32, batch_size=6)
    med_mc = score_mc_rotated(backend, med_items, depth=32, batch_size=6)
    trust_seqs = build_trust_seqs(backend.tokenizer, max_docs=64)

    plan = [
        ("Huginn_S0_ARC_HumanGold_10560", arc_items, None, "human_gold", None, 0.0),
        ("Huginn_S1_ARC_ZeroLabel_10560", arc_items, arc_mc, "hard_pseudo", None, 0.0),
        ("Huginn_S1_ARC_FixW_MeanMu_5280", arc_items, arc_mc, "hard_pseudo", mu_hug, 0.0),
        ("Huginn_S1_ARC_Pareto_lam100_10560", arc_items, arc_mc, "hard_pseudo", None, 1.0),
        ("Huginn_S3_MedMCQA_HumanGold_10560", med_items, None, "human_gold", None, 0.0),
        ("Huginn_S3_MedMCQA_ZeroLabel_10560", med_items, med_mc, "hard_pseudo", None, 0.0),
    ]
    for arm_name, tr_items, mcp, obj, fw, lam in plan:
        if arm_name in res["arms"]:
            continue
        net, meta = train_universal_rank1(
            backend, tr_items, objective=obj, mc_probs=mcp, fixed_w=fw,
            train_depths=[8, 16, 32], steps=280, micro_bs=2, grad_accum=2,
            trust_lam=lam, trust_seqs=trust_seqs, seed=26112701, tag=arm_name
        )
        ev = evaluate_dev_suite(backend, suite, net=net, depth=32, bs_cloze=4, bs_mmlu=6)
        fid = evaluate_wikitext_fidelity(backend, net=net, depth=32, max_docs=32, batch_size=2, nat_cache=nat_fid_cache)
        ckpt = P2_OUT / "weights" / f"{arm_name}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()}, "meta": meta}, ckpt)
        res["arms"][arm_name] = {
            "train": meta, "eval": ev, "fidelity": fid,
            "delta_macro4_pp": (ev["macro4"] - res["native"]["macro4"]) * 100.0,
            "delta_cloze3_pp": (ev["cloze3"] - res["native"]["cloze3"]) * 100.0,
            "delta_mmlu_pp": (ev["mmlu_val"] - res["native"]["mmlu_val"]) * 100.0,
        }
        print(f"  => [Huginn] {arm_name:36s} P={meta['n_params']:5d} dMac4={res['arms'][arm_name]['delta_macro4_pp']:+6.2f} dClz3={res['arms'][arm_name]['delta_cloze3_pp']:+6.2f} dMMLU={res['arms'][arm_name]['delta_mmlu_pp']:+5.2f} dNLL={fid['delta_nll']:+.3f}", flush=True)
        save_json(out_path, res)

    res["done"] = True
    save_json(out_path, res)
    backend.cleanup()
    return res


# ---------------------------------------------------------------------------
# Module 4: Ouro-1.4B Recurrence Depth Scaling K in {2, 4, 8}
# ---------------------------------------------------------------------------
def run_module4_ouro14b_depths(suite: dict):
    print("\n================ MODULE 4: OURO-1.4B DEPTH SCALING K in {2, 4, 8} ================", flush=True)
    out_path = P2_OUT / "MODULE4_OURO_1P4B_DEPTHS.json"
    res = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {
        "model": "Ouro-1.4B", "depth_scaling": {}
    }
    if res.get("done"):
        print("  Module 4 already complete, skipping.", flush=True)
        return res

    cfg = json.loads(E0_CONFIG_PATH.read_text(encoding="utf-8"))
    backend = OuroBackend(Path(cfg["model"]), name="Ouro-1.4B", default_depth=4)
    arc_items = load_source_items("arc_challenge", max_items=800)
    nat_fid_cache = {}

    for K in (2, 4, 8):
        kkey = f"K{K}"
        if kkey in res["depth_scaling"]:
            continue
        st = measure_sigma_h(backend, max_docs=64, depth=K, batch_size=4)
        ev_nat = evaluate_dev_suite(backend, suite, net=None, depth=K, bs_cloze=6, bs_mmlu=8)
        prof_arc, _ = profile_source_cfit(backend, arc_items[:200], depth=K, batch_size=8)
        arc_mc_full = score_mc_rotated(backend, arc_items, depth=K, batch_size=8)
        net_k, meta_k = train_universal_rank1(
            backend, arc_items, objective="hard_pseudo", mc_probs=arc_mc_full,
            train_depths=[K], steps=560, micro_bs=2, grad_accum=2, seed=26112701, tag=f"ZeroLabel_{kkey}"
        )
        ev_ctl = evaluate_dev_suite(backend, suite, net=net_k, depth=K, bs_cloze=6, bs_mmlu=8)
        fid_ctl = evaluate_wikitext_fidelity(backend, net=net_k, depth=K, max_docs=32, batch_size=2, nat_cache=nat_fid_cache)
        res["depth_scaling"][kkey] = {
            "K": K,
            "geometry": {k: v for k, v in st.items() if not isinstance(v, torch.Tensor)},
            "arc_profile_200": prof_arc,
            "native": ev_nat,
            "zero_label_ctrl": ev_ctl,
            "fidelity": fid_ctl,
            "train": meta_k,
            "delta_macro4_pp": (ev_ctl["macro4"] - ev_nat["macro4"]) * 100.0,
            "delta_cloze3_pp": (ev_ctl["cloze3"] - ev_nat["cloze3"]) * 100.0,
            "delta_mmlu_pp": (ev_ctl["mmlu_val"] - ev_nat["mmlu_val"]) * 100.0,
        }
        print(
            f"  [Ouro-1.4B {kkey}] sr_unc={st['stable_rank_uncent']:.2f} mu_f={st['mu_energy_frac']*100:.1f}% | "
            f"ARC gap={prof_arc['mc_minus_cloze_pp']:+.1f} | Nat Mac4={ev_nat['macro4']*100:.2f}% -> "
            f"dMac4={res['depth_scaling'][kkey]['delta_macro4_pp']:+.2f} dClz3={res['depth_scaling'][kkey]['delta_cloze3_pp']:+.2f} dMMLU={res['depth_scaling'][kkey]['delta_mmlu_pp']:+.2f}",
            flush=True,
        )
        save_json(out_path, res)

    res["done"] = True
    save_json(out_path, res)
    backend.cleanup()
    return res


def main():
    torch.set_num_threads(4)
    P2_OUT.mkdir(parents=True, exist_ok=True)
    suite = load_dev_suite()
    print(f"Loaded Dev Suite: " + ", ".join(f"{k}={len(v)}" for k, v in suite.items()), flush=True)

    run_module1_qwen(suite)
    run_module2_ouro26b(suite)
    run_module3_huginn(suite)
    run_module4_ouro14b_depths(suite)
    print("\nALL_PHASE2_CROSS_MODEL_DONE", flush=True)


if __name__ == "__main__":
    main()
