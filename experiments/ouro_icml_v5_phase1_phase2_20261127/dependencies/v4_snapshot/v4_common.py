"""Shared utilities for v4 RELMT + CFIT experiments on Ouro-1.4B Base."""

import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

V3_ROOT = Path("/root/ouro-v3-controllability-20261126")
E0_CONFIG_PATH = V3_ROOT / "stage1_e0" / "E0_CONFIG.json"
ENV_MANIFEST_PATH = V3_ROOT / "stage0_env" / "ENV_MANIFEST.json"
EXT_DEV_DIR = V3_ROOT / "stage0_env" / "UNIFIED_EXT_DEV"
UNLABELED_FIT = V3_ROOT / "stage0_env" / "UNLABELED_CORPUS" / "fit_unlabeled.jsonl"
UNLABELED_AUDIT = V3_ROOT / "stage0_env" / "UNLABELED_CORPUS" / "audit_unlabeled.jsonl"

DEPTHS = (2, 3, 4)
RECIPE = {"effective_batch": 4, "learning_rate": 0.001, "weight_decay": 0.03, "clip": 1.0}
BASE_SHA = "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_base_runtime():
    import minimal
    cfg = json.loads(E0_CONFIG_PATH.read_text(encoding="utf-8"))
    archive, model, tokenizer, pin = minimal.runtime(Path(cfg["archive"]), Path(cfg["previous"]), Path(cfg["model"]))
    assert pin["model_sha256"] == BASE_SHA, f"Model hash mismatch: {pin['model_sha256']}"
    return cfg, archive, model, tokenizer


def load_controller_weights(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if "state" in obj:
        st = obj["state"]
        u = st["out.weight"].double().flatten()
        w = st["encode.weight"].double().flatten()
    elif "u_A0" in obj and "w_A0" in obj:
        u = obj["u_A0"].double().flatten()
        w = obj["w_A0"].double().flatten()
    else:
        raise KeyError(f"Unrecognized controller format at {path}: {list(obj.keys())}")
    return u, w


def load_minimal_controller(path: Path, device="cuda"):
    import minimal
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if "arch" in obj:
        net = minimal.Minimal(**obj["arch"])
        net.load_state_dict(obj["state"])
    elif "state" in obj:
        st = obj["state"]
        rank = int(st["encode.weight"].shape[0])
        alpha = float(obj.get("alpha", obj.get("meta", {}).get("alpha", 0.45)))
        mode = obj.get("mode", "linear")
        net = minimal.Minimal(mode, rank, alpha)
        net.load_state_dict(st)
    else:
        raise KeyError(f"Cannot load Minimal controller from {path}: {list(obj.keys())}")
    return net.to(device).eval()


def save_rank1_controller(path: Path, net, meta=None):
    import minimal
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    minimal.save_net(path, net, meta or {})
    return sha256_file(path)


def extract_q_opts_gold(row):
    from e0_external_dev import options_and_gold
    return options_and_gold(row)


def make_candidate_row(sample_id, question, options, gold_idx, dataset="v4"):
    labels = [chr(65 + i) for i in range(len(options))]
    return {
        "sample_id": str(sample_id),
        "dataset": dataset,
        "candidate": {
            "id": str(sample_id),
            "question": question,
            "choices": {"label": labels, "text": list(options)},
            "answerKey": labels[gold_idx],
        },
    }


def forward_with_hook_all_tokens(model, input_ids, attention_mask, depth, net=None, token_mask=None):
    """Run Ouro core for `depth` loops, injecting `net.delta(h)` before final loop (counter == depth - 1)."""
    core = model.model
    old_steps = getattr(core, "total_ut_steps", 4)
    core.total_ut_steps = depth
    counter = 0

    def pre_hook(_, args):
        nonlocal counter
        counter += 1
        if net is not None and counter == depth - 1:
            h = args[0]
            hf = h.float()
            dh = net.delta(hf)
            if token_mask is None:
                return ((hf + dh).to(h.dtype), *args[1:])
            m = token_mask.unsqueeze(-1).to(hf.dtype)
            return ((hf + dh * m).to(h.dtype), *args[1:])
        return None

    handle = core.norm.register_forward_pre_hook(pre_hook) if net is not None else None
    try:
        out = core(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        if isinstance(out, tuple) and len(out) >= 2 and isinstance(out[1], (list, tuple)) and len(out[1]) > 0:
            hidden = out[1][-1]
        else:
            hidden = out[0] if isinstance(out, tuple) else out
            if hasattr(hidden, "last_hidden_state"):
                hidden = hidden.last_hidden_state
    finally:
        if handle is not None:
            handle.remove()
        core.total_ut_steps = old_steps
    return hidden


@torch.inference_mode()
def capture_boundary_states(model, input_ids, attention_mask, depth=4):
    """Capture pre-final-loop hidden states (counter == depth - 1) at core.norm."""
    core = model.model
    old_steps = getattr(core, "total_ut_steps", 4)
    core.total_ut_steps = depth
    counter = 0
    captured = {}

    def pre_hook(_, args):
        nonlocal counter
        counter += 1
        if counter == depth - 1:
            captured["h"] = args[0].detach()
        return None

    handle = core.norm.register_forward_pre_hook(pre_hook)
    try:
        core(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
        core.total_ut_steps = old_steps
    return captured["h"]


@torch.inference_mode()
def compute_sigma_h_unlabeled(model, tokenizer, corpus_path=UNLABELED_AUDIT, max_docs=256, depth=4, batch_size=4):
    """Compute exact uncentered second moment Sigma_h = E[h h^T] and mean mu_h in FP64 on unlabeled prose."""
    rows = load_jsonl(corpus_path)[:max_docs]
    d = model.config.hidden_size
    moment = torch.zeros((d, d), dtype=torch.float64, device="cuda")
    mean_sum = torch.zeros((d,), dtype=torch.float64, device="cuda")
    total_toks = 0
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        enc = [tokenizer.encode(r["text"], add_special_tokens=False)[:384] for r in chunk]
        mx = max(len(x) for x in enc)
        ids = torch.full((len(enc), mx), pad_id, dtype=torch.long, device="cuda")
        mask = torch.zeros((len(enc), mx), dtype=torch.long, device="cuda")
        for b, seq in enumerate(enc):
            ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long, device="cuda")
            mask[b, :len(seq)] = 1
        h = capture_boundary_states(model, ids, mask, depth=depth)  # [B, T, d]
        valid = h[mask.bool()].double()  # [N, d]
        moment.addmm_(valid.T, valid)
        mean_sum.add_(valid.sum(dim=0))
        total_toks += valid.shape[0]
    sigma = (moment / total_toks).cpu()
    mu = (mean_sum / total_toks).cpu()
    return sigma, mu, total_toks


@torch.inference_mode()
def score_mc_letter_rotation_probs(model, tokenizer, questions_opts, depth=4, batch_size=8):
    """Compute exact cyclic-rotation-averaged shown_letter posterior q_0^MC(c | x) for m-way questions.
    questions_opts: list of (question_str, options_list).
    Returns: list of 1D float lists (probabilities aligned back to original option indices 0..m-1).
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Pre-resolve single-token IDs for " A", " B", ..., " J"
    letter_ids = {}
    for idx in range(10):
        ch = chr(65 + idx)
        toks = tokenizer.encode(" " + ch, add_special_tokens=False)
        letter_ids[idx] = toks[-1]

    # Build all rotated prompts
    all_prompts = []
    meta = []  # (q_idx, shift, m)
    for q_idx, (stem, opts) in enumerate(questions_opts):
        m = len(opts)
        if m > 10 or m < 2:
            raise ValueError(f"Unsupported choice count {m}")
        # Use all m cyclic shifts for m<=5, and 4 evenly spaced shifts for m>5 (e.g. QASC m=8)
        shifts = list(range(m)) if m <= 5 else [0, 2, 4, 6]
        for shift in shifts:
            content = [(pos - shift) % m for pos in range(m)]
            lines = "\n".join(f"{chr(65 + pos)}. {opts[content[pos]]}" for pos in range(m))
            prompt = f"{stem}\n{lines}\nAnswer:"
            all_prompts.append(prompt)
            meta.append((q_idx, shift, m, content))

    prob_sums = [torch.zeros(len(opts), dtype=torch.float64) for _, opts in questions_opts]
    counts = [0 for _ in questions_opts]

    for start in range(0, len(all_prompts), batch_size):
        batch_p = all_prompts[start:start + batch_size]
        batch_m = meta[start:start + batch_size]
        enc = [tokenizer.encode(p, add_special_tokens=False)[-1024:] for p in batch_p]
        mx = max(len(x) for x in enc)
        ids = torch.full((len(enc), mx), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(enc), mx), dtype=torch.long, device=device)
        last_pos = []
        for b, seq in enumerate(enc):
            ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[b, :len(seq)] = 1
            last_pos.append(len(seq) - 1)
        hidden = forward_with_hook_all_tokens(model, ids, mask, depth=depth, net=None)
        last_h = hidden[torch.arange(len(enc), device=device), torch.tensor(last_pos, device=device)]
        logits = model.lm_head(last_h).float()  # [B, V]
        for b, (q_idx, shift, m, content) in enumerate(batch_m):
            c_ids = [letter_ids[pos] for pos in range(m)]
            slot_logits = logits[b, c_ids]
            slot_probs = torch.softmax(slot_logits, dim=0).double().cpu()
            # Map slot probabilities back to original option index content[pos]
            for pos in range(m):
                prob_sums[q_idx][content[pos]] += slot_probs[pos]
            counts[q_idx] += 1

    return [(s / c).tolist() for s, c in zip(prob_sums, counts)]


@torch.inference_mode()
def score_cloze_probs(archive, model, tokenizer, rows, depth=4, net=None, batch_size=6):
    """Return exact Cloze candidate score vectors (16*sum/chars, matching repo utility) & softmax probs."""
    from advantages import choice_batch, forward
    all_scores, all_probs = [], []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        options = choice_batch(archive, tokenizer, chunk)
        logits, _ = forward(model, options, depth, net)
        lp = -F.cross_entropy(logits, options.packed[-1], reduction="none")
        sums = lp.new_zeros(len(options.packed[0])).index_add_(0, options.packed[1], lp)
        sizes = [len(m["characters"]) for m in options.metadata]
        chars = sums.new_tensor([c for m in options.metadata for c in m["characters"]])
        # Note: repo candidate acc_norm uses sums/chars (length-normalized logprob); repo CE ranking uses 16*sums/chars
        raw_norm = (sums / chars).split(sizes)
        scaled = (16.0 * sums / chars).split(sizes)
        for rn, sc in zip(raw_norm, scaled):
            all_scores.append(rn.detach().float().cpu().tolist())
            all_probs.append(torch.softmax(sc.detach().float(), dim=0).cpu().tolist())
    return all_scores, all_probs


def cloze_option_logits_differentiable(archive, model, tokenizer, rows, depth, net):
    """Differentiable 16*sum/chars candidate option logits matching e0_reuse / train_generative.ranking."""
    from advantages import choice_batch, forward
    options = choice_batch(archive, tokenizer, rows)
    logits, _ = forward(model, options, depth, net)
    lp = -F.cross_entropy(logits, options.packed[-1], reduction="none")
    sums = lp.new_zeros(len(options.packed[0])).index_add_(0, options.packed[1], lp)
    sizes = [len(m["characters"]) for m in options.metadata]
    chars = sums.new_tensor([c for m in options.metadata for c in m["characters"]])
    scaled = (16.0 * sums / chars).split(sizes)
    work = options.packed[0].numel() * depth
    return scaled, options.metadata, work


def letter_option_logits_differentiable(model, tokenizer, prompts_m, depth, net):
    """Differentiable single-letter (' A'..' J') logits at the final prompt token with controller hook."""
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    letter_ids = [tokenizer.encode(" " + chr(65 + idx), add_special_tokens=False)[-1] for idx in range(10)]
    enc = [tokenizer.encode(p, add_special_tokens=False)[-1024:] for p, _ in prompts_m]
    mx = max(len(x) for x in enc)
    ids = torch.full((len(enc), mx), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(enc), mx), dtype=torch.long, device=device)
    tmask = torch.zeros((len(enc), mx), dtype=torch.bool, device=device)
    last_pos = []
    for b, seq in enumerate(enc):
        ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attn[b, :len(seq)] = 1
        lp_idx = len(seq) - 1
        tmask[b, lp_idx] = True
        last_pos.append(lp_idx)
    hidden = forward_with_hook_all_tokens(model, ids, attn, depth=depth, net=net, token_mask=tmask)
    last_h = hidden[torch.arange(len(enc), device=device), torch.tensor(last_pos, device=device)]
    vocab_logits = model.lm_head(last_h).float()
    out_logits = [vocab_logits[b, letter_ids[:m]] for b, (_, m) in enumerate(prompts_m)]
    return out_logits, ids.numel() * depth


@torch.inference_mode()
def evaluate_unlabeled_manifold(model, tokenizer, net, corpus_path=UNLABELED_AUDIT, max_docs=128, depth=4, batch_size=4):
    """Measure exact natural-text NLL, delta_NLL, KL(p_nat || p_ctrl), and EOS prob shift on unlabeled prose."""
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
        # Predict positions 0..T-2 -> target 1..T-1
        h_nat = forward_with_hook_all_tokens(model, ids, attn, depth=depth, net=None)
        h_ctl = forward_with_hook_all_tokens(model, ids, attn, depth=depth, net=net)
        # Only score non-pad next tokens
        valid_mask = attn[:, 1:].bool()
        targets = ids[:, 1:][valid_mask]
        z_nat = model.lm_head(h_nat[:, :-1][valid_mask]).float()
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
    nll_ctl = nll_ctrl_sum / total_toks
    return {
        "n_docs": len(rows),
        "n_tokens": total_toks,
        "nll_nat": nll_nat,
        "nll_ctrl": nll_ctl,
        "delta_nll": nll_ctl - nll_nat,
        "kl_nat_to_ctrl": kl_sum / total_toks,
        "eos_abs_shift": eos_abs_sum / total_toks,
        "eos_signed_shift": eos_signed_sum / total_toks,
    }


class FastUnifiedDevEvaluator:
    """Exact in-process replica of tools/score_shared_dev.py keeping the frozen model resident on GPU."""

    def __init__(self, cfg, archive, model, tokenizer):
        self.cfg = cfg
        self.archive = archive
        self.model = model
        self.tokenizer = tokenizer
        for key in ("dev_stage2_code", "unified", "followup"):
            p = cfg[key]
            if p not in sys.path:
                sys.path.insert(0, p)
        from e0_external_dev import candidate_rows, letter_rows, pro_items
        from evaluate_benchmarks import candidate_row, mmlu_metrics
        import evaluate_generation
        import generation_followup_run as decoder
        import seven_unified
        import stage2_score

        self.candidate_row = candidate_row
        self.mmlu_metrics = mmlu_metrics
        self.evaluate_generation = evaluate_generation
        self.decoder = decoder
        self.seven_unified = seven_unified
        self.stage2_score = stage2_score
        self.options = json.loads(Path(cfg["mmlu_scoring_settings"]).read_text(encoding="utf-8"))
        self.encoder = stage2_score.HarnessEncoder(tokenizer, self.options["max_length"])

        self.data = {}
        for task in ("sciq_val", "arc_easy_val", "cosmosqa_val"):
            rows = load_jsonl(EXT_DEV_DIR / f"{task}.jsonl")
            self.data[task] = candidate_rows(rows)
        self.mmlu_all_rows = letter_rows(load_jsonl(EXT_DEV_DIR / "mmlu_val.jsonl"))
        self.mmlu_dev158_rows = letter_rows(load_jsonl(EXT_DEV_DIR / "mmlu_dev.jsonl"))
        pro_raw = load_jsonl(EXT_DEV_DIR / "mmlu_pro_val.jsonl")
        mx, batch = decoder.TASKS["mmlu_pro"]
        self.pro_mx = mx
        self.pro_batch = batch
        self.pro_raw = pro_raw
        self.pro_items_fn = pro_items
        self.pro_n = len(pro_raw)

    @torch.inference_mode()
    def evaluate(self, net=None, include_mmlu_dev158=False):
        import minimal
        if net is not None:
            net.eval()
        out = {}
        for task in ("sciq_val", "arc_easy_val", "cosmosqa_val"):
            preds, _ = self.evaluate_generation.candidates(
                self.archive, self.model, self.tokenizer, net, self.data[task], 4, 12
            )
            out[task] = {
                "acc": sum(r["metrics"]["acc"] for r in preds) / len(preds),
                "acc_norm": sum(r["metrics"]["acc_norm"] for r in preds) / len(preds),
            }
        # mmlu_val (658)
        cur, _ = self.seven_unified.shard_score(
            self.stage2_score, minimal, self.model, self.encoder, net,
            self.mmlu_all_rows, self.options, self.model.device
        )
        mpreds = [self.candidate_row(r, self.seven_unified.UNIFIED) for r in cur]
        out["mmlu_val"] = {"acc": sum(r["metrics"]["acc"] for r in mpreds) / len(mpreds)}

        if include_mmlu_dev158:
            cur158, _ = self.seven_unified.shard_score(
                self.stage2_score, minimal, self.model, self.encoder, net,
                self.mmlu_dev158_rows, self.options, self.model.device
            )
            mp158 = [self.candidate_row(r, self.seven_unified.UNIFIED) for r in cur158]
            out["mmlu_dev158"] = {"acc": sum(r["metrics"]["acc"] for r in mp158) / len(mp158)}

        # mmlu_pro_val (59) — rebuild fresh blocks every call to prevent generator exhaustion or mutation!
        self.model.model.total_ut_steps = 4
        mode = "feedback" if net is not None else "native"
        pro_blocks = list(self.decoder.batches(
            self.pro_items_fn(self.pro_raw, self.tokenizer, self.decoder),
            self.pro_mx, self.pro_batch, 14500
        ))
        pro_preds = []
        for block in pro_blocks:
            raw, _ = self.decoder.greedy(
                "mmlu_pro", self.model, self.tokenizer, block, mode, net, 1.0,
                self.pro_mx, self.decoder.cache_type(self.model)
            )
            pro_preds += self.mmlu_metrics(raw, block, self.decoder)
        out["mmlu_pro_val"] = {
            "correct": sum(r["metrics"]["correct"] for r in pro_preds) / self.pro_n,
            "parseable": sum(r["metrics"]["parseable"] for r in pro_preds) / self.pro_n,
        }
        primary = [
            out["sciq_val"]["acc_norm"],
            out["arc_easy_val"]["acc_norm"],
            out["cosmosqa_val"]["acc_norm"],
            out["mmlu_val"]["acc"],
            out["mmlu_pro_val"]["correct"],
        ]
        cloze3 = primary[:3]
        non_mmlu4 = [primary[0], primary[1], primary[2], primary[4]]
        out["macro5"] = sum(primary) / 5.0
        out["cloze3"] = sum(cloze3) / 3.0
        out["non_mmlu4"] = sum(non_mmlu4) / 4.0
        return out
