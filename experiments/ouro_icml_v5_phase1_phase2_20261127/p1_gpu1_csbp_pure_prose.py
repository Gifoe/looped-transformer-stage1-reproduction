"""ICML Phase 1 — GPU 1:
Track A: Pure Unlabeled Prose Self-Bootstrapping (CSBP — ZERO External QA Datasets, ZERO Human Labels, ZERO External LLMs).
Incorporates ALL 4 Reviewer upgrades:
  1. Honest naming (`orig_recovery_rate`, NOT teacher factual accuracy).
  2. Mandatory Shortcut Audits: Option-Only (stem dropped) & Shuffled-Context (mismatched stem) MC controls.
  3. Exact Same-Source 5-Way Objective Ablation on identical prose probes:
     (1) Orig-Gold CE, (2) Unweighted MC Hard-Pseudo, (3) MC Soft-KL,
     (4) Conf-Weighted MC Hard-Pseudo, (5) CFIT-Filtered + Conf-Weighted Hard-Pseudo.
  4. Language-Fidelity Pareto Frontier (trust_lam in {0.0, 0.25, 0.5, 1.0, 2.0}) + strict unlabeled auto-selection.
"""
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/ouro-v4-cfit-20261126")
import minimal
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
    forward_with_hook_all_tokens,
    load_base_runtime,
    load_controller_weights,
    load_jsonl,
    make_candidate_row,
    save_json,
    save_rank1_controller,
    score_cloze_probs,
    score_mc_letter_rotation_probs,
)

V4_ROOT = Path("/root/autodl-tmp/ouro-v4-cfit-20261126")
P1_ROOT = Path("/root/autodl-tmp/ouro-v5-icml-phase1-20261127")
OUT_DIR = P1_ROOT / "track_a_csbp"
SEEDS = (26112701, 26112702)
STEPS = 840

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "as", "is", "was", "are", "were", "be", "been", "being", "that", "which", "who", "this", "these",
    "those", "it", "its", "he", "she", "they", "his", "her", "their", "has", "have", "had", "not",
    "into", "than", "also", "such", "more", "most", "other", "some", "can", "could", "would", "may",
}

CAUSAL_CUES = [
    "because", "causes", "caused", "converts", "results in", "resulted in", "leads to", "led to",
    "increases", "decreases", "allows", "produces", "produced", "prevents", "requires", "consists of",
    "depends on", "known as", "used to", "designed to", "referred to as", "characterized by",
    "due to", "therefore", "consequently", "thus", "instead of", "rather than", "compared to",
]

PIVOT_WORDS = {
    "is", "was", "are", "were", "becomes", "became", "causes", "caused", "produces", "produced",
    "creates", "created", "allows", "allowed", "requires", "required", "prevents", "converts",
    "increases", "decreases", "contains", "includes", "forms", "provides", "uses", "used",
    "into", "from", "with", "through", "between", "against", "towards", "toward", "because", "that",
}

DISCOURSE_STARTS = (
    "Therefore,", "Thus,", "Consequently,", "As a result,", "Because of this,",
    "However,", "In contrast,", "Although", "Because", "Since", "While", "When", "After",
)


def detokenize_wikitext(text: str) -> str:
    s = text.replace(" @-@ ", "-").replace(" @.@ ", ".").replace(" @,@ ", ",")
    s = re.sub(r"\s+([,.;:?!%)])", r"\1", s)
    s = re.sub(r"([(\[])\s+", r"\1", s)
    s = re.sub(r"\s+'(s|t|re|ve|ll|d|m)\b", r"'\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def content_words(s: str):
    toks = re.findall(r"[a-z]{3,}", s.lower())
    return {t for t in toks if t not in STOPWORDS}


def jaccard(a: str, b: str) -> float:
    sa, sb = content_words(a), content_words(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_8gram_set(texts):
    grams = set()
    for t in texts:
        toks = re.findall(r"[a-z0-9]+", t.lower())
        for i in range(max(0, len(toks) - 7)):
            grams.add(tuple(toks[i:i + 8]))
    return grams


def has_8gram_overlap(text: str, forbidden: set) -> bool:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    for i in range(max(0, len(toks) - 7)):
        if tuple(toks[i:i + 8]) in forbidden:
            return True
    return False


def load_forbidden_eval_8grams():
    """Collect 8-grams of actual question/choice text across UNIFIED_EXT_DEV + official test sets."""
    texts = []
    for p in EXT_DEV_DIR.glob("*.jsonl"):
        for r in load_jsonl(p):
            for k in ("question", "context", "passage", "stem", "text"):
                if isinstance(r.get(k), str):
                    texts.append(r[k])
            cand = r.get("candidate")
            if isinstance(cand, dict):
                if isinstance(cand.get("question"), str):
                    texts.append(cand["question"])
                ch = cand.get("choices")
                if isinstance(ch, dict) and isinstance(ch.get("text"), list):
                    texts.extend(str(x) for x in ch["text"])
    return build_8gram_set(texts)


def split_sentence_pivot(sent: str):
    """Split a clean declarative sentence near 40%-65% right after a pivot verb/preposition."""
    words = sent.split()
    n = len(words)
    if n < 14 or n > 42:
        return None
    lo, hi = max(6, int(n * 0.38)), min(n - 5, int(n * 0.68))
    best_idx = None
    best_dist = 999
    target = n * 0.52
    for i in range(lo, hi + 1):
        w = re.sub(r"[^a-z]", "", words[i - 1].lower())
        if w in PIVOT_WORDS:
            d = abs(i - target)
            if d < best_dist:
                best_dist = d
                best_idx = i
    if best_idx is None:
        return None
    lead = " ".join(words[:best_idx]).strip()
    tail = " ".join(words[best_idx:]).strip()
    if len(tail.split()) < 5 or len(tail.split()) > 16:
        return None
    if not tail.endswith("."):
        return None
    # Short tail stub: ONLY last 3..5 words of lead (stripping premise & subject nouns!)
    lead_words = lead.split()
    stub_len = min(4, max(3, len(lead_words) // 3))
    stub = " ".join(lead_words[-stub_len:]).strip()
    if stub:
        stub = stub[0].upper() + stub[1:]
    return lead, tail, stub


def extract_prose_items(forbidden_8grams):
    """Extract clean Route A (self-contained single sentences) & Route B (multi-sentence passages) from UNLABELED_FIT."""
    rows = load_jsonl(UNLABELED_FIT)
    route_a_causal, route_a_general, route_b = [], [], []
    n_dropped_8gram = 0
    seen_tails = set()

    for doc_idx, r in enumerate(rows):
        doc_id = str(r.get("document_id", r["sample_id"]))
        txt = detokenize_wikitext(r["text"])
        if has_8gram_overlap(txt, forbidden_8grams):
            n_dropped_8gram += 1
            continue
        # Split into complete interior sentences (drop first & last partial fragments!)
        raw_sents = re.split(r"(?<=[.!?])\s+", txt)
        sents = []
        for s in raw_sents[1:-1]:
            s = s.strip()
            if len(s) < 55 or len(s) > 260:
                continue
            if not s[0].isupper() or not s.endswith("."):
                continue
            if any(ch in s for ch in ("=", "@", '"', "(", ")", "[", "]", ";")):
                continue
            # Avoid pure date/number trivia sentences (>2 4-digit years)
            if len(re.findall(r"\b\d{4}\b", s)) >= 2:
                continue
            sents.append(s)

        for si, s in enumerate(sents):
            sp = split_sentence_pivot(s)
            if sp is None:
                continue
            lead, c0, stub = sp
            tkey = c0.lower()
            if tkey in seen_tails:
                continue
            seen_tails.add(tkey)
            low = s.lower()
            is_causal = any(cue in low for cue in CAUSAL_CUES)
            item_a = {
                "id": f"ra_{doc_idx}_{si}", "doc_id": doc_id, "route": "A_causal" if is_causal else "A_gen",
                "context": "", "lead": lead, "c0": c0, "stub": stub,
            }
            if is_causal:
                route_a_causal.append(item_a)
            else:
                route_a_general.append(item_a)

            # Route B: 2..3 preceding sentences as context, with <=50% content-word copy guard!
            if si >= 1:
                ctx_sents = sents[max(0, si - 3):si]
                ctx = " ".join(ctx_sents)
                if 20 <= len(ctx.split()) <= 95:
                    cw_c0 = content_words(c0)
                    cw_ctx = content_words(ctx)
                    copy_frac = len(cw_c0 & cw_ctx) / max(len(cw_c0), 1)
                    if copy_frac <= 0.50:
                        starts_disc = any(s.startswith(ds) for ds in DISCOURSE_STARTS) or is_causal
                        route_b.append({
                            "id": f"rb_{doc_idx}_{si}", "doc_id": doc_id,
                            "route": "B_disc" if starts_disc else "B_gen",
                            "context": ctx, "lead": lead, "c0": c0, "stub": stub,
                            "copy_frac": copy_frac, "priority": 1 if starts_disc else 0,
                        })

    route_b.sort(key=lambda x: (-x["priority"], x["copy_frac"]))
    audit = {
        "n_docs": len(rows),
        "n_dropped_8gram": n_dropped_8gram,
        "n_route_a_causal": len(route_a_causal),
        "n_route_a_general": len(route_a_general),
        "n_route_b_total": len(route_b),
        "n_route_b_discourse": sum(x["priority"] for x in route_b),
    }
    print("PROSE EXTRACTION AUDIT:", json.dumps(audit, indent=2), flush=True)
    return route_a_causal, route_a_general, route_b, audit


@torch.inference_mode()
def generate_k4_blind_distractors(model, tokenizer, items, n_samples=8, batch_size=8):
    """G1_K4Blind: Same-depth K=4 context-stripped short-tail blind completion (direct K=4 loop, no HF cache bug!)."""
    device = next(model.parameters()).device
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    period_ids = set()
    for ch in (".", " .", ".\n"):
        t = tokenizer.encode(ch, add_special_tokens=False)
        if t:
            period_ids.add(t[-1])

    t0 = time.perf_counter()
    results = []
    torch.manual_seed(26112701)
    for st in range(0, len(items), batch_size):
        chunk = items[st:st + batch_size]
        rep_stubs = [it["stub"] for it in chunk for _ in range(n_samples)]
        enc_list = [tokenizer.encode(s, add_special_tokens=False) for s in rep_stubs]
        B = len(enc_list)
        max_in = max(len(s) for s in enc_list)
        # Left-pad so every row predicts at the exact final column!
        ids = torch.full((B, max_in), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((B, max_in), dtype=torch.long, device=device)
        for b, s in enumerate(enc_list):
            ids[b, max_in - len(s):] = torch.tensor(s, dtype=torch.long, device=device)
            attn[b, max_in - len(s):] = 1
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        gen_tokens = []
        for step_t in range(22):
            h = forward_with_hook_all_tokens(model, ids, attn, depth=4, net=None)
            logits = model.lm_head(h[:, -1]).float() / 0.85
            # Top-p = 0.92 nucleus sampling
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            mask_rm = cum > 0.92
            mask_rm[:, 1:] = mask_rm[:, :-1].clone()
            mask_rm[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(mask_rm, -1e9)
            sample_pos = torch.multinomial(torch.softmax(sorted_logits, dim=-1), 1)
            nxt = sorted_idx.gather(-1, sample_pos).squeeze(-1)
            nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
            gen_tokens.append(nxt)
            ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
            attn = torch.cat([attn, (~finished).long().unsqueeze(1)], dim=1)
            # Mark finished after hitting period (once >=6 tokens generated) or EOS
            if step_t >= 6:
                is_end = (nxt == eos_id) if eos_id is not None else torch.zeros_like(finished)
                for pid in period_ids:
                    is_end = is_end | (nxt == pid)
                finished = finished | is_end
                if finished.all():
                    break
        gen_mat = torch.stack(gen_tokens, dim=1).cpu()
        # Decode and filter per item
        for bi, it in enumerate(chunk):
            c0 = it["c0"]
            cands = []
            seen = {c0.lower()}
            for si in range(n_samples):
                seq = gen_mat[bi * n_samples + si].tolist()
                txt = tokenizer.decode(seq, skip_special_tokens=True).strip()
                txt = txt.split("\n")[0].strip()
                m = re.search(r"^([^.!?]{12,150}\.)", txt)
                if not m:
                    continue
                cand = re.sub(r"\s+", " ", m.group(1)).strip()
                if any(ch in cand for ch in ('"', "(", ")", "[", "]", "=", "@")):
                    continue
                if not (0.55 * len(c0) <= len(cand) <= 1.65 * len(c0)):
                    continue
                if cand.lower() in seen:
                    continue
                if jaccard(cand, c0) > 0.45:
                    continue
                if any(jaccard(cand, prev) > 0.45 for prev in cands):
                    continue
                seen.add(cand.lower())
                cands.append(cand)
            if len(cands) >= 3:
                results.append({**it, "distractors": cands[:3], "gen_type": "G1_K4Blind"})
        if (st // batch_size) % 10 == 0:
            print(f"  [G1_K4Blind] {min(st+batch_size, len(items))}/{len(items)} kept={len(results)} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return results


@torch.inference_mode()
def build_human_retrieval_distractors(model, tokenizer, items, pool_items):
    """G2_HumanRetr: 100% real human endings retrieved by frozen Ouro-1.4B embeddings (ZERO style shift!)."""
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model.model.total_ut_steps = 4

    def embed_texts(texts, bs=32):
        vecs = []
        for i in range(0, len(texts), bs):
            sub = texts[i:i + bs]
            enc = [tokenizer.encode(t, add_special_tokens=False)[:48] for t in sub]
            mx = max(len(x) for x in enc)
            ids = torch.full((len(sub), mx), pad_id, dtype=torch.long, device=device)
            mask = torch.zeros((len(sub), mx), dtype=torch.long, device=device)
            for b, s in enumerate(enc):
                ids[b, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
                mask[b, :len(s)] = 1
            h = forward_with_hook_all_tokens(model, ids, mask, depth=4, net=None)
            m = mask.unsqueeze(-1).float()
            v = (h.float() * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
            v = F.normalize(v, dim=-1).cpu()
            vecs.append(v)
        return torch.cat(vecs, dim=0)

    pool_c0 = [p["c0"] for p in pool_items]
    pool_emb = embed_texts(pool_c0)
    item_emb = embed_texts([it["c0"] for it in items])
    sims = item_emb @ pool_emb.T  # [N_it, N_pool]

    out = []
    for i, it in enumerate(items):
        c0 = it["c0"]
        top_idx = torch.topk(sims[i], k=min(80, len(pool_items))).indices.tolist()
        dists = []
        for j in top_idx:
            cand = pool_items[j]["c0"]
            if pool_items[j]["doc_id"] == it["doc_id"]:
                continue
            if not (0.65 * len(c0) <= len(cand) <= 1.50 * len(c0)):
                continue
            if jaccard(cand, c0) > 0.40:
                continue
            if any(jaccard(cand, prev) > 0.40 for prev in dists):
                continue
            dists.append(cand)
            if len(dists) == 3:
                break
        if len(dists) == 3:
            out.append({**it, "distractors": dists, "gen_type": "G2_HumanRetr"})
    return out


def build_random_distractors(items, pool_items, seed=26112701):
    rng = random.Random(seed)
    out = []
    for it in items:
        c0 = it["c0"]
        dists = []
        tries = 0
        while len(dists) < 3 and tries < 80:
            tries += 1
            p = pool_items[rng.randrange(len(pool_items))]
            if p["doc_id"] == it["doc_id"] or p["c0"] == c0 or p["c0"] in dists:
                continue
            dists.append(p["c0"])
        if len(dists) == 3:
            out.append({**it, "distractors": dists, "gen_type": "G0_Random"})
    return out


def make_shuffled_quads(items, seed=26112701):
    rng = random.Random(seed)
    quads = []
    for it in items:
        opts = [it["c0"]] + list(it["distractors"])
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        quads.append({
            **it,
            "options": [opts[k] for k in order],
            "_orig_idx": order.index(0),
        })
    return quads


@torch.inference_mode()
def score_natural_cloze_probs(model, tokenizer, stem_opts_list, depth=4, net=None, batch_size=6):
    """Exact natural continuation Cloze scoring: log p(opt | stem) / len(opt) * 16 -> softmax."""
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    all_probs = []
    for st in range(0, len(stem_opts_list), batch_size):
        chunk = stem_opts_list[st:st + batch_size]
        seqs, spans, lens_chr, counts = [], [], [], []
        for stem, opts in chunk:
            p_ids = tokenizer.encode(stem, add_special_tokens=False)[-384:]
            counts.append(len(opts))
            for o in opts:
                c_ids = tokenizer.encode(" " + o, add_special_tokens=False)[:64]
                if not c_ids:
                    c_ids = [pad_id]
                seqs.append(p_ids + c_ids)
                spans.append((len(p_ids), len(p_ids) + len(c_ids)))
                lens_chr.append(max(len(o), 1))
        mx = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), mx), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(seqs), mx), dtype=torch.long, device=device)
        for b, s in enumerate(seqs):
            ids[b, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            attn[b, :len(s)] = 1
        h = forward_with_hook_all_tokens(model, ids, attn, depth=depth, net=net)
        scores = []
        for b, ((a, e), nch) in enumerate(zip(spans, lens_chr)):
            # predict positions a-1 .. e-2 -> targets a .. e-1
            z = model.lm_head(h[b, a - 1:e - 1]).float()
            tgt = ids[b, a:e]
            lp = -F.cross_entropy(z, tgt, reduction="sum")
            scores.append(float(lp) / nch)
        ptr = 0
        for m in counts:
            sc = torch.tensor(scores[ptr:ptr + m], dtype=torch.float32) * 16.0
            all_probs.append(torch.softmax(sc, dim=0).tolist())
            ptr += m
    return all_probs


def natural_cloze_logits_differentiable(model, tokenizer, mb, depth, net):
    """Differentiable natural continuation Cloze option logits (16 * sum_logp / chars)."""
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    seqs, spans, lens_chr, counts = [], [], [], []
    for it in mb:
        stem = it["question"]
        opts = it["options"]
        p_ids = tokenizer.encode(stem, add_special_tokens=False)[-384:]
        counts.append(len(opts))
        for o in opts:
            c_ids = tokenizer.encode(" " + o, add_special_tokens=False)[:64]
            if not c_ids:
                c_ids = [pad_id]
            seqs.append(p_ids + c_ids)
            spans.append((len(p_ids), len(p_ids) + len(c_ids)))
            lens_chr.append(max(len(o), 1))
    mx = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), mx), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), mx), dtype=torch.long, device=device)
    tmask = torch.zeros((len(seqs), mx), dtype=torch.bool, device=device)
    for b, (s, (a, e)) in enumerate(zip(seqs, spans)):
        ids[b, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        attn[b, :len(s)] = 1
        tmask[b, a - 1:e - 1] = True
    h = forward_with_hook_all_tokens(model, ids, attn, depth=depth, net=net, token_mask=tmask)
    flat_scores = []
    for b, ((a, e), nch) in enumerate(zip(spans, lens_chr)):
        z = model.lm_head(h[b, a - 1:e - 1]).float()
        tgt = ids[b, a:e]
        lp = -F.cross_entropy(z, tgt, reduction="sum")
        flat_scores.append(16.0 * lp / nch)
    out, ptr = [], 0
    for m in counts:
        out.append(torch.stack(flat_scores[ptr:ptr + m]))
        ptr += m
    return out


def format_stem(it, tmpl: str) -> str:
    ctx, lead = it["context"], it["lead"]
    full = f"{ctx} {lead}".strip() if ctx else lead
    if tmpl == "T1_Direct":
        return full
    elif tmpl == "T2_Blank":
        return f"{full} ____"
    elif tmpl == "T3_Passage":
        return f"Passage: {ctx}\n{lead}" if ctx else f"Complete the sentence:\n{lead}"
    raise ValueError(tmpl)


def kl_div(q, p):
    qt = torch.tensor(q, dtype=torch.float64).clamp_min(1e-8)
    pt = torch.tensor(p, dtype=torch.float64).clamp_min(1e-8)
    return float((qt * (qt.log() - pt.log())).sum())


def run_preflight_diagnostic(archive, model, tokenizer, route_a, route_b, pool_all):
    out_path = OUT_DIR / "CSBP_PREFLIGHT_VERDICT.json"
    if out_path.exists():
        print("CSBP Preflight already cached, loading...", flush=True)
        return json.loads(out_path.read_text(encoding="utf-8"))

    print("\n================ CSBP PREFLIGHT + REVIEWER SHORTCUT AUDIT ================", flush=True)
    # Build 220-item preflight slices for Route A and Route B under BOTH G1_K4Blind and G2_HumanRetr
    sub_a = route_a[:260]
    sub_b = route_b[:260]

    print("Generating G1_K4Blind preflight probes...", flush=True)
    g1_a = make_shuffled_quads(generate_k4_blind_distractors(model, tokenizer, sub_a, n_samples=8, batch_size=8)[:200], seed=101)
    g1_b = make_shuffled_quads(generate_k4_blind_distractors(model, tokenizer, sub_b, n_samples=8, batch_size=8)[:200], seed=102)
    print("Building G2_HumanRetr preflight probes...", flush=True)
    g2_a = make_shuffled_quads(build_human_retrieval_distractors(model, tokenizer, sub_a[:220], pool_all)[:200], seed=201)
    g2_b = make_shuffled_quads(build_human_retrieval_distractors(model, tokenizer, sub_b[:220], pool_all)[:200], seed=202)

    combos = [
        ("RouteA", "G1_K4Blind", g1_a),
        ("RouteB", "G1_K4Blind", g1_b),
        ("RouteA", "G2_HumanRetr", g2_a),
        ("RouteB", "G2_HumanRetr", g2_b),
    ]
    templates = ("T1_Direct", "T2_Blank", "T3_Passage")
    matrix = {}

    for rname, gname, quads in combos:
        if len(quads) < 50:
            continue
        # Score Cloze-QA and Cloze-Nat ONCE per (Route, Gen)
        stems_raw = [f"{q['context']} {q['lead']}".strip() if q["context"] else q["lead"] for q in quads]
        crows = [make_candidate_row(q["id"], st, q["options"], 0) for q, st in zip(quads, stems_raw)]
        _, clz_qa = score_cloze_probs(archive, model, tokenizer, crows, depth=4, net=None, batch_size=6)
        clz_nat = score_natural_cloze_probs(model, tokenizer, [(st, q["options"]) for st, q in zip(stems_raw, quads)], depth=4, net=None, batch_size=6)

        # Reviewer's Shortcut Audit 1: Option-Only MC (stem completely dropped!)
        opt_only_mc = score_mc_letter_rotation_probs(model, tokenizer, [("Choose the best option:", q["options"]) for q in quads], depth=4, batch_size=8)
        opt_only_rec = sum(int(torch.tensor(p).argmax()) == q["_orig_idx"] for p, q in zip(opt_only_mc, quads)) / len(quads)

        # Reviewer's Shortcut Audit 2: Shuffled-Context MC (mismatched stem from (i+17)%n!)
        shuf_qopts = [(format_stem(quads[(i + 17) % len(quads)], "T1_Direct"), q["options"]) for i, q in enumerate(quads)]
        shuf_mc = score_mc_letter_rotation_probs(model, tokenizer, shuf_qopts, depth=4, batch_size=8)
        shuf_rec = sum(int(torch.tensor(p).argmax()) == q["_orig_idx"] for p, q in zip(shuf_mc, quads)) / len(quads)

        clz_qa_rec = sum(int(torch.tensor(p).argmax()) == q["_orig_idx"] for p, q in zip(clz_qa, quads)) / len(quads)
        clz_nat_rec = sum(int(torch.tensor(p).argmax()) == q["_orig_idx"] for p, q in zip(clz_nat, quads)) / len(quads)

        for tmpl in templates:
            mc_qopts = [(format_stem(q, tmpl), q["options"]) for q in quads]
            mc_p = score_mc_letter_rotation_probs(model, tokenizer, mc_qopts, depth=4, batch_size=8)
            full_rec = sum(int(torch.tensor(p).argmax()) == q["_orig_idx"] for p, q in zip(mc_p, quads)) / len(quads)
            ctx_lift = (full_rec - max(opt_only_rec, shuf_rec)) * 100.0

            # Evaluate CFIT filter on both Cloze-QA and Cloze-Nat
            for cname, clz_p, clz_rec in [("ClozeQA", clz_qa, clz_qa_rec), ("ClozeNat", clz_nat, clz_nat_rec)]:
                kept = []
                for q, mp, cp in zip(quads, mc_p, clz_p):
                    sp = sorted(mp, reverse=True)
                    conf = max(0.0, (sp[0] - 0.25) / 0.75)
                    margin = sp[0] - sp[1]
                    kl = kl_div(mp, cp)
                    if conf >= 0.45 and margin >= 0.25 and kl >= 0.15:
                        kept.append((q, mp, cp))
                k_rate = len(kept) / len(quads)
                f_mc_rec = sum(int(torch.tensor(mp).argmax()) == q["_orig_idx"] for q, mp, _ in kept) / max(len(kept), 1)
                f_clz_rec = sum(int(torch.tensor(cp).argmax()) == q["_orig_idx"] for q, _, cp in kept) / max(len(kept), 1)
                key = f"{rname}__{gname}__{tmpl}__{cname}"
                entry = {
                    "route": rname, "generator": gname, "template": tmpl, "cloze": cname, "n": len(quads),
                    "full_orig_rec": full_rec,
                    "opt_only_orig_rec": opt_only_rec,
                    "shuf_ctx_orig_rec": shuf_rec,
                    "context_lift_pp": ctx_lift,
                    "cloze_orig_rec": clz_rec,
                    "raw_gap_pp": (full_rec - clz_rec) * 100.0,
                    "cfit_keep_rate": k_rate,
                    "cfit_n_kept": len(kept),
                    "filtered_mc_orig_rec": f_mc_rec,
                    "filtered_clz_orig_rec": f_clz_rec,
                    "filtered_gap_pp": (f_mc_rec - f_clz_rec) * 100.0,
                }
                matrix[key] = entry
                print(
                    f"  {key:44s} | MC={full_rec*100:5.1f}% OptOnly={opt_only_rec*100:4.1f}% Shuf={shuf_rec*100:4.1f}% "
                    f"Lift={ctx_lift:+5.1f}pp | Clz={clz_rec*100:5.1f}% RawGap={entry['raw_gap_pp']:+5.1f} | "
                    f"CFIT({k_rate*100:4.1f}%): fMC={f_mc_rec*100:5.1f}% fGap={entry['filtered_gap_pp']:+5.1f}pp",
                    flush=True,
                )

    # Pick best winning (Route, Template) per generator maximizing genuine context_lift + filtered_gap
    def pick_best(gen_filter):
        cands = [v for v in matrix.values() if v["generator"] == gen_filter and v["cfit_keep_rate"] >= 0.12]
        if not cands:
            cands = [v for v in matrix.values() if v["generator"] == gen_filter]
        cands.sort(key=lambda v: (v["context_lift_pp"] >= 20.0, v["filtered_gap_pp"] * v["filtered_mc_orig_rec"] * math.sqrt(max(v["cfit_keep_rate"], 0.01))), reverse=True)
        return cands[0] if cands else None

    best_g1 = pick_best("G1_K4Blind")
    best_g2 = pick_best("G2_HumanRetr")
    verdict = {
        "best_g1": best_g1,
        "best_g2": best_g2,
        "matrix": matrix,
    }
    save_json(out_path, verdict)
    return verdict


def train_csbp_arm(
    archive, model, tokenizer, bank, objective, cloze_mode, seed,
    steps=STEPS, trust_seqs=None, trust_lam=0.0, tag=""
):
    """Exact same-source 5-way objective trainer for CSBP."""
    torch.manual_seed(seed)
    net = minimal.Minimal("linear", 1, 0.45).to(model.device)
    net.requires_grad_(True)
    opt = torch.optim.AdamW(net.parameters(), lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"])
    rng = random.Random(f"csbp/{tag}/{seed}")
    order = list(range(len(bank)))
    rng.shuffle(order)
    ptr = 0
    nat_cache = {}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

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

    def nll_pen(seqs, depth):
        device = next(model.parameters()).device
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
                nat_cache[key] = float(F.cross_entropy(model.lm_head(h0[:, :-1][valid]).float(), targets).detach())
        h1 = forward_with_hook_all_tokens(model, ids, attn, depth=depth, net=net)
        nll1 = F.cross_entropy(model.lm_head(h1[:, :-1][valid]).float(), targets)
        return F.relu(nll1 - nat_cache[key] - 0.02)

    t0 = time.perf_counter()
    losses = []
    for step in range(1, steps + 1):
        depth = DEPTHS[(step - 1) % len(DEPTHS)]
        mb = next_mb(RECIPE["effective_batch"])
        use_nat = (cloze_mode == "ClozeNat") or (cloze_mode == "ClozeMix50" and step % 2 == 0)
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            if use_nat:
                opt_logits = natural_cloze_logits_differentiable(model, tokenizer, mb, depth, net)
            else:
                crows = [make_candidate_row(it["id"], it["question"], it["options"], 0) for it in mb]
                opt_logits, _, _ = cloze_option_logits_differentiable(archive, model, tokenizer, crows, depth, net)
            item_losses, weights = [], []
            for lg, it in zip(opt_logits, mb):
                log_p = F.log_softmax(lg, dim=0)
                if objective == "orig_gold_ce":
                    l_i = -log_p[int(it["orig_idx_for_selfsup_ablation"])]
                    w_i = 1.0
                elif objective == "mc_hard_unif":
                    l_i = -log_p[int(it["pseudo_label"])]
                    w_i = 1.0
                elif objective == "mc_soft_kl":
                    q = torch.tensor(it["q_mc"], dtype=torch.float32, device=lg.device).clamp_min(1e-8)
                    l_i = -(q * log_p).sum() + (q * q.log()).sum()
                    w_i = max(0.05, float(it["conf"]))
                elif objective == "mc_conf_hard":
                    l_i = -log_p[int(it["pseudo_label"])]
                    w_i = max(0.05, float(it["conf"]))
                else:
                    raise ValueError(objective)
                item_losses.append(l_i * w_i)
                weights.append(w_i)
            loss = sum(item_losses) / sum(weights)
            if trust_lam > 0.0 and trust_seqs and step % 2 == 0:
                t_mb = [trust_seqs[(step + j) % len(trust_seqs)] for j in range(2)]
                loss = loss + trust_lam * nll_pen(t_mb, depth)
            loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), RECIPE["clip"])
        assert grad.isfinite().item(), f"Non-finite grad in {tag} step={step}"
        opt.step()
        losses.append(float(loss.detach()))
        if step % 280 == 0 or step == steps:
            print(f"  [CSBP {tag} s={seed}] step={step}/{steps} loss={sum(losses[-70:])/min(len(losses),70):.4f} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return net, {"steps": steps, "seconds": time.perf_counter() - t0, "final_loss_avg70": sum(losses[-70:]) / min(len(losses), 70)}


def build_scored_full_bank(archive, model, tokenizer, quads, tmpl, name):
    cache = OUT_DIR / "banks" / f"{name}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    stems_raw = [f"{q['context']} {q['lead']}".strip() if q["context"] else q["lead"] for q in quads]
    mc_qopts = [(format_stem(q, tmpl), q["options"]) for q in quads]
    mc_p = score_mc_letter_rotation_probs(model, tokenizer, mc_qopts, depth=4, batch_size=8)
    crows = [make_candidate_row(q["id"], st, q["options"], 0) for q, st in zip(quads, stems_raw)]
    _, clz_qa = score_cloze_probs(archive, model, tokenizer, crows, depth=4, net=None, batch_size=6)
    clz_nat = score_natural_cloze_probs(model, tokenizer, [(st, q["options"]) for st, q in zip(stems_raw, quads)], depth=4, net=None, batch_size=6)

    bank = []
    for q, st, mp, cqa, cnat in zip(quads, stems_raw, mc_p, clz_qa, clz_nat):
        sp = sorted(mp, reverse=True)
        conf = max(0.0, (sp[0] - 0.25) / 0.75)
        margin = sp[0] - sp[1]
        bank.append({
            "id": q["id"],
            "question": st,
            "options": q["options"],
            "q_mc": mp,
            "conf": conf,
            "margin": margin,
            "pseudo_label": int(torch.tensor(mp).argmax()),
            "kl_qa": kl_div(mp, cqa),
            "kl_nat": kl_div(mp, cnat),
            "orig_idx_for_selfsup_ablation": q["_orig_idx"],  # ONLY used by Obj1_OrigGold_CE ablation!
        })
    save_json(cache, bank)
    return bank


@torch.inference_mode()
def heldout_unlab_agree(archive, model, tokenizer, va_bank, net, cloze_mode):
    if cloze_mode == "ClozeNat":
        probs = score_natural_cloze_probs(model, tokenizer, [(it["question"], it["options"]) for it in va_bank], depth=4, net=net, batch_size=6)
    else:
        crows = [make_candidate_row(it["id"], it["question"], it["options"], 0) for it in va_bank]
        _, probs = score_cloze_probs(archive, model, tokenizer, crows, depth=4, net=net, batch_size=6)
    agree = sum(int(torch.tensor(p).argmax()) == int(it["pseudo_label"]) for p, it in zip(probs, va_bank)) / len(va_bank)
    kls = [kl_div(it["q_mc"], p) for it, p in zip(va_bank, probs)]
    return agree, sum(kls) / len(kls)


def main():
    torch.set_num_threads(4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg, archive, model, tokenizer = load_base_runtime()
    evaluator = FastUnifiedDevEvaluator(cfg, archive, model, tokenizer)
    native_ev = evaluator.evaluate(None, include_mmlu_dev158=True)
    u_t1, w_t1 = load_controller_weights(Path(cfg["historical_t1"]))
    print(f"Native UNIFIED_EXT_DEV: macro5={native_ev['macro5']*100:.2f}% cloze3={native_ev['cloze3']*100:.2f}%", flush=True)

    forbidden = load_forbidden_eval_8grams()
    ra_c, ra_g, rb, audit = extract_prose_items(forbidden)
    save_json(OUT_DIR / "CORPUS_DEDUP_AUDIT.json", audit)
    route_a_all = ra_c + ra_g  # causal prioritized first!
    pool_all = route_a_all

    # Step 1: Zero-Training Preflight + Shortcut Audits
    verdict = run_preflight_diagnostic(archive, model, tokenizer, route_a_all, rb, pool_all)
    best_g1 = verdict["best_g1"]
    best_g2 = verdict["best_g2"]
    print("\n=== PREFLIGHT WINNERS ===", flush=True)
    print("  Best G1_K4Blind:", json.dumps(best_g1, indent=2), flush=True)
    print("  Best G2_HumanRetr:", json.dumps(best_g2, indent=2), flush=True)

    # Step 2: Build Full Unlabeled Prose Banks (up to 1,600 items per generator) using winning Route & Template
    src_g1 = route_a_all[:1500] if best_g1["route"] == "RouteA" else rb[:1500]
    src_g2 = route_a_all[:1500] if best_g2["route"] == "RouteA" else rb[:1500]

    g1_bank_path = OUT_DIR / "banks" / "bank_G1_K4Blind_full.json"
    if not g1_bank_path.exists():
        print(f"\nBuilding full G1_K4Blind bank on {len(src_g1)} {best_g1['route']} items...", flush=True)
        g1_quads = make_shuffled_quads(generate_k4_blind_distractors(model, tokenizer, src_g1, n_samples=8, batch_size=8), seed=2611271)
        bank_g1 = build_scored_full_bank(archive, model, tokenizer, g1_quads, best_g1["template"], "bank_G1_K4Blind_full")
    else:
        bank_g1 = json.loads(g1_bank_path.read_text(encoding="utf-8"))

    g2_bank_path = OUT_DIR / "banks" / "bank_G2_HumanRetr_full.json"
    if not g2_bank_path.exists():
        print(f"\nBuilding full G2_HumanRetr bank on {len(src_g2)} {best_g2['route']} items...", flush=True)
        g2_quads = make_shuffled_quads(build_human_retrieval_distractors(model, tokenizer, src_g2, pool_all), seed=2611272)
        bank_g2 = build_scored_full_bank(archive, model, tokenizer, g2_quads, best_g2["template"], "bank_G2_HumanRetr_full")
    else:
        bank_g2 = json.loads(g2_bank_path.read_text(encoding="utf-8"))

    g0_bank_path = OUT_DIR / "banks" / "bank_G0_Random_full.json"
    if not g0_bank_path.exists():
        g0_quads = make_shuffled_quads(build_random_distractors(src_g1[:1100], pool_all), seed=2611270)
        bank_g0 = build_scored_full_bank(archive, model, tokenizer, g0_quads, best_g1["template"], "bank_G0_Random_full")
    else:
        bank_g0 = json.loads(g0_bank_path.read_text(encoding="utf-8"))

    def split_and_filter(bank, kl_key):
        n_tr = int(len(bank) * 0.85)
        tr_all, va_all = bank[:n_tr], bank[n_tr:]
        tr_cfit = [x for x in tr_all if x["conf"] >= 0.45 and x["margin"] >= 0.25 and x[kl_key] >= 0.15]
        # Relax slightly ONLY if fewer than 200 items survive strict filter (logged transparently)
        mode_used = "strict_c045_m025_kl015"
        if len(tr_cfit) < 200:
            tr_cfit = [x for x in tr_all if x["conf"] >= 0.35 and x["margin"] >= 0.18 and x[kl_key] >= 0.10]
            mode_used = "relaxed_c035_m018_kl010"
        return tr_all, va_all, tr_cfit, mode_used

    g1_tr, g1_va, g1_cfit, g1_fmode = split_and_filter(bank_g1, "kl_qa" if best_g1["cloze"] == "ClozeQA" else "kl_nat")
    g2_tr, g2_va, g2_cfit, g2_fmode = split_and_filter(bank_g2, "kl_qa" if best_g2["cloze"] == "ClozeQA" else "kl_nat")
    g0_tr, g0_va, _, _ = split_and_filter(bank_g0, "kl_qa")
    print(f"Banks ready: G1={len(g1_tr)}/{len(g1_va)} (CFIT={len(g1_cfit)}, {g1_fmode}) | G2={len(g2_tr)}/{len(g2_va)} (CFIT={len(g2_cfit)}, {g2_fmode}) | G0={len(g0_tr)}/{len(g0_va)}", flush=True)

    trust_rows = load_jsonl(UNLABELED_FIT)[:128]
    trust_seqs = [s for s in (tokenizer.encode(r["text"], add_special_tokens=False)[:192] for r in trust_rows) if len(s) >= 64]

    # Step 3: Exact 5-Way Same-Source Objective Ablation + Cloze Format + NLL Pareto Frontier
    abl_path = OUT_DIR / "CSBP_ABLATION_RESULTS.json"
    res = json.loads(abl_path.read_text(encoding="utf-8")) if abl_path.exists() else {
        "native": native_ev,
        "preflight_winners": {"g1": best_g1, "g2": best_g2},
        "bank_sizes": {"g1_tr": len(g1_tr), "g1_cfit": len(g1_cfit), "g1_fmode": g1_fmode, "g2_tr": len(g2_tr), "g2_cfit": len(g2_cfit), "g2_fmode": g2_fmode},
        "arms": {},
    }

    plan = [
        # Rung 1: Random distractors + hard_pseudo
        ("Rung1_G0Rand_ConfHard_ClzQA", g0_tr, g0_va, "mc_conf_hard", "ClozeQA", 0.0),
        # 5-Way Same-Source Objective Ablation on G2_HumanRetr (100% Real Human Text, zero style gap!)
        ("G2_Obj1_OrigGold_CE", g2_tr, g2_va, "orig_gold_ce", "ClozeQA", 0.0),
        ("G2_Obj2_MCHard_Unif", g2_tr, g2_va, "mc_hard_unif", "ClozeQA", 0.0),
        ("G2_Obj3_MCSoft_KL", g2_tr, g2_va, "mc_soft_kl", "ClozeQA", 0.0),
        ("G2_Obj4_MCConfHard_Unfilt", g2_tr, g2_va, "mc_conf_hard", "ClozeQA", 0.0),
        ("G2_Obj5_CFIT_ConfHard_ClzQA", g2_cfit, g2_va, "mc_conf_hard", "ClozeQA", 0.0),
        ("G2_Obj5_CFIT_ConfHard_ClzNat", g2_cfit, g2_va, "mc_conf_hard", "ClozeNat", 0.0),
        ("G2_Obj5_CFIT_ConfHard_Mix50", g2_cfit, g2_va, "mc_conf_hard", "ClozeMix50", 0.0),
        # 5-Way Same-Source Objective Ablation on G1_K4Blind (K=4 Short-Tail Blind Completion!)
        ("G1_Obj1_OrigGold_CE", g1_tr, g1_va, "orig_gold_ce", "ClozeQA", 0.0),
        ("G1_Obj2_MCHard_Unif", g1_tr, g1_va, "mc_hard_unif", "ClozeQA", 0.0),
        ("G1_Obj3_MCSoft_KL", g1_tr, g1_va, "mc_soft_kl", "ClozeQA", 0.0),
        ("G1_Obj4_MCConfHard_Unfilt", g1_tr, g1_va, "mc_conf_hard", "ClozeQA", 0.0),
        ("G1_Obj5_CFIT_ConfHard_ClzQA", g1_cfit, g1_va, "mc_conf_hard", "ClozeQA", 0.0),
        ("G1_Obj5_CFIT_ConfHard_ClzNat", g1_cfit, g1_va, "mc_conf_hard", "ClozeNat", 0.0),
        ("G1_Obj5_CFIT_ConfHard_Mix50", g1_cfit, g1_va, "mc_conf_hard", "ClozeMix50", 0.0),
        # Language-Fidelity Pareto sweep on G1 & G2 CFIT winners
        ("G1_CFIT_Pareto_lam025", g1_cfit, g1_va, "mc_conf_hard", "ClozeQA", 0.25),
        ("G1_CFIT_Pareto_lam050", g1_cfit, g1_va, "mc_conf_hard", "ClozeQA", 0.50),
        ("G1_CFIT_Pareto_lam100", g1_cfit, g1_va, "mc_conf_hard", "ClozeQA", 1.00),
        ("G2_CFIT_Pareto_lam050", g2_cfit, g2_va, "mc_conf_hard", "ClozeQA", 0.50),
        ("G2_CFIT_Pareto_lam100", g2_cfit, g2_va, "mc_conf_hard", "ClozeQA", 1.00),
    ]

    # Native held-out agreement baseline for strictly unlabeled selection
    nat_ag = {
        "g0": heldout_unlab_agree(archive, model, tokenizer, g0_va, None, "ClozeQA")[0],
        "g1_qa": heldout_unlab_agree(archive, model, tokenizer, g1_va, None, "ClozeQA")[0],
        "g1_nat": heldout_unlab_agree(archive, model, tokenizer, g1_va, None, "ClozeNat")[0],
        "g2_qa": heldout_unlab_agree(archive, model, tokenizer, g2_va, None, "ClozeQA")[0],
        "g2_nat": heldout_unlab_agree(archive, model, tokenizer, g2_va, None, "ClozeNat")[0],
    }
    res["native_heldout_unlab_agree"] = nat_ag

    for cname, tr_b, va_b, obj, cmode, lam in plan:
        for seed in SEEDS:
            arm_id = f"{cname}.s{seed}"
            if arm_id in res["arms"]:
                continue
            net, meta = train_csbp_arm(
                archive, model, tokenizer, tr_b, obj, cmode, seed,
                steps=STEPS, trust_seqs=trust_seqs, trust_lam=lam, tag=cname
            )
            ev = evaluator.evaluate(net, include_mmlu_dev158=True)
            unlab = evaluate_unlabeled_manifold(model, tokenizer, net, UNLABELED_AUDIT, max_docs=64)
            va_ag, va_kl = heldout_unlab_agree(archive, model, tokenizer, va_b, net, "ClozeNat" if cmode == "ClozeNat" else "ClozeQA")
            ckpt = OUT_DIR / "weights" / f"{arm_id}.pt"
            sha = save_rank1_controller(ckpt, net, {"arm": arm_id, "obj": obj, "cloze": cmode, "lam": lam, "seed": seed, "zero_qa_zero_labels": obj != "orig_gold_ce"})
            u_a, w_a = load_controller_weights(ckpt)
            cu = float((u_a @ u_t1) / (u_a.norm() * u_t1.norm()).clamp_min(1e-12))
            d_m5 = (ev["macro5"] - native_ev["macro5"]) * 100
            d_c3 = (ev["cloze3"] - native_ev["cloze3"]) * 100
            res["arms"][arm_id] = {
                "group": cname, "objective": obj, "cloze_mode": cmode, "trust_lam": lam, "seed": seed,
                "n_train": len(tr_b), "sha256": sha, "train": meta, "eval": ev, "unlabeled": unlab,
                "heldout_unlab_agree": va_ag, "heldout_unlab_kl": va_kl,
                "delta_macro5_pp": d_m5, "delta_cloze3_pp": d_c3, "cos_u_t1": cu,
            }
            print(
                f"  => {arm_id:38s} dMac5={d_m5:+6.2f} dClz3={d_c3:+6.2f} MMLU={ev['mmlu_val']['acc']*100:.2f}% "
                f"dNLL={unlab['delta_nll']:+.3f} ag={va_ag*100:.1f}% uT1={cu:+.3f}",
                flush=True,
            )
            save_json(abl_path, res)

    print("\nALL_PHASE1_GPU1_CSBP_DONE", flush=True)


if __name__ == "__main__":
    main()
