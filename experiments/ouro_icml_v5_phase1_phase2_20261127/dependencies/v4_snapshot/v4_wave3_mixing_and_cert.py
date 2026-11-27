"""Wave 3: Exp T5 — Theorem 2 Mixing Laws, Convex Capacity Certificate, & Optimal 4,096-Param Rank-1 Controllers.
Part A: Exact 3-task functional gradient geometry & nuclear-norm / rank-r Pareto capacity certificate
        across (Cloze-ARC, MMLU-Letter, Unlabeled-Prose-NLL).
Part B: Controlled Theorem-2 mixing & deconfounding experiments (all matched 1,050 steps, cold-start, 2 seeds):
        1. T5_Uniform_All12          (naive uniform mix over all 12 candidate sources)
        2. T5_CFIT_BLUE_All12        (Theorem-2 exact alpha_i ~ P_deconf * M_pres over all 12 sources + trust region)
        3. T5_CFIT_Top6_4096         (Top-6 CFIT sources weighted by P_deconf * M_pres, 4,096-param Rank-1)
        4. T5_CFIT_NonARC_4096       (ZERO ARC-C, ZERO ARC-E, ZERO T1 weights! Top non-ARC CFIT sources)
        5. T5_Poison_Raw             (50% ARC+CSQA + 50% raw BoolQ+MedMCQA+WinoGrande)
        6. T5_Poison_CFIT_Rescued    (50% ARC+CSQA + 50% CFIT Regime-III filtered MedMCQA)
        7. T5_CFIT_Top6_MMLUGuard    (Top-6 CFIT + benign-margin NLL trust region + MMLU-500 cloze_full diversity, 4,096-param Rank-1)
        8. T5_Rank2_CapacityProbe    (Identical to T5_CFIT_Top6_4096 but Rank-2 = 8,192 params to test Rank-1 capacity bottleneck!)
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import minimal
from v4_common import (
    DEPTHS,
    EXT_DEV_DIR,
    RECIPE,
    UNLABELED_AUDIT,
    UNLABELED_FIT,
    FastUnifiedDevEvaluator,
    cloze_option_logits_differentiable,
    evaluate_unlabeled_manifold,
    extract_q_opts_gold,
    forward_with_hook_all_tokens,
    letter_option_logits_differentiable,
    load_base_runtime,
    load_controller_weights,
    load_jsonl,
    load_minimal_controller,
    make_candidate_row,
    save_json,
    save_rank1_controller,
)


def resolve_pool_path(src: str) -> Path:
    p1 = Path("/root/ouro-loop-aware-functional-identification-20261125/data/pools_complete") / f"{src}.jsonl"
    p2 = Path("/root/ouro-v3-controllability-20261126/stage0_env/PROSPECTIVE_TRAIN_DRAFT") / f"{src}.jsonl"
    return p1 if p1.exists() else p2


def load_source_items(src: str, max_items=6000):
    raw = load_jsonl(resolve_pool_path(src))
    rng = random.Random(f"t5_load/{src}")
    if len(raw) > max_items:
        idx = list(range(len(raw)))
        rng.shuffle(idx)
        raw = [raw[i] for i in sorted(idx[:max_items])]
    items = []
    for r in raw:
        q, opts, g = extract_q_opts_gold(r)
        if 2 <= len(opts) <= 10:
            items.append({"id": r["sample_id"], "dataset": src, "question": q, "options": opts, "gold": g})
    return items


def run_capacity_certificate(out_root: Path, archive, model, tokenizer):
    """Exact 3-task receiver-boundary gradient matrices (M_clz, M_mmlu_let, M_prose_nll) & nuclear-norm / rank-r certificate."""
    cert_path = out_root / "exp_t5_mixing" / "CAPACITY_CERTIFICATE.json"
    if cert_path.exists():
        print("[SKIP] CAPACITY_CERTIFICATE.json already exists.", flush=True)
        return json.loads(cert_path.read_text(encoding="utf-8"))

    print("\n================ EXP T5-CERT: CONVEX CAPACITY & CONFLICT CERTIFICATE ================", flush=True)
    ops14 = torch.load(out_root / "cfit_14_features" / "SOURCE_OPERATORS_14.pt", map_location="cpu", weights_only=False)
    sd = torch.load(out_root / "SIGMA_H.pt", map_location="cpu", weights_only=False)
    S = sd["sigma_unlab"].double()
    ev, ec = torch.linalg.eigh((S + S.T) * 0.5)
    S_reg = (ec * (ev.clamp_min(0.0) + float(ev.mean()) * 0.05)) @ ec.T
    S_half = (ec * (ev.clamp_min(0.0) + float(ev.mean()) * 0.05).sqrt()) @ ec.T
    S_half_inv = (ec * (ev.clamp_min(0.0) + float(ev.mean()) * 0.05).rsqrt()) @ ec.T

    # Helper: compute exact receiver-boundary gradient matrix G = E[g h^T] at depth=4 for a differentiable scalar loss
    def grad_matrix_for_mmlu_letter(n_items=256):
        raw = load_jsonl(EXT_DEV_DIR / "mmlu_validation.jsonl")[:n_items]
        G = torch.zeros((2048, 2048), dtype=torch.float64, device="cuda")
        # Use a zero-initialized dummy probe delta_A = u w^T via an explicit 2048x2048 matrix A at h^(3)!
        A = torch.zeros((2048, 2048), dtype=torch.float32, device="cuda", requires_grad=True)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        letter_ids = [tokenizer.encode(" " + chr(65 + i), add_special_tokens=False)[-1] for i in range(4)]
        core = model.model
        old_steps = getattr(core, "total_ut_steps", 4)
        count = 0
        for i in range(0, len(raw), 4):
            chunk = raw[i:i + 4]
            prompts, golds = [], []
            for r in chunk:
                q, opts, g = extract_q_opts_gold(r)
                lines = "\n".join(f"{chr(65 + k)}. {t}" for k, t in enumerate(opts[:4]))
                subj = r.get("subject", "").replace("_", " ")
                prompts.append(f"The following are multiple choice questions (with answers) about {subj}.\n\n{q}\n{lines}\nAnswer:")
                golds.append(g)
            enc = [tokenizer.encode(p, add_special_tokens=False)[-1024:] for p in prompts]
            mx = max(len(x) for x in enc)
            ids = torch.full((len(enc), mx), pad_id, dtype=torch.long, device="cuda")
            attn = torch.zeros((len(enc), mx), dtype=torch.long, device="cuda")
            last_pos = [len(s) - 1 for s in enc]
            for b, s in enumerate(enc):
                ids[b, :len(s)] = torch.tensor(s, dtype=torch.long, device="cuda")
                attn[b, :len(s)] = 1
            if A.grad is not None:
                A.grad.zero_()
            core.total_ut_steps = 4
            ctr = 0
            def hook(_, args):
                nonlocal ctr
                ctr += 1
                if ctr == 3:
                    h = args[0]
                    dh = torch.einsum("btd,ed->bte", h.float(), A)
                    return ((h.float() + dh).to(h.dtype), *args[1:])
                return None
            hd = core.norm.register_forward_pre_hook(hook)
            try:
                with torch.enable_grad():
                    out = core(input_ids=ids, attention_mask=attn, use_cache=False)
                    hid = out[1][-1]
                    lh = hid[torch.arange(len(enc), device="cuda"), torch.tensor(last_pos, device="cuda")]
                    lg = model.lm_head(lh).float()[:, letter_ids]
                    # Ascent direction on -CE (i.e. +log p(gold))
                    obj = -F.cross_entropy(lg, torch.tensor(golds, device="cuda"), reduction="sum")
                    obj.backward()
            finally:
                hd.remove()
                core.total_ut_steps = old_steps
            G.add_(A.grad.detach().double())
            count += len(enc)
        return (G / count).cpu()

    def grad_matrix_for_prose_nll(n_docs=64):
        rows = load_jsonl(UNLABELED_FIT)[:n_docs]
        G = torch.zeros((2048, 2048), dtype=torch.float64, device="cuda")
        A = torch.zeros((2048, 2048), dtype=torch.float32, device="cuda", requires_grad=True)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        core = model.model
        old_steps = getattr(core, "total_ut_steps", 4)
        count = 0
        for i in range(0, len(rows), 2):
            chunk = rows[i:i + 2]
            enc = [tokenizer.encode(r["text"], add_special_tokens=False)[:192] for r in chunk]
            mx = max(len(x) for x in enc)
            ids = torch.full((len(enc), mx), pad_id, dtype=torch.long, device="cuda")
            attn = torch.zeros((len(enc), mx), dtype=torch.long, device="cuda")
            for b, s in enumerate(enc):
                ids[b, :len(s)] = torch.tensor(s, dtype=torch.long, device="cuda")
                attn[b, :len(s)] = 1
            valid = attn[:, 1:].bool()
            targets = ids[:, 1:][valid]
            if A.grad is not None:
                A.grad.zero_()
            core.total_ut_steps = 4
            ctr = 0
            def hook(_, args):
                nonlocal ctr
                ctr += 1
                if ctr == 3:
                    h = args[0]
                    dh = torch.einsum("btd,ed->bte", h.float(), A)
                    return ((h.float() + dh).to(h.dtype), *args[1:])
                return None
            hd = core.norm.register_forward_pre_hook(hook)
            try:
                with torch.enable_grad():
                    out = core(input_ids=ids, attention_mask=attn, use_cache=False)
                    hid = out[1][-1]
                    z = model.lm_head(hid[:, :-1][valid]).float()
                    obj = -F.cross_entropy(z, targets, reduction="mean") * len(enc)
                    obj.backward()
            finally:
                hd.remove()
                core.total_ut_steps = old_steps
            G.add_(A.grad.detach().double())
            count += len(enc)
        return (G / count).cpu()

    # Load ARC-Challenge ordinary gradient matrix from v3 shards directly
    f_inputs = json.loads(Path("/root/ouro-v3-controllability-20261126/stage4_d3/FORECAST_INPUTS.json").read_text())
    M_arc = None
    tot_arc = 0
    for sh in f_inputs["arc_challenge"]["shards"]:
        ck = torch.load(sh, map_location="cpu", weights_only=False)
        M_arc = ck["sums"]["ordinary"].double() if M_arc is None else M_arc + ck["sums"]["ordinary"].double()
        tot_arc += int(ck["processed"])
    M_arc = M_arc / tot_arc

    M_mmlu = grad_matrix_for_mmlu_letter(n_items=256)
    M_prose = grad_matrix_for_prose_nll(n_docs=64)

    # Transform to H_{Sigma_h}-whitened coordinates: M_tilde = M @ S_half_inv
    # In this space, <A_tilde, M_tilde>_F = <A, M> AND ||A_tilde||_F = ||A||_{Sigma_h}!
    def whiten(M):
        W = M.double() @ S_half_inv
        return W / W.norm().clamp_min(1e-12)

    W_arc = whiten(M_arc)
    W_mmlu = whiten(M_mmlu)
    W_prose = whiten(M_prose)

    # Pairwise functional-space cosines & singular spectra!
    def mat_cos(A, B):
        return float((A * B).sum() / (A.norm() * B.norm()).clamp_min(1e-12))

    def spec_info(W):
        s = torch.linalg.svdvals(W)
        e = s.pow(2)
        return {
            "s1_frac_l1": float(s[0] / s.sum()),
            "s1_frac_l2": float(e[0] / e.sum()),
            "s12_frac_l2": float(e[:2].sum() / e.sum()),
            "s14_frac_l2": float(e[:4].sum() / e.sum()),
            "stable_rank": float(e.sum() / e[0]),
            "nuclear_over_op": float(s.sum() / s[0]),
            "top5_sv": [float(x) for x in s[:5]],
        }

    # Global Convex Capacity Upper Bound Certificate for multi-objective M_comb(gamma) = W_arc + gamma * W_mmlu!
    # For any rank-r operator ||A||_{Sigma_h} <= 1:
    #   max_{rank(A)<=r, ||A||_{Sigma_h}<=1} <A, W_comb> = sqrt(sum_{i=1}^r sigma_i(W_comb)^2)
    #   global unconstrained (full-rank!) upper bound      = ||W_comb||_F = sqrt(sum_i sigma_i^2)
    #   Rank-1 global optimality ratio = sigma_1(W_comb) / ||W_comb||_F !
    pareto = {}
    for gamma in (0.0, 0.25, 0.5, 1.0, 2.0):
        Wc = W_arc + gamma * W_mmlu
        U, s, Vt = torch.linalg.svd(Wc, full_matrices=False)
        frob = float(s.norm())
        # Rank-1 optimal operator A1 = U[:,0] Vt[0]
        A1 = torch.outer(U[:, 0], Vt[0])
        A2 = (U[:, :2] * s[:2]) @ Vt[:2, :]
        A2 = A2 / A2.norm().clamp_min(1e-12)
        Afull = Wc / frob
        pareto[f"gamma_{gamma:.2f}"] = {
            "rank1_global_opt_ratio": float(s[0] / frob),
            "rank2_global_opt_ratio": float(s[:2].norm() / frob),
            "rank1_vs_rank2_gap_pct": float((s[:2].norm() - s[0]) / frob * 100),
            "r1_arc_gain": float((A1 * W_arc).sum()),
            "r1_mmlu_gain": float((A1 * W_mmlu).sum()),
            "r1_prose_gain": float((A1 * W_prose).sum()),
            "r2_arc_gain": float((A2 * W_arc).sum()),
            "r2_mmlu_gain": float((A2 * W_mmlu).sum()),
            "r2_prose_gain": float((A2 * W_prose).sum()),
            "full_arc_gain": float((Afull * W_arc).sum()),
            "full_mmlu_gain": float((Afull * W_mmlu).sum()),
            "top4_sv": [float(x) for x in s[:4]],
        }

    # Also evaluate exact projection of ARC Rank-1 onto MMLU & Prose
    U_a, s_a, Vt_a = torch.linalg.svd(W_arc, full_matrices=False)
    U_m, s_m, Vt_m = torch.linalg.svd(W_mmlu, full_matrices=False)
    U_p, s_p, Vt_p = torch.linalg.svd(W_prose, full_matrices=False)
    cert = {
        "pairwise_full_matrix_cos_sigma": {
            "arc_vs_mmlu_letter": mat_cos(W_arc, W_mmlu),
            "arc_vs_prose_nll": mat_cos(W_arc, W_prose),
            "mmlu_letter_vs_prose_nll": mat_cos(W_mmlu, W_prose),
        },
        "pairwise_top1_axis_cos_sigma": {
            "u_arc_vs_u_mmlu": float(U_a[:, 0] @ U_m[:, 0]),
            "v_arc_vs_v_mmlu": float(Vt_a[0] @ Vt_m[0]),
            "op1_arc_vs_mmlu": float((U_a[:, 0] @ U_m[:, 0]) * (Vt_a[0] @ Vt_m[0])),
            "op1_arc_vs_prose": float((U_a[:, 0] @ U_p[:, 0]) * (Vt_a[0] @ Vt_p[0])),
            "op1_mmlu_vs_prose": float((U_m[:, 0] @ U_p[:, 0]) * (Vt_m[0] @ Vt_p[0])),
        },
        "spectra": {"arc_cloze": spec_info(W_arc), "mmlu_letter": spec_info(W_mmlu), "prose_nll": spec_info(W_prose)},
        "pareto_rank_certificate": pareto,
    }
    save_json(cert_path, cert)
    print("CAPACITY CERTIFICATE:\n" + json.dumps(cert, indent=2), flush=True)
    return cert


def train_mixed_arm(archive, model, tokenizer, weighted_pools, arm_id, seed, steps=1050, rank=1, trust_seqs=None, trust_lam=0.0, trust_margin=0.55):
    """Train cold-start Rank-r controller on weighted multi-source pools with optional benign-margin unlabeled NLL trust region."""
    torch.manual_seed(seed)
    net = minimal.Minimal("linear", rank, 0.45).to(model.device)
    net.requires_grad_(True)
    opt = torch.optim.AdamW(net.parameters(), lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"])
    rng = random.Random(f"t5/{arm_id}/{seed}")

    names = [n for n, _, _ in weighted_pools]
    probs = [w for _, w, _ in weighted_pools]
    psum = sum(probs)
    probs = [w / psum for w in probs]
    pools = {n: list(items) for n, _, items in weighted_pools}
    ptrs = {n: 0 for n in names}
    orders = {}
    for n in names:
        o = list(range(len(pools[n])))
        rng.shuffle(o)
        orders[n] = o

    def draw_one():
        u = rng.random()
        acc = 0.0
        chosen = names[-1]
        for n, p in zip(names, probs):
            acc += p
            if u <= acc:
                chosen = n
                break
        if ptrs[chosen] >= len(orders[chosen]):
            rng.shuffle(orders[chosen])
            ptrs[chosen] = 0
        it = pools[chosen][orders[chosen][ptrs[chosen]]]
        ptrs[chosen] += 1
        return it

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    nat_cache = {}

    def trust_pen(seqs, depth):
        mx = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), mx), pad_id, dtype=torch.long, device=model.device)
        attn = torch.zeros((len(seqs), mx), dtype=torch.long, device=model.device)
        for b, s in enumerate(seqs):
            ids[b, :len(s)] = torch.tensor(s, dtype=torch.long, device=model.device)
            attn[b, :len(s)] = 1
        valid = attn[:, 1:].bool()
        targets = ids[:, 1:][valid]
        key = (depth, len(seqs), int(ids[0, 0]), int(ids[-1, -1]))
        if key not in nat_cache:
            with torch.no_grad():
                h0 = forward_with_hook_all_tokens(model, ids, attn, depth=depth, net=None)
                nat_cache[key] = float(F.cross_entropy(model.lm_head(h0[:, :-1][valid]).float(), targets).detach())
        h1 = forward_with_hook_all_tokens(model, ids, attn, depth=depth, net=net)
        nll1 = F.cross_entropy(model.lm_head(h1[:, :-1][valid]).float(), targets)
        return F.relu(nll1 - nat_cache[key] - trust_margin)

    t0 = time.perf_counter()
    losses = []
    for step in range(1, steps + 1):
        depth = DEPTHS[(step - 1) % len(DEPTHS)]
        mb = [draw_one() for _ in range(RECIPE["effective_batch"])]
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        max_len = max(len(it["question"]) + max(len(o) for o in it["options"]) for it in mb)
        max_opts = max(len(it["options"]) for it in mb)
        sub_bs = 1 if (max_len > 1400 or max_len * max_opts > 5200) else (2 if (max_len > 650 or max_len * max_opts > 2600) else len(mb))
        for j in range(0, len(mb), sub_bs):
            sub = mb[j:j + sub_bs]
            crows = [make_candidate_row(it["id"], it["question"], it["options"], it["gold"], dataset=it["dataset"]) for it in sub]
            with torch.enable_grad():
                opt_logits, _, _ = cloze_option_logits_differentiable(archive, model, tokenizer, crows, depth, net)
                loss_sub = sum(F.cross_entropy(lg.unsqueeze(0), torch.tensor([it["gold"]], device=lg.device))
                               for lg, it in zip(opt_logits, sub)) / len(mb)
                loss_sub.backward()
                step_loss += float(loss_sub.detach())
        if trust_lam > 0.0 and trust_seqs and step % 3 == 0:
            t_mb = [trust_seqs[(step + j) % len(trust_seqs)] for j in range(2)]
            with torch.enable_grad():
                pl = trust_lam * trust_pen(t_mb, depth)
                pl.backward()
                step_loss += float(pl.detach())
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), RECIPE["clip"])
        assert grad.isfinite().item()
        opt.step()
        losses.append(step_loss)
        if step % 350 == 0 or step == steps:
            print(f"  [T5 {arm_id}] step={step}/{steps} loss={sum(losses[-70:])/min(len(losses),70):.4f} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return net, {"final_loss_avg70": sum(losses[-70:]) / 70.0, "seconds": time.perf_counter() - t0, "rank": rank, "n_params": sum(p.numel() for p in net.parameters())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--arms", default="all", help="Comma-separated arm names or 'all' or 'cert'")
    ap.add_argument("--steps", type=int, default=1050)
    args = ap.parse_args()

    torch.set_num_threads(4)
    out_dir = args.out_root / "exp_t5_mixing"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, archive, model, tokenizer = load_base_runtime()

    if args.arms == "cert" or args.arms == "all":
        run_capacity_certificate(args.out_root, archive, model, tokenizer)
        if args.arms == "cert":
            return

    evaluator = FastUnifiedDevEvaluator(cfg, archive, model, tokenizer)
    native_ev = evaluator.evaluate(None, include_mmlu_dev158=True)
    print(f"[T5] Native: macro5={native_ev['macro5']*100:.2f}% cloze3={native_ev['cloze3']*100:.2f}% mmlu_dev158={native_ev['mmlu_dev158']['acc']*100:.2f}%", flush=True)

    # Load exact pre-computed CFIT-14 P_deconf * M_pres weights (ZERO fitted weights!)
    cfit14 = json.loads((args.out_root / "cfit_14_features" / "CFIT_14_SOURCE_FEATURES.json").read_text())["per_source"]
    cand12 = ["arc_challenge", "boolq", "csqa", "hellaswag", "medmcqa", "obqa", "piqa", "qasc", "race", "scienceqa", "social_iqa", "winogrande"]
    w_p0 = {s: max(0.0, cfit14[s]["P_deconf_op_sig"]) * cfit14[s]["M_pres_ord"] for s in cand12}

    # Pre-load pools lazily
    pool_cache = {}
    def get_pool(s):
        if s not in pool_cache:
            pool_cache[s] = load_source_items(s, max_items=6000)
        return pool_cache[s]

    # Regime-III filtered MedMCQA (top 512 by w_iii_mc from cfit_14_features/regime3_samples/medmcqa.jsonl)
    def get_medmcqa_r3():
        if "medmcqa_r3" not in pool_cache:
            scored = load_jsonl(args.out_root / "cfit_14_features" / "regime3_samples" / "medmcqa.jsonl")
            top_ids = {r["sample_id"] for r in sorted(scored, key=lambda x: x["w_iii_mc"], reverse=True)[:512]}
            raw = {r["sample_id"]: r for r in load_jsonl(resolve_pool_path("medmcqa"))}
            its = []
            for sid in sorted(top_ids):
                q, opts, g = extract_q_opts_gold(raw[sid])
                its.append({"id": sid, "dataset": "medmcqa_r3", "question": q, "options": opts, "gold": g})
            pool_cache["medmcqa_r3"] = its
        return pool_cache["medmcqa_r3"]

    def get_mmlu500_cloze():
        if "mmlu500_clz" not in pool_cache:
            its = []
            for r in load_jsonl(EXT_DEV_DIR / "mmlu_validation.jsonl"):
                q, opts, g = extract_q_opts_gold(r)
                if len(opts) == 4:
                    its.append({"id": r["sample_id"], "dataset": "mmlu500_clz", "question": q, "options": opts, "gold": g})
            pool_cache["mmlu500_clz"] = its
        return pool_cache["mmlu500_clz"]

    trust_seqs = [tokenizer.encode(r["text"], add_special_tokens=False)[:192] for r in load_jsonl(UNLABELED_FIT)[:128]]
    trust_seqs = [s for s in trust_seqs if len(s) >= 64]

    top6 = ["arc_challenge", "social_iqa", "obqa", "race", "scienceqa", "piqa"]
    non_arc6 = ["social_iqa", "obqa", "race", "scienceqa", "piqa", "hellaswag"]

    def build_spec(name):
        if name == "T5_Uniform_All12":
            return [(s, 1.0, get_pool(s)) for s in cand12], 1, 0.0, 0.55
        elif name == "T5_CFIT_BLUE_All12":
            return [(s, max(w_p0[s], 1e-6), get_pool(s)) for s in cand12 if w_p0[s] > 0.01], 1, 0.25, 0.55
        elif name == "T5_CFIT_Top6_4096":
            return [(s, w_p0[s], get_pool(s)) for s in top6], 1, 0.0, 0.55
        elif name == "T5_CFIT_Top6_Trust_4096":
            return [(s, w_p0[s], get_pool(s)) for s in top6], 1, 0.5, 0.50
        elif name == "T5_CFIT_NonARC_4096":
            return [(s, w_p0[s], get_pool(s)) for s in non_arc6], 1, 0.0, 0.55
        elif name == "T5_Poison_Raw":
            return [("arc_challenge", 0.25, get_pool("arc_challenge")), ("csqa", 0.25, get_pool("csqa")),
                    ("boolq", 0.17, get_pool("boolq")), ("medmcqa", 0.17, get_pool("medmcqa")), ("winogrande", 0.16, get_pool("winogrande"))], 1, 0.0, 0.55
        elif name == "T5_Poison_CFIT_Rescued":
            return [("arc_challenge", 0.25, get_pool("arc_challenge")), ("csqa", 0.25, get_pool("csqa")),
                    ("medmcqa_r3", 0.50, get_medmcqa_r3())], 1, 0.0, 0.55
        elif name == "T5_CFIT_Top6_PlusMMLUCloze":
            # Strictly 4,096-param Rank-1 adding MMLU-500 cloze_full (disjoint from mmlu_dev158 & mmlu_test!) + trust region
            wp = [(s, w_p0[s], get_pool(s)) for s in top6]
            tot = sum(w for _, w, _ in wp)
            wp.append(("mmlu500_clz", tot * 0.20, get_mmlu500_cloze()))
            return wp, 1, 0.25, 0.55
        elif name == "T5_Rank2_CapacityProbe":
            return [(s, w_p0[s], get_pool(s)) for s in top6], 2, 0.0, 0.55
        raise ValueError(name)

    all_arms = [
        "T5_Uniform_All12", "T5_CFIT_BLUE_All12", "T5_CFIT_Top6_4096", "T5_CFIT_Top6_Trust_4096",
        "T5_CFIT_NonARC_4096", "T5_Poison_Raw", "T5_Poison_CFIT_Rescued",
        "T5_CFIT_Top6_PlusMMLUCloze", "T5_Rank2_CapacityProbe",
    ]
    run_arms = all_arms if args.arms == "all" else [x.strip() for x in args.arms.split(",") if x.strip()]
    seeds = (26112801, 26112802)

    for arm_base in run_arms:
        res_path = out_dir / f"T5_{arm_base}.json"
        results = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {"arm": arm_base, "native": native_ev, "seeds": {}}
        wp, rank, lam, marg = build_spec(arm_base)
        for seed in seeds:
            sid = f"{arm_base}.s{seed}"
            if sid in results["seeds"]:
                continue
            ckpt = out_dir / f"{sid}.pt"
            if ckpt.exists():
                net = load_minimal_controller(ckpt, device=model.device)
                meta = {"reused": True, "rank": rank}
            else:
                net, meta = train_mixed_arm(archive, model, tokenizer, wp, sid, seed, steps=args.steps, rank=rank, trust_seqs=trust_seqs, trust_lam=lam, trust_margin=marg)
                meta["sha256"] = save_rank1_controller(ckpt, net, {"arm": sid, "rank": rank, "steps": args.steps, "seed": seed, "weights": [(n, w) for n, w, _ in wp]})
            ev = evaluator.evaluate(net, include_mmlu_dev158=True)
            unlab = evaluate_unlabeled_manifold(model, tokenizer, net, UNLABELED_AUDIT, max_docs=64)
            d_mac5 = (ev["macro5"] - native_ev["macro5"]) * 100
            d_clz3 = (ev["cloze3"] - native_ev["cloze3"]) * 100
            d_nm4 = (ev["non_mmlu4"] - native_ev["non_mmlu4"]) * 100
            d_m158 = (ev["mmlu_dev158"]["acc"] - native_ev["mmlu_dev158"]["acc"]) * 100
            results["seeds"][sid] = {
                "seed": seed, "train": meta, "eval": ev, "unlabeled": unlab,
                "delta_macro5_pp": d_mac5, "delta_cloze3_pp": d_clz3, "delta_non_mmlu4_pp": d_nm4, "delta_mmlu_dev158_pp": d_m158,
            }
            print(f"  => [T5] {sid:36s}: mac5={ev['macro5']*100:5.2f}% ({d_mac5:+5.2f}) clz3={d_clz3:+5.2f} nm4={d_nm4:+5.2f} m158={d_m158:+5.2f} dNLL={unlab['delta_nll']:+.3f} eos={unlab['eos_abs_shift']:.4f}", flush=True)
            save_json(res_path, results)
        sub = list(results["seeds"].values())
        results["summary"] = {
            "macro5_pct": sum(x["eval"]["macro5"] for x in sub) / len(sub) * 100,
            "delta_macro5_pp": sum(x["delta_macro5_pp"] for x in sub) / len(sub),
            "delta_cloze3_pp": sum(x["delta_cloze3_pp"] for x in sub) / len(sub),
            "delta_non_mmlu4_pp": sum(x["delta_non_mmlu4_pp"] for x in sub) / len(sub),
            "delta_mmlu_dev158_pp": sum(x["delta_mmlu_dev158_pp"] for x in sub) / len(sub),
            "delta_nll_unlab": sum(x["unlabeled"]["delta_nll"] for x in sub) / len(sub),
        }
        save_json(res_path, results)
        print(f"  ===> {arm_base:28s} MEAN: mac5={results['summary']['macro5_pct']:.2f}% ({results['summary']['delta_macro5_pp']:+.2f} pp) clz3={results['summary']['delta_cloze3_pp']:+.2f} m158={results['summary']['delta_mmlu_dev158_pp']:+.2f}", flush=True)
    print("EXP_T5_SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
