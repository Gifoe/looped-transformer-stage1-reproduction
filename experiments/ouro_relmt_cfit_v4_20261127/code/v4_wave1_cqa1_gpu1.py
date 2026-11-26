"""Wave 1 on cqa1 GPU 1 (vGPU 24GB):
1. Exp T2-B: Pure Unlabeled WikiText Prose Contrastive Ranking in FULL R^2048 + one-sided NLL preservation
   (ZERO QA templates, ZERO human labels — directly fixing BOTH fatal bugs of D2 U1_PMI!).
2. Exp T4-CFIT-14: Compute exact CFIT metrics across all 14 existing sources:
   - Exact Sigma_h on cqa1 (saved for downstream reuse)
   - Deconfounded Purity P_deconf, Deconfounded Sigma_h Fold Stability C_ctx, Cross-source Sigma_h Compatibility
     (reusing all 56 already-computed 2048x2048 COLLECT_RESUME.pt moment shards!)
   - Exact Regime-III Observability O_III (p_0^MC vs p_0^cloze + native loop emergence K4 vs K1)
     across ALL 14 sources (and saving per-sample scores on medmcqa/qasc/obqa/race/arc for Exp T3-B!)
   - Exact On-Manifold Preservation M_pres (delta_NLL_unlab + eos_abs_shift) of each source's SVD1 operator!
"""

import hashlib
import json
import math
import random
import time
from itertools import combinations
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
    V3_ROOT,
    FastUnifiedDevEvaluator,
    cloze_option_logits_differentiable,
    compute_sigma_h_unlabeled,
    evaluate_unlabeled_manifold,
    extract_q_opts_gold,
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

OUT_ROOT = Path("/root/autodl-tmp/ouro-v4-cfit-20261126")


def build_prose_contrastive_bank(tokenizer, corpus_path=UNLABELED_FIT, n_items=800):
    """Build pure-prose 4-way Cloze contrast tuples from WikiText alone (ZERO QA templates, ZERO labels)."""
    rows = load_jsonl(corpus_path)
    segments = []
    for r in rows:
        ids = tokenizer.encode(r["text"], add_special_tokens=False)
        if len(ids) < 160:
            continue
        # Prefix 110..150 tokens, continuation 18 tokens
        cut = min(140, len(ids) - 24)
        pref = tokenizer.decode(ids[:cut], skip_special_tokens=True).strip()
        cont = tokenizer.decode(ids[cut:cut + 18], skip_special_tokens=True).strip()
        if len(pref) > 60 and len(cont) >= 15:
            segments.append((r["sample_id"], r.get("document_id", r["sample_id"]), pref, cont))
    rng = random.Random(26112691)
    bank = []
    for i, (sid, doc_id, pref, cont) in enumerate(segments[:n_items]):
        # Draw 3 fluent distractors from distinct documents
        distractors = []
        tries = 0
        while len(distractors) < 3 and tries < 50:
            j = rng.randrange(len(segments))
            tries += 1
            if segments[j][1] != doc_id and segments[j][3] != cont and segments[j][3] not in distractors:
                distractors.append(segments[j][3])
        if len(distractors) == 3:
            opts = [cont] + distractors
            order = [0, 1, 2, 3]
            rng.shuffle(order)
            rot_opts = [opts[k] for k in order]
            gold = order.index(0)
            bank.append({"id": sid, "question": pref, "options": rot_opts, "gold": gold})
    print(f"Built pure-prose contrastive bank: {len(bank)} tuples from WikiText", flush=True)
    return bank


def run_exp_t2b(cfg, archive, model, tokenizer, evaluator):
    print("\n================ EXP T2-B: PURE UNLABELED WIKITEXT PROSE CONTRAST ================", flush=True)
    out_dir = OUT_ROOT / "exp_t2b_prose_contrast"
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = build_prose_contrastive_bank(tokenizer, UNLABELED_FIT, n_items=800)

    # Also keep raw sequences for one-sided real-text NLL preservation
    raw_seqs = [tokenizer.encode(r["text"], add_special_tokens=False)[:192] for r in load_jsonl(UNLABELED_FIT)[:128]]
    raw_seqs = [s for s in raw_seqs if len(s) >= 64]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    native_ev = evaluator.evaluate(None)
    u_t1, w_t1 = load_controller_weights(Path(cfg["historical_t1"]))
    results = {"native": native_ev, "arms": {}}
    nat_cache = {}

    def onesided_nll_pen(seqs, net, depth):
        device = next(model.parameters()).device
        mx = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), mx), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(seqs), mx), dtype=torch.long, device=device)
        for b, s in enumerate(seqs):
            ids[b, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
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
        return F.relu(nll1 - nat_cache[key] - 0.02)

    configs = [("prose_rank_lam0", 0.0), ("prose_rank_lam1", 1.0), ("prose_rank_lam3", 3.0)]
    seeds = (26112691, 26112692)
    for cname, lam in configs:
        for seed in seeds:
            arm_id = f"{cname}.s{seed}"
            ckpt = out_dir / f"{arm_id}.pt"
            if ckpt.exists():
                net = minimal.load_net(ckpt)
                meta = {"reused": True}
            else:
                torch.manual_seed(seed)
                net = minimal.Minimal("linear", 1, 0.45).to(model.device)
                net.requires_grad_(True)
                opt = torch.optim.AdamW(net.parameters(), lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"])
                rng = random.Random(f"t2b/{cname}/{seed}")
                order = list(range(len(bank)))
                rng.shuffle(order)
                ptr = 0
                t0 = time.perf_counter()
                losses = []
                for step in range(1, 841):
                    depth = DEPTHS[(step - 1) % len(DEPTHS)]
                    mb = []
                    for _ in range(RECIPE["effective_batch"]):
                        if ptr >= len(order):
                            rng.shuffle(order)
                            ptr = 0
                        mb.append(bank[order[ptr]])
                        ptr += 1
                    crows = [make_candidate_row(it["id"], it["question"], it["options"], it["gold"]) for it in mb]
                    opt.zero_grad(set_to_none=True)
                    with torch.enable_grad():
                        opt_logits, _, _ = cloze_option_logits_differentiable(archive, model, tokenizer, crows, depth, net)
                        loss = sum(F.cross_entropy(lg.unsqueeze(0), torch.tensor([it["gold"]], device=lg.device))
                                   for lg, it in zip(opt_logits, mb)) / len(mb)
                        if lam > 0.0 and step % 2 == 0:
                            t_mb = [raw_seqs[(step + j) % len(raw_seqs)] for j in range(2)]
                            loss = loss + lam * onesided_nll_pen(t_mb, net, depth)
                        loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(net.parameters(), RECIPE["clip"])
                    assert grad.isfinite().item()
                    opt.step()
                    losses.append(float(loss.detach()))
                    if step % 280 == 0:
                        print(f"  [T2B {arm_id}] step={step}/840 loss={sum(losses[-70:])/70:.4f} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
                sha = save_rank1_controller(ckpt, net, {"arm": arm_id, "lam": lam, "seed": seed, "zero_qa_zero_labels": True})
                meta = {"sha256": sha, "final_loss_avg70": sum(losses[-70:]) / 70.0, "seconds": time.perf_counter() - t0}
            ev = evaluator.evaluate(net)
            unlab = evaluate_unlabeled_manifold(model, tokenizer, net, UNLABELED_AUDIT, max_docs=64)
            u_a, w_a = load_controller_weights(ckpt)
            cu = float((u_a @ u_t1) / (u_a.norm() * u_t1.norm()))
            cw = float((w_a @ w_t1) / (w_a.norm() * w_t1.norm()))
            d_mac5 = (ev["macro5"] - native_ev["macro5"]) * 100
            d_clz3 = (ev["cloze3"] - native_ev["cloze3"]) * 100
            results["arms"][arm_id] = {
                "group": cname, "lam": lam, "seed": seed, "train": meta, "eval": ev, "unlabeled": unlab,
                "delta_macro5_pp": d_mac5, "delta_cloze3_pp": d_clz3, "cos_u_t1": cu, "cos_w_euc_t1": cw,
            }
            print(f"  => {arm_id:24s}: d_mac5={d_mac5:+6.2f} d_clz3={d_clz3:+6.2f} | uT1={cu:+.3f} wT1={cw:+.3f} | dNLL={unlab['delta_nll']:+.3f} eos={unlab['eos_abs_shift']:.4f}", flush=True)
            save_json(out_dir / "T2B_PROSE_CONTRAST.json", results)
    print("EXP_T2B_COMPLETE", flush=True)


def sigma_svd1(M, S_half, S_half_inv):
    """Exact rank-1 SVD of operator M in the H_{Sigma_h} inner product <A,B>_{Sigma_h} = tr(A Sigma_h B^T).
    Wait: M is the gradient matrix G = sum g h^T so <A, G>_F = tr(A^T G).
    The steepest-ascent rank-1 operator maximizing <A, G>_F subject to ||A||_{Sigma_h} = ||u|| * ||Sigma_h^{1/2} w|| = 1
    satisfies u, v = top singular vectors of (G @ Sigma_h^{-1/2}), with w = Sigma_h^{-1/2} v!
    And the Frobenius-norm-on-states natural metric inner product between two operators A1=u1 w1^T, A2=u2 w2^T
    is (u1^T u2) * (w1^T Sigma_h w2).
    """
    # Compute standard Euclidean top-1 (u_e, s_e, v_e) AND natural preconditioned (u_s, s_s, w_s)
    U, s, Vt = torch.linalg.svd(M.double(), full_matrices=False)
    u_e, w_e, s_e = U[:, 0], Vt[0], float(s[0])
    # Natural preconditioned operator: M_nat = M @ S_half_inv -> max <u w^T, M> s.t. ||u||=1, w^T S w = 1
    M_nat = M.double() @ S_half_inv
    Un, sn, Vnt = torch.linalg.svd(M_nat, full_matrices=False)
    u_n = Un[:, 0]
    w_n = S_half_inv @ Vnt[0]
    w_n = w_n / torch.clamp(w_n @ (S_half @ (S_half @ w_n)), min=1e-12).sqrt()
    return (u_e, w_e, s_e), (u_n, w_n, float(sn[0]))


def op_cos_sigma(u1, w1, u2, w2, S):
    cu = float((u1 @ u2) / (u1.norm() * u2.norm()).clamp_min(1e-12))
    n1 = float(torch.clamp(w1 @ (S @ w1), min=1e-12).sqrt())
    n2 = float(torch.clamp(w2 @ (S @ w2), min=1e-12).sqrt())
    cw = float((w1 @ (S @ w2)) / (n1 * n2))
    return cu * cw, cu, cw


def run_cfit_14_source_features(cfg, archive, model, tokenizer):
    print("\n================ EXP T4-CFIT: 14-SOURCE EXACT CFIT FEATURES ================", flush=True)
    out_dir = OUT_ROOT / "cfit_14_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    sigma_path = OUT_ROOT / "SIGMA_H.pt"
    if sigma_path.exists():
        sd = torch.load(sigma_path, map_location="cpu", weights_only=False)
        S = sd["sigma_unlab"].double()
    else:
        S, mu, n_tok = compute_sigma_h_unlabeled(model, tokenizer, UNLABELED_AUDIT, max_docs=256, depth=4)
        torch.save({"sigma_unlab": S.float(), "mu_unlab": mu.float(), "n_tokens": n_tok}, sigma_path)
    # Regularized S^{1/2} and S^{-1/2} for stable natural-metric SVD
    evals, evecs = torch.linalg.eigh((S + S.T) * 0.5)
    reg = float(evals.mean()) * 0.05
    evals_reg = evals.clamp_min(0.0) + reg
    S_reg = (evecs * evals_reg) @ evecs.T
    S_half = (evecs * evals_reg.sqrt()) @ evecs.T
    S_half_inv = (evecs * evals_reg.rsqrt()) @ evecs.T

    f_inputs = json.loads((V3_ROOT / "stage4_d3" / "FORECAST_INPUTS.json").read_text(encoding="utf-8"))
    f_v3 = json.loads((V3_ROOT / "stage4_d3" / "FORECAST_LOCK_V3.json").read_text(encoding="utf-8"))

    # Locate all 14 source jsonl files
    source_paths = {}
    for s in sorted(f_inputs.keys()):
        p1 = Path("/root/ouro-loop-aware-functional-identification-20261125/data/pools_complete") / f"{s}.jsonl"
        p2 = V3_ROOT / "stage0_env" / "PROSPECTIVE_TRAIN_DRAFT" / f"{s}.jsonl"
        source_paths[s] = p1 if p1.exists() else p2
        assert source_paths[s].exists(), f"Missing pool for {s}"

    # 1. Moment-tensor deconfounding & Sigma_h geometry from the 56 already-computed COLLECT_RESUME.pt shards!
    print("Loading all 56 COLLECT_RESUME.pt moment shards...", flush=True)
    source_moments = {}
    for s, info in sorted(f_inputs.items()):
        fold_ord, fold_ctx, fold_lp = [], [], []
        tot_ord = tot_ctx = tot_lp = None
        tot_n = 0
        for sh in info["shards"]:
            ck = torch.load(sh, map_location="cpu", weights_only=False)
            n_f = int(ck["processed"])
            M_ord = ck["sums"]["ordinary"].double() / max(n_f, 1)
            M_ctx = ck["sums"]["true_receiver"].double() / max(n_f, 1)
            M_lp = ck["sums"]["loop_contrast"].double() / max(n_f, 1)
            fold_ord.append(M_ord)
            fold_ctx.append(M_ctx)
            fold_lp.append(M_lp)
            tot_ord = ck["sums"]["ordinary"].double() if tot_ord is None else tot_ord + ck["sums"]["ordinary"].double()
            tot_ctx = ck["sums"]["true_receiver"].double() if tot_ctx is None else tot_ctx + ck["sums"]["true_receiver"].double()
            tot_lp = ck["sums"]["loop_contrast"].double() if tot_lp is None else tot_lp + ck["sums"]["loop_contrast"].double()
            tot_n += n_f
        M_ord = tot_ord / tot_n
        M_ctx = tot_ctx / tot_n
        M_lp = tot_lp / tot_n
        M_cf = M_ord - M_ctx  # Note: ordinary is unweighted sum g0 h0^T while true_receiver is e_gap-weighted sum(g0 h0^T - g1 h1^T)
        # Deconfounding purity: alignment between ordinary training SVD1 and contextual true_receiver matrix!
        (u_ord, w_ord, s_ord), (u_ord_s, w_ord_s, s_ord_s) = sigma_svd1(M_ord, S_half, S_half_inv)
        (u_ctx, w_ctx, s_ctx), (u_ctx_s, w_ctx_s, s_ctx_s) = sigma_svd1(M_ctx, S_half, S_half_inv)

        # Normalize operator alignment <A_ord, M_ctx>_F / (||A_ord||_F * ||M_ctx||_2) and cosine between A_ord and A_ctx!
        op_cos_ctx_euc = float((u_ord @ u_ctx) * (w_ord @ w_ctx))
        op_cos_ctx_sig, cu_ctx, cw_ctx_sig = op_cos_sigma(u_ord, w_ord, u_ctx, w_ctx, S_reg)
        # Projected fraction of M_ctx along A_ord
        proj_ord_on_ctx = float((u_ord @ M_ctx @ w_ord) / max(s_ctx, 1e-12))
        proj_ctx_on_ord = float((u_ctx @ M_ord @ w_ctx) / max(s_ord, 1e-12))
        # Relative scale: s_ctx / s_ord (how big is deconfounded contextual signal vs raw ordinary?)
        ctx_to_ord_ratio = float(s_ctx / max(s_ord, 1e-12))

        # 4-fold stability of DECONFOUNDED contextual matrix M_ctx in Euclidean AND Sigma_h metric!
        f_axes_ctx = [sigma_svd1(Fm, S_half, S_half_inv)[0] for Fm in fold_ctx]
        f_axes_ord = [sigma_svd1(Fm, S_half, S_half_inv)[0] for Fm in fold_ord]
        c_ctx_euc = [float((u1 @ u2) * (w1 @ w2)) for (u1, w1, _), (u2, w2, _) in combinations(f_axes_ctx, 2)]
        c_ctx_sig = [op_cos_sigma(u1, w1, u2, w2, S_reg)[0] for (u1, w1, _), (u2, w2, _) in combinations(f_axes_ctx, 2)]
        c_ord_sig = [op_cos_sigma(u1, w1, u2, w2, S_reg)[0] for (u1, w1, _), (u2, w2, _) in combinations(f_axes_ord, 2)]

        # On-Manifold Preservation M_pres(S): evaluate finite-dose Rank-1 controller along A_ord AND A_ctx on audit_unlabeled!
        def make_probe_net(u_vec, w_vec, norm_prod=2.5):
            net = minimal.Minimal("linear", 1, 0.45).to(model.device)
            with torch.no_grad():
                un = u_vec / u_vec.norm().clamp_min(1e-12) * math.sqrt(norm_prod)
                wn = w_vec / w_vec.norm().clamp_min(1e-12) * math.sqrt(norm_prod)
                net.out.weight.copy_(un[:, None].to(net.out.weight.dtype).to(model.device))
                net.encode.weight.copy_(wn[None, :].to(net.encode.weight.dtype).to(model.device))
            return net.eval()

        net_ord = make_probe_net(u_ord, w_ord, norm_prod=2.5)
        net_ctx = make_probe_net(u_ctx, w_ctx, norm_prod=2.5)
        man_ord = evaluate_unlabeled_manifold(model, tokenizer, net_ord, UNLABELED_AUDIT, max_docs=64)
        man_ctx = evaluate_unlabeled_manifold(model, tokenizer, net_ctx, UNLABELED_AUDIT, max_docs=64)
        m_pres_ord = math.exp(-max(0.0, man_ord["delta_nll"]) - 5.0 * man_ord["eos_abs_shift"])
        m_pres_ctx = math.exp(-max(0.0, man_ctx["delta_nll"]) - 5.0 * man_ctx["eos_abs_shift"])

        source_moments[s] = {
            "M_ord": M_ord, "M_ctx": M_ctx, "u_ord": u_ord, "w_ord": w_ord, "s_ord": s_ord,
            "u_ctx": u_ctx, "w_ctx": w_ctx, "s_ctx": s_ctx,
            "stats": {
                "s_ord": s_ord, "s_ctx": s_ctx, "ctx_to_ord_ratio": ctx_to_ord_ratio,
                "P_deconf_op_euc": op_cos_ctx_euc, "P_deconf_op_sig": op_cos_ctx_sig,
                "P_deconf_proj_ord_on_ctx": proj_ord_on_ctx, "P_deconf_proj_ctx_on_ord": proj_ctx_on_ord,
                "C_ctx_euc": sum(c_ctx_euc) / len(c_ctx_euc),
                "C_ctx_sig": sum(c_ctx_sig) / len(c_ctx_sig),
                "C_ord_sig": sum(c_ord_sig) / len(c_ord_sig),
                "man_ord": man_ord, "man_ctx": man_ctx,
                "M_pres_ord": m_pres_ord, "M_pres_ctx": m_pres_ctx,
            },
        }
        print(f"  [Moments {s:13s}] P_deconf_sig={op_cos_ctx_sig:+.3f} C_ctx_sig={sum(c_ctx_sig)/6:+.3f} dNLL_ord={man_ord['delta_nll']:+.3f} eos_ord={man_ord['eos_abs_shift']:.4f} M_pres={m_pres_ord:.3f}", flush=True)

    # 2. Cross-source Sigma_h compatibility matrix & Leave-One-Out Consensus!
    srcs = sorted(source_moments.keys())
    compat_ctx_sig = {s: {} for s in srcs}
    compat_ord_sig = {s: {} for s in srcs}
    for s1 in srcs:
        for s2 in srcs:
            if s1 == s2:
                continue
            d1, d2 = source_moments[s1], source_moments[s2]
            c_ctx, _, _ = op_cos_sigma(d1["u_ctx"], d1["w_ctx"], d2["u_ctx"], d2["w_ctx"], S_reg)
            c_ord, _, _ = op_cos_sigma(d1["u_ord"], d1["w_ord"], d2["u_ord"], d2["w_ord"], S_reg)
            compat_ctx_sig[s1][s2] = c_ctx
            compat_ord_sig[s1][s2] = c_ord
        # Median leave-one-out in Sigma_h metric
        vals_ctx = sorted(compat_ctx_sig[s1].values())
        vals_ord = sorted(compat_ord_sig[s1].values())
        med_ctx = vals_ctx[len(vals_ctx) // 2]
        med_ord = vals_ord[len(vals_ord) // 2]
        source_moments[s1]["stats"]["A_cons_ctx_sig"] = med_ctx
        source_moments[s1]["stats"]["A_cons_ord_sig"] = med_ord
        # Also projection of A_ord onto other sources' M_ctx normalized by s_ctx
        proj_others_ctx = sorted(float((source_moments[s1]["u_ord"] @ source_moments[s2]["M_ctx"] @ source_moments[s1]["w_ord"]) / max(source_moments[s2]["s_ctx"], 1e-12)) for s2 in srcs if s2 != s1)
        source_moments[s1]["stats"]["A_cons_proj_ctx"] = proj_others_ctx[len(proj_others_ctx) // 2]

    # 3. Regime-III Observability O_III (p_0^MC vs p_0^cloze + Native Loop Emergence K4 vs K1) across ALL 14 sources!
    print("\nComputing exact Regime-III Observability O_III across all 14 sources...", flush=True)
    per_sample_dir = out_dir / "regime3_samples"
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    # Save larger per-sample banks (2048) for T3-B reversal sources, 512 for others
    reversal_targets = {"medmcqa", "qasc", "obqa", "race", "arc_challenge"}

    for s in srcs:
        grp = f_v3["per_source"][s]["group"]
        raw = load_jsonl(source_paths[s])
        # Deterministic hash order matching v3
        raw_sorted = sorted(raw, key=lambda r: hashlib.sha256(f"26112641:{r['sample_id']}".encode()).hexdigest())
        lim = 2048 if s in reversal_targets else 512
        sub = raw_sorted[:min(lim, len(raw_sorted))]

        if grp == "candidate":
            items = []
            for r in sub:
                q, opts, g = extract_q_opts_gold(r)
                if 2 <= len(opts) <= 10:
                    items.append((r["sample_id"], q, opts, g))
            crows = [make_candidate_row(sid, q, opts, g) for sid, q, opts, g in items]
            # Native Cloze K4 and K2 (loop emergence!)
            _, clz_p4 = score_cloze_probs(archive, model, tokenizer, crows, depth=4, net=None, batch_size=6)
            _, clz_p2 = score_cloze_probs(archive, model, tokenizer, crows, depth=2, net=None, batch_size=6)
            # Rotation-averaged MC letter K4
            mc_p4 = score_mc_letter_rotation_probs(model, tokenizer, [(q, opts) for _, q, opts, _ in items], depth=4, batch_size=8)

            records = []
            for (sid, _, opts, g), p4, p2, pm in zip(items, clz_p4, clz_p2, mc_p4):
                m = len(opts)
                egap = 1.0 - p4[g]
                p_lat_mc = max(0.0, (pm[g] - 1.0 / m) / (1.0 - 1.0 / m))
                w_iii_mc = egap * p_lat_mc
                # Loop emergence probability gain: p4[g] - p2[g]
                loop_gain = p4[g] - p2[g]
                w_iii_loop = egap * max(0.0, loop_gain)
                # Regime II indicator: high e_gap AND low latent MC knowledge
                w_ii = egap * (1.0 - p_lat_mc)
                records.append({
                    "sample_id": sid, "m": m, "gold": g,
                    "p_cloze_k4": p4[g], "p_cloze_k2": p2[g], "p_mc_rot": pm[g],
                    "e_gap": egap, "p_lat_mc": p_lat_mc, "w_iii_mc": w_iii_mc,
                    "loop_gain": loop_gain, "w_iii_loop": w_iii_loop, "w_ii": w_ii,
                    "clz_correct": int(torch.tensor(p4).argmax()) == g,
                    "mc_correct": int(torch.tensor(pm).argmax()) == g,
                })
            # Save per-sample records for T3-B!
            (per_sample_dir / f"{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            n = len(records)
            o_iii = sum(r["w_iii_mc"] for r in records) / n
            o_ii = sum(r["w_ii"] for r in records) / n
            o_loop = sum(r["w_iii_loop"] for r in records) / n
            clz_acc = sum(r["clz_correct"] for r in records) / n
            mc_acc = sum(r["mc_correct"] for r in records) / n
            # Pure Regime-III fraction: MC correct & Cloze wrong!
            r3_frac = sum(1 for r in records if r["mc_correct"] and not r["clz_correct"]) / n
            r2_frac = sum(1 for r in records if (not r["mc_correct"]) and (not r["clz_correct"])) / n
            r1_frac = sum(1 for r in records if r["clz_correct"]) / n
            source_moments[s]["stats"].update({
                "O_III_mc": o_iii, "O_II_mc": o_ii, "O_III_over_II": o_iii / max(o_ii, 1e-6),
                "O_III_loop": o_loop, "native_clz_acc": clz_acc, "native_mc_acc": mc_acc,
                "readout_gap_pp": (mc_acc - clz_acc) * 100,
                "regime3_frac": r3_frac, "regime2_frac": r2_frac, "regime1_frac": r1_frac,
                "n_regime_probed": n,
            })
            print(f"  [Regime3 {s:13s}] n={n:4d} clz={clz_acc*100:5.1f}% mc={mc_acc*100:5.1f}% gap={(mc_acc-clz_acc)*100:+5.1f}pp | O_III={o_iii:.3f} O_II={o_ii:.3f} ratio={o_iii/max(o_ii,1e-6):.2f} R3={r3_frac*100:4.1f}% R2={r2_frac*100:4.1f}%", flush=True)
        else:
            # Generation sources (squad, gsm8k): no MC-letter gap, record O_III_mc = 0.0 and measure loop emergence on response NLL
            source_moments[s]["stats"].update({
                "O_III_mc": 0.0, "O_II_mc": 1.0, "O_III_over_II": 0.0,
                "O_III_loop": 0.0, "native_clz_acc": None, "native_mc_acc": None,
                "readout_gap_pp": 0.0, "regime3_frac": 0.0, "regime2_frac": 0.0, "regime1_frac": 0.0,
                "n_regime_probed": 0,
            })
            print(f"  [Regime3 {s:13s}] generation source (non-contrastive NLL protocol)", flush=True)

    # Assemble clean CFIT intrinsic single-source score I_CFIT(S) and save full summary
    out_summary = {"per_source": {}, "compat_ctx_sig": compat_ctx_sig, "compat_ord_sig": compat_ord_sig}
    for s in srcs:
        st = source_moments[s]["stats"]
        # Purely intrinsic zero-reference CFIT score (ZERO other datasets, ZERO T1!)
        p_dec = max(0.0, st["P_deconf_op_sig"])
        c_ctx = max(0.0, st["C_ctx_sig"])
        i_cfit = st["O_III_mc"] * p_dec * c_ctx * st["M_pres_ord"]
        i_cfit_ratio = st["O_III_over_II"] * p_dec * st["M_pres_ord"]
        st["I_CFIT_intrinsic"] = i_cfit
        st["I_CFIT_ratio"] = i_cfit_ratio
        st["A_cons_cfit"] = st["A_cons_ord_sig"] * st["M_pres_ord"]
        out_summary["per_source"][s] = st

    # Save operators (u_ord, w_ord, u_ctx, w_ctx) for downstream self-anchor alignment & Theorem 2 mixing!
    ops_save = {
        s: {
            "u_ord": d["u_ord"].float(), "w_ord": d["w_ord"].float(), "s_ord": d["s_ord"],
            "u_ctx": d["u_ctx"].float(), "w_ctx": d["w_ctx"].float(), "s_ctx": d["s_ctx"],
        }
        for s, d in source_moments.items()
    }
    torch.save(ops_save, out_dir / "SOURCE_OPERATORS_14.pt")
    save_json(out_dir / "CFIT_14_SOURCE_FEATURES.json", out_summary)
    print("EXP_T4_CFIT_14_COMPLETE", flush=True)


def main():
    torch.set_num_threads(4)
    cfg, archive, model, tokenizer = load_base_runtime()
    evaluator = FastUnifiedDevEvaluator(cfg, archive, model, tokenizer)
    run_exp_t2b(cfg, archive, model, tokenizer, evaluator)
    run_cfit_14_source_features(cfg, archive, model, tokenizer)
    print("ALL_CQA1_GPU1_WAVE1_DONE", flush=True)


if __name__ == "__main__":
    main()
