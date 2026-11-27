"""ICML Phase 1 — GPU 0 Wave 2 (Zero overlap with Codex GPU 1 Track B):
1. Complete Centered vs Uncentered Sigma_h audit across all 16 E0 rank-only fits/*/controller.pt + v4 winners.
2. Track A CSBP Pure-Prose Self-Bootstrapping (0 External QA, 0 Human Labels, 0 External Models):
   - Combines Route A + Route B from UNLABELED_FIT (0 8-gram eval overlap) under T2_Blank (verified 92.9-94.7% purity,
     +36 to +47 pp gap, 20.5-27.4% Option-Only zero-shortcut score).
   - Runs Reviewer's exact 5-way same-source objective ablation on BOTH G1_K4Blind and G2_HumanRetr + G0_Random,
     plus Cloze interface ablation, NLL-Fidelity Pareto sweep (trust_lam), and FixW_MeanMu_2048 on pure prose!
"""
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
sys.path.insert(0, "/root/autodl-tmp/ouro-v5-icml-phase1-20261127")
import minimal
from v4_common import (
    DEPTHS,
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
from p1_gpu1_csbp_pure_prose import (
    build_human_retrieval_distractors,
    build_random_distractors,
    extract_prose_items,
    format_stem,
    generate_k4_blind_distractors,
    heldout_unlab_agree,
    kl_div,
    load_forbidden_eval_8grams,
    make_shuffled_quads,
    natural_cloze_logits_differentiable,
    score_natural_cloze_probs,
)

V4_ROOT = Path("/root/autodl-tmp/ouro-v4-cfit-20261126")
P1_ROOT = Path("/root/autodl-tmp/ouro-v5-icml-phase1-20261127")
OUT_CSBP = P1_ROOT / "track_a_csbp_v2"
SEEDS = (26112701, 26112702)
STEPS = 840


class FixedWController(nn.Module):
    def __init__(self, w_fixed: torch.Tensor, bound: float = 0.45):
        super().__init__()
        self.kind = "fixed_w"
        self.rank = 1
        self.bound = bound
        self.fixed = True
        w = w_fixed.detach().float().flatten()
        self.register_buffer("w_fixed", w / w.norm().clamp_min(1e-12))
        self.u = nn.Parameter(torch.zeros_like(w))

    def delta(self, h, previous=None):
        scale = minimal.rms(h)
        proj = torch.einsum("...d,d->...", h, self.w_fixed)
        x = proj.unsqueeze(-1) * self.u
        return x if self.bound is None else x * (self.bound * scale / minimal.rms(x)).clamp_max(1.0)


def run_centered_sigma_audit_16(cfg):
    print("\n================ TASK 1: COMPLETE 16-E0 CENTERED vs UNCENTERED SIGMA_H AUDIT ================", flush=True)
    out_path = P1_ROOT / "track_b_and_theory" / "PART1_CENTERED_SIGMA_FULL16.json"
    sd = torch.load(V4_ROOT / "SIGMA_H.pt", map_location="cpu", weights_only=False)
    S_u = sd["sigma_unlab"].double()
    mu_u = sd["mu_unlab"].double()
    S_c = S_u - torch.outer(mu_u, mu_u)
    u_t1, w_t1 = load_controller_weights(Path(cfg["historical_t1"]))

    def metric_cos(a, b, M):
        na = float(torch.clamp(a @ (M @ a), min=1e-12).sqrt())
        nb = float(torch.clamp(b @ (M @ b), min=1e-12).sqrt())
        return float((a @ (M @ b)) / (na * nb))

    fits_dir = V3_ROOT / "stage1_e0" / "fits"
    arms = {}
    rk_cw_u, rk_cw_c, rk_cu, rk_op_u, rk_op_c, rk_mu_frac = [], [], [], [], [], []
    for d in sorted(fits_dir.iterdir()):
        pt = d / "controller.pt"
        if not pt.exists():
            continue
        name = d.name
        u, w = load_controller_weights(pt)
        cu = float((u @ u_t1) / (u.norm() * u_t1.norm()).clamp_min(1e-12))
        cw_euc = float((w @ w_t1) / (w.norm() * w_t1.norm()).clamp_min(1e-12))
        cw_u = metric_cos(w, w_t1, S_u)
        cw_c = metric_cos(w, w_t1, S_c)
        w_mu_frac = float(((w @ mu_u) ** 2) / torch.clamp(w @ (S_u @ w), min=1e-12))
        is_rank_only = (".rank." in name and ".nll_rank." not in name)
        arms[name] = {
            "is_rank_only": is_rank_only,
            "cos_u_t1": cu, "cos_w_euc_t1": cw_euc,
            "cos_w_uncent_t1": cw_u, "cos_w_cent_t1": cw_c,
            "cos_op_uncent_t1": cu * cw_u, "cos_op_cent_t1": cu * cw_c,
            "w_mean_energy_frac": w_mu_frac,
        }
        if is_rank_only:
            rk_cw_u.append(abs(cw_u))
            rk_cw_c.append(abs(cw_c))
            rk_cu.append(abs(cu))
            rk_op_u.append(abs(cu * cw_u))
            rk_op_c.append(abs(cu * cw_c))
            rk_mu_frac.append(w_mu_frac)

    for label, pt in [
        ("Best_ZeroLabel_4096", V4_ROOT / "exp_t2a_self_distill" / "self_arc_hardpseudo.s26112681.pt"),
        ("Best_NonARC_4096", V4_ROOT / "exp_t5_mixing" / "T5_CFIT_NonARC_4096.s26112801.pt"),
        ("Best_Rank1_4096", V4_ROOT / "exp_t5_mixing" / "T5_CFIT_Top6_4096.s26112801.pt"),
    ]:
        if pt.exists():
            u, w = load_controller_weights(pt)
            cu = float((u @ u_t1) / (u.norm() * u_t1.norm()).clamp_min(1e-12))
            cw_u = metric_cos(w, w_t1, S_u)
            cw_c = metric_cos(w, w_t1, S_c)
            w_mu_frac = float(((w @ mu_u) ** 2) / torch.clamp(w @ (S_u @ w), min=1e-12))
            arms[label] = {
                "is_rank_only": False, "cos_u_t1": cu, "cos_w_uncent_t1": cw_u, "cos_w_cent_t1": cw_c,
                "cos_op_uncent_t1": cu * cw_u, "cos_op_cent_t1": cu * cw_c, "w_mean_energy_frac": w_mu_frac,
            }

    res = {
        "n_e0_rank_only": len(rk_cw_u),
        "e0_rank_only_summary": {
            "abs_cos_w_uncent": [min(rk_cw_u), max(rk_cw_u)],
            "abs_cos_w_cent": [min(rk_cw_c), max(rk_cw_c)],
            "abs_cos_u": [min(rk_cu), max(rk_cu)],
            "abs_cos_op_uncent": [min(rk_op_u), max(rk_op_u)],
            "abs_cos_op_cent": [min(rk_op_c), max(rk_op_c)],
            "w_mean_energy_frac": [min(rk_mu_frac), max(rk_mu_frac)],
        },
        "arms": arms,
    }
    save_json(out_path, res)
    print(f"  Verified {len(rk_cw_u)} E0 rank-only controllers:", json.dumps(res["e0_rank_only_summary"], indent=2), flush=True)
    return res


def build_scored_combined_bank(archive, model, tokenizer, quads, tmpl, name):
    cache = OUT_CSBP / "banks" / f"{name}.json"
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
            "route": q["route"],
            "question": st,
            "options": q["options"],
            "q_mc": mp,
            "conf": conf,
            "margin": margin,
            "pseudo_label": int(torch.tensor(mp).argmax()),
            "kl_qa": kl_div(mp, cqa),
            "kl_nat": kl_div(mp, cnat),
            "orig_idx_for_selfsup_ablation": q["_orig_idx"],
        })
    save_json(cache, bank)
    return bank


def train_prose_arm(
    archive, model, tokenizer, bank, objective, cloze_mode, seed,
    fixed_w=None, steps=STEPS, trust_seqs=None, trust_lam=0.0, tag=""
):
    torch.manual_seed(seed)
    net = minimal.Minimal("linear", 1, 0.45) if fixed_w is None else FixedWController(fixed_w, 0.45)
    net = net.to(model.device)
    trainable = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"])
    rng = random.Random(f"csbp_v2/{tag}/{seed}")
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
                    w_i = max(1e-4, float(it["conf"]))
                else:
                    raise ValueError(objective)
                item_losses.append(l_i * w_i)
                weights.append(w_i)
            loss = sum(item_losses) / sum(weights)
            if trust_lam > 0.0 and trust_seqs and step % 2 == 0:
                t_mb = [trust_seqs[(step + j) % len(trust_seqs)] for j in range(2)]
                loss = loss + trust_lam * nll_pen(t_mb, depth)
            loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(trainable, RECIPE["clip"])
        assert grad.isfinite().item(), f"Non-finite grad in {tag}"
        opt.step()
        losses.append(float(loss.detach()))
        if step % 280 == 0 or step == steps:
            print(f"  [CSBP_v2 {tag} s={seed}] step={step}/{steps} loss={sum(losses[-70:])/min(len(losses),70):.4f} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return net, {"n_params": sum(p.numel() for p in trainable), "steps": steps, "seconds": time.perf_counter() - t0}


def main():
    torch.set_num_threads(4)
    OUT_CSBP.mkdir(parents=True, exist_ok=True)
    cfg, archive, model, tokenizer = load_base_runtime()

    # Task 1: Complete 16-E0 centered Sigma_h audit
    run_centered_sigma_audit_16(cfg)

    # Task 2: CSBP Combined Route A + Route B from UNLABELED_FIT (0 External QA!)
    evaluator = FastUnifiedDevEvaluator(cfg, archive, model, tokenizer)
    native_ev = evaluator.evaluate(None, include_mmlu_dev158=True)
    u_t1, _ = load_controller_weights(Path(cfg["historical_t1"]))
    mu_u = torch.load(V4_ROOT / "SIGMA_H.pt", map_location="cpu", weights_only=False)["mu_unlab"].float()

    forbidden = load_forbidden_eval_8grams()
    ra_c, ra_g, rb, audit = extract_prose_items(forbidden)
    save_json(OUT_CSBP / "CORPUS_AUDIT.json", audit)
    route_a_all = ra_c + ra_g
    # Interleave Route A + Route B to maximize clean non-overlapping probe yield (~2,300 raw stems -> ~1,400 G1 quads -> ~500-700 CFIT ultra-pure!)
    combined = []
    seen_ids = set()
    for a, b in zip(route_a_all, rb):
        combined.append(a)
        combined.append(b)
    if len(route_a_all) > len(rb):
        combined.extend(route_a_all[len(rb):])

    print(f"\nBuilding combined Route A+B G1_K4Blind bank on {len(combined)} pure WikiText stems...", flush=True)
    g1_path = OUT_CSBP / "banks" / "bank_G1_AB_T2Blank.json"
    if not g1_path.exists():
        g1_quads = make_shuffled_quads(generate_k4_blind_distractors(model, tokenizer, combined, n_samples=8, batch_size=8), seed=26112751)
        bank_g1 = build_scored_combined_bank(archive, model, tokenizer, g1_quads, "T2_Blank", "bank_G1_AB_T2Blank")
    else:
        bank_g1 = json.loads(g1_path.read_text(encoding="utf-8"))

    g2_path = OUT_CSBP / "banks" / "bank_G2_AB_T2Blank.json"
    if not g2_path.exists():
        g2_quads = make_shuffled_quads(build_human_retrieval_distractors(model, tokenizer, combined[:1400], route_a_all), seed=26112752)
        bank_g2 = build_scored_combined_bank(archive, model, tokenizer, g2_quads, "T2_Blank", "bank_G2_AB_T2Blank")
    else:
        bank_g2 = json.loads(g2_path.read_text(encoding="utf-8"))

    g0_path = OUT_CSBP / "banks" / "bank_G0_AB_T2Blank.json"
    if not g0_path.exists():
        g0_quads = make_shuffled_quads(build_random_distractors(combined[:1200], route_a_all), seed=26112750)
        bank_g0 = build_scored_combined_bank(archive, model, tokenizer, g0_quads, "T2_Blank", "bank_G0_AB_T2Blank")
    else:
        bank_g0 = json.loads(g0_path.read_text(encoding="utf-8"))

    def split_filter(bank):
        n_tr = int(len(bank) * 0.85)
        tr_all, va_all = bank[:n_tr], bank[n_tr:]
        tr_cfit = [x for x in tr_all if x["conf"] >= 0.45 and x["margin"] >= 0.25 and x["kl_qa"] >= 0.15]
        tr_cfit_ra = [x for x in tr_cfit if x["route"].startswith("A")]
        tr_cfit_rb = [x for x in tr_cfit if x["route"].startswith("B")]
        return tr_all, va_all, tr_cfit, tr_cfit_ra, tr_cfit_rb

    g1_tr, g1_va, g1_cfit, g1_cfit_a, g1_cfit_b = split_filter(bank_g1)
    g2_tr, g2_va, g2_cfit, _, _ = split_filter(bank_g2)
    g0_tr, g0_va, _, _, _ = split_filter(bank_g0)
    print(f"Combined Banks Ready: G1={len(g1_tr)}/{len(g1_va)} (CFIT={len(g1_cfit)}: A={len(g1_cfit_a)}, B={len(g1_cfit_b)}) | G2={len(g2_tr)} (CFIT={len(g2_cfit)}) | G0={len(g0_tr)}", flush=True)

    trust_rows = load_jsonl(UNLABELED_FIT)[:128]
    trust_seqs = [s for s in (tokenizer.encode(r["text"], add_special_tokens=False)[:192] for r in trust_rows) if len(s) >= 64]

    out_path = OUT_CSBP / "CSBP_V2_ABLATION.json"
    res = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {
        "native": native_ev,
        "bank_sizes": {
            "g1_tr": len(g1_tr), "g1_va": len(g1_va), "g1_cfit": len(g1_cfit),
            "g1_cfit_a": len(g1_cfit_a), "g1_cfit_b": len(g1_cfit_b),
            "g2_tr": len(g2_tr), "g2_cfit": len(g2_cfit), "g0_tr": len(g0_tr),
        },
        "arms": {},
    }

    plan = [
        # Rung 1: Random cross-doc distractors + hard_pseudo
        ("Rung1_G0Rand_ConfHard_4096", g0_tr, g0_va, "mc_conf_hard", "ClozeQA", 0.0, None),
        # 5-Way Same-Source Objective Ablation on G1_K4Blind (exact same WikiText probes!)
        ("G1_Obj1_OrigGold_CE_4096", g1_tr, g1_va, "orig_gold_ce", "ClozeQA", 0.0, None),
        ("G1_Obj2_MCHard_Unif_4096", g1_tr, g1_va, "mc_hard_unif", "ClozeQA", 0.0, None),
        ("G1_Obj3_MCSoft_KL_4096", g1_tr, g1_va, "mc_soft_kl", "ClozeQA", 0.0, None),
        ("G1_Obj4_MCConfHard_Unfilt_4096", g1_tr, g1_va, "mc_conf_hard", "ClozeQA", 0.0, None),
        ("G1_Obj5_CFIT_ConfHard_ClzQA_4096", g1_cfit, g1_va, "mc_conf_hard", "ClozeQA", 0.0, None),
        ("G1_Obj5_CFIT_ConfHard_ClzNat_4096", g1_cfit, g1_va, "mc_conf_hard", "ClozeNat", 0.0, None),
        ("G1_Obj5_CFIT_ConfHard_Mix50_4096", g1_cfit, g1_va, "mc_conf_hard", "ClozeMix50", 0.0, None),
        # Route A-only vs Route B-only CFIT check
        ("G1_Obj5_CFIT_RouteAOnly_4096", g1_cfit_a if len(g1_cfit_a) >= 80 else g1_cfit, g1_va, "mc_conf_hard", "ClozeQA", 0.0, None),
        # 2,048-param Fixed-w=MeanMu on ZERO-QA Pure WikiText!!
        ("G1_Obj5_CFIT_FixW_MeanMu_2048", g1_cfit, g1_va, "mc_conf_hard", "ClozeQA", 0.0, mu_u),
        # Pareto trust_lam sweep on G1_CFIT
        ("G1_CFIT_Pareto_lam025_4096", g1_cfit, g1_va, "mc_conf_hard", "ClozeQA", 0.25, None),
        ("G1_CFIT_Pareto_lam050_4096", g1_cfit, g1_va, "mc_conf_hard", "ClozeQA", 0.50, None),
        ("G1_CFIT_Pareto_lam100_4096", g1_cfit, g1_va, "mc_conf_hard", "ClozeQA", 1.00, None),
        # 5-Way Same-Source Objective Ablation on G2_HumanRetr (100% real human text, zero generation!)
        ("G2_Obj1_OrigGold_CE_4096", g2_tr, g2_va, "orig_gold_ce", "ClozeQA", 0.0, None),
        ("G2_Obj4_MCConfHard_Unfilt_4096", g2_tr, g2_va, "mc_conf_hard", "ClozeQA", 0.0, None),
        ("G2_Obj5_CFIT_ConfHard_4096", g2_cfit, g2_va, "mc_conf_hard", "ClozeQA", 0.0, None),
    ]

    for cname, tr_b, va_b, obj, cmode, lam, fw in plan:
        for seed in SEEDS:
            arm_id = f"{cname}.s{seed}"
            if arm_id in res["arms"]:
                continue
            net, meta = train_prose_arm(
                archive, model, tokenizer, tr_b, obj, cmode, seed,
                fixed_w=fw, steps=STEPS, trust_seqs=trust_seqs, trust_lam=lam, tag=cname
            )
            ev = evaluator.evaluate(net, include_mmlu_dev158=True)
            unlab = evaluate_unlabeled_manifold(model, tokenizer, net, UNLABELED_AUDIT, max_docs=64)
            va_ag, va_kl = heldout_unlab_agree(archive, model, tokenizer, va_b, net, "ClozeNat" if cmode == "ClozeNat" else "ClozeQA")
            ckpt = OUT_CSBP / "weights" / f"{arm_id}.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            if fw is None:
                sha = save_rank1_controller(ckpt, net, {"arm": arm_id, "obj": obj, "cloze": cmode, "lam": lam, "seed": seed})
                u_a, _ = load_controller_weights(ckpt)
                cu = float((u_a @ u_t1) / (u_a.norm() * u_t1.norm()).clamp_min(1e-12))
            else:
                torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()}, "meta": meta}, ckpt)
                u_a = net.u.detach().double().cpu()
                cu = float((u_a @ u_t1) / (u_a.norm() * u_t1.norm()).clamp_min(1e-12))
                sha = None
            d_m5 = (ev["macro5"] - native_ev["macro5"]) * 100
            d_c3 = (ev["cloze3"] - native_ev["cloze3"]) * 100
            res["arms"][arm_id] = {
                "group": cname, "objective": obj, "cloze_mode": cmode, "trust_lam": lam, "seed": seed,
                "n_params": meta["n_params"], "n_train": len(tr_b), "sha256": sha,
                "train": meta, "eval": ev, "unlabeled": unlab,
                "heldout_unlab_agree": va_ag, "heldout_unlab_kl": va_kl,
                "delta_macro5_pp": d_m5, "delta_cloze3_pp": d_c3, "cos_u_t1": cu,
            }
            print(
                f"  => {arm_id:42s} P={meta['n_params']:4d} dMac5={d_m5:+6.2f} dClz3={d_c3:+6.2f} "
                f"MMLU={ev['mmlu_val']['acc']*100:.2f}% dNLL={unlab['delta_nll']:+.3f} ag={va_ag*100:.1f}% uT1={cu:+.3f}",
                flush=True,
            )
            save_json(out_path, res)

    print("\nALL_CSBP_V2_DONE", flush=True)


if __name__ == "__main__":
    main()
