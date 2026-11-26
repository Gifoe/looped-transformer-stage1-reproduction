"""Wave 1 on westc (RTX 4090 D):
1. Exp T1: Exact Sigma_h functional metric geometry audit across all 26 E0 controllers + anchors + U1_PMI.
2. Exp T3-A: Same-Question Protocol Inversion on ARC-Challenge (1,011 4-choice) AND MMLU-500 (mmlu_validation).
"""

import json
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
    V3_ROOT,
    FastUnifiedDevEvaluator,
    capture_boundary_states,
    cloze_option_logits_differentiable,
    compute_sigma_h_unlabeled,
    evaluate_unlabeled_manifold,
    extract_q_opts_gold,
    letter_option_logits_differentiable,
    load_base_runtime,
    load_controller_weights,
    load_minimal_controller,
    load_jsonl,
    make_candidate_row,
    save_json,
    save_rank1_controller,
)

OUT_ROOT = Path("/root/ouro-v4-cfit-20261126")


def cos_euc(a, b):
    na, nb = float(a.norm()), float(b.norm())
    return float((a @ b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def cos_sigma(a, b, S):
    sa = S @ a
    sb = S @ b
    na = float(torch.clamp(a @ sa, min=0.0).sqrt())
    nb = float(torch.clamp(b @ sb, min=0.0).sqrt())
    return float((a @ sb) / (na * nb)) if na > 0 and nb > 0 else 0.0


@torch.inference_mode()
def collect_cand_states(archive, model, tokenizer, rows, max_rows=200, depth=4):
    """Collect receiver boundary states h_t on candidate prediction tokens (matching controller hook)."""
    from advantages import choice_batch
    states = []
    for start in range(0, min(len(rows), max_rows), 8):
        chunk = rows[start:start + 8]
        batch = choice_batch(archive, tokenizer, chunk)
        ids, seq_idx, pos_idx, _ = batch.packed
        attn = (ids != (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)).long()
        h = capture_boundary_states(model, ids.cuda(), attn.cuda(), depth=depth)
        states.append(h[seq_idx.cuda(), pos_idx.cuda()].float().cpu())
    H = torch.cat(states, dim=0)  # [N, d]
    return H


def run_exp_t1(cfg, archive, model, tokenizer):
    print("\n================ EXP T1: SIGMA_H FUNCTIONAL METRIC AUDIT ================", flush=True)
    t0 = time.perf_counter()
    out_dir = OUT_ROOT / "exp_t1"
    out_dir.mkdir(parents=True, exist_ok=True)

    sigma_unlab, mu_unlab, n_unlab = compute_sigma_h_unlabeled(model, tokenizer, UNLABELED_AUDIT, max_docs=256, depth=4)
    print(f"Computed Sigma_h^unlab on {n_unlab} tokens in {time.perf_counter()-t0:.1f}s", flush=True)

    from e0_external_dev import candidate_rows
    ae_rows = [dict(r, dataset="arc_easy_val") for r in candidate_rows(load_jsonl(EXT_DEV_DIR / "arc_easy_val.jsonl"))]
    H_cand = collect_cand_states(archive, model, tokenizer, ae_rows, max_rows=200, depth=4)  # [N, 2048]
    H_cand_d = H_cand.double()
    sigma_cand = (H_cand_d.T @ H_cand_d) / H_cand_d.shape[0]
    print(f"Collected H_cand shape={tuple(H_cand.shape)}", flush=True)

    # Spectrum summary of Sigma_h
    ev_u = torch.linalg.eigvalsh(sigma_unlab).flip(0).clamp_min(0)
    ev_c = torch.linalg.eigvalsh(sigma_cand).flip(0).clamp_min(0)
    spec_summary = {
        "unlab_n_tokens": n_unlab,
        "unlab_top1_frac": float(ev_u[0] / ev_u.sum()),
        "unlab_top16_frac": float(ev_u[:16].sum() / ev_u.sum()),
        "unlab_top64_frac": float(ev_u[:64].sum() / ev_u.sum()),
        "unlab_stable_rank": float(ev_u.sum() / ev_u[0]),
        "cand_n_tokens": int(H_cand.shape[0]),
        "cand_top1_frac": float(ev_c[0] / ev_c.sum()),
        "cand_top16_frac": float(ev_c[:16].sum() / ev_c.sum()),
        "cand_top64_frac": float(ev_c[:64].sum() / ev_c.sum()),
        "cand_stable_rank": float(ev_c.sum() / ev_c[0]),
    }
    print("Sigma_h spectrum:", json.dumps(spec_summary, indent=2), flush=True)

    # Save compact float32 Sigma_h (16 MB each)
    torch.save({"sigma_unlab": sigma_unlab.float(), "mu_unlab": mu_unlab.float(), "sigma_cand": sigma_cand.float()}, out_dir / "SIGMA_H.pt")

    # Load anchors
    u_t1, w_t1 = load_controller_weights(Path(cfg["historical_t1"]))
    u_mx, w_mx = load_controller_weights(Path(cfg["historical_mixed"]))

    # Precompute clipped physical delta_H for T1 & Mixed on H_cand
    def clipped_delta(u_vec, w_vec, H_mat):
        # H_mat: [N, d] float32
        u32 = u_vec.float()
        w32 = w_vec.float()
        scale = H_mat.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        raw = (H_mat @ w32)[:, None] * u32[None, :]
        r_rms = raw.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return raw * (0.45 * scale / r_rms).clamp_max(1.0)

    dH_t1 = clipped_delta(u_t1, w_t1, H_cand)
    dH_mx = clipped_delta(u_mx, w_mx, H_cand)

    def phys_cos(dH_a, dH_b):
        num = float((dH_a * dH_b).sum())
        den = float(dH_a.norm() * dH_b.norm())
        return num / den if den > 0 else 0.0

    # Load E0 macro scores for table correlation
    e0_raw = json.loads((V3_ROOT / "stage1_e0" / "E0_RAW_RESULTS.json").read_text(encoding="utf-8"))
    e0_ev = e0_raw["evaluation"]

    candidates = []
    # Anchors & D2 U1_PMI
    candidates.append(("t2", Path("/root/ouro-functional-controllability-20261123/results/weights/rank1_t2.pt")))
    candidates.append(("mixed", Path(cfg["historical_mixed"])))
    d2_ref = V3_ROOT / "stage2_d2" / "D2_REF_OPERATOR.pt"
    if d2_ref.exists():
        candidates.append(("d2_u1_pmi", d2_ref))
    for p in sorted((V3_ROOT / "stage1_e0" / "fits").glob("*/controller.pt")):
        candidates.append((p.parent.name, p))

    results = {}
    print(f"\n{'arm':46s} | {'ext%':5s} | {'cos_u':6s} {'w_Euc':6s} {'w_SigU':6s} {'w_SigC':6s} | {'op_Euc':6s} {'op_Sig':6s} {'dH_T1':6s} | {'dH_Mx':6s}", flush=True)
    for name, path in candidates:
        if not path.exists():
            continue
        u, w = load_controller_weights(path)
        cu = cos_euc(u, u_t1)
        cw_e = cos_euc(w, w_t1)
        cw_su = cos_sigma(w, w_t1, sigma_unlab)
        cw_sc = cos_sigma(w, w_t1, sigma_cand)
        cop_e = cu * cw_e
        cop_su = cu * cw_su
        cop_sc = cu * cw_sc
        dH = clipped_delta(u, w, H_cand)
        dh_t1 = phys_cos(dH, dH_t1)
        dh_mx = phys_cos(dH, dH_mx)
        # External macro if in E0
        ext = None
        if name in e0_ev:
            m = e0_ev[name]["development"]["metrics"]
            ext = 100.0 * (m["sciq_val"]["acc_norm"] + m["arc_easy_val"]["acc_norm"] + m["cosmosqa_val"]["acc_norm"] + m["mmlu_val"]["acc"] + m["mmlu_pro_val"]["correct"]) / 5.0
        results[name] = {
            "external_macro_pct": ext,
            "norm_u": float(u.norm()),
            "norm_w": float(w.norm()),
            "cos_u_t1": cu,
            "cos_w_euc_t1": cw_e,
            "cos_w_sigma_unlab_t1": cw_su,
            "cos_w_sigma_cand_t1": cw_sc,
            "cos_op_euc_t1": cop_e,
            "cos_op_sigma_unlab_t1": cop_su,
            "cos_op_sigma_cand_t1": cop_sc,
            "phys_deltaH_cos_t1": dh_t1,
            "phys_deltaH_cos_mixed": dh_mx,
        }
        es = f"{ext:5.2f}" if ext is not None else "  —  "
        print(f"{name:46s} | {es} | {cu:+6.3f} {cw_e:+6.3f} {cw_su:+6.3f} {cw_sc:+6.3f} | {cop_e:+6.3f} {cop_sc:+6.3f} {dh_t1:+6.3f} | {dh_mx:+6.3f}", flush=True)

    save_json(out_dir / "T1_SIGMA_METRIC_AUDIT.json", {"spectrum": spec_summary, "arms": results})
    print("EXP_T1_COMPLETE", flush=True)
    return sigma_unlab, sigma_cand


def train_protocol_arm(archive, model, tokenizer, items, protocol, seed, steps=840):
    """Train cold-start Rank-1 on identical 4-choice items under `cloze_full`, `shown_full`, or `shown_letter`."""
    torch.manual_seed(seed)
    net = minimal.Minimal("linear", 1, 0.45).to(model.device)
    net.requires_grad_(True)
    opt = torch.optim.AdamW(net.parameters(), lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"])
    rng = random.Random(f"t3a/{protocol}/{seed}")
    order = list(range(len(items)))
    ptr = 0
    rng.shuffle(order)

    def next_batch(bs=4):
        nonlocal ptr
        out = []
        for _ in range(bs):
            if ptr >= len(order):
                rng.shuffle(order)
                ptr = 0
            out.append(items[order[ptr]])
            ptr += 1
        return out

    t0 = time.perf_counter()
    losses = []
    for step in range(1, steps + 1):
        depth = DEPTHS[(step - 1) % len(DEPTHS)]
        mb = next_batch(RECIPE["effective_batch"])
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            if protocol == "cloze_full":
                crows = [make_candidate_row(it["id"], it["question"], it["options"], it["gold"]) for it in mb]
                opt_logits, _, _ = cloze_option_logits_differentiable(archive, model, tokenizer, crows, depth, net)
                loss = sum(F.cross_entropy(lg.unsqueeze(0), torch.tensor([it["gold"]], device=lg.device))
                           for lg, it in zip(opt_logits, mb)) / len(mb)
            elif protocol == "shown_full":
                crows = []
                golds = []
                for it in mb:
                    m = len(it["options"])
                    shift = rng.randrange(m)
                    content = [(pos - shift) % m for pos in range(m)]
                    rot_opts = [it["options"][idx] for idx in content]
                    rot_gold = (it["gold"] + shift) % m
                    lines = "\n".join(f"{chr(65 + pos)}. {txt}" for pos, txt in enumerate(rot_opts))
                    q_shown = f"{it['question']}\n{lines}"
                    crows.append(make_candidate_row(it["id"], q_shown, rot_opts, rot_gold))
                    golds.append(rot_gold)
                opt_logits, _, _ = cloze_option_logits_differentiable(archive, model, tokenizer, crows, depth, net)
                loss = sum(F.cross_entropy(lg.unsqueeze(0), torch.tensor([g], device=lg.device))
                           for lg, g in zip(opt_logits, golds)) / len(mb)
            elif protocol == "shown_letter":
                prompts_m = []
                golds = []
                for it in mb:
                    m = len(it["options"])
                    shift = rng.randrange(m)
                    content = [(pos - shift) % m for pos in range(m)]
                    rot_opts = [it["options"][idx] for idx in content]
                    rot_gold = (it["gold"] + shift) % m
                    lines = "\n".join(f"{chr(65 + pos)}. {txt}" for pos, txt in enumerate(rot_opts))
                    prefix = f"The following are multiple choice questions (with answers) about {it['subject']}.\n\n" if it.get("subject") else ""
                    prompt = f"{prefix}{it['question']}\n{lines}\nAnswer:"
                    prompts_m.append((prompt, m))
                    golds.append(rot_gold)
                opt_logits, _ = letter_option_logits_differentiable(model, tokenizer, prompts_m, depth, net)
                loss = sum(F.cross_entropy(lg.unsqueeze(0), torch.tensor([g], device=lg.device))
                           for lg, g in zip(opt_logits, golds)) / len(mb)
            else:
                raise ValueError(protocol)
            loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), RECIPE["clip"])
        assert grad.isfinite().item(), f"Nonfinite grad at {protocol} step {step}"
        opt.step()
        losses.append(float(loss.detach()))
        if step % 210 == 0 or step == steps:
            print(f"  [T3A {protocol} s={seed}] step={step}/{steps} loss={sum(losses[-70:])/min(len(losses),70):.4f} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return net, {"final_loss_avg70": sum(losses[-70:]) / 70.0, "seconds": time.perf_counter() - t0}


def run_exp_t3a(cfg, archive, model, tokenizer, evaluator):
    print("\n================ EXP T3-A: SAME-QUESTION PROTOCOL INVERSION ================", flush=True)
    out_dir = OUT_ROOT / "exp_t3a"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build ARC-Challenge 4-choice items (1,011 questions, 100% disjoint from UNIFIED_EXT_DEV)
    arc_raw = load_jsonl(Path(cfg["pools"]["arc_challenge"]["path"]))
    arc_items = []
    for r in arc_raw:
        q, opts, g = extract_q_opts_gold(r)
        if len(opts) == 4:
            arc_items.append({"id": r["sample_id"], "question": q, "options": opts, "gold": g})
    print(f"ARC 4-choice items: {len(arc_items)}", flush=True)

    # Build MMLU-500 items from mmlu_validation.jsonl (100% disjoint from sciq, arc_easy, cosmosqa, mmlu_pro, AND mmlu_dev 158!)
    mmlu_raw = load_jsonl(EXT_DEV_DIR / "mmlu_validation.jsonl")
    mmlu_items = []
    for r in mmlu_raw:
        q, opts, g = extract_q_opts_gold(r)
        if len(opts) == 4:
            mmlu_items.append({"id": r["sample_id"], "question": q, "options": opts, "gold": g, "subject": r.get("subject", "").replace("_", " ")})
    print(f"MMLU-500 items: {len(mmlu_items)}", flush=True)

    # Verify evaluator parity with T1 + Native baseline (including mmlu_dev158)
    print("Evaluating Native & Historical T1 baseline (with disjoint mmlu_dev158)...", flush=True)
    native_ev = evaluator.evaluate(None, include_mmlu_dev158=True)
    t1_net = load_minimal_controller(Path(cfg["historical_t1"]), device=model.device)
    t1_ev = evaluator.evaluate(t1_net, include_mmlu_dev158=True)
    print(f"  Native: macro5={native_ev['macro5']*100:.2f}% non_mmlu4={native_ev['non_mmlu4']*100:.2f}% mmlu_dev158={native_ev['mmlu_dev158']['acc']*100:.2f}%", flush=True)
    print(f"  HistT1: macro5={t1_ev['macro5']*100:.2f}% non_mmlu4={t1_ev['non_mmlu4']*100:.2f}% mmlu_dev158={t1_ev['mmlu_dev158']['acc']*100:.2f}%", flush=True)

    plan = [
        ("arc_cloze_full", arc_items, "cloze_full"),
        ("arc_shown_full", arc_items, "shown_full"),
        ("arc_shown_letter", arc_items, "shown_letter"),
        ("mmlu500_cloze_full", mmlu_items, "cloze_full"),
        ("mmlu500_shown_letter", mmlu_items, "shown_letter"),
    ]
    seeds = (26112671, 26112672)
    results = {"native": native_ev, "historical_t1": t1_ev, "arms": {}}

    for arm_base, items, proto in plan:
        for seed in seeds:
            arm_id = f"{arm_base}.s{seed}"
            ckpt = out_dir / f"{arm_id}.pt"
            if ckpt.exists():
                net = load_minimal_controller(ckpt, device=model.device)
                meta = {"reused": True}
            else:
                net, meta = train_protocol_arm(archive, model, tokenizer, items, proto, seed, steps=840)
                sha = save_rank1_controller(ckpt, net, {"arm": arm_id, "protocol": proto, "seed": seed, "n_items": len(items)})
                meta["sha256"] = sha
            ev = evaluator.evaluate(net, include_mmlu_dev158=True)
            unlab = evaluate_unlabeled_manifold(model, tokenizer, net, UNLABELED_AUDIT, max_docs=64)
            d_mac5 = (ev["macro5"] - native_ev["macro5"]) * 100
            d_nm4 = (ev["non_mmlu4"] - native_ev["non_mmlu4"]) * 100
            d_clz = (ev["cloze3"] - native_ev["cloze3"]) * 100
            d_m158 = (ev["mmlu_dev158"]["acc"] - native_ev["mmlu_dev158"]["acc"]) * 100
            results["arms"][arm_id] = {
                "group": arm_base, "protocol": proto, "seed": seed, "train": meta,
                "eval": ev, "unlabeled": unlab,
                "delta_macro5_pp": d_mac5, "delta_non_mmlu4_pp": d_nm4,
                "delta_cloze3_pp": d_clz, "delta_mmlu_dev158_pp": d_m158,
            }
            print(f"  => {arm_id:28s}: d_mac5={d_mac5:+6.2f} d_nonMMLU4={d_nm4:+6.2f} d_clz3={d_clz:+6.2f} d_m158={d_m158:+5.2f} dNLL={unlab['delta_nll']:+.3f} eos={unlab['eos_abs_shift']:.4f}", flush=True)
            save_json(out_dir / "T3A_PROTOCOL_INVERSION.json", results)

    # Group summary
    summary = {}
    for arm_base, _, _ in plan:
        sub = [v for v in results["arms"].values() if v["group"] == arm_base]
        summary[arm_base] = {
            "delta_macro5_pp": sum(x["delta_macro5_pp"] for x in sub) / len(sub),
            "delta_non_mmlu4_pp": sum(x["delta_non_mmlu4_pp"] for x in sub) / len(sub),
            "delta_cloze3_pp": sum(x["delta_cloze3_pp"] for x in sub) / len(sub),
            "delta_mmlu_dev158_pp": sum(x["delta_mmlu_dev158_pp"] for x in sub) / len(sub),
            "delta_nll_unlab": sum(x["unlabeled"]["delta_nll"] for x in sub) / len(sub),
        }
    results["summary"] = summary
    save_json(out_dir / "T3A_PROTOCOL_INVERSION.json", results)
    print("\n=== T3-A SUMMARY ===", json.dumps(summary, indent=2), flush=True)
    print("EXP_T3A_COMPLETE", flush=True)


def main():
    torch.set_num_threads(4)
    cfg, archive, model, tokenizer = load_base_runtime()
    if not (OUT_ROOT / "exp_t1" / "T1_SIGMA_METRIC_AUDIT.json").exists():
        run_exp_t1(cfg, archive, model, tokenizer)
    else:
        print("Reusing completed Exp T1 at exp_t1/T1_SIGMA_METRIC_AUDIT.json", flush=True)
    evaluator = FastUnifiedDevEvaluator(cfg, archive, model, tokenizer)
    run_exp_t3a(cfg, archive, model, tokenizer, evaluator)
    print("ALL_WESTC_WAVE1_DONE", flush=True)


if __name__ == "__main__":
    main()
