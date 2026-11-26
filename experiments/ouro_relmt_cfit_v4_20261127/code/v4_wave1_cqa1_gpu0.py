"""Wave 1 on cqa1 GPU 0 (vGPU 24GB):
Exp T2-A: ZERO-HUMAN-LABEL Protocol Self-Distillation (MC-Letter -> Cloze) in FULL R^2048.
Tests RELMT's core falsifiable prediction: can a 4,096-param Rank-1 controller recover +4..+7 pp
cross-task gains using ZERO ground-truth labels y* and ZERO historical T1/Rank-8 weights?
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
    UNLABELED_FIT,
    V3_ROOT,
    FastUnifiedDevEvaluator,
    cloze_option_logits_differentiable,
    evaluate_unlabeled_manifold,
    extract_q_opts_gold,
    forward_with_hook_all_tokens,
    load_base_runtime,
    load_controller_weights,
    load_minimal_controller,
    load_jsonl,
    make_candidate_row,
    save_json,
    save_rank1_controller,
    score_cloze_probs,
    score_mc_letter_rotation_probs,
)

OUT_ROOT = Path("/root/autodl-tmp/ouro-v4-cfit-20261126")


def build_unlabeled_teacher_bank(archive, model, tokenizer, name, raw_rows, max_items=None):
    """Strip gold labels completely (keeping gold ONLY for offline diagnostic readout-gap logging, NEVER training!)."""
    items = []
    for r in raw_rows:
        q, opts, g = extract_q_opts_gold(r)
        if 2 <= len(opts) <= 5:
            items.append({"id": r["sample_id"], "question": q, "options": opts, "_diag_gold_never_trained": g})
    if max_items is not None and len(items) > max_items:
        rng = random.Random(f"bank/{name}")
        idx = list(range(len(items)))
        rng.shuffle(idx)
        items = [items[i] for i in sorted(idx[:max_items])]

    t0 = time.perf_counter()
    q_opts = [(it["question"], it["options"]) for it in items]
    mc_probs = score_mc_letter_rotation_probs(model, tokenizer, q_opts, depth=4, batch_size=8)
    crows = [make_candidate_row(it["id"], it["question"], it["options"], 0) for it in items]  # dummy gold=0
    _, cloze_probs = score_cloze_probs(archive, model, tokenizer, crows, depth=4, net=None, batch_size=6)

    # Diagnostic only: measure frozen Native Latent-to-Readout Gap!
    mc_acc = sum(int(torch.tensor(p).argmax()) == it["_diag_gold_never_trained"] for p, it in zip(mc_probs, items)) / len(items)
    clz_acc = sum(int(torch.tensor(p).argmax()) == it["_diag_gold_never_trained"] for p, it in zip(cloze_probs, items)) / len(items)
    agree = sum(int(torch.tensor(pm).argmax()) == int(torch.tensor(pc).argmax()) for pm, pc in zip(mc_probs, cloze_probs)) / len(items)
    mean_conf = sum(max(0.0, (max(p) - 1.0 / len(p)) / (1.0 - 1.0 / len(p))) for p in mc_probs) / len(items)
    print(f"  Bank[{name}] n={len(items)} built in {time.perf_counter()-t0:.1f}s | diag_MC={mc_acc*100:.2f}% diag_Cloze={clz_acc*100:.2f}% gap={(mc_acc-clz_acc)*100:+.2f}pp agree={agree*100:.1f}% mean_conf={mean_conf:.3f}", flush=True)

    # Strictly strip diagnostic gold before returning training records!
    clean_bank = []
    for it, mp in zip(items, mc_probs):
        m = len(it["options"])
        conf = max(0.0, (max(mp) - 1.0 / m) / (1.0 - 1.0 / m))
        clean_bank.append({
            "id": it["id"],
            "question": it["question"],
            "options": it["options"],
            "q_mc": mp,
            "conf": conf,
            "pseudo_label": int(torch.tensor(mp).argmax()),
        })
    diag = {"n": len(items), "diag_mc_acc": mc_acc, "diag_cloze_acc": clz_acc, "diag_gap_pp": (mc_acc - clz_acc) * 100, "agree": agree, "mean_conf": mean_conf}
    return clean_bank, diag


def prepare_unlabeled_trust_batches(tokenizer, corpus_path=UNLABELED_FIT, max_docs=128):
    rows = load_jsonl(corpus_path)[:max_docs]
    seqs = [tokenizer.encode(r["text"], add_special_tokens=False)[:192] for r in rows]
    return [s for s in seqs if len(s) >= 64]


def unlabeled_onesided_nll_penalty(model, tokenizer, seqs, net, depth, nat_nll_cache):
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
    if key not in nat_nll_cache:
        with torch.no_grad():
            h0 = forward_with_hook_all_tokens(model, ids, attn, depth=depth, net=None)
            z0 = model.lm_head(h0[:, :-1][valid]).float()
            nat_nll_cache[key] = float(F.cross_entropy(z0, targets).detach())
    h1 = forward_with_hook_all_tokens(model, ids, attn, depth=depth, net=net)
    z1 = model.lm_head(h1[:, :-1][valid]).float()
    nll1 = F.cross_entropy(z1, targets)
    return F.relu(nll1 - nat_nll_cache[key] - 0.02)


def train_self_distill_arm(archive, model, tokenizer, bank, mode, seed, steps=840, trust_seqs=None, trust_lam=0.0):
    """Train 4,096-param Rank-1 in FULL R^2048 using ZERO human labels."""
    torch.manual_seed(seed)
    net = minimal.Minimal("linear", 1, 0.45).to(model.device)
    net.requires_grad_(True)
    opt = torch.optim.AdamW(net.parameters(), lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"])
    rng = random.Random(f"t2a/{mode}/{seed}")
    order = list(range(len(bank)))
    rng.shuffle(order)
    ptr = 0
    nat_cache = {}

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
    for step in range(1, steps + 1):
        depth = DEPTHS[(step - 1) % len(DEPTHS)]
        mb = next_mb(RECIPE["effective_batch"])
        # Note: dummy gold=0 is NEVER used in loss! Only q_mc or pseudo_label from frozen model!
        crows = [make_candidate_row(it["id"], it["question"], it["options"], 0) for it in mb]
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            opt_logits, _, _ = cloze_option_logits_differentiable(archive, model, tokenizer, crows, depth, net)
            item_losses = []
            weights = []
            for lg, it in zip(opt_logits, mb):
                log_p = F.log_softmax(lg, dim=0)
                if mode == "kl":
                    q = torch.tensor(it["q_mc"], dtype=torch.float32, device=lg.device).clamp_min(1e-8)
                    l_i = -(q * log_p).sum() + (q * q.log()).sum()
                    w_i = 1.0
                elif mode == "conf_kl":
                    q = torch.tensor(it["q_mc"], dtype=torch.float32, device=lg.device).clamp_min(1e-8)
                    l_i = -(q * log_p).sum() + (q * q.log()).sum()
                    w_i = max(0.05, it["conf"])
                elif mode == "hard_pseudo":
                    l_i = -log_p[it["pseudo_label"]]
                    w_i = max(0.05, it["conf"])
                else:
                    raise ValueError(mode)
                item_losses.append(l_i * w_i)
                weights.append(w_i)
            loss = sum(item_losses) / sum(weights)
            if trust_lam > 0.0 and trust_seqs and step % 2 == 0:
                t_mb = [trust_seqs[(step + j) % len(trust_seqs)] for j in range(2)]
                loss = loss + trust_lam * unlabeled_onesided_nll_penalty(model, tokenizer, t_mb, net, depth, nat_cache)
            loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), RECIPE["clip"])
        assert grad.isfinite().item(), f"Nonfinite grad at step {step}"
        opt.step()
        losses.append(float(loss.detach()))
        if step % 210 == 0 or step == steps:
            print(f"  [T2A {mode} lam={trust_lam} s={seed}] step={step}/{steps} loss={sum(losses[-70:])/min(len(losses),70):.4f} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return net, {"final_loss_avg70": sum(losses[-70:]) / 70.0, "seconds": time.perf_counter() - t0}


@torch.inference_mode()
def heldout_unlabeled_distill_kl(archive, model, tokenizer, bank_val, net):
    """Strictly label-free validation metric: KL(q_0^MC || p_net^cloze) on held-out unlabeled fold."""
    crows = [make_candidate_row(it["id"], it["question"], it["options"], 0) for it in bank_val]
    _, probs = score_cloze_probs(archive, model, tokenizer, crows, depth=4, net=net, batch_size=6)
    kls = []
    for it, p in zip(bank_val, probs):
        q = torch.tensor(it["q_mc"], dtype=torch.float64).clamp_min(1e-8)
        pt = torch.tensor(p, dtype=torch.float64).clamp_min(1e-8)
        kls.append(float((q * (q.log() - pt.log())).sum()))
    return sum(kls) / len(kls)


def main():
    torch.set_num_threads(4)
    out_dir = OUT_ROOT / "exp_t2a_self_distill"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, archive, model, tokenizer = load_base_runtime()
    evaluator = FastUnifiedDevEvaluator(cfg, archive, model, tokenizer)

    print("Evaluating Native & Historical T1 baseline on cqa1...", flush=True)
    native_ev = evaluator.evaluate(None, include_mmlu_dev158=True)
    t1_net = load_minimal_controller(Path(cfg["historical_t1"]), device=model.device)
    t1_ev = evaluator.evaluate(t1_net, include_mmlu_dev158=True)
    u_t1, w_t1 = load_controller_weights(Path(cfg["historical_t1"]))
    print(f"  Native: macro5={native_ev['macro5']*100:.2f}% | T1: macro5={t1_ev['macro5']*100:.2f}%", flush=True)

    # Build 3 zero-label teacher banks (split 85/15 for strictly label-free held-out KL selection!)
    arc_bank, arc_diag = build_unlabeled_teacher_bank(
        archive, model, tokenizer, "arc_unlabeled", load_jsonl(Path(cfg["pools"]["arc_challenge"]["path"]))
    )
    csqa_bank, csqa_diag = build_unlabeled_teacher_bank(
        archive, model, tokenizer, "csqa_unlabeled", load_jsonl(Path(cfg["pools"]["csqa"]["path"])), max_items=1500
    )
    mmlu_bank, mmlu_diag = build_unlabeled_teacher_bank(
        archive, model, tokenizer, "mmlu500_unlabeled", load_jsonl(EXT_DEV_DIR / "mmlu_validation.jsonl")
    )
    trust_seqs = prepare_unlabeled_trust_batches(tokenizer, UNLABELED_FIT, max_docs=128)

    # Split 85/15 unlabeled val fold for purely label-free anchor selection
    arc_tr, arc_va = arc_bank[:860], arc_bank[860:]
    csqa_tr, csqa_va = csqa_bank[:1250], csqa_bank[1250:]
    mmlu_tr, mmlu_va = mmlu_bank[:420], mmlu_bank[420:]

    nat_kl_arc = heldout_unlabeled_distill_kl(archive, model, tokenizer, arc_va, None)
    nat_kl_csqa = heldout_unlabeled_distill_kl(archive, model, tokenizer, csqa_va, None)
    print(f"Native held-out unlabeled KL(MC||Cloze): arc={nat_kl_arc:.4f}, csqa={nat_kl_csqa:.4f}", flush=True)

    plan = [
        ("self_arc_kl", arc_tr, arc_va, "kl", 0.0),
        ("self_arc_confkl", arc_tr, arc_va, "conf_kl", 0.0),
        ("self_arc_hardpseudo", arc_tr, arc_va, "hard_pseudo", 0.0),
        ("self_arc_confkl_trust", arc_tr, arc_va, "conf_kl", 1.0),
        ("self_csqa_confkl", csqa_tr, csqa_va, "conf_kl", 0.0),
        ("self_csqa_confkl_trust", csqa_tr, csqa_va, "conf_kl", 1.0),
        ("self_mmlu_confkl", mmlu_tr, mmlu_va, "conf_kl", 0.0),
    ]
    seeds = (26112681, 26112682)

    results = {
        "native": native_ev,
        "historical_t1": t1_ev,
        "bank_diagnostics": {"arc": arc_diag, "csqa": csqa_diag, "mmlu500": mmlu_diag},
        "native_unlabeled_kl": {"arc_va": nat_kl_arc, "csqa_va": nat_kl_csqa},
        "arms": {},
    }

    best_unlab_score = -1e9
    best_unlab_arm = None

    for base_name, tr_bank, va_bank, mode, lam in plan:
        for seed in seeds:
            arm_id = f"{base_name}.s{seed}"
            ckpt = out_dir / f"{arm_id}.pt"
            if ckpt.exists():
                net = load_minimal_controller(ckpt, device=model.device)
                meta = {"reused": True}
            else:
                net, meta = train_self_distill_arm(archive, model, tokenizer, tr_bank, mode, seed, steps=840, trust_seqs=trust_seqs, trust_lam=lam)
                sha = save_rank1_controller(ckpt, net, {"arm": arm_id, "mode": mode, "trust_lam": lam, "seed": seed, "human_labels_used": 0})
                meta["sha256"] = sha

            # Strictly label-free validation score (for unbiased anchor selection)
            val_kl = heldout_unlabeled_distill_kl(archive, model, tokenizer, va_bank, net)
            unlab = evaluate_unlabeled_manifold(model, tokenizer, net, UNLABELED_AUDIT, max_docs=64)
            # Evaluate on UNIFIED_EXT_DEV (reported transparently, NEVER used for anchor selection!)
            ev = evaluator.evaluate(net, include_mmlu_dev158=True)
            u_a, w_a = load_controller_weights(ckpt)
            cu = float((u_a @ u_t1) / (u_a.norm() * u_t1.norm()).clamp_min(1e-12))
            cw = float((w_a @ w_t1) / (w_a.norm() * w_t1.norm()).clamp_min(1e-12))
            d_mac5 = (ev["macro5"] - native_ev["macro5"]) * 100
            d_clz3 = (ev["cloze3"] - native_ev["cloze3"]) * 100
            d_nm4 = (ev["non_mmlu4"] - native_ev["non_mmlu4"]) * 100

            # Purely unlabeled selection score (only from arc/csqa banks, not mmlu): KL reduction minus NLL penalty
            if base_name.startswith("self_csqa") or base_name.startswith("self_arc"):
                ref_kl = nat_kl_arc if base_name.startswith("self_arc") else nat_kl_csqa
                unlab_sel = (ref_kl - val_kl) - 2.0 * max(0.0, unlab["delta_nll"]) - 5.0 * unlab["eos_abs_shift"]
                if unlab_sel > best_unlab_score:
                    best_unlab_score = unlab_sel
                    best_unlab_arm = (arm_id, ckpt, unlab_sel)

            results["arms"][arm_id] = {
                "group": base_name, "mode": mode, "trust_lam": lam, "seed": seed,
                "train": meta, "heldout_unlabeled_kl": val_kl, "unlabeled": unlab,
                "eval": ev, "delta_macro5_pp": d_mac5, "delta_cloze3_pp": d_clz3, "delta_non_mmlu4_pp": d_nm4,
                "cos_u_t1": cu, "cos_w_euc_t1": cw, "cos_op_euc_t1": cu * cw,
            }
            print(f"  => {arm_id:28s}: d_mac5={d_mac5:+6.2f} d_clz3={d_clz3:+6.2f} d_nm4={d_nm4:+6.2f} | uT1={cu:+.3f} wT1={cw:+.3f} | unlab_kl={val_kl:.3f} dNLL={unlab['delta_nll']:+.3f} eos={unlab['eos_abs_shift']:.4f}", flush=True)
            save_json(out_dir / "T2A_SELF_DISTILL.json", results)

    # Freeze strictly label-free winning anchor V4_SELF_ANCHOR.pt
    if best_unlab_arm is not None:
        win_id, win_ckpt, win_sc = best_unlab_arm
        u_win, w_win = load_controller_weights(win_ckpt)
        anchor_path = OUT_ROOT / "V4_SELF_ANCHOR.pt"
        torch.save({"u_self": u_win.float(), "w_self": w_win.float(), "source_arm": win_id, "unlabeled_selection_score": win_sc}, anchor_path)
        results["frozen_self_anchor"] = {"arm": win_id, "path": str(anchor_path), "unlabeled_selection_score": win_sc}
        print(f"\nFROZEN LABEL-FREE ANCHOR: {win_id} (unlab_score={win_sc:.4f}) -> {anchor_path}", flush=True)

    # Group means
    summary = {}
    for base_name, _, _, _, _ in plan:
        sub = [v for v in results["arms"].values() if v["group"] == base_name]
        summary[base_name] = {
            "delta_macro5_pp": sum(x["delta_macro5_pp"] for x in sub) / len(sub),
            "delta_cloze3_pp": sum(x["delta_cloze3_pp"] for x in sub) / len(sub),
            "delta_non_mmlu4_pp": sum(x["delta_non_mmlu4_pp"] for x in sub) / len(sub),
            "delta_nll_unlab": sum(x["unlabeled"]["delta_nll"] for x in sub) / len(sub),
            "mean_abs_cos_u_t1": sum(abs(x["cos_u_t1"]) for x in sub) / len(sub),
        }
    results["summary"] = summary
    save_json(out_dir / "T2A_SELF_DISTILL.json", results)
    print("\n=== T2-A SUMMARY ===", json.dumps(summary, indent=2), flush=True)
    print("ALL_CQA1_GPU0_WAVE1_DONE", flush=True)


if __name__ == "__main__":
    main()
