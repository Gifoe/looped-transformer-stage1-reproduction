"""Quick 4-minute follow-up on Exp T2-A:
1. Test ZERO-HUMAN-LABEL `hard_pseudo` on CSQA and MMLU-500 (proving soft-KL entropy mismatch vs hard-pseudo ranking alignment!).
2. Select strictly label-free `V4_SELF_ANCHOR.pt` using scale-invariant held-out unlabeled MC-Cloze agreement on the highest-confidence unlabeled bank (ZERO human labels, ZERO dev scores!).
"""

import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, "/root/autodl-tmp/ouro-v4-cfit-20261126")
from v4_common import (
    EXT_DEV_DIR, UNLABELED_AUDIT, FastUnifiedDevEvaluator,
    evaluate_unlabeled_manifold, load_base_runtime, load_controller_weights,
    load_jsonl, load_minimal_controller, make_candidate_row,
    save_json, save_rank1_controller, score_cloze_probs,
)
from v4_wave1_cqa1_gpu0 import build_unlabeled_teacher_bank, train_self_distill_arm

OUT_ROOT = Path("/root/autodl-tmp/ouro-v4-cfit-20261126")


@torch.inference_mode()
def heldout_unlabeled_mc_agreement(archive, model, tokenizer, bank_val, net):
    """Strictly label-free agreement between controller's Cloze argmax and frozen native MC-letter argmax."""
    crows = [make_candidate_row(it["id"], it["question"], it["options"], 0) for it in bank_val]
    _, probs = score_cloze_probs(archive, model, tokenizer, crows, depth=4, net=net, batch_size=6)
    agree = sum(int(torch.tensor(p).argmax()) == it["pseudo_label"] for it, p in zip(bank_val, probs))
    return agree / len(bank_val)


def main():
    torch.set_num_threads(4)
    out_dir = OUT_ROOT / "exp_t2a_self_distill"
    res_path = out_dir / "T2A_SELF_DISTILL.json"
    results = json.loads(res_path.read_text(encoding="utf-8"))

    cfg, archive, model, tokenizer = load_base_runtime()
    evaluator = FastUnifiedDevEvaluator(cfg, archive, model, tokenizer)
    native_ev = results["native"]
    u_t1, w_t1 = load_controller_weights(Path(cfg["historical_t1"]))

    arc_bank, arc_diag = build_unlabeled_teacher_bank(
        archive, model, tokenizer, "arc_unlabeled", load_jsonl(Path(cfg["pools"]["arc_challenge"]["path"]))
    )
    csqa_bank, csqa_diag = build_unlabeled_teacher_bank(
        archive, model, tokenizer, "csqa_unlabeled", load_jsonl(Path(cfg["pools"]["csqa"]["path"])), max_items=1500
    )
    mmlu_bank, mmlu_diag = build_unlabeled_teacher_bank(
        archive, model, tokenizer, "mmlu500_unlabeled", load_jsonl(EXT_DEV_DIR / "mmlu_validation.jsonl")
    )
    arc_tr, arc_va = arc_bank[:860], arc_bank[860:]
    csqa_tr, csqa_va = csqa_bank[:1250], csqa_bank[1250:]
    mmlu_tr, mmlu_va = mmlu_bank[:420], mmlu_bank[420:]

    extra_plan = [
        ("self_csqa_hardpseudo", csqa_tr, csqa_va, "hard_pseudo", 0.0),
        ("self_mmlu_hardpseudo", mmlu_tr, mmlu_va, "hard_pseudo", 0.0),
    ]
    for base_name, tr_bank, va_bank, mode, lam in extra_plan:
        for seed in (26112681, 26112682):
            arm_id = f"{base_name}.s{seed}"
            ckpt = out_dir / f"{arm_id}.pt"
            if ckpt.exists():
                net = load_minimal_controller(ckpt, device=model.device)
                meta = {"reused": True}
            else:
                net, meta = train_self_distill_arm(archive, model, tokenizer, tr_bank, mode, seed, steps=840)
                meta["sha256"] = save_rank1_controller(ckpt, net, {"arm": arm_id, "mode": mode, "seed": seed, "human_labels_used": 0})
            ev = evaluator.evaluate(net, include_mmlu_dev158=True)
            unlab = evaluate_unlabeled_manifold(model, tokenizer, net, UNLABELED_AUDIT, max_docs=64)
            u_a, w_a = load_controller_weights(ckpt)
            cu = float((u_a @ u_t1) / (u_a.norm() * u_t1.norm()).clamp_min(1e-12))
            cw = float((w_a @ w_t1) / (w_a.norm() * w_t1.norm()).clamp_min(1e-12))
            d_mac5 = (ev["macro5"] - native_ev["macro5"]) * 100
            d_clz3 = (ev["cloze3"] - native_ev["cloze3"]) * 100
            d_nm4 = (ev["non_mmlu4"] - native_ev["non_mmlu4"]) * 100
            results["arms"][arm_id] = {
                "group": base_name, "mode": mode, "trust_lam": lam, "seed": seed,
                "train": meta, "unlabeled": unlab, "eval": ev,
                "delta_macro5_pp": d_mac5, "delta_cloze3_pp": d_clz3, "delta_non_mmlu4_pp": d_nm4,
                "cos_u_t1": cu, "cos_w_euc_t1": cw, "cos_op_euc_t1": cu * cw,
            }
            print(f"  => [EXTRA] {arm_id:28s}: d_mac5={d_mac5:+6.2f} d_clz3={d_clz3:+6.2f} d_nm4={d_nm4:+6.2f} | uT1={cu:+.3f} dNLL={unlab['delta_nll']:+.3f} eos={unlab['eos_abs_shift']:.4f}", flush=True)

    # Strictly label-free anchor selection on the highest-confidence unlabeled bank (arc_unlabeled: mean_conf=0.653 vs csqa=0.316, mmlu=0.425)
    # Selection metric: held-out unlabeled MC-Cloze agreement on arc_va, gated by eos_abs_shift < 0.01 & delta_nll < 1.0!
    best_arm = None
    best_agree = -1.0
    for arm_id, rec in sorted(results["arms"].items()):
        if not arm_id.startswith("self_arc_"):
            continue
        ckpt = out_dir / f"{arm_id}.pt"
        net = load_minimal_controller(ckpt, device=model.device)
        ag = heldout_unlabeled_mc_agreement(archive, model, tokenizer, arc_va, net)
        rec["heldout_unlabeled_mc_agree"] = ag
        ok_man = rec["unlabeled"]["eos_abs_shift"] < 0.01 and rec["unlabeled"]["delta_nll"] < 1.0
        print(f"  [UNLAB SELECT] {arm_id:28s}: heldout_unlab_agree={ag*100:5.2f}% ok_man={ok_man} (diag d_mac5={rec['delta_macro5_pp']:+.2f})", flush=True)
        if ok_man and ag > best_agree:
            best_agree = ag
            best_arm = (arm_id, ckpt, ag)

    win_id, win_ckpt, win_ag = best_arm
    u_win, w_win = load_controller_weights(win_ckpt)
    anchor_path = OUT_ROOT / "V4_SELF_ANCHOR.pt"
    torch.save({
        "u_self": u_win.float(), "w_self": w_win.float(),
        "source_arm": win_id, "selection_rule": "highest_conf_unlabeled_bank_argmax_agreement_on_manifold",
        "heldout_unlabeled_mc_agree": win_ag, "human_labels_used": 0,
    }, anchor_path)
    results["frozen_self_anchor"] = {"arm": win_id, "path": str(anchor_path), "heldout_unlabeled_mc_agree": win_ag}
    save_json(res_path, results)
    print(f"\nUPDATED STRICTLY LABEL-FREE ANCHOR: {win_id} (heldout_unlab_agree={win_ag*100:.2f}%) -> {anchor_path}", flush=True)
    print("T2A_HARDPSEUDO_FOLLOWUP_DONE", flush=True)


if __name__ == "__main__":
    main()
