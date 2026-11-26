"""Wave 2A: Compute exact CFIT + v3-baseline features on unseen prospective pools (ZERO training!).
Can be sharded across GPUs via `--sources quartz,swag,...`.
Each source computes:
  - 4-fold deconfounded moments (M_ord, M_ctx, M_lp) matching collect_signed.py bit-for-bit on 512 samples
  - Exact Regime-III observability O_III_mc, O_II_mc, O_III_loop (K4 vs K2 Cloze + rotation-averaged MC letter)
  - Exact On-Manifold Preservation M_pres_ord & M_pres_ctx on audit_unlabeled
  - Exact v3 unsigned / signed_true baseline statistics (for apples-to-apples prospective comparison)
"""

import argparse
import hashlib
import json
import math
import time
from itertools import combinations
from pathlib import Path

import torch

import minimal
from collect_signed import MOMENTS, measure, mismatched, source_stem
from d3_scores import mismatch_pairs
from v4_common import (
    UNLABELED_AUDIT,
    V3_ROOT,
    compute_sigma_h_unlabeled,
    evaluate_unlabeled_manifold,
    extract_q_opts_gold,
    load_base_runtime,
    load_jsonl,
    make_candidate_row,
    save_json,
    score_cloze_probs,
    score_mc_letter_rotation_probs,
    sha256_file,
)


def sigma_svd1(M, S_half, S_half_inv):
    U, s, Vt = torch.linalg.svd(M.double(), full_matrices=False)
    return U[:, 0], Vt[0], float(s[0])


def op_cos_sigma(u1, w1, u2, w2, S):
    cu = float((u1 @ u2) / (u1.norm() * u2.norm()).clamp_min(1e-12))
    n1 = float(torch.clamp(w1 @ (S @ w1), min=1e-12).sqrt())
    n2 = float(torch.clamp(w2 @ (S @ w2), min=1e-12).sqrt())
    cw = float((w1 @ (S @ w2)) / (n1 * n2))
    return cu * cw, cu, cw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--pools-dir", type=Path, required=True)
    ap.add_argument("--sources", required=True, help="Comma-separated unseen source names")
    ap.add_argument("--limit", type=int, default=512)
    ap.add_argument("--seed", type=int, default=26112641)
    args = ap.parse_args()

    torch.set_num_threads(4)
    feat_dir = args.out_root / "prospective_unseen" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    cfg, archive, model, tokenizer = load_base_runtime()

    # Load or compute Sigma_h^unlab
    sigma_path = args.out_root / "SIGMA_H.pt"
    if not sigma_path.exists() and (args.out_root / "exp_t1" / "SIGMA_H.pt").exists():
        sigma_path = args.out_root / "exp_t1" / "SIGMA_H.pt"
    if sigma_path.exists():
        sd = torch.load(sigma_path, map_location="cpu", weights_only=False)
        S = sd["sigma_unlab"].double()
    else:
        S, mu, n_tok = compute_sigma_h_unlabeled(model, tokenizer, UNLABELED_AUDIT, max_docs=256, depth=4)
        torch.save({"sigma_unlab": S.float(), "mu_unlab": mu.float(), "n_tokens": n_tok}, args.out_root / "SIGMA_H.pt")
    evals, evecs = torch.linalg.eigh((S + S.T) * 0.5)
    reg = float(evals.mean()) * 0.05
    evals_reg = evals.clamp_min(0.0) + reg
    S_reg = (evecs * evals_reg) @ evecs.T
    S_half = (evecs * evals_reg.sqrt()) @ evecs.T
    S_half_inv = (evecs * evals_reg.rsqrt()) @ evecs.T

    # Load D2 anchor ONLY so we can compute exact v3 baseline P1_signed_true for honest prospective comparison
    d2_ref = torch.load(V3_ROOT / "stage2_d2" / "D2_REF_OPERATOR.pt", map_location="cpu", weights_only=False)
    d2_u = d2_ref["u_A0"].float().cuda()
    d2_w = d2_ref["w_A0"].float().cuda()

    sources = [x.strip() for x in args.sources.split(",") if x.strip()]
    for name in sources:
        out_json = feat_dir / f"{name}.json"
        out_pt = feat_dir / f"{name}_ops.pt"
        if out_json.exists() and out_pt.exists():
            print(f"[SKIP] {name} already extracted.", flush=True)
            continue
        t0 = time.perf_counter()
        pool_path = args.pools_dir / f"{name}.jsonl"
        full = load_jsonl(pool_path)
        pool_sha = sha256_file(pool_path)
        index = {r["sample_id"]: r for r in full}
        order = sorted(index, key=lambda k: hashlib.sha256(f"{args.seed}:{k}".encode()).hexdigest())
        eligible = order[:min(args.limit, len(order))]
        swaps = mismatch_pairs([dict(id=k, stem=source_stem(index[k])) for k in index], args.seed)

        # 1. Exact 4-fold deconfounded moments matching collect_signed.py
        fold_ord, fold_ctx, fold_lp = [], [], []
        tot_ord = tot_ctx = tot_lp = None
        all_recs = []
        tot_n = 0
        for fold in range(4):
            used = eligible[fold::4]
            chosen = [index[k] for k in used]
            foil = [dict(mismatched(r, index[swaps[r["sample_id"]]]), _mismatch_source_id=swaps[r["sample_id"]]) for r in chosen]
            sums = {m: torch.zeros((2048, 2048), device="cuda") for m in MOMENTS}
            for i in range(0, len(chosen), 4):
                recs = measure(chosen[i:i + 4], foil[i:i + 4], archive, model, tokenizer, "candidate", 4, d2_u, d2_w, sums)
                all_recs.extend(recs)
            n_f = len(chosen)
            Mo = sums["ordinary"].double().cpu() / n_f
            Mc = sums["true_receiver"].double().cpu() / n_f
            Ml = sums["loop_contrast"].double().cpu() / n_f
            fold_ord.append(Mo)
            fold_ctx.append(Mc)
            fold_lp.append(Ml)
            tot_ord = sums["ordinary"].double().cpu() if tot_ord is None else tot_ord + sums["ordinary"].double().cpu()
            tot_ctx = sums["true_receiver"].double().cpu() if tot_ctx is None else tot_ctx + sums["true_receiver"].double().cpu()
            tot_lp = sums["loop_contrast"].double().cpu() if tot_lp is None else tot_lp + sums["loop_contrast"].double().cpu()
            tot_n += n_f

        M_ord = tot_ord / tot_n
        M_ctx = tot_ctx / tot_n
        M_lp = tot_lp / tot_n
        u_ord, w_ord, s_ord = sigma_svd1(M_ord, S_half, S_half_inv)
        u_ctx, w_ctx, s_ctx = sigma_svd1(M_ctx, S_half, S_half_inv)
        u_lp, w_lp, s_lp = sigma_svd1(M_lp, S_half, S_half_inv)

        op_cos_ctx_euc = float((u_ord @ u_ctx) * (w_ord @ w_ctx))
        op_cos_ctx_sig, _, _ = op_cos_sigma(u_ord, w_ord, u_ctx, w_ctx, S_reg)
        proj_ord_on_ctx = float((u_ord @ M_ctx @ w_ord) / max(s_ctx, 1e-12))

        f_axes_ctx = [sigma_svd1(Fm, S_half, S_half_inv) for Fm in fold_ctx]
        f_axes_ord = [sigma_svd1(Fm, S_half, S_half_inv) for Fm in fold_ord]
        c_ctx_euc = [float((u1 @ u2) * (w1 @ w2)) for (u1, w1, _), (u2, w2, _) in combinations(f_axes_ctx, 2)]
        c_ctx_sig = [op_cos_sigma(u1, w1, u2, w2, S_reg)[0] for (u1, w1, _), (u2, w2, _) in combinations(f_axes_ctx, 2)]
        c_ord_sig = [op_cos_sigma(u1, w1, u2, w2, S_reg)[0] for (u1, w1, _), (u2, w2, _) in combinations(f_axes_ord, 2)]

        # 2. On-Manifold Preservation M_pres
        def make_probe_net(u_vec, w_vec, norm_prod=2.5):
            net = minimal.Minimal("linear", 1, 0.45).to(model.device)
            with torch.no_grad():
                un = u_vec / u_vec.norm().clamp_min(1e-12) * math.sqrt(norm_prod)
                wn = w_vec / w_vec.norm().clamp_min(1e-12) * math.sqrt(norm_prod)
                net.out.weight.copy_(un[:, None].to(net.out.weight.dtype).to(model.device))
                net.encode.weight.copy_(wn[None, :].to(net.encode.weight.dtype).to(model.device))
            return net.eval()

        man_ord = evaluate_unlabeled_manifold(model, tokenizer, make_probe_net(u_ord, w_ord), UNLABELED_AUDIT, max_docs=64)
        man_ctx = evaluate_unlabeled_manifold(model, tokenizer, make_probe_net(u_ctx, w_ctx), UNLABELED_AUDIT, max_docs=64)
        m_pres_ord = math.exp(-max(0.0, man_ord["delta_nll"]) - 5.0 * man_ord["eos_abs_shift"])
        m_pres_ctx = math.exp(-max(0.0, man_ctx["delta_nll"]) - 5.0 * man_ctx["eos_abs_shift"])

        # 3. Exact Regime-III Observability O_III (K4 & K2 Cloze + rotation-averaged MC letter)
        items = []
        for k in eligible:
            r = index[k]
            q, opts, g = extract_q_opts_gold(r)
            if 2 <= len(opts) <= 10:
                items.append((r["sample_id"], q, opts, g))
        crows = [make_candidate_row(sid, q, opts, g, dataset=name) for sid, q, opts, g in items]
        _, clz_p4 = score_cloze_probs(archive, model, tokenizer, crows, depth=4, net=None, batch_size=6)
        _, clz_p2 = score_cloze_probs(archive, model, tokenizer, crows, depth=2, net=None, batch_size=6)
        mc_p4 = score_mc_letter_rotation_probs(model, tokenizer, [(q, opts) for _, q, opts, _ in items], depth=4, batch_size=8)

        o_iii_list, o_ii_list, o_loop_list = [], [], []
        r3_cnt = r2_cnt = r1_cnt = clz_ok = mc_ok = 0
        for (_, _, opts, g), p4, p2, pm in zip(items, clz_p4, clz_p2, mc_p4):
            m = len(opts)
            egap = 1.0 - p4[g]
            p_lat_mc = max(0.0, (pm[g] - 1.0 / m) / (1.0 - 1.0 / m))
            o_iii_list.append(egap * p_lat_mc)
            o_ii_list.append(egap * (1.0 - p_lat_mc))
            o_loop_list.append(egap * max(0.0, p4[g] - p2[g]))
            c_ok = int(torch.tensor(p4).argmax()) == g
            m_ok = int(torch.tensor(pm).argmax()) == g
            clz_ok += int(c_ok)
            mc_ok += int(m_ok)
            if m_ok and not c_ok:
                r3_cnt += 1
            elif (not m_ok) and (not c_ok):
                r2_cnt += 1
            else:
                r1_cnt += 1

        n_it = len(items)
        o_iii = sum(o_iii_list) / n_it
        o_ii = sum(o_ii_list) / n_it
        o_loop = sum(o_loop_list) / n_it
        p_dec = max(0.0, op_cos_ctx_sig)
        c_ctx_s = max(0.0, sum(c_ctx_sig) / len(c_ctx_sig))

        # v3 baseline scalars
        b_v3 = {k: sum(r[k] for r in all_recs) / len(all_recs)
                for k in ("signed_true", "signed_proxy", "signed_full", "unsigned_matrix", "unsigned_gradient", "error_gap")}
        # Also compute v3 B0_source_only = mean(signed_true) / ... Wait: B0 in v3 was s_ord * ... let's store fold_ctx/ord norms & unsigned_ctv
        feat = {
            "source": name, "pool_sha256": pool_sha, "pool_size": len(full), "n_probed": tot_n,
            "s_ord": s_ord, "s_ctx": s_ctx, "s_lp": s_lp, "ctx_to_ord_ratio": s_ctx / max(s_ord, 1e-12),
            "P_deconf_op_euc": op_cos_ctx_euc, "P_deconf_op_sig": op_cos_ctx_sig, "P_deconf_proj_ord_on_ctx": proj_ord_on_ctx,
            "C_ctx_euc": sum(c_ctx_euc) / len(c_ctx_euc), "C_ctx_sig": sum(c_ctx_sig) / len(c_ctx_sig),
            "C_ord_sig": sum(c_ord_sig) / len(c_ord_sig),
            "man_ord": man_ord, "man_ctx": man_ctx, "M_pres_ord": m_pres_ord, "M_pres_ctx": m_pres_ctx,
            "O_III_mc": o_iii, "O_II_mc": o_ii, "O_III_over_II": o_iii / max(o_ii, 1e-6), "O_III_loop": o_loop,
            "native_clz_acc": clz_ok / n_it, "native_mc_acc": mc_ok / n_it, "readout_gap_pp": (mc_ok - clz_ok) / n_it * 100,
            "regime3_frac": r3_cnt / n_it, "regime2_frac": r2_cnt / n_it, "regime1_frac": r1_cnt / n_it,
            "I_CFIT_intrinsic": o_iii * p_dec * c_ctx_s * m_pres_ord,
            "I_CFIT_ratio": (o_iii / max(o_ii, 1e-6)) * p_dec * m_pres_ord,
            "v3_baselines": b_v3,
            "seconds": time.perf_counter() - t0,
        }
        torch.save({
            "u_ord": u_ord.float(), "w_ord": w_ord.float(), "s_ord": s_ord,
            "u_ctx": u_ctx.float(), "w_ctx": w_ctx.float(), "s_ctx": s_ctx,
            "u_lp": u_lp.float(), "w_lp": w_lp.float(), "s_lp": s_lp,
            "M_ord": M_ord.float(), "M_ctx": M_ctx.float(), "M_lp": M_lp.float(),
        }, out_pt)
        save_json(out_json, feat)
        print(f"  [UNSEEN {name:14s}] {feat['seconds']:5.1f}s | clz={feat['native_clz_acc']*100:4.1f}% mc={feat['native_mc_acc']*100:4.1f}% gap={feat['readout_gap_pp']:+5.1f} | O_III={o_iii:.3f} P_dec={op_cos_ctx_sig:+.3f} C_ctx={feat['C_ctx_sig']:+.3f} M_pres={m_pres_ord:.3f} | I_CFIT={feat['I_CFIT_intrinsic']:.4f}", flush=True)
    print("UNSEEN_FEATURES_SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
