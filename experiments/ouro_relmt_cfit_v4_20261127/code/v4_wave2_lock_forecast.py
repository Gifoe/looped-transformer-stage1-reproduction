"""Wave 2 Lock: Freeze FORECAST_LOCK_V4_PROSPECTIVE.json + .sha256 BEFORE any training on unseen pools!
Combines purely pre-training features of the 11 unseen sources with:
  - Frozen SIGMA_H.pt
  - Frozen 0-human-label V4_SELF_ANCHOR.pt (from Exp T2-A)
  - Frozen 14-source operators SOURCE_OPERATORS_14.pt (from Exp T4-CFIT-14)
Computes zero-free-parameter CFIT predictors + exact v3 baselines and locks SHA-256!
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch


def op_cos_sigma(u1, w1, u2, w2, S):
    u1, w1, u2, w2 = u1.double(), w1.double(), u2.double(), w2.double()
    cu = float((u1 @ u2) / (u1.norm() * u2.norm()).clamp_min(1e-12))
    n1 = float(torch.clamp(w1 @ (S @ w1), min=1e-12).sqrt())
    n2 = float(torch.clamp(w2 @ (S @ w2), min=1e-12).sqrt())
    cw = float((w1 @ (S @ w2)) / (n1 * n2))
    return cu * cw, cu, cw


def op_cos_euc(u1, w1, u2, w2):
    u1, w1, u2, w2 = u1.double(), w1.double(), u2.double(), w2.double()
    cu = float((u1 @ u2) / (u1.norm() * u2.norm()).clamp_min(1e-12))
    cw = float((w1 @ w2) / (w1.norm() * w2.norm()).clamp_min(1e-12))
    return cu * cw


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    args = ap.parse_args()

    feat_dir = args.out_root / "prospective_unseen" / "features"
    lock_path = args.out_root / "prospective_unseen" / "FORECAST_LOCK_V4_PROSPECTIVE.json"
    sha_path = lock_path.with_suffix(".sha256")
    assert not lock_path.exists(), f"Refusing to overwrite existing lock {lock_path}!"

    # Verify ZERO unseen trained checkpoints exist yet!
    trained_dir = args.out_root / "prospective_unseen" / "trained"
    if trained_dir.exists() and list(trained_dir.glob("*.pt")):
        raise RuntimeError("CONTAMINATION GUARD: trained checkpoints already exist before forecast lock!")

    sd = torch.load(args.out_root / "SIGMA_H.pt", map_location="cpu", weights_only=False)
    S = sd["sigma_unlab"].double()
    evals, evecs = torch.linalg.eigh((S + S.T) * 0.5)
    reg = float(evals.mean()) * 0.05
    S_reg = (evecs * (evals.clamp_min(0.0) + reg)) @ evecs.T

    self_anchor_path = args.out_root / "V4_SELF_ANCHOR.pt"
    self_anchor = torch.load(self_anchor_path, map_location="cpu", weights_only=False)
    u_self, w_self = self_anchor["u_self"].double(), self_anchor["w_self"].double()

    ops14_path = args.out_root / "cfit_14_features" / "SOURCE_OPERATORS_14.pt"
    ops14 = torch.load(ops14_path, map_location="cpu", weights_only=False)
    cand12_names = [k for k in sorted(ops14.keys()) if k not in ("squad", "gsm8k")]

    feat_files = sorted(feat_dir.glob("*.json"))
    assert len(feat_files) >= 6, f"Expected >=6 unseen feature files, found {len(feat_files)}"

    unseen_feats = {}
    unseen_ops = {}
    for fp in feat_files:
        d = json.loads(fp.read_text(encoding="utf-8"))
        name = d["source"]
        unseen_feats[name] = d
        unseen_ops[name] = torch.load(feat_dir / f"{name}_ops.pt", map_location="cpu", weights_only=False)

    per_source = {}
    for name in sorted(unseen_feats.keys()):
        d = unseen_feats[name]
        op = unseen_ops[name]
        u_o, w_o = op["u_ord"], op["w_ord"]
        u_c, w_c = op["u_ctx"], op["w_ctx"]
        u_l, w_l = op["u_lp"], op["w_lp"]

        # 1. Zero-human-label Self-Anchor alignment in H_{Sigma_h}
        a_self_ord, cu_self_o, cw_self_o = op_cos_sigma(u_o, w_o, u_self, w_self, S_reg)
        a_self_ctx, cu_self_c, cw_self_c = op_cos_sigma(u_c, w_c, u_self, w_self, S_reg)
        # Signed natural projection <A_self, M_ctx> / s_ctx and <A_self, M_ord> / s_ord
        # Note: A_self sign convention: check against M_ord / M_ctx (u_self w_self^T has positive alignment with good Cloze ascent!)
        proj_self_ord = float((u_self @ op["M_ord"].double() @ w_self) / max(op["s_ord"] * float(u_self.norm()) * float(w_self.norm()), 1e-12)) if "M_ord" in op else None
        proj_self_ctx = float((u_self @ op["M_ctx"].double() @ w_self) / max(op["s_ctx"] * float(u_self.norm()) * float(w_self.norm()), 1e-12)) if "M_ctx" in op else None

        # 2. Cross-source consensus (median Sigma_h cosine against the 12 historical candidate sources)
        cons_ord_sig_vals = sorted(op_cos_sigma(u_o, w_o, ops14[s]["u_ord"], ops14[s]["w_ord"], S_reg)[0] for s in cand12_names)
        cons_ctx_sig_vals = sorted(op_cos_sigma(u_c, w_c, ops14[s]["u_ctx"], ops14[s]["w_ctx"], S_reg)[0] for s in cand12_names)
        cons_ord_euc_vals = sorted(op_cos_euc(u_o, w_o, ops14[s]["u_ord"], ops14[s]["w_ord"]) for s in cand12_names)
        cons_lp_euc_vals = sorted(op_cos_euc(u_l, w_l, ops14[s]["u_ord"], ops14[s]["w_ord"]) for s in cand12_names)
        med_cons_ord_sig = cons_ord_sig_vals[len(cons_ord_sig_vals) // 2]
        med_cons_ctx_sig = cons_ctx_sig_vals[len(cons_ctx_sig_vals) // 2]
        med_cons_ord_euc = cons_ord_euc_vals[len(cons_ord_euc_vals) // 2]

        # 3. Pre-registered zero-free-parameter CFIT predictors (ALL locked before training!)
        o_iii = d["O_III_mc"]
        o_ratio = d["O_III_over_II"]
        p_dec = max(0.0, d["P_deconf_op_sig"])
        c_ctx = max(0.0, d["C_ctx_sig"])
        m_pres = d["M_pres_ord"]

        # Note: a_self_ord is SIGNED! (In H_{Sigma_h}, w_o and w_self both align with the same cone, so u_o^T u_self * w_o^T S w_self has exact physical sign!)
        preds = {
            # Flagship Theorem-1 Zero-Reference Single-Source CFIT predictors:
            "CFIT_P0_Deconf_Pres": p_dec * m_pres,
            "CFIT_P0b_Deconf_Pure": d["P_deconf_op_sig"],
            "CFIT_P0c_Deconf_SqrtPres": p_dec * (m_pres ** 0.5),
            # Zero-human-label Self-Anchor CFIT predictors:
            "CFIT_P1_SelfAnchor_Pres": a_self_ord * m_pres,
            "CFIT_P1b_SelfAnchor_Pure": a_self_ord,
            "CFIT_P2_SelfAnchor_Ctx": a_self_ctx * m_pres,
            # QA Regime-III Intrinsic CFIT predictors:
            "CFIT_P3_Intrinsic_OIII_Pdec_Cctx_Mpres": o_iii * p_dec * c_ctx * m_pres,
            "CFIT_P4_Intrinsic_OIII_Ratio_Mpres": o_ratio * p_dec * m_pres,
            "CFIT_P5_OIII_Pure": o_iii,
            "CFIT_P6_Hybrid_OIII_Self_Mpres": o_iii * max(0.0, a_self_ord) * m_pres,
            # Multi-source Sigma_h Consensus CFIT predictors:
            "CFIT_P7_Consensus_Sig_Mpres": med_cons_ord_sig * m_pres,
            "CFIT_P7b_Consensus_Sig_Pure": med_cons_ord_sig,
            "CFIT_P8_Deconf_Consensus_Sig": med_cons_ctx_sig * m_pres,
            # Historical v3 baselines (for direct head-to-head prospective audit):
            "v3_P1_signed_true_d2": d["v3_baselines"]["signed_true"],
            "v3_B3_unsigned_ctv": d["v3_baselines"]["unsigned_matrix"],
            "v3_B1_ord_euc_consensus": med_cons_ord_euc,
            "v3_P4_unlabeled_pres_only": m_pres,
        }
        per_source[name] = {
            "pool_sha256": d["pool_sha256"],
            "pool_size": d["pool_size"],
            "n_probed": d["n_probed"],
            "diagnostics": {
                "native_clz_acc": d["native_clz_acc"],
                "native_mc_acc": d["native_mc_acc"],
                "readout_gap_pp": d["readout_gap_pp"],
                "O_III_mc": o_iii,
                "O_II_mc": d["O_II_mc"],
                "O_III_over_II": o_ratio,
                "P_deconf_op_sig": d["P_deconf_op_sig"],
                "C_ctx_sig": d["C_ctx_sig"],
                "M_pres_ord": m_pres,
                "delta_nll_ord": d["man_ord"]["delta_nll"],
                "eos_shift_ord": d["man_ord"]["eos_abs_shift"],
                "a_self_ord_sig": a_self_ord,
                "cu_self_ord": cu_self_o,
                "cw_self_ord_sig": cw_self_o,
                "a_self_ctx_sig": a_self_ctx,
                "proj_self_ord": proj_self_ord,
                "proj_self_ctx": proj_self_ctx,
                "med_cons_ord_sig": med_cons_ord_sig,
                "med_cons_ctx_sig": med_cons_ctx_sig,
            },
            "predictors": preds,
        }

    # Build explicit predicted rankings for every predictor
    rankings = {}
    pred_names = list(next(iter(per_source.values()))["predictors"].keys())
    for pn in pred_names:
        rankings[pn] = sorted(per_source.keys(), key=lambda s: per_source[s]["predictors"][pn], reverse=True)

    lock_obj = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strict_pre_training_lock": True,
        "self_anchor_path": str(self_anchor_path),
        "self_anchor_sha256": sha256_file(self_anchor_path),
        "self_anchor_arm": self_anchor.get("source_arm"),
        "sigma_h_sha256": sha256_file(args.out_root / "SIGMA_H.pt"),
        "ops14_sha256": sha256_file(ops14_path),
        "n_unseen_sources": len(per_source),
        "sources": sorted(per_source.keys()),
        "rankings": rankings,
        "per_source": per_source,
    }
    lock_path.write_text(json.dumps(lock_obj, indent=2) + "\n", encoding="utf-8")
    digest = sha256_file(lock_path)
    sha_path.write_text(f"{digest}  {lock_path.name}\n", encoding="utf-8")
    print(f"FROZEN CRYPTOGRAPHIC FORECAST LOCK ({len(per_source)} unseen sources): {digest}", flush=True)
    for pn in ("CFIT_P0_Deconf_Pres", "CFIT_P0b_Deconf_Pure", "CFIT_P1_SelfAnchor_Pres", "CFIT_P7_Consensus_Sig_Mpres", "v3_B3_unsigned_ctv"):
        print(f"  RANK[{pn:38s}]: {rankings[pn]}", flush=True)


if __name__ == "__main__":
    main()
