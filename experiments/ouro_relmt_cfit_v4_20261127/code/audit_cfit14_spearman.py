"""CPU-only exact 14-source diagnostic Spearman/Kendall audit comparing CFIT v4 vs v3 baselines."""
import json
from pathlib import Path
import torch

ROOT = Path("/root/autodl-tmp/ouro-v4-cfit-20261126")
V3 = Path("/root/ouro-v3-controllability-20261126/stage4_d3")

def rankdata(xs):
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks

def spearman(xs, ys):
    rx, ry = rankdata(xs), rankdata(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0

def op_cos_sigma(u1, w1, u2, w2, S):
    u1, w1, u2, w2 = u1.double(), w1.double(), u2.double(), w2.double()
    cu = float((u1 @ u2) / (u1.norm() * u2.norm()).clamp_min(1e-12))
    n1 = float(torch.clamp(w1 @ (S @ w1), min=1e-12).sqrt())
    n2 = float(torch.clamp(w2 @ (S @ w2), min=1e-12).sqrt())
    cw = float((w1 @ (S @ w2)) / (n1 * n2))
    return cu * cw

cfit14 = json.loads((ROOT / "cfit_14_features" / "CFIT_14_SOURCE_FEATURES.json").read_text())
audit = json.loads((V3 / "D3A_FORECAST_AUDIT.json").read_text())
lock3 = json.loads((V3 / "FORECAST_LOCK_V3.json").read_text())
ops14 = torch.load(ROOT / "cfit_14_features" / "SOURCE_OPERATORS_14.pt", map_location="cpu", weights_only=False)
sa = torch.load(ROOT / "V4_SELF_ANCHOR.pt", map_location="cpu", weights_only=False)
sd = torch.load(ROOT / "SIGMA_H.pt", map_location="cpu", weights_only=False)
S = sd["sigma_unlab"].double()
ev, ec = torch.linalg.eigh((S + S.T) * 0.5)
S_reg = (ec * (ev.clamp_min(0.0) + float(ev.mean()) * 0.05)) @ ec.T
u_self, w_self = sa["u_self"].double(), sa["w_self"].double()

# Extract ground-truth Cold & D2Warm macro deltas
gt_cold, gt_warm, gt_clz = {}, {}, {}
for src, d in audit["subject_values"].items():
    gt_cold[src] = d["Cold_FixedWork"]["macro_delta_pp"]
    gt_warm[src] = d["D2Warm_FixedWork"]["macro_delta_pp"]
    gt_clz[src] = d["Cold_FixedWork"]["cloze_ood_delta_pp"]

all14 = sorted(gt_cold.keys())
cand12 = [s for s in all14 if s not in ("squad", "gsm8k")]
prosp4 = [s for s in all14 if s in ("piqa", "race", "social_iqa", "squad")]
hist10 = [s for s in all14 if s not in prosp4]

# Compute per-source predictor values
feats = {}
for s in all14:
    st = cfit14["per_source"][s]
    op = ops14[s]
    a_self_o = op_cos_sigma(op["u_ord"], op["w_ord"], u_self, w_self, S_reg)
    a_self_c = op_cos_sigma(op["u_ctx"], op["w_ctx"], u_self, w_self, S_reg)
    p_dec = max(0.0, st["P_deconf_op_sig"])
    m_pres = st["M_pres_ord"]
    o_iii = st["O_III_mc"]
    o_rat = st["O_III_over_II"]
    v3p = lock3["per_source"][s].get("predictors", lock3["per_source"][s])
    feats[s] = {
        "CFIT_P0_Deconf_Pres": p_dec * m_pres,
        "CFIT_P0b_Deconf_Pure": st["P_deconf_op_sig"],
        "CFIT_P0c_Deconf_SqrtPres": p_dec * (m_pres ** 0.5),
        "CFIT_P1_SelfAnchor_Pres": a_self_o * m_pres,
        "CFIT_P1b_SelfAnchor_Pure": a_self_o,
        "CFIT_P2_SelfAnchor_Ctx": a_self_c * m_pres,
        "CFIT_P3_Intrinsic_OIII": st["I_CFIT_intrinsic"],
        "CFIT_P4_OIII_Ratio_Pdec_Mpres": o_rat * p_dec * m_pres,
        "CFIT_P5_OIII_Pure": o_iii,
        "CFIT_P7_Consensus_Sig_Mpres": st["A_cons_ord_sig"] * m_pres,
        "CFIT_P7b_Consensus_Sig_Pure": st["A_cons_ord_sig"],
        "v3_P1_signed_true": v3p.get("P1_signed_true", 0.0),
        "v3_B0_source_only": v3p.get("B0_source_only", 0.0),
        "v3_B1_ordinary_func": v3p.get("B1_ordinary_functional", 0.0),
        "v3_B3_unsigned_ctv": v3p.get("B3_unsigned_ctv", 0.0),
    }

print(f"{'Predictor':30s} | {'All14 Cold':>10s} {'Cand12 Cold':>11s} {'All14 Warm':>10s} {'Prosp4 Cold':>11s} | {'Top1(All14)':>14s}")
print("-" * 98)
out_table = {}
for pn in feats[all14[0]].keys():
    r14c = spearman([feats[s][pn] for s in all14], [gt_cold[s] for s in all14])
    r12c = spearman([feats[s][pn] for s in cand12], [gt_cold[s] for s in cand12])
    r14w = spearman([feats[s][pn] for s in all14], [gt_warm[s] for s in all14])
    r4c = spearman([feats[s][pn] for s in prosp4], [gt_cold[s] for s in prosp4])
    top1 = max(all14, key=lambda s: feats[s][pn])
    regret = max(gt_cold.values()) - gt_cold[top1]
    out_table[pn] = {"all14_cold": r14c, "cand12_cold": r12c, "all14_warm": r14w, "prosp4_cold": r4c, "top1": top1, "top1_regret_pp": regret}
    print(f"{pn:30s} | {r14c:+10.3f} {r12c:+11.3f} {r14w:+10.3f} {r4c:+11.3f} | {top1:>13s} ({regret:.2f})")

print("\nPer-source diagnostic table (sorted by Cold_FixedWork macro delta):")
print(f"{'Source':14s} | {'Cold':>6s} {'Warm':>6s} | {'P_dec':>6s} {'M_pres':>6s} {'P0':>6s} {'A_self':>7s} {'O_III':>6s} {'ConsSig':>7s}")
for s in sorted(all14, key=lambda k: gt_cold[k], reverse=True):
    f = feats[s]
    st = cfit14["per_source"][s]
    print(f"{s:14s} | {gt_cold[s]:+6.2f} {gt_warm[s]:+6.2f} | {st['P_deconf_op_sig']:+6.3f} {st['M_pres_ord']:6.3f} {f['CFIT_P0_Deconf_Pres']:6.3f} {f['CFIT_P1b_SelfAnchor_Pure']:+7.3f} {st['O_III_mc']:6.3f} {st['A_cons_ord_sig']:+7.3f}")

(ROOT / "cfit_14_features" / "CFIT_14_DIAGNOSTIC_SPEARMAN.json").write_text(json.dumps({"table": out_table, "per_source": feats, "gt_cold": gt_cold, "gt_warm": gt_warm}, indent=2) + "\n")
