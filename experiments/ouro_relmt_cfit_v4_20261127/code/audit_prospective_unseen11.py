"""Evaluate pre-registered FORECAST_LOCK_V4_PROSPECTIVE.json against all 11 trained unseen sources.
Verifies SHA-256 lock integrity, computes Spearman rho, Kendall tau, Top-1 regret, and saves
PROSPECTIVE_UNSEEN11_AUDIT.json!
"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path


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


def kendall_tau(xs, ys):
    conc = disc = 0
    for i, j in combinations(range(len(xs)), 2):
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        if dx * dy > 0:
            conc += 1
        elif dx * dy < 0:
            disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/root/autodl-tmp/ouro-v4-cfit-20261126"))
    args = ap.parse_args()

    pdir = args.out_root / "prospective_unseen"
    lock_path = pdir / "FORECAST_LOCK_V4_PROSPECTIVE.json"
    sha_path = pdir / "FORECAST_LOCK_V4_PROSPECTIVE.sha256"
    expected_sha = sha_path.read_text(encoding="utf-8").strip().split()[0]
    actual_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    assert actual_sha == expected_sha, f"Lock SHA mismatch: {actual_sha} != {expected_sha}"

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    sources = lock["sources"]
    trained_dir = pdir / "trained"

    actual_mac5, actual_clz3 = {}, {}
    missing = []
    for s in sources:
        fp = trained_dir / f"TRAIN_{s}.json"
        if not fp.exists():
            missing.append(s)
            continue
        d = json.loads(fp.read_text(encoding="utf-8"))
        if len(d.get("arms", {})) < 2:
            missing.append(f"{s}(incomplete:{len(d.get('arms', {}))})")
            continue
        actual_mac5[s] = d["mean_delta_macro5_pp"]
        actual_clz3[s] = d["mean_delta_cloze3_pp"]

    print(f"LOCK SHA-256 VERIFIED: {actual_sha}")
    print(f"Completed unseen sources: {len(actual_mac5)}/{len(sources)} | Missing: {missing}")
    present = sorted(actual_mac5.keys())
    present_no_swag = [s for s in present if s != "swag"]

    pred_names = list(lock["per_source"][present[0]]["predictors"].keys())
    table = {}
    print(f"\n{'Predictor':38s} | {'All rho':>7s} {'All tau':>7s} {'NoSwag rho':>10s} {'Clz3 rho':>8s} | {'Top-1 (Regret)':>22s}")
    print("-" * 105)
    for pn in pred_names:
        xs = [lock["per_source"][s]["predictors"][pn] for s in present]
        ys = [actual_mac5[s] for s in present]
        yc = [actual_clz3[s] for s in present]
        r_all = spearman(xs, ys)
        t_all = kendall_tau(xs, ys)
        r_clz = spearman(xs, yc)
        r_ns = spearman([lock["per_source"][s]["predictors"][pn] for s in present_no_swag],
                        [actual_mac5[s] for s in present_no_swag]) if len(present_no_swag) >= 3 else 0.0
        top1 = max(present, key=lambda s: lock["per_source"][s]["predictors"][pn])
        regret = max(actual_mac5.values()) - actual_mac5[top1]
        table[pn] = {
            "spearman_macro5": r_all, "kendall_macro5": t_all,
            "spearman_macro5_excl_swag": r_ns, "spearman_cloze3": r_clz,
            "top1_source": top1, "top1_actual_pp": actual_mac5[top1], "top1_regret_pp": regret,
        }
        print(f"{pn:38s} | {r_all:+7.3f} {t_all:+7.3f} {r_ns:+10.3f} {r_clz:+8.3f} | {top1:>13s} ({regret:5.2f}pp)")

    print("\nActual Unseen Prospective Ranking (sorted by Cold-840 Macro5 delta):")
    print(f"{'Source':15s} | {'Macro5':>7s} {'Clz3':>7s} | {'P_dec':>6s} {'M_pres':>6s} {'P0':>6s} {'A_self':>7s} {'ConsSig':>7s} {'v3_B3':>7s}")
    for s in sorted(present, key=lambda k: actual_mac5[k], reverse=True):
        diag = lock["per_source"][s]["diagnostics"]
        pr = lock["per_source"][s]["predictors"]
        print(f"{s:15s} | {actual_mac5[s]:+7.2f} {actual_clz3[s]:+7.2f} | {diag['P_deconf_op_sig']:+6.3f} {diag['M_pres_ord']:6.3f} {pr['CFIT_P0_Deconf_Pres']:6.3f} {diag['a_self_ord_sig']:+7.3f} {diag['med_cons_ord_sig']:+7.3f} {pr['v3_B3_unsigned_ctv']:7.2f}")

    out_path = pdir / "PROSPECTIVE_UNSEEN11_AUDIT.json"
    out_path.write_text(json.dumps({
        "lock_sha256": actual_sha,
        "n_completed": len(present),
        "missing": missing,
        "actual_macro5_pp": actual_mac5,
        "actual_cloze3_pp": actual_clz3,
        "predictor_audit": table,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nSAVED {out_path}")


if __name__ == "__main__":
    main()
