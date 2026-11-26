"""Close Wave-3 capacity expansion & dual-source poison summaries, then freeze SELECTION_V4.json + FREEZE.json."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/ouro-v4-cfit-20261126")
T5 = ROOT / "exp_t5_mixing"
OFF_ROOT = ROOT / "official_run"

sys.path.insert(0, str(ROOT / "handoff_codex"))
from d2_worker import digest, save_json


def main():
    # 1. Assemble CAPACITY_EXPANSION_DEVELOPMENT.json
    base_p = T5 / "T5_T5_CFIT_Top6_4096.json"
    gated_p = T5 / "capacity_expansion" / "gated_rank1.json"
    ortho_p = T5 / "capacity_expansion" / "ortho_rank2.json"
    for p in (base_p, gated_p, ortho_p):
        assert p.exists(), f"Missing required capacity file: {p}"

    base = json.loads(base_p.read_text(encoding="utf-8"))
    gated = json.loads(gated_p.read_text(encoding="utf-8"))
    ortho = json.loads(ortho_p.read_text(encoding="utf-8"))

    comp = {
        "T5_CFIT_Top6_4096": {
            "n_params": 4096,
            "mean_protected_macro5": base["mean_protected_macro5"],
            "source_sha256": digest(base_p),
        },
        "gated_rank1": {
            "n_params": next(iter(gated["seeds"].values()))["train"].get("n_params", 4098),
            "mean_protected_macro5": gated["mean_protected_macro5"],
            "source_sha256": digest(gated_p),
        },
        "ortho_rank2": {
            "n_params": next(iter(ortho["seeds"].values()))["train"].get("n_params", 8192),
            "mean_protected_macro5": ortho["mean_protected_macro5"],
            "source_sha256": digest(ortho_p),
        },
    }
    best_exp = max(("gated_rank1", "ortho_rank2"), key=lambda k: comp[k]["mean_protected_macro5"])
    gain_vs_r1 = (comp[best_exp]["mean_protected_macro5"] - comp["T5_CFIT_Top6_4096"]["mean_protected_macro5"]) * 100
    exp_summary = {
        "certificate_sha256": digest(T5 / "CAPACITY_CERTIFICATE.json"),
        "protocol_sha256": digest(T5 / "capacity_expansion" / "PROTOCOL.json"),
        "comparison": comp,
        "best_expansion_family": best_exp,
        "best_expansion_minus_rank1_pp": gain_vs_r1,
        "rank1_retained_for_official": True,
        "official_test_used": False,
    }
    save_json(T5 / "CAPACITY_EXPANSION_DEVELOPMENT.json", exp_summary)
    print("CAPACITY_EXPANSION_DEVELOPMENT:", json.dumps(exp_summary, indent=2), flush=True)

    # 2. Dual-source poison summary (if both finished)
    dp_raw_p = T5 / "dual_source_poison" / "raw.json"
    dp_r3_p = T5 / "dual_source_poison" / "regime3.json"
    if dp_raw_p.exists() and dp_r3_p.exists():
        dp_raw = json.loads(dp_raw_p.read_text(encoding="utf-8"))
        dp_r3 = json.loads(dp_r3_p.read_text(encoding="utf-8"))
        dp_sum = {
            "protocol_sha256": digest(T5 / "dual_source_poison" / "PROTOCOL.json"),
            "raw_protected_macro5": dp_raw["mean_protected_macro5"],
            "regime3_protected_macro5": dp_r3["mean_protected_macro5"],
            "rescue_gain_pp": (dp_r3["mean_protected_macro5"] - dp_raw["mean_protected_macro5"]) * 100,
        }
        save_json(T5 / "DUAL_SOURCE_POISON_SUMMARY.json", dp_sum)
        print("DUAL_SOURCE_POISON_SUMMARY:", json.dumps(dp_sum, indent=2), flush=True)

    # 3. Freeze SELECTION_V4.json + .sha256 + FREEZE.json for all 3 aliases!
    py = sys.executable
    off_script = str(ROOT / "handoff_codex" / "v4_official.py")
    subprocess.run([py, "-u", off_script, "--work", str(ROOT), "--root", str(OFF_ROOT), "--phase", "selection"], check=True)
    for alias in ("Best_Rank1_4096", "Best_NonARC_4096", "Best_ZeroLabel_4096"):
        subprocess.run([py, "-u", off_script, "--root", str(OFF_ROOT), "--phase", "freeze", "--alias", alias], check=True)
    print("ALL_SELECTION_AND_FREEZE_DONE", flush=True)


if __name__ == "__main__":
    main()
