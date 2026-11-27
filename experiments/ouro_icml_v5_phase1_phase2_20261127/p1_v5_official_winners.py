"""Run exact 43,769-question Official 8-Benchmark LEGACY suite for our 3 ICML Phase 1 Breakout Winners:
1. FixW_MeanMu_2048       (ONLY 2,048 trainable params, w=mu/||mu|| frozen from WikiText, 0 human labels!)
2. Pareto_ZeroNLL_4096    (4,096 params, 0 human labels, dNLL = +0.0030 [PPL x1.003], MMLU above Native!)
3. ZeroQA_WikiText_4096   (4,096 params, 0 external QA datasets, 0 human labels, 0 external LLMs — pure WikiText!)
Uses the exact unmodified official_v3 + aggregate_v3 pipeline with SHA-256 locks and 10,000-replicate paired bootstrap.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/ouro-v4-cfit-20261126")
import minimal
import aggregate_v3 as pairing
import official_v3 as official
from d2_worker import digest, save_json
from v4_common import E0_CONFIG_PATH, save_rank1_controller

P1_ROOT = Path("/root/autodl-tmp/ouro-v5-icml-phase1-20261127")
OUT_ROOT = P1_ROOT / "official_v5_winners"

ALIASES = ("FixW_MeanMu_2048", "Pareto_ZeroNLL_4096", "ZeroQA_WikiText_4096")
LEGACY = ("arc_challenge", "arc_easy", "sciq", "mmlu", "hellaswag", "winogrande", "piqa", "mmlu_pro")
BASELINES = official.BASELINE_ARMS
official.ALIASES = ALIASES
pairing.ALIASES = ALIASES


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def copy_pinned(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        part = target.with_suffix(".partial")
        shutil.copyfile(source, part)
        part.replace(target)
    pin = digest(target)
    if pin != digest(source):
        raise ValueError(f"Copied inputs differ: {target}")
    return pin


def convert_fixed_w_to_minimal(src_pt: Path, dst_pt: Path):
    """Convert FixedWController (w_fixed buffer + u param, 2,048 trainable) to bit-exact Minimal('linear',1,0.45)."""
    obj = torch.load(src_pt, map_location="cpu", weights_only=False)
    st = obj["state"]
    w = st["w_fixed"].float().flatten()
    u = st["u"].float().flatten()
    net = minimal.Minimal("linear", 1, 0.45)
    with torch.no_grad():
        net.encode.weight.copy_(w.view(1, -1))
        net.out.weight.copy_(u.view(-1, 1))
    # Verify bit-exact equivalence on random boundary state
    torch.manual_seed(12345)
    h_test = torch.randn(4, 16, w.numel())
    scale = minimal.rms(h_test)
    proj = torch.einsum("...d,d->...", h_test, w)
    x_ref = proj.unsqueeze(-1) * u
    ref = x_ref * (0.45 * scale / minimal.rms(x_ref)).clamp_max(1.0)
    got = net.delta(h_test)
    max_diff = float((ref - got).abs().max())
    assert max_diff < 1e-6, f"FixedW conversion mismatch: {max_diff}"
    save_rank1_controller(dst_pt, net, {"converted_from": str(src_pt), "trainable_params": 2048, "max_diff": max_diff})
    print(f"  Converted FixW_MeanMu_2048 -> {dst_pt} (max_diff={max_diff:.2e}, trainable=2048)", flush=True)


def pick_best_zeroqa_ckpt():
    """Select the highest-scoring NLL-safe (dNLL <= 0.10) pure-WikiText checkpoint from CSBP_V2_ABLATION.json."""
    csbp_json = P1_ROOT / "track_a_csbp_v2" / "CSBP_V2_ABLATION.json"
    d = read(csbp_json)
    cands = []
    for arm_id, rec in d["arms"].items():
        if rec["n_params"] != 4096:
            continue
        dnll = float(rec["unlabeled"]["delta_nll"])
        if dnll <= 0.10:
            cands.append((float(rec["delta_macro5_pp"]), -dnll, arm_id, rec))
    if not cands:
        # fallback to best overall pure-WikiText
        for arm_id, rec in d["arms"].items():
            if rec["n_params"] == 4096:
                cands.append((float(rec["delta_macro5_pp"]), -float(rec["unlabeled"]["delta_nll"]), arm_id, rec))
    cands.sort(reverse=True)
    best_dm5, _, best_id, best_rec = cands[0]
    ckpt = P1_ROOT / "track_a_csbp_v2" / "weights" / f"{best_id}.pt"
    assert ckpt.exists(), f"Missing CSBP checkpoint {ckpt}"
    print(f"  Selected ZeroQA_WikiText_4096 = {best_id} (dMac5={best_dm5:+.2f} pp, dNLL={best_rec['unlabeled']['delta_nll']:+.4f})", flush=True)
    return ckpt, csbp_json, best_id, best_dm5


def seal_selection():
    seal = OUT_ROOT / "SELECTION_V5_WINNERS.json"
    sig = seal.with_suffix(".sha256")
    if seal.is_file():
        official.checked_selection(seal)
        return seal
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. FixW_MeanMu_2048 (best seed on dev: s26112702, dMac5=+8.76 pp, MMLU=68.39%)
    fixw_src = P1_ROOT / "track_b_and_theory" / "weights" / "S1_ZeroLabel__FixW_MeanMu_2048.s26112702.pt"
    fixw_conv = OUT_ROOT / "converted" / "FixW_MeanMu_2048.minimal.pt"
    convert_fixed_w_to_minimal(fixw_src, fixw_conv)
    p2_json = P1_ROOT / "track_b_and_theory" / "PART2_FIXED_W_AND_MECHANISM.json"

    # 2. Pareto_ZeroNLL_4096 (best dNLL<=0.01 seed-mean group: pareto_lam4.00.s26112702, dMac5=+6.70 pp, dNLL=+0.0030)
    pareto_src = P1_ROOT / "track_b_and_theory" / "weights" / "pareto_lam4.00.s26112702.pt"
    assert pareto_src.exists()

    # 3. ZeroQA_WikiText_4096 (best NLL-safe pure WikiText CSBP winner)
    zeroqa_src, csbp_json, zeroqa_id, zeroqa_dm5 = pick_best_zeroqa_ckpt()

    roster = {
        "FixW_MeanMu_2048": (fixw_conv, p2_json, "S1_ZeroLabel__FixW_MeanMu_2048.s26112702", 2048),
        "Pareto_ZeroNLL_4096": (pareto_src, p2_json, "pareto_lam4.00.s26112702", 4096),
        "ZeroQA_WikiText_4096": (zeroqa_src, csbp_json, zeroqa_id, 4096),
    }

    final = {}
    for alias, (ck, dev_json, tag, n_params) in roster.items():
        rec = dict(selected_arm=tag, n_params=n_params)
        for kind, src in (("controller", ck), ("development", dev_json), ("training", dev_json)):
            dst = OUT_ROOT / "selection_inputs" / alias / (kind + src.suffix)
            rec[kind] = str(dst)
            rec[kind + "_sha256"] = copy_pinned(src, dst)
        rec.update(historical=rec["development"], historical_sha256=rec["development_sha256"])
        final[alias] = rec

    val = dict(
        status="FROZEN_BEFORE_OFFICIAL_EVALUATION",
        development_only=True,
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        candidates=final,
        env_sha256=read(E0_CONFIG_PATH)["approved_env_sha256"],
    )
    save_json(seal, val)
    sig.write_text(f"{digest(seal)}  {seal.name}\n", encoding="utf-8")
    official.checked_selection(seal)
    print("V5_WINNERS_SELECTION_SEALED", digest(seal), flush=True)
    return seal


def run_all_official(seal: Path):
    for alias in ALIASES:
        official.main(SimpleNamespace(
            selection=seal, config=E0_CONFIG_PATH, out=OUT_ROOT / "official",
            alias=alias, phase="freeze", task=None, arm="new",
        ))
    t_all = time.perf_counter()
    for alias in ALIASES:
        for task in LEGACY:
            t0 = time.perf_counter()
            print(f"\n>>> OFFICIAL START {alias} / {task}", flush=True)
            official.main(SimpleNamespace(
                selection=seal, config=E0_CONFIG_PATH, out=OUT_ROOT / "official",
                alias=alias, phase="legacy", task=task, arm="new",
            ))
            print(f"<<< OFFICIAL DONE  {alias} / {task} in {time.perf_counter()-t0:.1f}s", flush=True)
    print(f"\nALL_3_WINNERS_OFFICIAL_DONE in {(time.perf_counter()-t_all)/60:.1f} min", flush=True)


def aggregate_official(seal: Path):
    from aggregate_full import legacy_rows
    from run_full import LEGACY as N_LEGACY

    auth = official.checked_selection(seal)
    launch = SimpleNamespace(selection=seal, config=E0_CONFIG_PATH, out=OUT_ROOT / "official")
    out = launch.out
    paired, table = {}, {}
    for alias in auth["candidates"]:
        launch.alias = alias
        runtime = official.arguments(launch, auth)
        runtime.release = runtime.panel.parents[1]
        runtime.minimal = runtime.minimal_code
        receipt = read(runtime.out / "FREEZE.json")
        methods = {}
        for task, n in N_LEGACY.items():
            rows, clusters = legacy_rows(runtime, task, n)
            report = pairing.summarize(task, rows, clusters)
            methods[task] = report
            metric = receipt["primary"][task]
            scores = {arm: report["paired"][arm][metric]["baseline"] for arm in BASELINES}
            cur = table.setdefault(task, dict(n=n, metric=metric, scores=scores))
            cur["scores"][alias] = report["paired"]["native"][metric]["new"]
        paired[alias] = methods

    model_names = (*BASELINES, *ALIASES)
    macro = {arm: float(np.mean([table[t]["scores"][arm] for t in LEGACY])) for arm in model_names}
    save_json(out / "V5_WINNERS_OFFICIAL8_TABLE.json", dict(table=table, macro_8=macro, total_questions=sum(N_LEGACY.values()), selection_sha256=digest(seal)))
    save_json(out / "V5_WINNERS_OFFICIAL8_PAIRED.json", dict(models=paired, selection_sha256=digest(seal), paired_draws=10000))

    print("\n================ V5 WINNERS 43,769-QUESTION OFFICIAL 8-BENCHMARK TABLE ================", flush=True)
    for t in LEGACY:
        sc = table[t]["scores"]
        print(f"  {t:14s}: Native={sc['native']*100:5.2f}  T1={sc['t1']*100:5.2f}  | FixW2048={sc['FixW_MeanMu_2048']*100:5.2f}  ParetoZeroNLL={sc['Pareto_ZeroNLL_4096']*100:5.2f}  ZeroQA_Wiki={sc['ZeroQA_WikiText_4096']*100:5.2f}", flush=True)
    print(f"  {'MACRO-8':14s}: Native={macro['native']*100:5.2f}  T1={macro['t1']*100:5.2f}  | FixW2048={macro['FixW_MeanMu_2048']*100:5.2f}  ParetoZeroNLL={macro['Pareto_ZeroNLL_4096']*100:5.2f}  ZeroQA_Wiki={macro['ZeroQA_WikiText_4096']*100:5.2f}", flush=True)
    print("V5_WINNERS_OFFICIAL8_AGGREGATED", flush=True)


def main():
    seal = seal_selection()
    run_all_official(seal)
    aggregate_official(seal)


if __name__ == "__main__":
    main()
