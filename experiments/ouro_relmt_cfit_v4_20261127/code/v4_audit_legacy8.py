"""Aggregate complete 8-Benchmark (43,769-question) Official Results + 10,000-draw Paired Bootstrap CIs & McNemar p-values!"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path("/root/autodl-tmp/ouro-v4-cfit-20261126")
OFF_ROOT = ROOT / "official_run"
sys.path.insert(0, str(ROOT / "handoff_codex"))

import aggregate_v3 as pairing
import official_v3 as official
from aggregate_full import legacy_rows
from d2_worker import digest, save_json
from run_full import LEGACY as N_LEGACY
from v4_common import E0_CONFIG_PATH

ALIASES = ("Best_Rank1_4096", "Best_NonARC_4096", "Best_ZeroLabel_4096")
LEGACY = ("arc_challenge", "arc_easy", "sciq", "mmlu", "hellaswag", "winogrande", "piqa", "mmlu_pro")
BASELINES = official.BASELINE_ARMS
official.ALIASES = ALIASES
pairing.ALIASES = ALIASES


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    seal = OFF_ROOT / "SELECTION_V4.json"
    auth = official.checked_selection(seal)
    launch = SimpleNamespace(selection=seal, config=E0_CONFIG_PATH, out=OFF_ROOT / "official")
    out = launch.out
    paired, table = {}, {}

    for alias in ALIASES:
        launch.alias = alias
        runtime = official.arguments(launch, auth)
        runtime.release = runtime.panel.parents[1]
        runtime.minimal = runtime.minimal_code
        receipt = read(runtime.out / "FREEZE.json")
        methods = {}
        for task in LEGACY:
            n = N_LEGACY[task]
            mode = "generated" if task == "mmlu_pro" else "candidate"
            rows, clusters = legacy_rows(runtime, task, n)
            report = pairing.summarize(task, rows, clusters)
            methods[f"{task}/{mode}"] = report
            metric = receipt["primary"][task]
            scores = {arm: report["paired"][arm][metric]["baseline"] for arm in BASELINES}
            current = table.setdefault(task, dict(n=n, metric=metric, scores=scores))
            for arm in BASELINES:
                if abs(scores[arm] - current["scores"][arm]) > 1e-12:
                    raise ValueError(f"Historical baseline score mismatch on {task}/{arm}")
            current["scores"][alias] = report["paired"]["native"][metric]["new"]
        paired[alias] = methods

    model_names = (*BASELINES, *ALIASES)
    macro_8 = {arm: float(np.mean([table[t]["scores"][arm] for t in LEGACY])) for arm in model_names}
    macro_7_no_mmlu = {arm: float(np.mean([table[t]["scores"][arm] for t in LEGACY if t != "mmlu"])) for arm in model_names}

    # Add best_loopcd row (per-task max of fixed vs adaptive)
    best_loopcd_scores = {t: max(table[t]["scores"]["fixed"], table[t]["scores"]["adaptive"]) for t in LEGACY}
    macro_8["loopcd_best"] = float(np.mean(list(best_loopcd_scores.values())))

    save_json(out / "FINAL_8_OFFICIAL_TABLE.json", dict(
        table=table, macro_8=macro_8, macro_7_no_mmlu=macro_7_no_mmlu,
        historical_count=sum(N_LEGACY.values()), selection_sha256=digest(seal),
    ))
    save_json(out / "PAIRED_SIGNIFICANCE_LEGACY8.json", dict(
        models=paired, selection_sha256=digest(seal), paired_draws=10000,
    ))

    # Print readable table
    hdr = ["Benchmark", "N", "native", "fixed", "adaptive", "mixed", "t1", "rank8", *ALIASES]
    print("| " + " | ".join(hdr) + " |")
    print("| " + " | ".join(["---"] * len(hdr)) + " |")
    for t in LEGACY:
        sc = table[t]["scores"]
        vals = [t, str(table[t]["n"]), *(f"{sc[a]*100:.2f}%" for a in hdr[2:])]
        print("| " + " | ".join(vals) + " |")
    print("| " + " | ".join(["**Macro-8**", str(sum(N_LEGACY.values())), *(f"**{macro_8[a]*100:.2f}%**" for a in hdr[2:])]) + " |")
    print("| " + " | ".join(["**Macro-7 (ex MMLU)**", "29727", *(f"**{macro_7_no_mmlu[a]*100:.2f}%**" for a in hdr[2:])]) + " |")
    print(f"\nLoopCD per-task best Macro-8: {macro_8['loopcd_best']*100:.2f}%")

    print("\n=== PAIRED SIGNIFICANCE vs T1 & NATIVE ===")
    for alias in ALIASES:
        print(f"\n[{alias}] ({auth['candidates'][alias]['selected_arm']}):")
        for t in LEGACY:
            mode = "generated" if t == "mmlu_pro" else "candidate"
            metric = table[t]["metric"]
            rep = paired[alias][f"{t}/{mode}"]["paired"]
            dn = rep["native"][metric]
            dt = rep["t1"][metric]
            print(f"  {t:14s}: {dn['new']*100:5.2f}% | vs Nat {dn['delta']*100:+5.2f}pp (p={dn.get('mcnemar_p', dn.get('p_value', 1.0)):.2e}) | vs T1 {dt['delta']*100:+5.2f}pp (p={dt.get('mcnemar_p', dt.get('p_value', 1.0)):.2e})")
    print("V4_LEGACY8_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
