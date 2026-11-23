"""Compare five equal-budget training sources after their second confirmation lock."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch

from analyze_transfer import differences, jsonl


SOURCES = ("arc", "csqa", "obqa", "qasc", "mixed")
SEEDS = (261123137, 261123271)
CONTINUOUS_FEATURES = ("signed_source_moment", "clipped_proxy", "signed_mean_gradient",
                       "target_train_moment", "target_train_clipped_proxy")
TASKS = ("arc", "csqa")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rho(x, y):
    r = float(spearmanr(x, y).statistic)
    return r if np.isfinite(r) else None


def vec(net):
    state = net["state"]
    u, w = state["out.weight"].flatten(), state["encode.weight"].flatten()
    return u / u.norm(), w / w.norm()


def transfer():
    root = Path("results")
    forecast = read(root/"forecast_round2/PREDICTIONS.json")["predictions"]
    outcomes, joint, factors = [], {}, {}
    rng = np.random.default_rng(261123713)
    full_details = {}
    for source in SOURCES:
        for seed in SEEDS:
            suite = "depth" if source == "arc" and seed == SEEDS[0] else "source"
            prefix = f"{source}/{suite}/s{seed}"
            work = root/"HOLDOUT"/prefix
            confirm = work/"confirm/r2"
            summary = read(confirm/"SUMMARY.json")
            raw_cost = read(work/"k234/TRAIN.json")
            factors[(source, seed)] = vec(torch.load(work/"k234/controller.pt", weights_only=True))
            full_details[f"{source}/{seed}"] = dict(training_cost={k:raw_cost[k] for k in
                                                      ("wall_seconds","padded_input_token_loops","scored_target_tokens")})
            for target in TASKS:
                name = f"{target}_round2_confirm"
                rows = jsonl(confirm/f"{name}_PREDICTIONS.jsonl")
                count = len(rows)
                if count != (70 if target == "arc" else 299):
                    raise ValueError(f"Unexpected {target} size")
                by_depth = []
                for k in range(2, 7):
                    key = f"k234_k{k}"
                    forecast_cell = forecast[f"{prefix}/k234"][target][f"h{k-1}_m1"]
                    item = summary["evaluation"][name]["results"][key]["statistics"]
                    actual = differences(rows, f"native_k{k}", key)
                    by_depth.append(actual)
                    result = dict(source=source, seed=seed, target=target, n=count, depth=k,
                                  norm_acc=item["acc_norm"], raw_acc=item["acc"], log_odds=float(actual[:,0].mean()),
                                  forecast={f:forecast_cell[f] for f in CONTINUOUS_FEATURES},
                                  normalized_margin=float(actual[:,1].mean()))
                    outcomes.append(result)
                joint[(source, seed, target)] = np.stack(by_depth).mean(0)
    seed_align = []
    for source in SOURCES:
        u, w = factors[(source,SEEDS[0])]
        v, z = factors[(source,SEEDS[1])]
        sign = 1 if float(w@z)>=0 else -1
        seed_align.append(dict(source=source, read_cos=float((w@z)*sign), write_cos=float((u@v)*sign)))
    across_align = []
    for i, source in enumerate(SOURCES):
        for other in SOURCES[i+1:]:
            for seed in SEEDS:
                u,w=factors[(source,seed)]
                v,z=factors[(other,seed)]
                sign = 1 if float(w@z)>=0 else -1
                across_align.append(dict(source=source, other=other, seed=seed, read_cos=float((w@z)*sign),
                                         write_cos=float((u@v)*sign)))

    draws = {target:rng.integers(size, size=(4000,size)) for target,size in (("arc",70),("csqa",299))}
    theoretical = {f:[] for f in CONTINUOUS_FEATURES}
    source_summary = []
    for source in SOURCES:
        values = {}
        for target in TASKS:
            q = np.stack([joint[(source,seed,target)] for seed in SEEDS])
            avg = q.mean(0)
            values[f"{target}_avg_acc_norm_pp"] = float(100*avg[:,3].mean())
            values[f"{target}_avg_acc_norm_pp_ci95"] = np.quantile(100*avg[draws[target],3].mean(1),[.025,.975]).tolist()
            values[f"{target}_avg_log_odds"] = float(avg[:,0].mean())
            values[f"{target}_by_seed_acc_norm_pp"] = (100*q[:,:,3].mean(1)).tolist()
        by_source = [row for row in outcomes if row["source"]==source]
        for f in CONTINUOUS_FEATURES:
            pred = {target:float(np.mean([r["forecast"][f] for r in by_source if r["target"]==target]))
                    for target in TASKS}
            theoretical[f].append(dict(source=source, target_scores=pred))
        values.update(source=source, maximin_acc_norm_pp=min(values["arc_avg_acc_norm_pp"], values["csqa_avg_acc_norm_pp"]))
        source_summary.append(values)
    theory_summary = {}
    for f in CONTINUOUS_FEATURES:
        predictions = [r["forecast"][f] for r in outcomes]
        sign = lambda v: np.sign(np.asarray(v))
        eq = sign(predictions) == sign([r["log_odds"] for r in outcomes])
        theory_summary[f] = dict(
            pooled_sign=float(eq.mean()), pooled_log_odds_rank=rho(predictions,[r["log_odds"] for r in outcomes]),
            pooled_acc_norm_rank=rho(predictions,[r["norm_acc"]["mean_gain"] for r in outcomes]),
            by_target_rank={t:rho([r["forecast"][f] for r in outcomes if r["target"]==t],
                                  [r["log_odds"] for r in outcomes if r["target"]==t]) for t in TASKS},
            source_scores=theoretical[f],
        )
    realized = sum(r["log_odds"]>0 for r in outcomes)
    out = root/"ROUND2_AUDIT"
    out.mkdir(parents=True, exist_ok=True)
    result=dict(cell_count=len(outcomes), always_positive_sign_rate=realized/len(outcomes),
                source_aggregate=source_summary, predictor_audit=theory_summary, seeds=SEEDS,
                per_cell=outcomes, training=full_details, within_source_seed_alignment=seed_align,
                between_source_alignment=across_align)
    (out/"RESULTS.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
                                    encoding="utf-8")
    print(json.dumps(source_summary,ensure_ascii=False),flush=True)
    print(json.dumps({k:{f:v for f,v in rec.items() if f!="source_scores"}
                      for k,rec in theory_summary.items()},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    transfer()
