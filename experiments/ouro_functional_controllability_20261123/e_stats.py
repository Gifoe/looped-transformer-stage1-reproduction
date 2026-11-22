"""Prespecified group-level endpoints and Train-only lexical bias."""
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import binomtest

from e_design import FAMILIES, OPTIONS, VARIANTS, grouped


BOOTSTRAP_SEED = 2026112399
REPLICATES = 10000
LEXICAL_L2 = 0.01


def fit_lexical_bias(rows, native):
    index = {word: index for index, word in enumerate(OPTIONS)}
    features = np.zeros((len(rows), len(index)), dtype=np.float64)
    gold_margins = []
    for i, row in enumerate(rows):
        gold, bad = row["gold_content"], next(w for w in row["choices"]["text"] if w != row["gold_content"])
        gold_position = row["choices"]["text"].index(gold)
        features[i, index[gold]] = 1.0
        features[i, index[bad]] = -1.0
        gold_margins.append(native[i][gold_position] - native[i][1-gold_position])
    margin = np.array(gold_margins)

    def objective(weights):
        predictor = margin + features @ weights
        loss = np.logaddexp(0, -predictor).mean() + 0.5 * LEXICAL_L2 * (weights @ weights)
        gradient = (-expit(-predictor) @ features) / len(rows) + LEXICAL_L2 * weights
        return float(loss), gradient

    result = minimize(objective, np.zeros(len(index)), method="L-BFGS-B", jac=True,
                      options={"ftol": 1e-13, "gtol": 1e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Lexical-calibration fit failed: {result.message}")
    return dict(
        estimator="L2-regularized offset to K4 native candidate log probability, independent Synthetic Train only",
        objective=float(result.fun), uncalibrated_objective=objective(np.zeros(len(index)))[0],
        l2=LEXICAL_L2, n=len(rows), weights=dict(zip(OPTIONS, result.x.tolist(), strict=True)),
    )


def interval(values, bootstrap):
    samples = values[bootstrap].mean(axis=1)
    return [float(bound) for bound in np.quantile(samples, [.025, .975])]


def group_statistics(row_group, method):
    cell = {(r["variant"], r["reversal"], r["remap"]): r for r in row_group}
    variant_rows = {}
    canonical_top, canonical_grades = None, None
    n = len(row_group)
    for variant in VARIANTS:
        grades, margins, top = {}, {}, {}
        for r in (0, 1):
            for m in (0, 1):
                row = cell[variant, r, m]
                score = row["scores"][method]
                gold = row["choices"]["text"].index(row["gold_content"])
                top[r, m] = row["choices"]["text"][int(np.argmax(score))]
                grades[r, m] = top[r, m] == row["gold_content"]
                margins[r, m] = score[gold] - score[1 - gold]
        if variant == "canonical":
            canonical_top, canonical_grades = top, grades
        variant_rows[variant] = dict(
            accuracy=float(np.mean(list(grades.values()))),
            margin=float(np.mean(list(margins.values()))),
            both_directions=float(np.mean([grades[0, m] and grades[1, m] for m in (0, 1)])),
            reversal_equivariance=float(np.mean([top[0, m] != top[1, m] for m in (0, 1)])),
            option_remap_invariance=float(np.mean([top[r, 0] == top[r, 1] for r in (0, 1)])),
            all_four_correct=float(all(grades.values())),
            canonical_content_invariance=float(np.mean([value == canonical_top[key] for key, value in top.items()])),
            canonical_and_variant_both_correct=float(np.mean([value and canonical_grades[key]
                                                             for key, value in grades.items()])),
        )
    if n != 12:
        raise ValueError("Expected 12 held-out variants for each group")
    complete = float(all(variant_rows[v]["all_four_correct"] for v in VARIANTS))
    variant_rows["pooled"] = {
        **{key: float(np.mean([variant_rows[v][key] for v in VARIANTS])) for key in variant_rows["canonical"]},
        "all_twelve_correct": complete,
    }
    return variant_rows


def blocked_values(ordered, details, arm):
    blocks = defaultdict(list)
    for tag, group in ordered:
        pair = tuple(sorted(group[0]["choices"]["text"]))
        blocks[pair].append(details[tag][arm])
    sizes = {len(components) for components in blocks.values()}
    if len(sizes) != 1:
        raise ValueError("Unequal cluster sizes require a weighted bootstrap")
    return {variant: {metric: np.array([
        np.mean([member[variant][metric] for member in block]) for block in blocks.values()])
        for metric in next(iter(details.values()))[arm][variant]}
        for variant in (*VARIANTS, "pooled")}


def semantic_summary(rows):
    methods = list(rows[0]["scores"])
    groups = grouped(rows)
    stats = {tag: {arm: group_statistics(group, arm) for arm in methods} for tag, group in groups.items()}
    summary = {}
    pairs = [("rank8", "native"), ("shared_fixed", "native"), ("shared_learned", "native"),
             ("shared_fixed", "reversed"), ("shared_fixed", "random"),
             ("shared_learned", "lexical"), ("shared_fixed", "lexical")]
    for family in (*FAMILIES, "pooled"):
        ordered = [(tag, group) for tag, group in sorted(groups.items())
                   if family == "pooled" or group[0]["family"] == family]
        block = {arm: blocked_values(ordered, stats, arm) for arm in methods}
        n_blocks = len(block["native"]["pooled"]["accuracy"])
        bootstrap = np.random.default_rng(BOOTSTRAP_SEED).integers(n_blocks, size=(REPLICATES, n_blocks))
        metrics = {}
        for variant in (*VARIANTS, "pooled"):
            metrics[variant] = {}
            for metric in block["native"][variant]:
                measures = {arm: block[arm][variant][metric] for arm in methods}
                row = {arm: dict(mean=float(measures[arm].mean()), ci95=interval(measures[arm], bootstrap))
                       for arm in methods}
                differences = {f"{a}_minus_{b}": measures[a] - measures[b] for a, b in pairs}
                if "fixed_clamped" in methods:
                    differences["axis_by_clamp_interaction"] = (
                        measures["shared_fixed"] - measures["fixed_clamped"]
                        - measures["random"] + measures["random_clamped"]
                    )
                row["paired"] = {name: dict(mean=float(value.mean()), ci95=interval(value, bootstrap))
                                 for name, value in differences.items()}
                metrics[variant][metric] = row
        summary[family] = dict(groups=len(ordered), candidate_pair_clusters=n_blocks, results=metrics)
    pooled = summary["pooled"]["results"]["canonical"]
    bidirectional = pooled["both_directions"]["paired"]
    branch = pooled["margin"]["paired"].get("axis_by_clamp_interaction")
    gate = dict(
        fixed_vs_native_both_directions_ci95=bidirectional["shared_fixed_minus_native"]["ci95"],
        fixed_vs_lexical_both_directions_ci95=bidirectional["shared_fixed_minus_lexical"]["ci95"],
        fixed_vs_random_both_directions_ci95=bidirectional["shared_fixed_minus_random"]["ci95"],
        axis_by_locked_attention_clamp_margin_ci95=branch["ci95"] if branch else None,
    )
    gate["criterion_met"] = all(value is not None and value[0] > 0 for value in gate.values())
    return dict(
        primary=dict(variant="canonical", family="pooled", outcomes=["margin", "both_directions", "all_four_correct"],
                     design="Candidate-pair blocked bootstrap. Each pair contributes one group in each family."),
        uncertainty=dict(seed=BOOTSTRAP_SEED, replicates=REPLICATES, units="proportions or log-probability differences",
                         interval="Two-sided 95% percentile paired cluster bootstrap, descriptive across endpoints"),
        populations=summary, fixed_axis_semantic_specificity_gate=gate,
    )


def transfer_summary(rows):
    rates = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(rows)
    for k in (1, 2, 3, 4):
        for metric in ("acc", "acc_norm"):
            predictions = {}
            for method in ("native", "rank8", "frozen"):
                scores = np.array([row["scores"][f"{method}_k{k}"] for row in rows])
                if metric == "acc_norm":
                    scores /= np.array([row["characters"] for row in rows])
                predictions[method] = scores.argmax(axis=1) == np.array([row["gold"] for row in rows])
            comparisons = {}
            for base, arm in (("native", "rank8"), ("native", "frozen"), ("rank8", "frozen")):
                old, new = predictions[base], predictions[arm]
                rescue, damage = int((~old & new).sum()), int((old & ~new).sum())
                draws = rng.multinomial(n, [damage/n, (n-rescue-damage)/n, rescue/n], size=REPLICATES)
                comparisons[f"{arm}_vs_{base}"] = dict(
                    rescued=rescue, damaged=damage, delta_pp=100*(rescue-damage)/n,
                    ci95_pp=np.quantile(100*(draws[:, 2]-draws[:, 0])/n, [.025, .975]).tolist(),
                    mcnemar=float(binomtest(rescue, rescue+damage).pvalue) if rescue+damage else 1.0,
                )
            rates[f"{metric}_k{k}"] = dict(
                arms={name: dict(correct=int(mask.sum()), total=n, pct=100*float(mask.mean()))
                      for name, mask in predictions.items()},
                paired=comparisons,
            )
    for contrast in ("rank8_vs_native", "frozen_vs_native", "frozen_vs_rank8"):
        values = [item["paired"][contrast] for item in rates.values()]
        adjusted = 0.0
        for index, row in enumerate(sorted(values, key=lambda value: value["mcnemar"])):
            adjusted = max(adjusted, min(1.0, (len(values)-index)*row["mcnemar"]))
            row["mcnemar_holm_across_metrics_depths"] = adjusted
    return dict(n=n, metrics=rates, uncertainty="Pairwise question bootstrap, 10,000 replicates")
