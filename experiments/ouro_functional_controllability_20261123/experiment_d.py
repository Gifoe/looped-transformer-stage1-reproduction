"""Signed K4 dose response and matched-RMS propagation for Ouro."""

import argparse
from pathlib import Path

import numpy as np
import torch

from c_common import (
    BRANCHES, METRICS, arguments, bootstrap, candidate_trace, cosine, digest, forward,
    hash_key, load_runtime, make_manifest, paired, population, profile, question_trace,
    replay_batches, rms, save, score,
)

SMALL = .025
SENSITIVITY = .05
DOSES = (-.70, -.45, -.25, -.10, -SENSITIVITY, -SMALL, 0, SMALL, SENSITIVITY, .10, .25, .45, .70)
HELD_OUT = (-.70, -.45, -.25, -.10, .10, .25, .45, .70)
EARLY_DOSES = (-SMALL, SMALL, .45)


def tag(turn, dose):
    return "native" if dose == 0 else f"turn{turn}/{dose:+.3f}"


def cases():
    return [(turn, dose) for turn in (1, 2, 3)
            for dose in (EARLY_DOSES if turn < 3 else DOSES) if dose != 0]


def trajectory(active, native, anchor):
    edit = active["edits"]
    stats = {
        "anchor_rms": rms(native["raw"][anchor]),
        "edit_abs_rms": rms(edit),
        "edit_relative_rms": rms(edit) / rms(native["raw"][anchor]).clamp_min(1e-20),
        "visit_same_anchor_max_error": (active["raw"][anchor] - native["raw"][anchor]).abs().amax(dim=-1),
    }
    for turn in range(anchor, 5):
        before = native["normalized"][turn]
        diff = active["normalized"][turn] - before
        stats[f"turn{turn}/normalized_abs_rms"] = rms(diff)
        stats[f"turn{turn}/normalized_relative_rms"] = rms(diff) / rms(before).clamp_min(1e-20)
        stats[f"turn{turn}/normalized_cos_to_edit"] = cosine(diff, edit)
        if turn > anchor:
            raw = native["raw"][turn]
            propagated = active["raw"][turn] - raw
            stats[f"turn{turn}/raw_abs_rms"] = rms(propagated)
            stats[f"turn{turn}/raw_relative_rms"] = rms(propagated) / rms(raw).clamp_min(1e-20)
            stats[f"turn{turn}/raw_cos_to_edit"] = cosine(propagated, edit)
    for (turn, branch), deltas in active["changes"].items():
        stats[f"turn{turn}/{branch}_write_delta_rms"] = deltas
    return stats


def current_batch(archive, model, axis, packed, subset):
    branches = {(turn, layer, branch) for turn in (2, 3, 4)
                for layer in range(24) for branch in BRANCHES}
    hidden, base = forward(model, packed, branch_layers=branches)
    scores = {"native": score(archive, model, packed, subset, hidden)}
    traces = {}
    for turn, dose in cases():
        edit = dose * rms(base["raw"][turn]).unsqueeze(-1) * axis
        probe = dose == .45 or (turn == 3 and dose in (-.70, -.45, .70))
        active_layers = {key for key in branches if key[0] > turn} if probe else ()
        hidden, active = forward(model, packed, turn, edit, branch_layers=active_layers,
                                 reference=base, trace_turns=(2, 3, 4))
        if not torch.equal(base["raw"][turn], active["raw"][turn]):
            raise RuntimeError("A dose changed the native state before its injection")
        name = tag(turn, dose)
        scores[name] = score(archive, model, packed, subset, hidden)
        traces[name] = candidate_trace(packed, len(subset), trajectory(active, base, turn))
    return dict(scores=scores, traces=traces)


def responses(info, metric, turn, dose):
    return profile(info, tag(turn, dose), metric) - profile(info, "native", metric)


def derivatives(info, metric, turn, step=SMALL):
    high = responses(info, metric, turn, step)
    low = responses(info, metric, turn, -step)
    return (high - low) / (2 * step), (high + low) / step**2


def error(actual, predicted):
    absolute = np.abs(actual - predicted)
    scale = np.abs(actual).sum()
    nonzero = actual != 0
    return dict(mae=float(absolute.mean()), rmse=float(np.sqrt(np.square(actual - predicted).mean())),
                absolute_error_over_absolute_actual=float(absolute.sum() / scale) if scale else None,
                nonzero_actual=int(nonzero.sum()),
                sign_match_nonzero=float((np.sign(predicted[nonzero]) == np.sign(actual[nonzero])).mean())
                if nonzero.any() else None)


def quadratic_optimum(slope, curvature, limit=.45):
    best_endpoint = np.where(slope >= 0, limit, -limit)
    concave = curvature < 0
    best_endpoint[concave] = np.clip(-slope[concave] / curvature[concave], -limit, limit)
    best_endpoint[(slope == 0) & (curvature == 0)] = 0
    return best_endpoint


def dose_summary(info, sample, metric):
    slopes, curves = derivatives(info, metric, 3)
    alt_slopes, alt_curves = derivatives(info, metric, 3, SENSITIVITY)
    local = quadratic_optimum(slopes, curves)
    ordered = sorted((dose for dose in DOSES if abs(dose) <= .45),
                     key=lambda dose: (abs(dose), dose < 0))
    dose_response = np.stack([responses(info, metric, 3, dose) for dose in ordered])
    best = np.asarray(ordered)[dose_response.argmax(axis=0)]
    best_gain = dose_response.max(axis=0)
    forecasts = {}
    for dose in HELD_OUT:
        actual = responses(info, metric, 3, dose)
        forecasts[tag(3, dose)] = {
            "linear": error(actual, dose * slopes),
            "quadratic": error(actual, dose * slopes + (dose**2 / 2) * curves),
            "linear_larger_calibration": error(actual, dose * alt_slopes),
            "quadratic_larger_calibration": error(actual, dose * alt_slopes + (dose**2 / 2) * alt_curves),
        }
    asymmetry = {str(dose): paired(responses(info, metric, 3, dose) +
                                  responses(info, metric, 3, -dose), sample)
                 for dose in (.10, .25, .45, .70)}
    return dict(forecasts=forecasts, plus_minus_even_response=asymmetry,
                calibration_derivative_pearson=float(np.corrcoef(slopes, alt_slopes)[0, 1]),
                local_optimum=dict(negative=int((local < 0).sum()), zero=int((local == 0).sum()),
                                   positive=int((local > 0).sum()),
                                   absolute_at_cap=int((np.abs(local) == .45).sum())),
                empirical_optimum_at_measured_grid={str(dose): int((best == dose).sum()) for dose in ordered},
                optimum_sign_agreement=float((np.sign(local) == np.sign(best)).mean()),
                optimum_oracle_gain=paired(best_gain, sample),
                per_question=[dict(id=row["id"], derivative=float(slopes[i]), curvature=float(curves[i]),
                                   alternative_derivative=float(alt_slopes[i]),
                                   alternative_curvature=float(alt_curves[i]),
                                   optimum_quadratic=float(local[i]), optimum_grid=float(best[i]),
                                   grid_gain=float(best_gain[i])) for i, row in enumerate(info)])


def norm_predictor(feature, actual, folds):
    result = np.zeros_like(actual)
    for held in (0, 1):
        train = folds != held
        x = np.column_stack((np.ones(len(feature)), feature))
        weights, *_ = np.linalg.lstsq(x[train], actual[train], rcond=None)
        result[~train] = x[~train] @ weights
    return result


def correlation(left, right):
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def horizon(info, sample):
    guess, actual, norm, tracks = {}, {}, {}, {}
    folds = np.empty(len(info), dtype=np.int32)
    order = sorted(range(len(info)), key=lambda i: hash_key("D-norm-feature-crossfit", info[i]["id"]))
    folds[order] = np.arange(len(info)) % 2
    utility = "utility_norm"
    for turn in (1, 2, 3):
        name = tag(turn, .45)
        slope, _ = derivatives(info, utility, turn)
        target = responses(info, utility, turn, .45)
        magnitude = question_trace(info, name, "turn4/normalized_relative_rms")
        tracks[turn] = {key: paired(question_trace(info, name, key), sample)
                        for key in info[0]["trace"][name]}
        guess[turn] = .45 * slope
        actual[turn] = target
        norm[turn] = magnitude
        tracks[turn]["norm_vs_signed_utility_pearson"] = correlation(magnitude, target)
        tracks[turn]["slope_vs_signed_utility_pearson"] = correlation(slope, target)
        tracks[turn]["nonlinear_utility"] = paired(target, sample)
        norm_prediction = norm_predictor(magnitude, target, folds)
        tracks[turn]["norm_only_crossfit"] = error(target, norm_prediction)
        tracks[turn]["signed_slope"] = error(target, guess[turn])
        tracks[turn]["mae_improvement_over_norm_only"] = paired(
            np.abs(target - norm_prediction) - np.abs(target - guess[turn]), sample)
    comparisons = {}
    for early in (1, 2):
        late_gain = actual[3] - actual[early]
        late_norm = norm[3] - norm[early]
        comparisons[f"turn3_minus_turn{early}"] = dict(
            utility=paired(late_gain, sample), norm=paired(late_norm, sample),
            gold_margin=paired(responses(info, "margin", 3, .45) -
                               responses(info, "margin", early, .45), sample),
            gold_margin_norm=paired(responses(info, "margin_norm", 3, .45) -
                                    responses(info, "margin_norm", early, .45), sample),
            utility_and_norm_opposite_direction=int(((late_gain * late_norm) < 0).sum()),
            late_gain_with_smaller_norm=int(((late_gain > 0) & (late_norm < 0)).sum()),
            early_gain_with_smaller_norm=int(((late_gain < 0) & (late_norm > 0)).sum()))
    return dict(trajectory=tracks, comparisons=comparisons)


def collect_metrics(info, sample):
    stats = {}
    for turn, dose in [(3, 0), *cases()]:
        name = tag(turn, dose)
        metrics = {}
        for metric in METRICS:
            values = profile(info, name, metric)
            delta = values - profile(info, "native", metric)
            metrics[metric] = dict(mean=float(values.mean()), paired_native=paired(delta, sample))
            if metric.startswith("acc"):
                metrics[metric].update(count=int(values.sum()), rescue=int((delta > 0).sum()),
                                       damage=int((delta < 0).sum()))
        if name != "native":
            metrics["trace"] = {key: float(question_trace(info, name, key).mean())
                                for key in info[0]["trace"][name]}
        stats[name] = metrics
    return stats


def threshold_summary(info, metric):
    stock = profile(info, "native", metric)
    result = {}
    for sign in (-1, 1):
        changed = np.zeros(len(info), dtype=bool)
        for amplitude in sorted(dose for dose in DOSES if dose > 0):
            cur = profile(info, tag(3, sign * amplitude), metric)
            first = ~changed & (cur != stock)
            result[tag(3, sign * amplitude)] = {
                "first_measured_rescue": int((first & (cur > stock)).sum()),
                "first_measured_damage": int((first & (cur < stock)).sum()),
            }
            changed |= first
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("Expected a positive batch size")
    out = args.out
    archive, model, tokenizer, runtime, axis = load_runtime(args)
    items, info = population(archive, tokenizer, args.corpus, "validation")
    if len(info) != 299:
        raise RuntimeError("D requires all 299 ARC Validation questions")
    make_manifest(out, dict(depth=4, split="ARC-Challenge validation", runtime=runtime,
                            batch_size=args.batch_size, direction_sha256=digest(args.direction),
                            parquet_sha256=digest(args.corpus / "validation.parquet"),
                            scoring_source_sha256=digest(args.archive / "run_arc.py"),
                            script_sha256=digest(__file__),
                            common_sha256=digest(Path(__file__).with_name("c_common.py")),
                            dose_grid=list(DOSES), early_doses=list(EARLY_DOSES),
                            question_id_sha256=hash_key("questions", "/".join(row["id"] for row in info)),
                            positive_sign="Supplied axis sign, unmodified",
                            intervention="All eligible causal prediction positions before pass output RMSNorm",
                            candidate_weighting="Mean over predictor tokens, candidates, then questions",
                            utility="Gold score minus logsumexp of wrong option scores",
                            calibration="Central finite differences around zero in native forward precision",
                            norm_comparison="Prespecified two-fold ID hash crossfit with intercept; norm computed at held-out dose"))
    replay_batches(archive, items, info, args.batch_size, out,
                   lambda packed, subset: current_batch(archive, model, axis, packed, subset))
    sample = bootstrap(info)
    result = dict(questions=len(info), candidates=len(items), conditions=collect_metrics(info, sample),
                  dose_forecasting={metric: dose_summary(info, sample, metric)
                                    for metric in ("utility", "utility_norm", "margin", "margin_norm")},
                  horizon=horizon(info, sample),
                  first_measured_correctness_flip={metric: threshold_summary(info, metric)
                                                   for metric in ("acc", "acc_norm")})
    save(out / "D_SUMMARY.json", result)
    print(f"D complete with {len(info)} questions and {len(result['conditions'])} conditions", flush=True)


if __name__ == "__main__":
    main()
