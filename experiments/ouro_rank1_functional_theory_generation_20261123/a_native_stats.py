"""Summarize native-to-Rank-1 matched question probes and Train-only forecasts."""

from itertools import combinations
import json

import numpy as np


DOSE = .35
SMALL = .025
CELLS = {"physical_to_k1": ("physical21", 1), "k1_to_k2": (1, 2), "k1_to_k3": (1, 3),
         "k1_to_k4": (1, 4), "k2_to_k3": (2, 3), "k2_to_k4": (2, 4), "k3_to_k4": (3, 4)}
COLUMNS = {"features": ("state_rms", "learned_state_cosine", "sensitive_state_cosine", "normalized_rank1_encoder",
                        "state_grad_pc1", "state_grad_pc2", "state_grad_pc3", "state_grad_pc4"),
           "geometry": ("outgoing_relative_rms", "outgoing_learned_cosine", "outgoing_sensitive_cosine",
                        "outgoing_rotation_cosine", "outgoing_energy_in_grad_span8", "state_energy_in_grad_span8"),
           "incoming": ("incoming_relative_rms", "incoming_learned_cosine", "incoming_outgoing_cosine"),
           "treatments": ("actual_edit_relative_rms", "terminal_change_relative_rms", "terminal_natural_step_cosine",
                          "terminal_edit_cosine", "applied_requested_cosine", "applied_signed_learned_axis_ratio")}
FIXED = ("learned", "learned_law", "future_native", "functional", "rotation", "past_native")
COUNT = 10_000
BOOT_SEED = 26112372


def save(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def rms(x):
    return x.square().mean(-1).sqrt()


def unit(x):
    return x / rms(x).unsqueeze(-1).clamp_min(1e-20)


def cosine(a, b):
    return (unit(a) * unit(b)).mean(-1).clamp(-1, 1)


def jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def distribution(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(x.mean()), q05_median_q95=np.quantile(x, [.05, .50, .95]).tolist())


def paired(deltas, boot):
    x = np.asarray(deltas, dtype=float)
    return dict(mean=float(x.mean()), ci95=np.quantile(x[boot].mean(1), [.025, .975]).tolist(),
                positive=int((x > 0).sum()), negative=int((x < 0).sum()), zero=int((x == 0).sum()))


def scores(row, arm, normalize=True):
    z = np.asarray(row["scores"][arm], dtype=float)
    return z / np.asarray(row["characters"]) if normalize else z


def utility(z, gold):
    return z[gold] - np.logaddexp.reduce(np.delete(z, gold))


def profile(rows, arm):
    utility_raw, utility_norm, acc, acc_norm, margins, margins_norm = [], [], [], [], [], []
    for row in rows:
        gold = row["gold"]
        z, z_norm = scores(row, arm, False), scores(row, arm)
        utility_norm.append(utility(z_norm, gold))
        utility_raw.append(utility(z, gold))
        acc_norm.append(int(z_norm.argmax() == gold))
        acc.append(int(z.argmax() == gold))
        margins.append(z[gold] - np.delete(z, gold).max())
        margins_norm.append(z_norm[gold] - np.delete(z_norm, gold).max())
    return {k: np.asarray(x) for k, x in dict(acc=acc, acc_norm=acc_norm, utility=utility_raw,
                     utility_norm=utility_norm, margin=margins, margin_norm=margins_norm).items()}


def summary_columns(rows, name):
    return {name: distribution(col) for name, col in zip(COLUMNS[name], np.asarray([r[name] for r in rows]).T, strict=True)}


def vector_cosine(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / n) if n > 1e-12 else None


def score_change_vectors(rows, arm):
    movement = []
    for row in rows:
        diff = scores(row, arm) - scores(row, "native")
        movement.append(diff - diff.mean())
    return movement


def output_response(rows, arm):
    delta = score_change_vectors(rows, arm)
    report = dict(candidate_vector_norm=distribution([np.linalg.norm(x) for x in delta]))
    if "native_anchor_scores" in rows[0]:
        natural = [scores(row, "native") - np.asarray(row["native_anchor_scores"]) / row["characters"] for row in rows]
        similarities = [vector_cosine(d, x-x.mean()) for d, x in zip(delta, natural, strict=True)]
        nonzero = [x for x in similarities if x is not None]
        report["cosine_to_native_anchor_to_terminal_score_progress"] = dict(valid=len(nonzero), **distribution(nonzero)) if nonzero else None
    return report


def output_agreement(rows, name, boot):
    left, right = [score_change_vectors(rows, arm) for arm in ("learned", name)]
    values = np.asarray([x if x is not None else np.nan for x in map(vector_cosine, left, right)])
    valid = np.isfinite(values)
    if not bool(valid.any()):
        return None
    resampled = values[boot]
    counts = np.isfinite(resampled).sum(axis=1)
    means = np.nansum(resampled[counts > 0], axis=1) / counts[counts > 0]
    return dict(n=int(valid.sum()), quantiles=distribution(values[valid]),
                mean_ci95=np.quantile(means, [.025, .975]).tolist())


def forecast_data(rows):
    feat = np.asarray([row["features"] for row in rows])
    profiles = {name: profile(rows, name)["utility_norm"] for name in ("native", "learned", "learned_small_plus", "learned_small_minus")}
    y = profiles["learned"] - profiles["native"]
    slope = (profiles["learned_small_plus"] - profiles["learned_small_minus"]) / (2*SMALL)
    return feat, y, slope


def ridge(train_x, train_y, validation_x):
    lam = 4.
    average = train_x.mean(0)
    spread = np.maximum(train_x.std(0), 1e-10)
    x = (train_x - average) / spread
    y = train_y - train_y.mean()
    beta = np.linalg.solve(x.T @ x + lam * np.eye(train_x.shape[1]), x.T @ y)
    prediction = train_y.mean() + ((validation_x - average) / spread) @ beta
    return prediction, dict(feature_mean=average.tolist(), feature_sd=spread.tolist(), beta=beta.tolist(),
                            intercept=float(train_y.mean()), penalty=lam)


def validation_forecasts(train, val, boot):
    x, y, slope = forecast_data(train)
    xx, yy, ss = forecast_data(val)
    predictions = {"train_mean": np.full(yy.shape, y.mean()), "local_linear": DOSE*ss}
    fits = {}
    for tag, a, b in (("norm_only", x[:, :1], xx[:, :1]),
                      ("norm_and_encoder_magnitude", np.column_stack((x[:, 0], abs(x[:, 3]))),
                       np.column_stack((xx[:, 0], abs(xx[:, 3])))),
                      ("current_state", x, xx), ("calibrated_slope", slope[:, None], ss[:, None]),
                      ("current_state_and_slope", np.column_stack((x, slope)), np.column_stack((xx, ss)))):
        predictions[tag], fits[tag] = ridge(a, y, b)
    losses = {name: abs(pred - yy) for name, pred in predictions.items()}
    scores = {}
    for method, pred in predictions.items():
        diff = yy - pred
        score = dict(mae=float(np.mean(abs(diff))), rmse=float(np.sqrt(np.mean(diff**2))),
                     pearson=float(np.corrcoef(yy, pred)[0, 1]) if float(np.std(pred)) > 1e-12 else None,
                     sign_agreement_nonzero=float(np.mean(np.sign(pred[yy != 0]) == np.sign(yy[yy != 0])))
                     if bool((yy != 0).any()) else None)
        score["mae_improvement_over_norm_only"] = paired(losses["norm_only"] - losses[method], boot)
        scores[method] = score
    rows = [dict(id=row["id"], actual=float(yy[i]), **{name: float(pred[i]) for name, pred in predictions.items()})
            for i, row in enumerate(val)]
    return dict(n_train=len(y), n_validation=len(yy), predictors=scores, train_fits=fits, predictions=rows,
                target="smooth character-normalized candidate utility change at fixed +0.35 RMS")


def summarize(root, directions):
    result, data = {}, {}
    for key, (anchor, depth) in CELLS.items():
        train, val = (jsonl(root / f"PROBES_{split}_{key}.jsonl") for split in ("train", "validation"))
        if len({x["id"] for x in val}) != 299 or {r["id"] for r in val} & {r["id"] for r in train}:
            raise RuntimeError(f"Unexpected ARC population at {key}")
        data[key] = val
        boot = np.random.default_rng(BOOT_SEED).integers(0, len(val), size=(COUNT, len(val)))
        names = [name for name in FIXED if name in val[0]["scores"]]
        profiles = {m: profile(val, m) for m in ("native", *names)}
        conditions = {method: dict(metrics={tag: int(x.sum()) if tag.startswith("acc") else float(x.mean())
                                            for tag, x in metrics.items()}) for method, metrics in profiles.items()}
        for method in names:
            diagnostics = np.asarray([r["treatments"][method] for r in val])
            conditions[method].update(diagnostics={name: distribution(col) for name, col in
                                      zip(COLUMNS["treatments"], diagnostics.T, strict=True)},
                                      finite_output_response=output_response(val, method))
        comparisons = [(name, "native") for name in names] + [("learned", name) for name in names if name != "learned"]
        contrasts = {f"{a}_vs_{b}": {name: paired(x - profiles[b][name], boot) for name, x in profiles[a].items()}
                     for a, b in comparisons}
        fit = directions["cells"][key]
        basis = fit["basis"].numpy()
        align = {label: [(float(np.square(basis[:r] @ v.numpy()).sum()) / v.numel()) for r in (1, 2, 4, 8)]
                 for label, v in (("learned", directions["learned"]), ("random_rotation", directions["rotation"]),
                                  ("functional_mean", fit["axis"]))}
        result[key] = dict(anchor=anchor, terminal_budget=depth, directions=dict(train_gradient_uncentered_pca=
            fit["eigenvalue_energy_fraction"].tolist(), projection_r1_r2_r4_r8=align,
            functional_mean_to_learned_cosine=float(cosine(fit["axis"], directions["learned"]))),
            geometry=summary_columns(val, "geometry"), conditions=conditions, paired=contrasts,
            learned_output_agreement={arm: output_agreement(val, arm, boot) for arm in names if arm != "learned"},
            prediction=validation_forecasts(train, val, boot))
        if "incoming" in val[0]:
            result[key]["incoming_geometry"] = summary_columns(val, "incoming")
    horizons = {}
    for anchor in (1, 2):
        cells = [key for key, (a, _) in CELLS.items() if a == anchor]
        for short, long in combinations(cells, 2):
            left, right = data[short], data[long]
            if [x["id"] for x in left] != [x["id"] for x in right]:
                raise RuntimeError("Horizon cohorts changed question order")
            boot = np.random.default_rng(BOOT_SEED).integers(0, len(left), size=(COUNT, len(left)))
            lvalue, rvalue = [forecast_data(p)[1] for p in (left, right)]
            angles = [vector_cosine(a, b) for a, b in zip(score_change_vectors(left, "learned"),
                                                        score_change_vectors(right, "learned"), strict=True)]
            angles = [x for x in angles if x is not None]
            horizons[f"{short}__{long}"] = dict(utility_delta_long_minus_short=paired(rvalue - lvalue, boot),
                response_vector_pearson=float(np.corrcoef(lvalue, rvalue)[0, 1]),
                centered_output_vector_cosine=dict(valid=len(angles), **distribution(angles)) if angles else None)
    save(root / "SUMMARY.json", dict(columns=COLUMNS, learned_axis_orientation=directions["orientation"], cells=result,
         same_anchor_horizon_contrasts=horizons, paired_draws=COUNT, candidate_unit="per question", finite_dose=DOSE))
