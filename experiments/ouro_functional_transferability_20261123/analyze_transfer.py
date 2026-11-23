"""Pair every sealed theory prediction with independent observed utility and accuracy."""

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from state_horizon import score_arrays


FEATURES = ("signed_source_moment", "clipped_proxy", "signed_mean_gradient", "state_read_cosine",
            "unsigned_gradient_pca_alignment", "unsupervised_native_displacement_cosine",
            "deployed_write_rms", "linearized_postnorm_rms")
UNSIGNED = {"unsigned_gradient_pca_alignment", "deployed_write_rms", "linearized_postnorm_rms"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def differences(rows, native, intervention):
    values = []
    for row in rows:
        a, b = (score_arrays(row["scores"][name], row["gold"], row["characters"]) for name in (native, intervention))
        values.append([b[key] - a[key] for key in ("utility", "margin", "acc", "acc_norm")])
    return np.array(values, dtype=np.float64)


def collection(root, predictions):
    records = []
    cache = {}
    horizon = root / "state_horizon" / "arc_confirm"
    for key, features in predictions["predictions"]["historical_t1"]["arc"].items():
        raw = horizon / f"{key}.jsonl"
        if not raw.is_file():
            continue
        rows = jsonl(raw)
        records.append(dict(controller="historical_t1", target="arc", condition=key, family="State × horizon",
                            features=features, ids=[r["id"] for r in rows],
                            outcomes=differences(rows, "native", "learned")))

    def visit(name, path, target, key, tag, native, family):
        if not path.is_file():
            return
        if path not in cache:
            cache[path] = jsonl(path)
        rows = cache[path]
        if tag not in rows[0]["scores"]:
            return
        measures=predictions["predictions"][name][target][key]
        records.append(dict(controller=name, target=target, condition=key, family=family,
                            features=measures, ids=[r["id"] for r in rows],
                            outcomes=differences(rows, native, tag)))

    for name, detail in predictions["controllers"].items():
        if name == "historical_t1":
            continue
        directory, group = name.rsplit("/", 1)
        source = detail["source"]
        home = root / "HOLDOUT" / directory / "confirm"
        for target in ("arc", "csqa"):
            maps = [(home, f"{target}_confirm")]
            if source == "arc" and target == "csqa":
                maps.append((home / "csqa_transfer", "csqa_confirm"))
            for folder, tagname in maps:
                path = folder / f"{tagname}_PREDICTIONS.jsonl"
                if not path.is_file():
                    continue
                for key, item in predictions["predictions"][name][target].items():
                    k, anchor = item["total_depth"], item["anchor"]
                    if detail["suite"] == "position":
                        family = "Trained position" if anchor == detail["anchor"] else "Unseen position"
                        tag = f"{group}_k{k}_p{anchor}"
                    else:
                        tag = f"{group}_k{k}"
                        if source != target:
                            family = "Cross-task"
                        elif k > max(detail["depths"]) and k >= 5:
                            family = "Outside K2–K4"
                        elif k in detail["depths"]:
                            family = "Seen depth"
                        else:
                            family = "Unseen K2–K4"
                    visit(name, path, target, key, tag, f"native_k{k}", family)
    return records


def correlation(left, right, axis=-1):
    left = left - left.mean(axis=axis, keepdims=True)
    right = right - right.mean(axis=axis, keepdims=True)
    norm = np.sqrt(np.square(left).sum(axis=axis) * np.square(right).sum(axis=axis))
    return (left * right).sum(axis=axis) / np.where(norm > 0, norm, np.nan)


def interval(x):
    good = x[np.isfinite(x)]
    return np.quantile(good, [.025, .975]).tolist() if len(good) else None


def summary(records, rng):
    if not records:
        return {}
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[tuple(row["ids"])].append(i)
    nboot = 2000
    boots = np.empty((nboot, len(records), 4))
    for ids, indexes in groups.items():
        draws = rng.integers(len(ids), size=(nboot, len(ids)))
        for i in indexes:
            boots[:, i, :] = records[i]["outcomes"][draws].mean(axis=1)
    measured = np.stack([r["outcomes"].mean(0) for r in records])
    out = {}
    for metric, j in [("utility", 0), ("margin", 1), ("acc", 2), ("acc_norm", 3)]:
        by_predictor = {}
        for feature in FEATURES:
            x = np.array([r["features"][feature] for r in records])
            y = measured[:, j]
            ybs = boots[:, :, j]
            rho = correlation(rankdata(x), rankdata(y))
            rho_bs = correlation(rankdata(x)[None], rankdata(ybs, axis=1), axis=1)
            sign = np.sign(x) == np.sign(y)
            testable = np.array([not (a <= 0 <= b) for a, b in np.quantile(ybs, [.025, .975], axis=0).T])
            by_predictor[feature] = dict(
                spearman=float(rho) if np.isfinite(rho) else None, spearman_question_bootstrap_ci95=interval(rho_bs),
                exact_sign_agreement=None if feature in UNSIGNED else float(sign.mean()),
                sign_ci95=None if feature in UNSIGNED else interval((np.sign(x)[None] == np.sign(ybs)).mean(1)),
                sign_among_nonzero_ci=None if feature in UNSIGNED or not testable.any() else float(sign[testable].mean()),
                conditions_with_ci_excluding_zero=int(testable.sum()),
                mean_absolute_utility_error=float(np.abs(x-y).mean()) if j==0 and feature in FEATURES[:2] else None,
            )
        out[metric] = by_predictor
    cells = []
    for i, r in enumerate(records):
        ci = np.quantile(boots[:, i, :], [.025, .975], axis=0).T
        cells.append(dict(controller=r["controller"], target=r["target"], condition=r["condition"], n=len(r["ids"]),
                          features=r["features"], mean=dict(zip(("utility", "margin", "acc", "acc_norm"), measured[i].tolist())),
                          ci95=dict(zip(("utility", "margin", "acc", "acc_norm"), ci.tolist())),
                          utility_per_question_digest=None))
    return dict(n_conditions=len(records), full_bootstrapped_cells=cells, predictors=out)


def figure1(root, out):
    data = read(root / "state_horizon" / "arc_confirm" / "STATE_HORIZON.json")
    treatments = ("learned", "tangent_component", "radial_component")
    labels = ("Deployed rank one", "Tangential component", "Radial component")
    matrices = [np.array([[data["conditions"][f"h{k}_m{m}"]["result"][arm]["utility"]["delta"]
                           for m in range(4)] for k in range(1, 4)]) for arm in treatments]
    vmax = max(np.max(np.abs(a)) for a in matrices)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    for ax, matrix, label in zip(axes, matrices, labels):
        artist = ax.imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax)
        for k, m in itertools.product(range(3), range(4)):
            ax.text(m, k, f"{matrix[k,m]:+.3f}", ha="center", va="center",
                    color="white" if abs(matrix[k,m])>.6*vmax else "#152238", fontsize=9)
        ax.set(xticks=range(4), yticks=range(3), yticklabels=("h1", "h2", "h3"),
               xlabel="Remaining loops", title=label)
    axes[0].set_ylabel("Receiving boundary")
    fig.colorbar(artist, ax=axes, shrink=.74, label="Paired change in log odds")
    fig.savefig(out / "FIGURE_1_STATE_HORIZON.pdf")
    fig.savefig(out / "FIGURE_1_STATE_HORIZON.png", dpi=180)
    plt.close(fig)


def figure3(records, out):
    shown = ("Unseen K2–K4", "Outside K2–K4", "Cross-task")
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.45), layout="constrained")
    for ax, family in zip(axes, shown, strict=True):
        xs = [r["features"]["clipped_proxy"] for r in records if r["family"] == family]
        ys = [r["outcomes"][:, 0].mean() for r in records if r["family"] == family]
        target = [r["target"] for r in records if r["family"] == family]
        ax.axvline(0, color="#acb5c3", linewidth=.9)
        ax.axhline(0, color="#acb5c3", linewidth=.9)
        for t, c in (("arc", "#27608b"), ("csqa", "#934650")):
            xy=[(x,y) for x,y,name in zip(xs,ys,target) if name == t]
            if xy:
                x, y=zip(*xy)
                ax.scatter(x, y, color=c, alpha=.8, label=t.upper())
        ax.set(xlabel="Forecast change in log odds", title=family)
    axes[0].set_ylabel("Measured change in log odds")
    ax = axes[-1]
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / "FIGURE_3_PREDICTED_VS_ACTUAL.pdf")
    fig.savefig(out / "FIGURE_3_PREDICTED_VS_ACTUAL.png", dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    args=parser.parse_args()
    root=args.results
    out=root / "PREDICTION_AUDIT"
    out.mkdir(parents=True, exist_ok=True)
    pred=read(root / "forecast" / "PREDICTIONS.json")
    rows=collection(root, pred)
    if not rows:
        raise RuntimeError("No frozen confirmation scores for prediction audit")
    rng=np.random.default_rng(261123311)
    result={name:summary([r for r in rows if name=="all" or r["family"]==name], rng)
            for name in ("all", *sorted({r["family"] for r in rows}))}
    (out/"AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if (root / "state_horizon" / "arc_confirm" / "STATE_HORIZON.json").is_file():
        figure1(root, out)
    figure3(rows, out)
    print("AUDIT", len(rows), {n: v["n_conditions"] for n,v in result.items()})


if __name__ == "__main__":
    main()
