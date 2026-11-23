"""CPU analysis of locked training moments, their subspaces, and rank-one transfer margin."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import torch

from dataset_transfer import digest, json_read, require, write


SOURCES = ("arc", "csqa")
RANKS = (1, 3, 5, 8, 12)
K_VALUES = (1, 2, 3)
MOMENT_KEYS = tuple(f"h{k}_m{m}" for k in K_VALUES for m in range(5-k))
MAIN_KEYS = tuple(key for key in MOMENT_KEYS if not key.endswith("_m0"))
SEED = 261123547
SVD_VIEW = 5


def read_moments(moment_root, manifest_path):
    manifest = json_read(manifest_path)
    moments, provenance = {}, {}
    for source in SOURCES:
        root = moment_root / source
        index_path = root / "MOMENTS.json"
        index = json_read(index_path)
        protocol = index["protocol"]
        history = manifest["cohorts"][f"{source}_train"]
        require(protocol["input_sha256"] == history["sha256"], f"{source} moments are not authorized Train fits")
        train_path = manifest_path.parent / history["file"]
        require(digest(train_path) == history["sha256"], f"{source} fitting cohort file is inconsistent")
        with train_path.open(encoding="utf-8") as stream:
            train_ids = {json.loads(line)["id"] for line in stream}
        selected = set(protocol["ids"])
        require(len(train_ids) == history["n"] and len(selected) == protocol["n"] and selected <= train_ids,
                f"Unexpected {source} fit population")
        require(set(index["conditions"]) == set(MOMENT_KEYS), f"Missing or augmented cells in {root}")
        provenance[source] = dict(index_sha256=digest(index_path), n=protocol["n"],
                                  fitting_file_sha256=history["sha256"], matrices={})
        moments[source] = {}
        for cell in MOMENT_KEYS:
            path = root / f"{cell}.pt"
            expected = index["conditions"][cell]["matrix_sha256"]
            require(digest(path) == expected, f"Moment checksum mismatch in {path}")
            metadata = json_read(root / f"{cell}.json")
            ids = [r["id"] for r in metadata["observations"]]
            require(len(ids) == len(selected) and set(ids) == selected
                    and metadata["matrix_sha256"] == expected,
                    f"Unexpected moment receipt in {cell}")
            state = torch.load(path, map_location="cpu", weights_only=True)
            arrays = {k: state[k].numpy().astype(np.float64, copy=False) for k in ("u", "s", "w")}
            matrix = state["matrix"].numpy()
            require(matrix.shape == (2048, 2048) and arrays["u"].shape == arrays["w"].shape == (2048, 12)
                    and arrays["s"].shape == (12,) and arrays["s"][0] > 0 and np.isfinite(matrix).all()
                    and all(np.isfinite(v).all() for v in arrays.values()), f"Unexpected matrix in {path}")
            for name in ("u", "w"):
                basis = arrays[name]
                require(np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1]))) < 1e-4,
                        f"SVD vectors not orthonormal in {cell}")
            moments[source][cell] = dict(**arrays, matrix=matrix, frobenius=state["frobenius"],
                                         stable_rank=state["stable_rank"], n=state["n"],
                                         token_count=state["token_count"])
            provenance[source]["matrices"][cell] = expected
    return moments, provenance


def angles(a, b, rank):
    overlap = a[:, :rank].T @ b[:, :rank]
    cosines = np.linalg.svd(overlap, compute_uv=False).clip(0., 1.)
    return dict(cosines=cosines.tolist(), angles_deg=np.rad2deg(np.arccos(cosines)).tolist(),
                mean_cosine_squared=float(np.mean(cosines**2)))


def alignment(a, b):
    left = float(np.dot(a["u"][:, 0], b["u"][:, 0]))
    right = float(np.dot(a["w"][:, 0], b["w"][:, 0]))
    return dict(
        operator_rank1_cosine=left * right,
        subspaces={str(rank): {side: angles(a[key], b[key], rank) for side, key in (("left", "u"), ("right", "w"))}
                   for rank in RANKS},
    )


def spectrum(moments):
    profiles = {}
    within = {}
    for source, cells in moments.items():
        profiles[source] = {}
        for cell, entry in cells.items():
            vals = entry["s"]
            frob = entry["frobenius"]
            profiles[source][cell] = dict(
                singular_values=vals.tolist(), relative_to_first=(vals / vals[0]).tolist(),
                top_k_frobenius_energy={str(k): float(np.sum(vals[:k]**2)/frob**2) for k in RANKS},
                stable_rank=float(entry["stable_rank"]), frobenius=float(frob),
                fit_questions=entry["n"], scored_positions=entry["token_count"],
            )
        within[source] = {f"{a}__{b}": alignment(cells[a], cells[b]) for a, b in combinations(cells, 2)}
    across = {key: alignment(moments["arc"][key], moments["csqa"][key]) for key in MOMENT_KEYS}
    return dict(profiles=profiles, within_source=within, across_source_matched=across)


def span(group, side, rank):
    stacked = np.concatenate([entry[side][:, :rank] for entry in group.values()], axis=1)
    u, s, _ = np.linalg.svd(stacked, full_matrices=False)
    return u[:, s > 1e-7 * s[0]]


def couples(vector, left_width):
    u, w = vector[:left_width], vector[left_width:]
    un, wn = np.linalg.norm(u), np.linalg.norm(w)
    return u / un, w / wn, un, wn


def projected_values(matrices, u, w):
    return np.einsum("cij,i,j->c", matrices, u, w, optimize=True)


def fit_gamma(group, span_rank, mode):
    # Keep the two spheres distinct under normalization of the free factors.
    left, right = span(group, "u", span_rank), span(group, "w", span_rank)
    du, dw = left.shape[1], right.shape[1]
    matrices = np.array([left.T @ entry["matrix"] @ right for entry in group.values()])
    norms = np.array([entry["s"][0] for entry in group.values()])
    scale = norms if mode == "relative" else np.full_like(norms, norms.max())
    objective_matrices = matrices / scale[:, None, None]
    restore_objective = float(norms.max()) if mode == "raw" else 1.0
    starts = {}
    for i, (tag, entry) in enumerate(group.items()):
        starts[tag] = np.r_[left.T @ entry["u"][:, 0], right.T @ entry["w"][:, 0]]
    u, _, wt = np.linalg.svd(objective_matrices.mean(0), full_matrices=False)
    starts["mean_svd"] = np.r_[u[:, 0], wt[0]]
    rng = np.random.default_rng(SEED)
    for i in range(8):
        starts[f"random_{i}"] = rng.normal(size=du+dw)

    def margin(x):
        u, w, *_ = couples(x, du)
        return float(projected_values(objective_matrices, u, w).min())

    def soft_objective(x, tau):
        u, w, nu, nw = couples(x, du)
        scores = projected_values(objective_matrices, u, w)
        values = -scores / tau
        normalizer = logsumexp(values)
        blend = np.einsum("c,cij->ij", np.exp(values-normalizer), objective_matrices)
        gu, gw = -blend @ w, -blend.T @ u
        return tau * normalizer, np.r_[(gu - u * np.dot(u, gu)) / nu, (gw - w * np.dot(w, gw)) / nw]

    best_name, best_x, best_score = None, None, -np.inf
    summaries = {}
    for name, start in starts.items():
        original = margin(start)
        best_local, best_value, current = start, original, start
        iterations = 0
        for tau in (.10, .025, .006, .0015):
            fit = minimize(soft_objective, current, args=(tau,), method="L-BFGS-B", jac=True,
                           options={"maxiter": 80, "ftol": 1e-11, "gtol": 1e-8})
            u, w, *_ = couples(fit.x, du)
            current = np.r_[u, w]
            score = margin(current)
            iterations += fit.nit
            if score > best_value:
                best_local, best_value = current, score
        summaries[name] = dict(initial_margin=original*restore_objective, best_margin=best_value*restore_objective,
                               iterations=iterations)
        if best_value > best_score:
            best_name, best_x, best_score = name, best_local, best_value
    u, w, *_ = couples(best_x, du)
    physical = [(left @ u), (right @ w)]
    values = {name: float(physical[0] @ (entry["matrix"] @ physical[1])) for name, entry in group.items()}
    projected = projected_values(matrices, u, w)
    require(np.allclose(list(values.values()), projected, atol=1e-10, rtol=1e-8),
            "Compressed and original matrix projections disagree")
    max_projected = min(np.linalg.svd(A, compute_uv=False)[0] for A in objective_matrices) * restore_objective
    max_full = min(entry["frobenius"] / s for entry, s in zip(group.values(), scale, strict=True)) * restore_objective
    return dict(
        candidate_lower_bound=float(best_score*restore_objective),
        full_frobenius_upper_bound=float(max_full), span_spectral_upper_bound=float(max_projected),
        best_start=best_name, best_initial_margin=max(m["initial_margin"] for m in summaries.values()),
        starts=summaries, span_basis_fit_rank=span_rank, span_dimensions=[du, dw],
        raw_scores=values, relative_scores={tag: values[tag]/n for tag,n in zip(group, norms, strict=True)},
        objective="raw u^T M w" if mode == "raw" else "per-condition u^T M w divided by estimated sigma1",
    ), dict(u=torch.from_numpy(physical[0]), w=torch.from_numpy(physical[1]))


def alpha_title(ax, label, title):
    ax.set_title(f"${label}$  {title}", loc="left", fontweight="bold", pad=8)


def spectrum_fig(result, out):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": .8,
        "legend.frameon": False, "figure.facecolor": "white", "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    labels = {"arc": "ARC-Challenge", "csqa": "CommonsenseQA"}
    plot_colors = {1: "#0F4D92", 2: "#42949E", 3: "#B64342"}
    styles = {1: "-", 2: (0, (6, 2)), 3: (0, (2, 2))}
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.0), layout="constrained")
    for i, source in enumerate(SOURCES):
        ax = axes[0, i]
        for entry in MAIN_KEYS:
            k, m = int(entry[1]), int(entry[-1])
            values = result["profiles"][source][entry]["relative_to_first"]
            ax.semilogy(range(1, len(values)+1), values, linewidth=1.6, color=plot_colors[k], ls=styles[m])
        alpha_title(ax, "ab"[i], labels[source] + " spectra")
        ax.set(xlabel="Singular index", ylabel=r"$\sigma_i / \sigma_1$",
               xticks=[1, 3, 5, 8, 12], xlim=(1, 12), ylim=(.007, 1.35))
    handles = [Line2D([], [], color=c, lw=1.6, label=fr"$h_{k}$") for k,c in plot_colors.items()]
    handles += [Line2D([], [], color="#444444", lw=1.4, ls=s, label=f"$m={m}$") for m,s in styles.items()]
    axes[0, 1].legend(handles=handles, ncol=2, loc="upper right", fontsize=7, handlelength=1.8)
    ax = axes[0, 2]
    at = np.arange(len(MAIN_KEYS))
    height = 0.
    for i, source in enumerate(SOURCES):
        bars = [result["profiles"][source][key]["stable_rank"] for key in MAIN_KEYS]
        height = max(height, max(bars))
        ax.bar(at + (-.18 if i == 0 else .18), bars, .34, color=("#0F4D92", "#B64342")[i], label=labels[source])
    alpha_title(ax, "c", "Stable rank")
    key_label = [rf"$({x[1]},{x[-1]})$" for x in MAIN_KEYS]
    ax.set(xlabel="(State, remaining loops)", ylabel=r"$\|M\|_F^2 / \sigma_1^2$",
           xticks=at, xticklabels=key_label, ylim=(0., 1.24*height))
    ax.legend(fontsize=7, loc="upper right")

    for j, source in enumerate(SOURCES):
        ax = axes[1, j]
        values = np.eye(len(K_VALUES))
        for i, k in enumerate(K_VALUES):
            for q in range(i+1, len(K_VALUES)):
                key = f"h{k}_m1__h{K_VALUES[q]}_m1"
                pair = result["within_source"][source][key]["subspaces"][str(SVD_VIEW)]
                values[i, q] = pair["left"]["mean_cosine_squared"]
                values[q, i] = pair["right"]["mean_cosine_squared"]
        ax.imshow(values, vmin=0, vmax=1, cmap="Blues")
        for i, row in enumerate(values):
            for k, value in enumerate(row):
                ax.text(k, i, f"{value:.2f}", va="center", ha="center",
                        color="white" if value > .55 else "#1a1a1a", fontsize=8)
        alpha_title(ax, "de"[j], labels[source] + " at $m=1$")
        ticks = [rf"$h_{k}$" for k in K_VALUES]
        ax.set(xticks=[0, 1, 2], yticks=[0, 1, 2], xticklabels=ticks, yticklabels=ticks,
               xlabel="Left above diagonal · right below")
    ax = axes[1, 2]
    pairs = result["across_source_matched"]
    for i, side in enumerate(("left", "right")):
        values = [pairs[key]["subspaces"][str(SVD_VIEW)][side]["mean_cosine_squared"] for key in MAIN_KEYS]
        ax.plot(at, values, color=("#0F4D92", "#B64342")[i], marker=("o", "s")[i], label=side.capitalize())
    alpha_title(ax, "f", "Cross-source subspace alignment")
    ax.set(xlabel="(State, remaining loops)", ylabel=r"Mean $\cos^2\theta$ (top 5)",
           xticks=at, xticklabels=key_label, ylim=(0, 1))
    ax.legend(fontsize=7)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(out / f"FIGURE_2_SPECTRUM.{suffix}", dpi=300)
    plt.close(fig)


def main(a):
    moments, provenance = read_moments(a.moments, a.manifest)
    result = spectrum(moments)
    groups = {source: {cell: moments[source][cell] for cell in MAIN_KEYS} for source in SOURCES}
    groups["joint"] = {f"{source}/{cell}": moments[source][cell] for source in SOURCES for cell in MAIN_KEYS}
    candidates, gamma = {}, {}
    for name, grouped in groups.items():
        candidates[name], gamma[name] = {}, {}
        for mode in ("raw", "relative"):
            gamma[name][mode], candidates[name][mode] = fit_gamma(grouped, a.span_rank, mode)
            print(name, mode, gamma[name][mode]["candidate_lower_bound"], flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    candidate_path = a.out / "TRAIN_ONLY_GAMMA_DIRECTIONS.pt"
    temporary = candidate_path.with_suffix(".part")
    torch.save(candidates, temporary)
    temporary.replace(candidate_path)
    result.update(
        provenance=provenance, gamma1=gamma, fit_manifest_sha256=digest(a.manifest),
        analyzer_sha256=digest(Path(__file__)), direction_file_sha256=digest(candidate_path),
        direct_readout="m=0 saved in tables, excluded from Gamma and Figure 2",
        method="Train-only feasible lower bound from multistart soft-min optimization in pooled leading singular spans",
        limitation="Approximate in-sample first-order proxy. No confirmation outcomes or finite-dose predictions used.",
    )
    spectrum_fig(result, a.out)
    write(a.out / "SPECTRUM.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("moments", "manifest", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--span-rank", type=int, default=SVD_VIEW)
    args = parser.parse_args()
    if not 1 <= args.span_rank <= max(RANKS):
        parser.error("--span-rank must be between 1 and 12")
    main(args)
