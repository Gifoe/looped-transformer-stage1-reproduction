"""Plot natural trajectories, semantic changes and held-out intervention effects."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from design import TASKS

NAMES = {"multihop": "Multi-hop", "arithmetic": "Arithmetic",
         "prontoqa_derived": "PrOntoQA-derived", "prontoqa": "PrOntoQA (proof MC)",
         "arc": "ARC-Challenge", "hellaswag": "HellaSwag"}
COLORS = {"multihop": "#0F4D92", "arithmetic": "#B64342", "prontoqa_derived": "#42949E",
          "prontoqa": "#9A4D8E", "arc": "#A06421", "hellaswag": "#767676"}
STYLES = {"semantic_minus_natural": ("Same answer vs native", "#0F4D92", "o"),
          "semantic_minus_irrelevant": ("Same answer vs irrelevant", "#42949E", "s"),
          "semantic_minus_random": ("Same answer vs random", "#B64342", "^")}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def export(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.1)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.1)
    plt.close(fig)


def trajectory_figure(root, out, behavior):
    rows = read_csv(root / "trajectory_results" / "layer_aggregates.csv")
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 5.8), layout="constrained", sharex=True, sharey=True)
    heat = np.stack([
        np.array([float(r["median_cosine"]) for r in rows if r["task"] == task and r["role"] == "readout"]).reshape(3,25)
        for task in NAMES
    ])
    lo = float(heat.min())
    for i, (ax, task) in enumerate(zip(axes.flat, NAMES)):
        im = ax.imshow(heat[i], aspect="auto", vmin=lo, vmax=1, cmap="viridis",
                       interpolation="nearest")
        n = next(r["base_count"] for r in behavior if r["task"] == task)
        ax.set_title(f"{NAMES[task]} ($n={n}$)", loc="left", fontweight="semibold")
        ax.set_xticks([0, 5, 11, 17, 24], ["0", "5", "11", "17", "Norm"])
        ax.set_yticks(range(3), ["K1→K2", "K2→K3", "K3→K4"])
        ax.axvline(23.5, color="white", lw=1.3)
        if i >= 3:
            ax.set_xlabel("Physical layer (zero-indexed)")
    fig.colorbar(im, ax=axes, shrink=.87, label="Median readout-state cosine")
    export(fig, out / "latent_trajectories")


def semantics_figure(root, out, behavior):
    pairs = read_csv(root / "trajectory_results" / "semantic_geometry.csv")
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.8), layout="constrained")
    for j, task in enumerate(TASKS):
        ax, lower = axes[:, j]
        ax.set_title(NAMES[task], loc="left", fontweight="semibold")
        values = [r for r in behavior if r["task"] == task]
        k = [int(r["K"]) for r in values]
        ax.plot(k, [int(r["correct"])/int(r["base_count"]) for r in values],
                color=COLORS[task], marker="o", label="Final answer")
        ax.plot(k, [int(r["middle_correct"])/int(r["middle_count"]) for r in values],
                color="#626262", ls="--", marker="s", label="Intermediate query")
        ax.set(xticks=k, ylim=(0, 1.04))
        for contrast, name, col in (("same_answer","First + compensating map","#0F4D92"),
                                    ("second","Map only","#626262"),
                                    ("irrelevant","Irrelevant premise","#B64342")):
            y = [float(r["median_relative_l2"]) for r in pairs if r["task"] == task
                 and r["contrast"] == contrast and r["role"] == "anchor" and r["event"] == "24"]
            lower.plot(k, y, color=col, marker="o", label=name)
        lower.set(xticks=k, xlabel="Native recurrence depth")
    axes[0,0].set_ylabel("Accuracy")
    axes[1,0].set_ylabel("Anchor relative L2\n(post-loop norm)")
    axes[0,-1].legend(loc="best", fontsize=8, frameon=False)
    axes[1,-1].legend(loc="best", fontsize=7.5, frameon=False)
    export(fig, out / "semantic_evolution")


def causal_figure(root, out):
    results = read_csv(root / "causal_effects.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), layout="constrained", sharey=True)
    for ax, task in zip(axes, TASKS):
        subset = [r for r in results if r["task"] == task]
        ax.set_title(f"{NAMES[task]} (K{subset[0]['site_K']}, L{subset[0]['site_layer_zero_index']})",
                     fontweight="semibold", loc="left")
        for idx, (key, (label, color, marker)) in enumerate(STYLES.items()):
            series = [r for r in subset if r["comparison"] == key]
            y = np.array([float(r["effect_nats"]) for r in series])
            low = np.array([float(r["lower"]) for r in series])
            high = np.array([float(r["upper"]) for r in series])
            x = np.arange(1, 5) + .085 * (idx-1)
            ax.errorbar(x, y, yerr=[np.maximum(y-low,0), np.maximum(high-y,0)],
                        color=color, marker=marker, capsize=2.8, lw=1.4, label=label)
        ax.axhline(0, color="#979797", lw=.8, ls=":")
        ax.set(xticks=range(1,5), xlabel="Readout depth")
    axes[0].set_ylabel("Counterfactual margin difference (nats)")
    axes[-1].legend(loc="best", frameon=False, fontsize=7.8)
    export(fig, out / "heldout_patch_effects")


def cross_figure(root, out):
    rows = read_csv(root / "cross_task_mechanisms.csv")
    all_rows = read_csv(root / "causal_effects.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.35), layout="constrained", sharey=True)
    for p, (key, label) in enumerate((
        ("intermediate_minus_random", "Intermediate-query effect\n(same answer vs random)"),
        ("full_minus_blocked", "Final-answer effect attenuated by readout block"),
    )):
        ax = axes[p]
        ax.axvline(0, color="#777777", lw=.8, ls=":")
        for j, row in enumerate(rows[::-1]):
            task = row["task"]
            r = next(r for r in all_rows if r["task"] == task and r["comparison"] == key and r["readout_K"] == "4")
            x, lo, hi = (float(r[col]) for col in ("effect_nats", "lower", "upper"))
            ax.errorbar(x, j, xerr=[[max(x-lo, 0)], [max(hi-x, 0)]], color=COLORS[task],
                        marker="o", lw=1.8, capsize=3)
        ax.set(yticks=range(len(rows)), yticklabels=[NAMES[r["task"]] for r in rows[::-1]],
               xlabel="K4 margin difference (nats)", ylim=(-.6, len(rows)-.4))
        ax.set_title(label, loc="left", fontweight="semibold")
    export(fig, out / "cross_task_pathway")


def followup_figure(root, out):
    prefix = root / "prefix_followup" / "causal_effects.csv"
    record = root / "record_followup" / "causal_effects.csv"
    if not prefix.exists():
        return
    fig, axes = plt.subplots(1, 3 if record.exists() else 2, figsize=(12.6, 3.5), layout="constrained")
    rows = read_csv(prefix)
    names = [NAMES[task] for task in TASKS]
    for ax in axes:
        ax.axvline(0, color="#777777", lw=.8, ls=":")
        ax.set_xlabel("K4 margin change (nats)")
    axes[0].set_title("Entire source prefix", loc="left", fontweight="semibold")
    axes[1].set_title("Which source tokens carry the effect?", loc="left", fontweight="semibold")

    def point(ax, value, y, coef="effect_nats", **style):
        v, lo, hi = (float(value[k]) for k in (coef, "lower", "upper"))
        ax.errorbar(v, y, xerr=[[max(v-lo, 0)], [max(hi-v, 0)]], capsize=2.8, lw=1.5, **style)

    for j, task in enumerate(TASKS[::-1]):
        for i, (k, color) in enumerate((("1","#0F4D92"), ("2","#B64342"))):
            chosen = next(r for r in rows if r["task"] == task and r["patch_K"] == k
                          and r["readout_K"] == "4" and r["comparison"] == "full_minus_natural")
            point(axes[0], chosen, j+(.12 if i else -.12), color=color, marker="o",
                  label=f"Patch at K{k}" if j == 0 else None)
        for i, (condition, label, color) in enumerate((
            ("changed_tokens_minus_natural", "Changed token IDs", "#0F4D92"),
            ("unchanged_tokens_minus_natural", "Unchanged token IDs", "#767676"),
        )):
            chosen = next(r for r in rows if r["task"] == task and r["patch_K"] == "2"
                          and r["readout_K"] == "4" and r["comparison"] == condition)
            point(axes[1], chosen, j+(.12 if i else -.12), color=color, marker="o",
                  label=label if j == 0 else None)
    axes[0].set_yticks(range(len(names)), names[::-1])
    axes[1].set_yticks(range(len(names)), [])
    for ax in axes[:2]:
        ax.set_ylim(-.55,2.55)
        ax.legend(loc="best", frameon=False, fontsize=7.8)
    if record.exists():
        axes[2].set_title("Queried-record test", loc="left", fontweight="semibold")
        target = read_csv(record)
        labels = (
            ("target_minus_native", "vs native"), ("target_minus_unqueried", "vs unqueried"),
            ("target_minus_random", "vs random"), ("target_minus_blocked", "vs restored readout"),
        )
        for j, (key, _) in enumerate(labels[::-1]):
            row = next(r for r in target if r["contrast"] == key and r["readout_K"] == "4")
            point(axes[2], row, j, coef="mean_nats", color="#42949E", marker="s")
        axes[2].set_yticks(range(len(labels)), [label for _, label in labels[::-1]])
        axes[2].set_ylim(-.6,3.6)
    export(fig, out / "source_record_transfer")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    out = args.root / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.linewidth": .95,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"})
    behavior = read_csv(args.root / "trajectory_results" / "behavior.csv")
    trajectory_figure(args.root, out, behavior)
    semantics_figure(args.root, out, behavior)
    causal_figure(args.root, out)
    cross_figure(args.root, out)
    followup_figure(args.root, out)


if __name__ == "__main__":
    main()
