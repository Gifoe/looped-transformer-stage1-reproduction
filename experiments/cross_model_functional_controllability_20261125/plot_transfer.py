"""Draw paired cross-version benchmark effects from full result receipts."""

import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from report_transfer import MODEL_LABELS


TASK_NAMES = {"arc_challenge": "ARC-Challenge", "arc_easy": "ARC-Easy", "sciq": "SciQ", "mmlu": "MMLU",
              "hellaswag": "HellaSwag", "winogrande": "WinoGrande", "piqa": "PIQA", "mmlu_pro": "MMLU-Pro"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def exports(figure, a, name):
    for suffix in ("svg", "pdf", "png"):
        figure.savefig(a.out / f"{name}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run(a):
    a.out.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    old = read(a.historical)["results"]
    axes_data = {"Ouro-1.4B Base": {k: row["paired"]["native"][row["metric"]] for k, row in old.items()}}
    history = {}
    for slug, name in MODEL_LABELS.items():
        raw = read(a.results / slug / "RESULTS.json")
        history[name] = raw
        axes_data[name] = {task: row["metrics"][row["primary"]] for task, row in raw.items()}
    if any(set(scores) != TASK_NAMES.keys() for scores in axes_data.values()):
        raise ValueError("The cross-version figure requires each complete benchmark population")
    colors = ("#A3652E", "#175584", "#298073", "#905278")
    fig, axes = plt.subplots(1, 4, figsize=(13.3, 4.2), sharex=True, sharey=True, layout="constrained")
    y = np.arange(len(TASK_NAMES))
    for ax, (title, scores), color in zip(axes, axes_data.items(), colors, strict=True):
        means = [100 * scores[k]["delta"] for k in TASK_NAMES]
        lo, hi = zip(*[(100 * x for x in scores[k]["paired_ci95"]) for k in TASK_NAMES])
        ax.axvline(0, color="#777777", lw=.7)
        ax.hlines(y, lo, hi, color=color, lw=1.25)
        ax.plot(means, y, "o", ms=4, color=color)
        ax.set(yticks=y, yticklabels=list(TASK_NAMES.values()), title=title, xlabel="T1 − Native (pp)")
    axes[0].invert_yaxis()
    exports(fig, a, "paired_accuracy_transfer")

    fig, ax = plt.subplots(figsize=(5.8, 3.8), layout="constrained")
    ax.axvline(0, color="#777777", lw=.7)
    ax.axhline(0, color="#777777", lw=.7)
    for (name, raw), color in zip(history.items(), colors[1:], strict=True):
        times = {}
        for arm in ("native", "t1"):
            values = [row["cost"][arm] for row in raw.values()]
            times[arm] = sum(p.get("seconds", p.get("elapsed_seconds")) for p in values)
        change = 100 * (times["t1"] / times["native"] - 1)
        gain = 100 * statistics.mean(axes_data[name][task]["delta"] for task in TASK_NAMES)
        ax.scatter(change, gain, color=color, label=name, s=38)
    ax.set(xlabel="T1 change in benchmark time (%)", ylabel="Eight-task mean accuracy change (pp)")
    ax.legend(frameon=False, loc="best")
    exports(fig, a, "accuracy_and_compute")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())
