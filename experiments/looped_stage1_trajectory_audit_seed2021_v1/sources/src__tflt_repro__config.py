"""YAML experiment configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_experiment(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a YAML experiment file."""

    p = Path(path)
    data = yaml.safe_load(p.read_text()) or {}
    if "runs" not in data or not isinstance(data["runs"], list):
        raise ValueError(f"{p} must contain a top-level runs: list")
    for idx, run in enumerate(data["runs"]):
        if "tag" not in run:
            raise ValueError(f"{p}: runs[{idx}] is missing tag")
        if "model" not in run:
            raise ValueError(f"{p}: runs[{idx}] is missing model")
    data["_path"] = str(p)
    return data


def merged_runs(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    """Return runs with top-level defaults merged in."""

    defaults = deepcopy(experiment.get("defaults", {}))
    runs = []
    for run in experiment["runs"]:
        merged = deepcopy(defaults)
        merged.update(deepcopy(run))
        runs.append(merged)
    return runs


def select_runs(runs: list[dict[str, Any]], names: list[str] | None) -> list[dict[str, Any]]:
    if not names or names == ["all"]:
        return runs
    wanted = set(names)
    selected = [run for run in runs if run["tag"] in wanted]
    missing = wanted - {run["tag"] for run in selected}
    if missing:
        available = ", ".join(run["tag"] for run in runs)
        raise ValueError(f"unknown run tag(s): {sorted(missing)}. Available: {available}")
    return selected

