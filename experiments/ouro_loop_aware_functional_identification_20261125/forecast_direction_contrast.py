"""Paired domain-level and per-question comparison of the two forecast axes."""

import argparse
from collections import defaultdict
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def rows_by_depth(folder):
    cohorts = defaultdict(dict)
    for path in sorted(folder.glob("*.jsonl")):
        receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        with path.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != receipt["predictions_sha256"]:
                raise ValueError(f"Changed intervention rows {path}")
        plan = receipt["plan"]
        cells = defaultdict(dict)
        for row in read_rows(path):
            source, direction = row["direction"].rsplit("__", 1)
            if direction in ("excess", "loop"):
                cells[row["id"], source, row["dose"]][direction] = row
        sources = sorted({key[1] for key in cells})
        for dose in plan["doses"]:
            vectors = []
            for id in plan["ids"]:
                axes = [cells[id, source, dose] for source in sources]
                effect = []
                for metric in ("loop_gain", "interaction", "edited_loop"):
                    effect.append(np.mean([cell["excess"][metric] - cell["loop"][metric] for cell in axes]))
                vectors.append(effect)
            cohorts[plan["depth"], dose][plan["target"]] = np.asarray(vectors)
    return cohorts


def interval(data, draws, seed):
    rng = np.random.default_rng(seed)
    domains, n, _ = data.shape
    chosen = rng.integers(domains, size=(draws, domains, 1))
    questions = rng.integers(n, size=(draws, domains, n))
    replicates = data[chosen, questions].mean(axis=(1, 2))
    return np.quantile(replicates, (.025, .975), axis=0).T.tolist()


def main(a):
    rows = []
    for (k, dose), domain in sorted(rows_by_depth(a.input).items()):
        names = sorted(domain)
        arrays = np.stack([domain[name] for name in names])
        rows.append(dict(depth=k, dose=dose, targets=names, n_per_target=arrays.shape[1],
                         endpoints=["absolute_loop", "interaction", "edited_accuracy"],
                         mean=arrays.mean(axis=(0, 1)).tolist(),
                         target_means=dict(zip(names, arrays.mean(axis=1).tolist(), strict=True)),
                         domain_question_ci95=interval(arrays, 10000, 261125137)))
    a.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
