"""Apply the preregistered equal-question weighting to frozen Experiment A traces."""

import argparse
import hashlib
from itertools import combinations
from collections import Counter
import json
from pathlib import Path

import numpy as np


DEPTHS = (1, 2, 3, 4)
EPS = 1e-20
BOOTSTRAP_SEED = 261123903


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def weights_for(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1 / (len(counts) * counts[inverse]), len(counts)


def weighted_stat(value, weights):
    order = np.argsort(value)
    index = np.searchsorted(np.cumsum(weights[order]), [.05, .5, .95])
    return dict(mean=float(value @ weights), q05_median_q95=value[order[index]].tolist())


def weighted_spectrum(x, w, centered, extra_upper):
    if centered:
        x = x - w @ x
    values, vectors = np.linalg.eigh((x.T * w) @ x)
    values = values[::-1].clip(0)
    within = float(values.sum())
    if within <= EPS:
        return dict(top1_interval=[0., 1.], axis=None)
    interval = [float(values[0] / (within + extra_upper)), float((values[0] + extra_upper) / (within + extra_upper))]
    return dict(projected_energy_fractions=(values / within).tolist(), variance_inside_u=within,
                outside_variance_upper=float(extra_upper), top1_interval=interval, axis=vectors[:, -1].tolist())


def select(keys, category):
    offset = keys[:, 2]
    if category == "last_prompt":
        return (keys[:, 1] == 0) & (offset == 0)
    if category == "continuation":
        return offset > 0
    return np.ones(len(offset), dtype=bool)


def question_cosines(unit, keys, weights, n_questions):
    means = []
    within = []
    for question in np.unique(keys[:, 0]):
        eligible = keys[:, 0] == question
        x, offsets = unit[eligible], keys[eligible, 2]
        means.append(x.mean(axis=0))
        distinct, counts = np.unique(offsets, return_counts=True)
        denominator = len(x)**2 - int(np.square(counts).sum())
        if denominator:
            same_offset = sum(float(np.square(x[offsets == i].sum(axis=0)).sum()) for i in distinct)
            within.append((float(np.square(x.sum(axis=0)).sum()) - same_offset) / denominator)
    full = weights @ unit
    diagonal = float(np.square(means).sum()) / n_questions**2
    return dict(cross_question=(float((full @ full - diagonal) / (1 - 1/n_questions)) if n_questions > 1 else None),
                cross_token_within_question=float(np.mean(within)) if within else None,
                within_question_n=len(within))


def measure(arrays, category):
    subset = select(arrays["keys"], category)
    keys = arrays["keys"][subset]
    coords = arrays["coords"][subset]
    scalars = arrays["scalars"][subset]
    if not len(keys):
        return {"n": 0, "questions": 0}
    w, n = weights_for(keys[:, 0])
    result = dict(n=len(keys), questions=n, clip_fraction=float(w @ (scalars[:, 4] > .45)),
                  state_rms=weighted_stat(scalars[:, 0], w), updates={})
    for i, name in enumerate(("raw", "clipped_before_cast", "actual_after_cast")):
        q = coords[:, i]
        off = arrays["cast_off"][subset] if i == 2 else np.zeros(len(q))
        ratio = scalars[:, i+4]
        alpha = scalars[:, i+7]
        active = ratio > EPS
        active_weights, _ = weights_for(keys[active, 0])
        cos = weighted_stat((alpha[active] / ratio[active]).clip(-1, 1), active_weights)
        edit = dict(ratio=weighted_stat(ratio, w), signed_ratio=weighted_stat(alpha, w), cosine=cos,
                    sign_negative=int((alpha < 0).sum()), sign_positive=int((alpha > 0).sum()), spectra={})
        for typ, divisor in (("edit_vector", np.ones(len(q))),
                             ("unit_direction", np.maximum(np.sqrt(np.square(q).sum(-1) + off), EPS))):
            projected = q / divisor[:, None]
            outside = float(w @ (off / divisor**2))
            edit["spectra"][typ] = {name: weighted_spectrum(projected, w, centered, outside)
                                    for name, centered in (("uncentered", False), ("centered", True))}
        result["updates"][name] = edit
    applied = result["updates"]["actual_after_cast"]
    lower, upper = applied["spectra"]["unit_direction"]["uncentered"]["top1_interval"]
    median = applied["cosine"]["q05_median_q95"][1]
    result["shares_by_preregistered_thresholds"] = ("fail" if upper < .95 or median < .98
                                                     else "pass" if lower >= .95 else "indeterminate")
    bounded = coords[:, 1]
    norms = np.linalg.norm(bounded, axis=1)
    active = norms > EPS
    cw, n = weights_for(keys[active, 0])
    result["clipped_pair_cosines"] = question_cosines(bounded[active] / norms[active, None], keys[active], cw, n)
    return result


def across_depth(traces):
    output = {}
    for x, y in combinations(DEPTHS, 2):
        tx, ty = traces[x], traces[y]
        look = {(tuple(key), h): i for i, (key, h) in enumerate(zip(tx["keys"], tx["prefix_hash"], strict=True))}
        pairs = np.asarray([(look[(tuple(key), h)], i) for i, (key, h) in enumerate(
            zip(ty["keys"], ty["prefix_hash"], strict=True)) if (tuple(key), h) in look])
        a, b = tx["coords"][pairs[:, 0], 1], ty["coords"][pairs[:, 1], 1]
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        dot = (a * b).sum(1) / np.maximum(denom, EPS)
        groups = tx["keys"][pairs[:, 0]]
        categories = {}
        for category in ("last_prompt", "continuation", "all"):
            valid = select(groups, category) & (denom > EPS)
            w, n = weights_for(groups[valid, 0])
            categories[category] = dict(n=int(valid.sum()), questions=n, **weighted_stat(dot[valid].clip(-1, 1), w))
        output[f"k{x}_k{y}"] = categories
    return output


def audit(root):
    weighted, cosines, sources = {}, {}, {}
    for family in ("candidate", "generation"):
        for split in ("arc_validation", "openbookqa_test"):
            data = {}
            for k in DEPTHS:
                stem = root / f"A1_{family}_{split}_k{k}"
                path = stem.with_suffix(".npz")
                computed = digest(path)
                if computed != read(stem.with_suffix(".json"))["coordinates_sha256"]:
                    raise RuntimeError(f"Geometry checksum mismatch for {path}")
                sources[path.name] = computed
                with np.load(path, allow_pickle=False) as arr:
                    data[k] = {key: arr[key] for key in ("keys", "prefix_hash", "coords", "scalars", "cast_off")}
                if len({tuple(key) for key in data[k]["keys"]}) != len(data[k]["keys"]):
                    raise RuntimeError(f"Duplicate question/choice/predictor keys in {path}")
                weighted[f"{family}/{split}/k{k}"] = {name: measure(data[k], name)
                                                      for name in ("last_prompt", "continuation", "all")}
            cosines[f"{family}/{split}"] = across_depth(data)
    cross_domain = {}
    for a, b in combinations(sorted(weighted), 2):
        family_a, split_a, ka = a.split("/")
        family_b, split_b, kb = b.split("/")
        if ka != kb or (family_a != family_b and split_a != split_b):
            continue
        angle = {}
        for category in ("last_prompt", "continuation", "all"):
            spectra = [weighted[k][category]["updates"]["clipped_before_cast"]["spectra"]["unit_direction"] for k in (a, b)]
            angle[category] = {name: float(abs(np.dot(spectra[0][name]["axis"], spectra[1][name]["axis"])))
                               for name in ("centered", "uncentered")}
        cross_domain[f"{a}__{b}"] = angle
    for receipt in sorted(root.glob("A2_FACTORIAL*.json")):
        suffix = receipt.stem.removeprefix("A2_FACTORIAL")
        a2 = read(receipt)
        if set(a2["depths"]) != {f"k{k}" for k in DEPTHS}:
            raise RuntimeError(f"Candidate score summary has incomplete depth coverage: {receipt}")
        for k, values in a2["depths"].items():
            for file, expected in [(f"A2_SCORES_{k}{suffix}.jsonl", values["candidate_scores_sha256"]),
                                   *((f"A2_scores_{k}_{name}{suffix}.jsonl", cell["scores_sha256"])
                                     for name, cell in values["costs"].items())]:
                actual = digest(root / file)
                if actual != expected:
                    raise RuntimeError(f"Candidate checksum mismatch for {file}")
                sources[file] = actual
    write(root / "A1_QUESTION_WEIGHTED.json", dict(reports=weighted, matched_prefix_across_depth=cosines,
          cross_domain_axis_cosines=cross_domain, source_sha256=sources,
          weighting="Each represented question contributes unit mass; observed positions within a question receive equal weight.",
          cast_centered="For centered actual-cast spectra, out-of-span uncentered energy is an explicit upper bound on centered extra variance."))


def jsonl(path):
    with path.open(encoding="utf-8") as source:
        by_id = {}
        for line in source:
            row = json.loads(line)
            if row["id"] in by_id:
                raise RuntimeError(f"Duplicate question in {path}")
            by_id[row["id"]] = row
        return by_id


def verify(path, expected):
    if digest(path) != expected:
        raise RuntimeError(f"File has differing hash: {path}")
    return jsonl(path)


def generation_audit(root, history):
    prefixes = ("", *(p.stem.removeprefix("A2_GENERATION") for p in sorted(root.glob("A2_GENERATION_learned_*.json"))))
    data = {}
    for suffix in prefixes:
        summary = read(root / f"A2_GENERATION{suffix}.json")
        if set(summary) != {f"arc_validation_k{k}" for k in DEPTHS}:
            raise RuntimeError(f"ARC generation summary has incomplete depth coverage: {suffix}")
        for key, cell in summary.items():
            depth = int(key[-1])
            rows = data.setdefault(depth, {})
            for method, expected in cell["historical_sources"].items():
                path = history / f"arc_validation__{method}__k{depth}.jsonl"
                rows[method] = verify(path, expected)
            for method, values in cell["methods"].items():
                path = root / f"A2_generation_{key}_{method}{suffix}.jsonl"
                rows[method + suffix.removeprefix("_learned")] = verify(path, values["cost"]["sha256"])
    order = list(data[1]["native"])
    if len(order) != 299 or any(set(records) != set(order) for cell in data.values() for records in cell.values()):
        raise RuntimeError("Generation has differing question coverage")

    metrics = {}
    counts = {}
    for depth, cell in sorted(data.items()):
        k = f"k{depth}"
        metrics[k], counts[k] = {}, {}
        for method, rows in cell.items():
            metrics[k][method] = {name: np.asarray([rows[q][name] for q in order], dtype=int) for name in ("correct", "valid")}
            counts[k][method] = dict(**{name: int(x.sum()) for name, x in metrics[k][method].items()},
                emitted_tokens=sum(len(r["tokens"]) for r in rows.values()),
                invalid_categories=dict(Counter(r["invalid_category"] for r in rows.values() if not r["valid"])))
    names = tuple(metrics["k1"])
    if any(set(cell) != set(names) for cell in metrics.values()):
        raise RuntimeError("Generation methods differ across depths")
    samples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 299, size=(10_000, 299))
    pooled, common_valid = {}, {}
    controls = ("native", "linear8_t1", "linear8_t2", "fixed", "shuffled")
    for target in names:
        for baseline in controls:
            if baseline == target:
                continue
            key = f"{target}_vs_{baseline}"
            pooled[key], common_valid[key] = {}, {}
            for metric in ("correct", "valid"):
                diff = np.stack([metrics[f"k{k}"][target][metric] - metrics[f"k{k}"][baseline][metric]
                                 for k in (2, 3, 4)]).mean(axis=0)
                pooled[key][metric] = dict(delta_pp=100*float(diff.mean()),
                    clustered_ci95_pp=(100 * np.quantile(diff[samples].mean(axis=1), [.025, .975])).tolist())
            for depth, cell in metrics.items():
                a, b = cell[target], cell[baseline]
                in_common = a["valid"] & b["valid"]
                common_valid[key][depth] = dict(n=int(in_common.sum()), correct_a=int((a["correct"] & in_common).sum()),
                    correct_b=int((b["correct"] & in_common).sum()),
                    rescued=int(((a["correct"] > b["correct"]) & in_common).sum()),
                    damaged=int(((a["correct"] < b["correct"]) & in_common).sum()))
    write(root / "A2_GENERATION_AUDIT.json", dict(counts=counts, k2_k4_question_clustered_effects=pooled,
        both_valid_descriptive=common_valid, question_count=len(order), bootstrap_draws=len(samples), seed=BOOTSTRAP_SEED))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--historical-generation", type=Path)
    args = parser.parse_args()
    audit(args.out)
    if args.historical_generation:
        generation_audit(args.out, args.historical_generation)
