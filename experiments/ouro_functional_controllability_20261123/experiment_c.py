"""Train-locked K4 direction-by-additive-path factorial for Ouro."""

import argparse
from pathlib import Path

import numpy as np
import torch

from c_common import (
    BRANCHES, METRICS, SEED, arguments, bootstrap, candidate_trace, cosine, digest, forward,
    hash_key, load, load_runtime, make_manifest, paired, population, profile,
    replay_batches, rms, save, score,
)

BANDS = {"early": (0, 8), "middle": (8, 16), "late": (16, 24)}
DIRECTIONS = ("learned", "rotated")
PRIMARY = "utility_norm"


def band_keys(branch, band):
    start, stop = BANDS[band]
    return {(4, layer, branch) for layer in range(start, stop)}


def rotation(axis):
    rng = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(axis.numel(), generator=rng).to(axis.device)
    signs = (2 * torch.randint(2, (axis.numel(),), generator=rng) - 1).to(axis.device)
    rotated = axis[perm] * signs
    return rotated / rms(rotated)


def branch_tag(direction, branch, band):
    return f"{direction}_clamped/{branch}/{band}"


def updates(base, axis, rotated, dose):
    h = base["raw"][3]
    scale = dose * rms(h).unsqueeze(-1)
    return dict(learned=scale * axis, rotated=scale * rotated)


def ensure_matched(edit, baseline, full):
    if not torch.equal(edit["raw"][3], baseline["raw"][3]):
        raise RuntimeError("The direct injection used a different pre-intervention state")
    if full is not None and not torch.equal(edit["edits"], full["edits"]):
        raise RuntimeError("Clamping changed the realized direct residual update")


def trace(edited, base):
    delta = edited["edits"]
    raw = edited["raw"][4] - base["raw"][4]
    readout = edited["normalized"][4] - base["normalized"][4]
    result = dict(direct_edit_rms=rms(delta),
                  direct_edit_relative_rms=rms(delta) / rms(base["raw"][3]).clamp_min(1e-20),
                  final_raw_delta_rms=rms(raw),
                  final_raw_direction_cos=cosine(raw, delta),
                  final_readout_delta_rms=rms(readout),
                  final_readout_direction_cos=cosine(readout, delta))
    for (turn, branch), values in edited["changes"].items():
        result[f"turn{turn}/{branch}_write_delta_rms"] = values
    return result


def evaluate(archive, model, axis, rotated, dose, locations, packed, subset):
    layers = set().union(*(band_keys(branch, band) for branch, band in locations))
    hidden, base = forward(model, packed, branch_layers=layers)
    predictions = {"native": score(archive, model, packed, subset, hidden)}
    traces = {}
    direct = updates(base, axis, rotated, dose)
    for direction, edit in direct.items():
        hidden, full = forward(model, packed, 3, edit, branch_layers=layers,
                               reference=base, trace_turns=(4,))
        ensure_matched(full, base, None)
        name = f"{direction}_live"
        predictions[name] = score(archive, model, packed, subset, hidden)
        traces[name] = candidate_trace(packed, len(subset), trace(full, base))
        for branch, band in locations:
            clamp = band_keys(branch, band)
            hidden, active = forward(model, packed, 3, edit, branch_layers=clamp,
                                     reference=base, clamp_layers=clamp)
            ensure_matched(active, base, full)
            name = branch_tag(direction, branch, band)
            predictions[name] = score(archive, model, packed, subset, hidden)
            traces[name] = candidate_trace(packed, len(subset), trace(active, base))
    return dict(scores=predictions, traces=traces)


def condition_table(info, sample, tags):
    result = {}
    for tag in tags:
        result[tag] = {}
        for metric in METRICS:
            values = profile(info, tag, metric)
            deltas = values - profile(info, "native", metric)
            stats = dict(mean=float(values.mean()), versus_native=paired(deltas, sample))
            if metric.startswith("acc"):
                stats.update(correct=int(values.sum()), rescued=int((deltas > 0).sum()),
                             damaged=int((deltas < 0).sum()))
            result[tag][metric] = stats
    return result


def grouped_traces(info, tags):
    stock = profile(info, "native", "acc_norm")
    live = profile(info, "learned_live", "acc_norm")
    delta = live - stock
    groups = dict(rescue=delta > 0, damage=delta < 0,
                  stable_correct=(delta == 0) & (stock == 1),
                  stable_wrong=(delta == 0) & (stock == 0))
    result = {}
    for label, include in groups.items():
        index = np.flatnonzero(include)
        arms = {}
        for tag in tags:
            if tag == "native" or len(index) == 0:
                continue
            row_traces = [info[i]["trace"][tag] for i in index]
            arm = {key: float(np.mean([np.mean(row[key]) for row in row_traces]))
                   for key in row_traces[0]}
            for metric in ("margin", "margin_norm", "utility_norm"):
                arm[f"{metric}_change_vs_native"] = float((
                    profile(info, tag, metric)[index] - profile(info, "native", metric)[index]).mean())
            arms[tag] = arm
        result[label] = dict(count=len(index), question_ids=[info[i]["id"] for i in index], traces=arms)
    return result


def interaction_table(info, sample, locations):
    contrasts = {}
    for branch, band in locations:
        a, b = "learned_live", branch_tag("learned", branch, band)
        c, d = "rotated_live", branch_tag("rotated", branch, band)
        interaction = {}
        for metric in METRICS:
            arms = {name: profile(info, name, metric) for name in (a, b, c, d)}
            learned = arms[a] - arms[b]
            rotated = arms[c] - arms[d]
            difference = learned - rotated
            interaction[metric] = dict(
                difference=paired(difference, sample),
                adjusted=paired(difference, sample, .05 / (len(locations) * len(METRICS))),
                learned_live_minus_clamp=paired(learned, sample),
                rotated_live_minus_clamp=paired(rotated, sample))
        contrasts[f"{branch}/{band}"] = interaction
    return contrasts


def run_factorial(args, archive, model, tokenizer, axis, rotated, split, locations, metadata):
    out = args.out / split
    train = split == "train"
    items, info = population(archive, tokenizer, args.corpus, split,
                             train_subset=args.train_count if train else None)
    if not train and len(info) != 299:
        raise RuntimeError("C requires all 299 ARC Validation questions")
    if train and len(info) != args.train_count:
        raise RuntimeError("ARC Train subset count differs from the manifest")
    make_manifest(out, dict(**metadata, split=split,
                            roles=["screen" if train else "confirm"], locations=[list(x) for x in locations],
                            parquet_sha256=digest(args.corpus / f"{split}.parquet"),
                            question_id_sha256=hash_key("questions", "/".join(row["id"] for row in info))))
    replay_batches(archive, items, info, args.batch_size, out,
                   lambda packed, subset: evaluate(archive, model, axis, rotated, args.dose,
                                                   locations, packed, subset))
    ts = list(info[0]["scores"])
    sample = bootstrap(info)
    result = dict(questions=len(info), candidates=len(items),
                  primary_metric=PRIMARY, question_ids=[row["id"] for row in info],
                  conditions=condition_table(info, sample, ts),
                  interaction=interaction_table(info, sample, locations),
                  grouped_responses=grouped_traces(info, ts))
    save(out / "C_SUMMARY.json", result)
    return result


def lock_locations(args, train, metadata):
    estimates = {name: row[PRIMARY]["difference"]["mean"] for name, row in train["interaction"].items()}
    locations = {branch: max(BANDS, key=lambda band: estimates[f"{branch}/{band}"]) for branch in BRANCHES}
    lock = dict(locations=locations, train_estimates=estimates,
                question_count=train["questions"], question_ids=train["question_ids"],
                direction_sha256=metadata["direction_sha256"], dose=args.dose,
                selection_rule=f"Highest ARC Train {PRIMARY} DiD within each branch, early then middle then late for ties",
                train_summary_sha256=digest(args.out / "train" / "C_SUMMARY.json"),
                zero_based_half_open_layers={key: list(value) for key, value in BANDS.items()})
    path = args.out / "LOCKED_LOCATIONS.json"
    if path.exists():
        if load(path) != lock:
            raise RuntimeError("Locked locations differ from the Train screen")
    else:
        save(path, lock)
    return lock


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    parser.add_argument("--stage", choices=("factorial", "paths"), default="factorial")
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--dose", type=float, default=.45)
    args = parser.parse_args()
    if args.batch_size < 1 or not 2 <= args.train_count <= 1119 or not np.isfinite(args.dose) or args.dose <= 0:
        parser.error("Expected positive finite dose and batch size and 2–1119 ARC Train questions")
    archive, model, tokenizer, runtime, axis = load_runtime(args)
    rotated = rotation(axis)
    metadata = dict(depth=4, runtime=runtime, dose=args.dose, batch_size=args.batch_size,
                    direction_sha256=digest(args.direction), script_sha256=digest(__file__),
                    common_sha256=digest(Path(__file__).with_name("c_common.py")),
                    scoring_source_sha256=digest(args.archive / "run_arc.py"),
                    rotation_rule="Seeded signed coordinate permutation subsequently divided by its RMS",
                    rotation_seed=SEED, axis_rotation_cosine=float(cosine(axis, rotated)),
                    injection_rule="RMS-scaled shared axis at all continuation predictor positions before pass 3 RMSNorm",
                    clamp_rule="Same-input native additive write at every position of the chosen branch and layers in pass 4",
                    utility="Gold score minus logsumexp of non-gold scores")
    if args.stage == "paths":
        from c_attention import run_paths
        run_paths(args, archive, model, tokenizer, axis, rotated, metadata)
        return
    screening = [(branch, band) for branch in BRANCHES for band in BANDS]
    train = run_factorial(args, archive, model, tokenizer, axis, rotated, "train", screening, metadata)
    lock = lock_locations(args, train, metadata)
    print("Locked C locations", lock["locations"], flush=True)
    result = run_factorial(args, archive, model, tokenizer, axis, rotated, "validation",
                           list(lock["locations"].items()), metadata)
    print("C complete", {k: v[PRIMARY]["difference"] for k, v in result["interaction"].items()}, flush=True)


if __name__ == "__main__":
    main()
