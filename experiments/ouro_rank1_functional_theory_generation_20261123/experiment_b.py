"""Rank-1 counterfactual interchange of semantically matched source edges."""

import argparse
from collections import Counter
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

import b_content_design as design
from b_content_path import EdgeProbe, matched_deltas, path_patches, rms

SITES = (15, 19, 23)
ROTATION_SEED = 20261123891
RESAMPLE_SEED = 20261123897
BOOTSTRAP = 6000
NORMAL = ("native", "rank1", "rotated", "natural")
SWAPS = ("native_target_swap", "native_control_swap", "rank1_target_swap", "rank1_control_swap")
RESTORE = ("rank1_target_restore", "rank1_control_restore",
           "rank1_target_restore_matched", "rank1_control_restore_matched")


def digest(path):
    import hashlib
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save_jsonl(path, records):
    import json
    temporary = path.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8") as output:
        for row in records:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def rotation(width, device):
    rng = torch.Generator().manual_seed(ROTATION_SEED)
    return (torch.randperm(width, generator=rng).to(device),
            (2 * torch.randint(2, (width,), generator=rng) - 1).to(device))


def reshape_requests(pairs, ranked):
    if ranked:
        return [(row, choice, ()) for pair in pairs for choice in (0, 1) for row in pair]
    return [(row, None, ()) for pair in pairs for row in pair]


def batch(requests, device):
    input_ids, owners, positions, targets, queries, sources = [], [], [], [], [], []
    for index, (row, candidate, continuation) in enumerate(requests):
        prefix = row["prefix"]
        cursor = len(prefix) - 1
        if candidate is None:
            path = prefix + list(continuation)
            scored = range(cursor, len(path))
            target = [0] * len(scored)
        else:
            target = row["candidates"][candidate]
            path = prefix + target[:-1]
            scored = range(cursor, len(path))
        input_ids.append(path)
        owners.extend([index] * len(scored))
        positions.extend(scored)
        targets.extend(target)
        queries.append(cursor)
        sources.append(row["source_positions"])
    width = max(map(len, input_ids))
    padded = [row + [0] * (width - len(row)) for row in input_ids]
    ids, group, positions, target, query, source = [
        torch.tensor(value, device=device) for value in (padded, owners, positions, targets, queries, sources)]
    return (ids, group, positions, target), query, source


class Runner:
    def __init__(self, model, net, edit_forward):
        self.model = model
        self.net = net
        self.forward = edit_forward
        self.device = next(model.parameters()).device
        self.permutation, self.signs = rotation(model.config.hidden_size, self.device)
        self.calls = 0
        self.padded = 0

    def run(self, packed, edit=None, probe=None):
        self.calls += 1
        self.padded += packed[0].numel()
        result = self.forward(self.model, packed, anchor=3 if edit is not None else None,
                              update=edit, attention_probe=probe)
        if probe is not None:
            probe.audit()
        return result

    def edits(self, reference):
        h = reference["raw"][3]
        rank1 = self.net.delta(h, reference["raw"][2]).float()
        natural = h - reference["raw"][2]
        natural_size = rms(natural)
        rank_size = rms(rank1)
        if ((natural_size == 0) & (rank_size > 0)).any():
            raise RuntimeError("Cannot norm-match a zero natural displacement")
        return dict(native=None, rank1=rank1, rotated=rank1[:, self.permutation] * self.signs,
                    natural=natural * (rank_size / natural_size.clamp_min(1e-20)).unsqueeze(-1))

    def scores(self, hidden, packed, ranked):
        if not ranked:
            logits = self.model.lm_head(hidden)
            return logits.argmax(-1).tolist()
        logits = self.model.lm_head(hidden).float().log_softmax(-1)
        return logits.gather(1, packed[-1].unsqueeze(-1)).reshape(-1, 2).tolist()

    def measure(self, requests, layer, screening=False):
        packed, query, sources = batch(requests, self.device)
        ranked = requests[0][1] is not None
        native_probe = EdgeProbe(layer, query, sources, record=True)
        native_out, baseline = self.run(packed, probe=native_probe)
        edit = self.edits(baseline)
        rank_probe = EdgeProbe(layer, query, sources, record=True)
        rank_out, rank_trace = self.run(packed, edit["rank1"], rank_probe)
        if not torch.equal(rank_trace["raw"][3], baseline["raw"][3]):
            raise RuntimeError("The Rank-1 edit received a different original anchor")
        scores = dict(native=self.scores(native_out, packed, ranked),
                      rank1=self.scores(rank_out, packed, ranked))
        arms = {}
        donor = torch.arange(len(requests), device=self.device) ^ 1
        patches = path_patches(native_probe.trace, rank_probe.trace, donor, screening)
        for name, delta in patches.items():
            arm = "native" if name.startswith("native_") else "rank1"
            probe = EdgeProbe(layer, query, sources, patch=delta)
            hidden, state = self.run(packed, edit[arm], probe)
            if arm == "rank1" and not torch.equal(rank_trace["edits"], state["edits"]):
                raise RuntimeError("The path edit changed the direct Rank-1 update")
            scores[name] = self.scores(hidden, packed, ranked)
            arms[name] = dict(requested_patch_rms=rms(delta).tolist(),
                              realized_patch_rms=probe.applied.tolist())
        if not screening:
            for name in ("rotated", "natural"):
                hidden, state = self.run(
                    packed, edit[name])
                scores[name] = self.scores(hidden, packed, ranked)
                arms[name] = dict(direct_update_rms=rms(state["edits"]).tolist())
        traces = {}
        for label, probe in (("native", native_probe), ("rank1", rank_probe)):
            traces[label] = dict(
                source_mass=probe.trace["weights"].sum(dim=2).mean(-1).tolist(),
                readout_reconstruction_relative_rms=probe.trace["relative_reconstruction_rms"].tolist())
        traces["rank1"]["direct_update_rms"] = rms(rank_trace["edits"]).tolist()
        return scores, arms, traces

    def continuation(self, requests, arm):
        packed, _, _ = batch(requests, self.device)
        normalized = arm if arm in NORMAL else "native" if arm.startswith("native_") else "rank1"
        hidden, baseline = self.run(packed)
        if normalized != "native":
            edit = self.edits(baseline)[normalized]
            hidden, _ = self.run(packed, edit)
        last = torch.bincount(packed[1], minlength=len(requests)).cumsum(0) - 1
        return self.scores(hidden[last], packed, ranked=False)


def rank_records(requests, scores):
    output = {}
    for i, (row, choice, _) in enumerate(requests):
        entry = output.setdefault(row["id"], dict(id=row["id"], group=row["group"],
                                   family=row["family"], options=row["options"],
                                   code_index=row["code_index"], scores={}))
        for arm, values in scores.items():
            accumulated = entry["scores"].setdefault(arm, [None, None])
            accumulated[choice] = values[i]
    if any(any(value is None for value in pair) for row in output.values() for pair in row["scores"].values()):
        raise RuntimeError("Expected two complete answer scores per question")
    return list(output.values())


def generation_result(row, method, tokens, tokenizer, stop):
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    clean = text.strip().casefold()
    candidates = "|".join(re.escape(value) for value in row["options"])
    first = re.match(rf"^({candidates})(?=$|[\s.,;:!?])", clean)
    strict = re.fullmatch(rf"({candidates})\.?\s*", clean)
    answer = first.group(1) if first else None
    strict_answer = strict.group(1) if strict else None
    gold = row["options"][row["code_index"]]
    return dict(id=row["id"], group=row["group"], family=row["family"], method=method, options=row["options"],
                gold=gold, text=text, tokens=tokens, stop=stop, parsed=answer, strict_answer=strict_answer,
                valid=answer is not None, correct=answer == gold, strict_correct=strict_answer == gold)


def free_generate(runner, requests, first_tokens, tokenizer, arm, maximum):
    active = list(range(len(requests)))
    ids = [[] for _ in requests]
    stop = ["limit"] * len(requests)
    predicted = first_tokens
    special = set(tokenizer.all_special_ids)
    for step in range(maximum):
        if step:
            suffix = [(requests[i][0], None, tuple(ids[i])) for i in active]
            predicted = runner.continuation(suffix, arm)
        continuing = []
        for i, token in zip(active, predicted, strict=True):
            if token in special:
                stop[i] = "special"
                continue
            ids[i].append(token)
            text = tokenizer.decode(ids[i], skip_special_tokens=True)
            if "\n" in text:
                stop[i] = "newline"
                continue
            continuing.append(i)
        active = continuing
        if not active:
            break
    return [generation_result(row, arm, ids[i], tokenizer, stop[i]) for i, (row, _, _) in enumerate(requests)]


def run_split(common, out, runner, pairs, layer, pair_batch, tokenizer, mode, max_tokens):
    chunk_dir = out / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    costs = Counter()
    interventions = []
    measurements = []
    for start in range(0, len(pairs), pair_batch):
        group = pairs[start:start + pair_batch]
        requests = reshape_requests(group, ranked=mode != "generate")
        identity = [[row["id"], choice] for row, choice, _ in requests]
        path = chunk_dir / f"{start // pair_batch:04d}.json"
        if path.exists():
            chunk = common.load(path)
            if chunk["identity"] != identity:
                raise RuntimeError(f"Resume input identity mismatch in {path}")
        else:
            runner.calls, runner.padded = 0, 0
            timer = time.perf_counter()
            scores, arms, traces = runner.measure(requests, layer, screening=mode == "dev")
            if mode == "generate":
                scored = [record for label, predictions in scores.items()
                          for record in free_generate(runner, requests, predictions, tokenizer, label, max_tokens)]
            else:
                scored = rank_records(requests, scores)
            chunk = dict(identity=identity, rows=scored, source_intervention=arms, source_measurements=traces,
                         cost=dict(calls=runner.calls, padded_tokens=runner.padded,
                                   seconds=time.perf_counter() - timer))
            common.save(path, chunk)
            if start % (4 * pair_batch) == 0:
                print(f"B {mode} layer {layer}, pairs {start + len(group)}/{len(pairs)}", flush=True)
        rows.extend(chunk["rows"])
        costs.update(chunk["cost"])
        interventions.append(chunk["source_intervention"])
        measurements.append(chunk["source_measurements"])
    save_jsonl(out / "PREDICTIONS.jsonl", rows)
    result = dict(pair_count=len(pairs), prompt_count=2 * len(pairs),
                  initial_model_sequences=len(pairs) * (2 if mode == "generate" else 4),
                  calls=costs["calls"], pass_calls=costs["calls"] * 4,
                  padded_tokens=costs["padded_tokens"], seconds=costs["seconds"])
    common.save(out / "COST.json", result)
    common.save(out / "INTERVENTION_AUDIT.json", intervention_audit(interventions, measurements))
    return rows, result


def intervention_audit(interventions, measurements):
    def merge(rows, name, key):
        return np.concatenate([row[name][key] for row in rows])

    def summary(values):
        return dict(mean=float(values.mean()), median=float(np.median(values)), max=float(values.max()))

    matched = {}
    for label, names in (("donor_swaps", SWAPS), ("native_restore", RESTORE[2:])):
        if names[0] not in interventions[0]:
            continue
        requested = np.stack([merge(interventions, name, "requested_patch_rms") for name in names])
        realized = np.stack([merge(interventions, name, "realized_patch_rms") for name in names])
        largest = requested.max(axis=0)
        valid = largest > 0
        matched[label] = dict(
            requested_rms=summary(largest),
            requested_zero=int((~valid).sum()),
            requested_max_across_arms_spread=float(np.ptp(requested, axis=0).max()),
            after_bf16_spread_divided_by_requested_arm_rms=summary(
                np.ptp(realized[:, valid], axis=0) / largest[valid]) if valid.any() else None,
            by_arm={name: dict(realized_rms=summary(realized[index]), realized_zero=int((realized[index] == 0).sum()))
                    for index, name in enumerate(names)})
    recorded = {}
    for name in ("native", "rank1"):
        mass = merge(measurements, name, "source_mass")
        recorded[name] = dict(
            target_attention_mass=summary(mass[:, 0]),
            control_attention_mass=summary(mass[:, 1]),
            relative_reconstruction_rms=summary(merge(measurements, name, "readout_reconstruction_relative_rms")))
    return dict(matched_patches=matched, readout=recorded)


def group_values(rows, metric):
    values = {}
    for row in rows:
        target = row["code_index"]
        for arm, numbers in row["scores"].items():
            scores = np.asarray(numbers)
            if metric == "first_margin":
                totals = scores[:, 0]
            else:
                totals = scores.sum(axis=-1)
                if metric == "margin_norm":
                    totals /= [len(x) for x in row["options"]]
            separation = totals[target] - totals[target ^ 1]
            group = values.setdefault(arm, {}).setdefault(row["group"], [])
            group.append(float(totals.argmax() == target) if metric in ("acc", "paired_correct") else separation)
    return {arm: np.array([np.prod(x) if metric == "paired_correct" else np.mean(x)
                           for x in groups.values()]) for arm, groups in values.items()}


def contrasts(values):
    learned = values["rank1_control_swap"] - values["rank1_target_swap"]
    native = values["native_control_swap"] - values["native_target_swap"]
    out = dict(rank1_site_swap=learned, native_site_swap=native, rank1_by_site=learned - native,
               rank1_over_native=values["rank1"] - values["native"])
    if "rank1_target_restore" in values:
        out.update(
            exact_restoration=values["rank1_control_restore"] - values["rank1_target_restore"],
            matched_restoration=values["rank1_control_restore_matched"] - values["rank1_target_restore_matched"],
            rotated_over_native=values["rotated"] - values["native"],
            natural_over_native=values["natural"] - values["native"])
    return out


def interval(x, sample, alpha=.05):
    lower, upper = np.quantile(x[sample].mean(axis=1), (alpha / 2, 1 - alpha / 2))
    return dict(mean=float(x.mean()), lower=float(lower), upper=float(upper))


def rank_summary(rows):
    groups = list(dict.fromkeys(row["group"] for row in rows))
    family = {row["group"]: row["family"] for row in rows}
    sample = np.random.default_rng(RESAMPLE_SEED).integers(len(groups), size=(BOOTSTRAP, len(groups)))
    metrics = {}
    for metric in ("margin", "margin_norm", "first_margin", "acc", "paired_correct"):
        arms = group_values(rows, metric)
        comparisons = {}
        for name, effect in contrasts(arms).items():
            comparisons[name] = dict(
                paired=interval(effect, sample),
                adjusted=interval(effect, sample, alpha=.05 / 6),
                families={tag: float(np.mean([effect[i] for i, key in enumerate(groups) if family[key] == tag]))
                          for tag in design.FAMILIES})
        metrics[metric] = dict(arms={arm: float(v.mean()) for arm, v in arms.items()}, contrasts=comparisons)
    site = metrics["margin"]["contrasts"]
    return dict(groups=len(groups), questions=len(rows), metrics=metrics,
                decision=dict(rank1_target_effect_supported=site["rank1_site_swap"]["adjusted"]["lower"] > 0,
                              rank1_extra_site_dependence_supported=site["rank1_by_site"]["adjusted"]["lower"] > 0,
                              positive_target_effect_both_families=all(
                                  m > 0 for m in site["rank1_site_swap"]["families"].values()),
                              positive_rank1_interaction_both_families=all(
                                  m > 0 for m in site["rank1_by_site"]["families"].values())),
                adjusted_test_family="Six planned contrasts on primary raw full-candidate margin and strict paired generation: rank1 site, native site, and rank1 × site in each. Other adjusted intervals use the same multiplier for comparison.")


def generation_summary(rows):
    arms = {}
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], {}).setdefault(row["group"], []).append(row)
    for name, groups in grouped.items():
        selected = [row for values in groups.values() for row in values]
        arms[name] = dict(rows=len(selected), pairs=len(groups), valid=sum(x["valid"] for x in selected),
                          correct=sum(x["correct"] for x in selected),
                          strict_correct=sum(x["strict_correct"] for x in selected),
                          both_correct=sum(all(x["correct"] for x in two) for two in groups.values()),
                          both_strict_correct=sum(all(x["strict_correct"] for x in two) for two in groups.values()),
                          stops=dict(Counter(x["stop"] for x in selected)),
                          generated_tokens=sum(len(x["tokens"]) for x in selected))
    sample = np.random.default_rng(RESAMPLE_SEED).integers(
        len(grouped["native"]), size=(BOOTSTRAP, len(grouped["native"])))
    changes = {}
    for metric in ("strict_correct", "correct", "valid"):
        values = {name: np.array([np.prod([row[metric] for row in pair]) for pair in groups.values()])
                  for name, groups in grouped.items()}
        changes[metric] = {name: dict(paired=interval(x, sample), adjusted=interval(x, sample, .05 / 6))
                           for name, x in contrasts(values).items()}
    return dict(arms=arms, paired=changes)


def lock_metadata(args, common, runtime, checkpoint):
    prior = args.prior_study
    here = Path(__file__).resolve().parent
    return dict(
        model=runtime, rank1_checkpoint_sha256=digest(checkpoint),
        prior_frozen_manifest_sha256=digest(prior / "results" / "FREEZE.json"),
        prior_source={name: digest(prior / name) for name in ("controller.py", "c_common.py")},
        source={name: digest(here / name) for name in
                ("experiment_b.py", "b_content_path.py", "b_content_design.py")},
        dev_layers_zero_based=list(SITES), pair_batch=args.pair_batch, max_new_tokens=args.max_new,
        depth=4, design_seed=design.SEEDS, rotation_seed=ROTATION_SEED, bootstrap_seed=RESAMPLE_SEED,
        intervention="Pass 4 pre-o_proj source contributions at first answer prediction; prefix unchanged. "
                     "Both matched fact spans hold identical lexical content in each world. Donor is the paired reversal. "
                     "Receiver's actual attention weights multiply donor changes to each value span. Scale all four "
                     "requested (Native, Rank1) × (target, control) deltas to the smallest per-recipient RMS.",
        direct_controls="Rank1 same hidden-RMS bound as published Rank1-T1; signed-permuted direction and "
                        "same-token pass 3 minus pass 2 raw displacement both normalized to Rank1 update RMS.",
        restoration="Exact computed native span contribution minus Rank1 span contribution, then separately "
                    "the equalized target/control pair of these same requested differences.",
        generation="Full vocabulary greedy with no answer masking. Only the first answer step receives a source "
                   "intervention; Rank1 and direction controls remain active at every predictor where applicable.")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "archive", "prior-study", "minimal-study", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--pair-batch", type=int, default=8, help="Each pair occupies four candidate sequences")
    parser.add_argument("--max-new", type=int, default=6)
    parser.add_argument("--stage", choices=("all", "rank", "generate"), default="all")
    args = parser.parse_args()
    if args.max_new < 2 or args.pair_batch < 1:
        parser.error("Expected positive pair batch and max-new at least two")
    sys.path[:0] = [str(path.resolve()) for path in (args.prior_study, args.minimal_study, args.archive)]
    import c_common as common
    from controller import load_controller
    import run_arc

    freeze = common.load(args.prior_study / "results" / "FREEZE.json")
    weight = freeze["weights"]["rank1_t1"]
    checkpoint = args.prior_study / "results" / "weights" / weight["file"]
    if digest(checkpoint) != weight["sha256"] or weight["form"] != "rank1":
        raise RuntimeError("The historical Rank-1 T1 snapshot differs from its frozen manifest")
    model, tokenizer, runtime = run_arc.load_runtime(args.model)
    model.requires_grad_(False)
    metadata = lock_metadata(args, common, runtime, checkpoint)
    common.make_manifest(args.out, metadata)
    tokenizer_rows = design.population()
    rows = {split: design.encode(tokenizer, pairs) for split, pairs in tokenizer_rows.items()}
    for split, pairs in rows.items():
        save_jsonl(args.out / f"{split.upper()}_DECK.jsonl", [row for pair in pairs for row in pair])
    model_device = next(model.parameters()).device
    runner = Runner(model, load_controller(checkpoint, device=model_device), common.forward)
    dev = {}
    for layer in SITES:
        subfolder = args.out / f"dev_layer_{layer:02}"
        pred, cost = run_split(common, subfolder, runner, rows["dev"], layer, args.pair_batch,
                               tokenizer, "dev", args.max_new)
        summary = rank_summary(pred)
        common.save(subfolder / "SUMMARY.json", summary)
        dev[layer] = dict(
            interaction=summary["metrics"]["margin"]["contrasts"]["rank1_by_site"],
            rank1_site=summary["metrics"]["margin"]["contrasts"]["rank1_site_swap"], cost=cost,
            predictions_sha256=digest(subfolder / "PREDICTIONS.jsonl"))
    selected = max(SITES, key=lambda layer: dev[layer]["interaction"]["paired"]["mean"])
    lock = dict(chosen_layer=selected, candidates={str(k): v for k, v in dev.items()},
                dev_deck_sha256=digest(args.out / "DEV_DECK.jsonl"),
                rule="Highest Dev24 raw two-token margin Rank1 × target-vs-irrelevant swap interaction; earliest layer wins ties.")
    lock_path = args.out / "LOCKED_SITE.json"
    if lock_path.exists():
        if common.load(lock_path) != lock:
            raise RuntimeError("The development-only lock differs")
    else:
        common.save(lock_path, lock)
    print(f"B locked dev layer {selected}; holding all remaining conditions fixed", flush=True)
    if args.stage in ("all", "rank"):
        pred, _ = run_split(common, args.out / "heldout", runner, rows["heldout"], selected,
                            args.pair_batch, tokenizer, "heldout", args.max_new)
        common.save(args.out / "heldout" / "SUMMARY.json", rank_summary(pred))
    if args.stage in ("all", "generate"):
        selected_pairs = design.generations(rows["heldout"])
        pred, _ = run_split(common, args.out / "generation", runner, selected_pairs, selected,
                            args.pair_batch, tokenizer, "generate", args.max_new)
        common.save(args.out / "generation" / "SUMMARY.json", generation_summary(pred))
    print("B complete", args.stage, flush=True)


if __name__ == "__main__":
    main()
