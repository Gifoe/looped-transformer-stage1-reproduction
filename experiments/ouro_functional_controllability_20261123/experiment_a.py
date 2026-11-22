"""Audit frozen Rank-8 geometry and isolate direction versus amplitude in Ouro."""

import argparse
from collections import defaultdict
from itertools import combinations
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from a_geometry import Geometry, SEED, EPS, across_depth, capture, geometry_summary
from a_geometry import position_masks, rms, shuffled_ratios, stats, unit_rms


DEPTHS = (1, 2, 3, 4)
SPLITS = ("arc_validation", "openbookqa_test")
ARMS = ("shared_projected", "shared_matched", "fixed", "shuffled", "residual_only",
        "residual_matched", "reversed", "random")
CONTRASTS = (("shared_projected", "fixed"), ("shared_projected", "shuffled"),
             ("shared_projected", "residual_only"), ("shared_matched", "reversed"),
             ("shared_matched", "random"), ("shared_matched", "residual_matched"))
SCORE_KEYS = ("id", "labels", "gold", "characters", "answer_tokens", "scored_input_tokens")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prefix_hash(tokens):
    return np.frombuffer(hashlib.blake2b(np.asarray(tokens, dtype="<i8").tobytes(), digest_size=8).digest(),
                         dtype="<u8")[0]


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def check_lock(minimal, path, state):
    if path.exists() and read(path) != json.loads(json.dumps(state)):
        raise RuntimeError(f"Existing lock mismatch: {path}")
    if not path.exists():
        minimal.dump(path, state)


def candidate_rows(old, tokenizer, rows):
    items, info = old.encode(tokenizer, rows)
    items.sort(key=lambda x: len(x[0]))
    return items, info


def candidate_keys(items):
    return np.asarray([(q, a, j) for _, n, q, a in items for j in range(n)], dtype=np.int32)


def candidate_hashes(items):
    return np.asarray([prefix_hash(seq[:len(seq) - n + j]) for seq, n, _, _ in items for j in range(n)])


def append_capture(collector, values, keys, hashes):
    for name, value in dict(values, keys=keys, prefix_hash=hashes).items():
        collector[name].append(value)


def reduce_capture(collector):
    return {k: np.concatenate(v) for k, v in collector.items()}


def generation_archive(minimal, data, directory, split, depth, method, max_new):
    path = directory / "fragments" / f"{split}__{method}__k{depth}.jsonl"
    receipt = read(path.with_suffix(".cost.json"))
    digest = minimal.sha(path)
    if receipt["raw_sha256"] != digest:
        raise RuntimeError(f"Historical generation checksum mismatch: {path}")
    rows = minimal.load_jsonl(path)
    if len(rows) != len(data):
        raise RuntimeError(f"Historical generation rows differ from input: {path}")
    for expected, row in zip(data, rows, strict=True):
        if any(row[key] != expected[key] for key in ("id", "prompt", "options", "gold", "official_labels")):
            raise RuntimeError(f"Historical generation protocol mismatch: {path}, {row['id']}")
        if (row["prompt_token_count"] != len(expected["prompt_ids"]) or len(row["tokens"]) != row["predictor_calls"]
                or not 0 < len(row["tokens"]) <= max_new or row["depth"] != depth or row["method"] != method):
            raise RuntimeError(f"Historical generation prefix mismatch: {path}, {row['id']}")
    return rows, digest


class Intervention:
    kind = "linear"

    def __init__(self, name, net, axis, random_axis, fixed, dtype, donor=None, learned=None):
        self.name = name
        self.net = net
        self.axis = axis
        self.random_axis = random_axis
        self.fixed = fixed
        self.dtype = dtype
        self.donor = donor
        self.learned = learned
        self.measurements = []
        self.final_indices = None
        self.final_measurements = []

    def delta(self, h, previous=None):
        original = self.net.delta(h)
        scale = rms(h).clamp_min(EPS)
        source = rms(original)
        alpha = (original @ self.axis)[:, None] / h.shape[-1]
        parallel = alpha * self.axis
        residual = original - parallel
        # In the exactly parallel case, residual-matched has no native residual
        # direction. Keep that edit zero and make the missing match auditable.
        if self.name == "shared_projected":
            edit = parallel
        elif self.name == "shared_matched":
            edit = alpha.sign() * source * self.axis
        elif self.name == "fixed":
            edit = self.fixed * scale * self.axis
        elif self.name == "shuffled":
            edit = self.donor[:, None] * scale * self.axis
        elif self.name == "residual_only":
            edit = residual
        elif self.name == "residual_matched":
            edit = unit_rms(residual) * source
        elif self.name == "reversed":
            edit = -alpha.sign() * source * self.axis
        elif self.name == "random":
            edit = alpha.sign() * source * self.random_axis
        elif self.name == "learned":
            edit = self.learned(h)
        else:
            raise ValueError(self.name)
        actual = (h + edit).to(self.dtype).float() - h
        reference = (h + original).to(self.dtype).float() - h
        data = torch.cat((rms(edit)/scale, rms(actual)/scale, source/scale, rms(reference)/scale,
                          rms(edit), rms(actual), source, rms(reference), scale), dim=1)
        values = data.detach().cpu().numpy()
        self.measurements.append(values)
        if self.final_indices is not None:
            self.final_measurements.append(values[self.final_indices])
        return edit

    def norms(self, final_only=False):
        values = np.concatenate(self.final_measurements if final_only else self.measurements)
        columns = ("proposed_ratio", "actual_ratio", "rank8_proposed_ratio", "rank8_actual_ratio",
                   "proposed_rms", "actual_rms", "rank8_proposed_rms", "rank8_actual_rms", "state_rms")
        result = {name: stats(values[:, i]) for i, name in enumerate(columns)}
        return dict(n=len(values), **result,
                    actual_rms_abs_gap_from_rank8=stats(np.abs(values[:, 5] - values[:, 7])),
                    proposed_rms_abs_gap_from_rank8=stats(np.abs(values[:, 4] - values[:, 6])),
                    zero_proposed_norm=int((values[:, 4] == 0).sum()))


def score_profiles(rows, method, normalized):
    gold, prediction, margin = [], [], []
    for row in rows:
        values = np.asarray(row["scores"][method], dtype=np.float64)
        if normalized:
            values /= row["characters"]
        if not np.isfinite(values).all():
            raise RuntimeError("Nonfinite complete-candidate score")
        gold.append(row["gold"])
        prediction.append(values.argmax())
        margin.append(values[row["gold"]] - np.delete(values, row["gold"]).max())
    return np.asarray(prediction) == gold, np.asarray(margin)


def score_summary(rows, comparisons, seed):
    from scipy.stats import binomtest

    sample = np.random.default_rng(seed).integers(0, len(rows), size=(10_000, len(rows)), dtype=np.int32)
    result = {}
    for metric, normalized in (("acc", False), ("acc_norm", True)):
        profiles = {method: score_profiles(rows, method, normalized) for method in rows[0]["scores"]}
        marginal = {method: dict(correct=int(correct.sum()), mean_gold_margin=float(margin.mean()))
                    for method, (correct, margin) in profiles.items()}
        paired = []
        for target, base in comparisons:
            if target not in profiles or base not in profiles:
                continue
            change = profiles[target][0].astype(int) - profiles[base][0]
            margin = profiles[target][1] - profiles[base][1]
            rescue, damage = int((change == 1).sum()), int((change == -1).sum())
            p = binomtest(rescue, rescue + damage).pvalue if rescue + damage else 1.
            paired.append(dict(target=target, base=base, rescued=rescue, damaged=damage, p_exact_mcnemar=p,
                               delta_pp=100 * float(change.mean()),
                               ci95_pp=(100 * np.quantile(change[sample].mean(-1), [.025, .975])).tolist(),
                               mean_margin_delta=float(margin.mean()),
                               mean_margin_delta_ci95=np.quantile(margin[sample].mean(-1), [.025, .975]).tolist()))
        result[metric] = dict(marginal=marginal, paired=paired)
    return result


class Experiment:
    def __init__(self, args):
        sys.path.insert(0, str(args.previous_study.resolve()))
        import minimal
        import generation
        import pyarrow.parquet as pq

        self.args = args
        self.minimal = minimal
        self.generation = generation
        self.results = args.previous_study / "results"
        self.generation_dir = self.results / "generation"
        self.old, self.model, self.tokenizer, self.runtime = minimal.runtime(args.archive, args.previous, args.model)
        self.model.eval()
        self.dtype = self.model.model.embed_tokens.weight.dtype
        freeze_path = self.results / "FREEZE.json"
        freeze = read(freeze_path)
        self.method = freeze["best"]
        item = freeze["arms"][self.method]
        self.net_path = self.results / item["file"]
        self.dose = item["dose"]
        if self.method != "linear8_t1" or item["sha256"] != minimal.sha(self.net_path) or self.dose != 1.:
            raise RuntimeError("A uses the original frozen Linear Rank-8 T1 at dose 1")
        self.net = minimal.load_net(self.net_path)
        if self.net.kind != "linear" or self.net.rank != 8 or self.net.bound != .45:
            raise RuntimeError("Unexpected original Rank-8 architecture or RMS cap")
        axis_path = self.results / "CAUSAL_ANALYSIS/mean_direction_rank1.pt"
        axis_data = torch.load(axis_path, map_location="cpu", weights_only=True)
        if axis_data["meta"]["source_checkpoint"] != item["sha256"] or not axis_data["meta"]["train_capture_hashes"]:
            raise RuntimeError("The shared direction has differing Rank-8 or Train provenance")
        self.axis = unit_rms(axis_data["state"]["out.weight"][:, 0].float().cuda())
        if not bool(torch.isfinite(self.axis).all()) or not bool(torch.isclose(rms(self.axis).squeeze(), torch.ones((), device="cuda"))):
            raise RuntimeError("The Train axis has invalid scale")
        rng = torch.Generator(device="cpu").manual_seed(SEED)
        random_axis = torch.randn(self.axis.shape, generator=rng).cuda()
        self.random_axis = unit_rms(random_axis - self.axis * (random_axis @ self.axis) / self.axis.numel())
        self.generation_lock_path = self.generation_dir / "GENERATION_LOCK.json"
        gen_lock = read(self.generation_lock_path)
        self.max_new = gen_lock["max_new"]
        self.generate_data, self.data_sources = generation.read_datasets(
            self.old, self.tokenizer, args.corpus, args.external)
        if (gen_lock["checkpoints"][self.method]["sha256"] != item["sha256"]
                or gen_lock["checkpoints"][self.method]["dose"] != self.dose
                or gen_lock["runtime"]["files_sha256"] != self.runtime["files_sha256"]
                or gen_lock["datasets"] != self.data_sources
                or gen_lock["minimal_source_sha256"] != minimal.sha(Path(minimal.__file__))):
            raise RuntimeError("The archived generation protocol does not match these inputs")
        raw = dict(arc_validation=self.old.read_data(args.corpus, "validation"),
                   openbookqa_test=pq.read_table(args.external).to_pylist())
        self.candidates = {split: candidate_rows(self.old, self.tokenizer, raw[split]) for split in SPLITS}
        self.lock = dict(model=self.runtime, phase_version=1, seed=SEED, data=self.data_sources,
                         rank8_sha256=item["sha256"], train_axis_sha256=minimal.sha(axis_path),
                         train_capture_sha256=axis_data["meta"]["train_capture_hashes"],
                         previous_freeze_sha256=minimal.sha(freeze_path),
                         previous_generation_lock_sha256=minimal.sha(self.generation_lock_path),
                         minimal_source_sha256=minimal.sha(Path(minimal.__file__)),
                         generation_source_sha256=minimal.sha(Path(generation.__file__)),
                         source={name: minimal.sha(Path(__file__).with_name(name))
                                 for name in ("experiment_a.py", "a_geometry.py")})
        check_lock(self.minimal, args.out / "A_LOCK.json", self.lock)
        self.learned = None
        if args.learned:
            # Put the current directory before the previous study when importing
            # the trainable controller shared with Experiment B.
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from controller import load_controller
            self.learned = load_controller(args.learned)
            if hasattr(self.learned, "delta"):
                self.learned = self.learned.delta
            if not callable(self.learned):
                raise TypeError("Experiment B load_controller must return a callable or an object exposing delta")

    def output(self, name, value):
        self.minimal.dump(self.args.out / name, value)

    def capture_candidates(self, split, depths, geometry):
        items, _ = self.candidates[split]
        data = {k: defaultdict(list) for k in depths}
        for batch in batches(items, self.args.candidate_batch):
            packed = self.old.pack(batch)
            keys = candidate_keys(batch)
            hashes = candidate_hashes(batch)
            anchors = capture(self.model, packed, depths)
            for depth, anchor in anchors.items():
                append_capture(data[depth], geometry.measure(anchor), keys, hashes)
        return {k: reduce_capture(value) for k, value in data.items()}

    def capture_generated(self, split, depth, geometry):
        data = self.generate_data[split]
        historical, digest = generation_archive(self.minimal, data, self.generation_dir, split, depth,
                                                 self.method, self.max_new)
        collector = defaultdict(list)
        prompts = [r["prompt_ids"] for r in data]
        prefixes = [r["tokens"][:-1] for r in historical]
        for indices in batches(list(range(len(prompts))), self.args.generation_batch):
            packed, _, _ = self.generation.pack_generation(prompts, prefixes, indices, 0, "cuda")
            # Causality lets one archived teacher-forced sequence represent all
            # of its actual past-free-generation predictors in a single pass.
            keys = np.asarray([(i, 0, j) for i in indices for j in range(len(historical[i]["tokens"]))], dtype=np.int32)
            hashes = np.asarray([prefix_hash(prompts[i] + prefixes[i][:j]) for i, _, j in keys])
            anchors = capture(self.model, packed, (depth,))
            append_capture(collector, geometry.measure(anchors[depth]), keys, hashes)
        return reduce_capture(collector), digest

    def save_geometry(self, family, split, depth, values, info, archive_hash=None):
        stem = f"A1_{family}_{split}_k{depth}"
        categories = ("last_prompt", "continuation", "all")
        coordinates = self.args.out / f"{stem}.npz"
        atomic_npz(coordinates, **{k: v for k, v in values.items() if k != "cast_off_vector"},
                   question_ids=np.asarray([row["id"] for row in info]))
        summary = {name: geometry_summary(values, position_masks(values["keys"], name), self.net.bound)
                   for name in categories}
        offsets = values["keys"][:, 2]
        scalars = values["scalars"]
        summary["by_answer_offset"] = {
            str(i): dict(n=int((offsets == i).sum()), raw_ratio=stats(scalars[offsets == i, 4]),
                         clipped_ratio=stats(scalars[offsets == i, 5]),
                         signed_amplitude_ratio=stats(scalars[offsets == i, 8]),
                         clip_fraction=float(np.mean(scalars[offsets == i, 4] > self.net.bound)))
            for i in np.unique(offsets)}
        self.output(f"{stem}.json", dict(summary=summary, coordinates_sha256=self.minimal.sha(coordinates),
                                        archive_generation_sha256=archive_hash))
        values.pop("cast_off_vector")
        return summary

    def cached_geometry(self, family, split, depth):
        stem = self.args.out / f"A1_{family}_{split}_k{depth}"
        if not stem.with_suffix(".json").exists():
            return None
        receipt = read(stem.with_suffix(".json"))
        path = stem.with_suffix(".npz")
        if receipt["coordinates_sha256"] != self.minimal.sha(path):
            raise RuntimeError(f"Stored geometry hash differs: {path}")
        if family == "generation":
            _, digest = generation_archive(self.minimal, self.generate_data[split], self.generation_dir,
                                           split, depth, self.method, self.max_new)
            if digest != receipt["archive_generation_sha256"]:
                raise RuntimeError(f"Historical generation changed: {path}")
        with np.load(path, allow_pickle=False) as data:
            return receipt["summary"], {k: data[k] for k in ("keys", "coords", "prefix_hash")}

    def run_geometry(self):
        clock = time.monotonic()
        geometry = Geometry(self.net, self.axis, self.dose)
        reports, across, axes = {}, {}, {}
        for split in self.args.splits:
            candidates = {k: self.cached_geometry("candidate", split, k) for k in self.args.depths}
            missing = [k for k, value in candidates.items() if value is None]
            c = self.capture_candidates(split, missing, geometry) if missing else {}
            g = {}
            for depth in self.args.depths:
                key = f"candidate/{split}/k{depth}"
                if depth in c:
                    reports[key] = self.save_geometry("candidate", split, depth, c[depth], self.candidates[split][1])
                else:
                    reports[key], c[depth] = candidates[depth]
                key = f"generation/{split}/k{depth}"
                cached = self.cached_geometry("generation", split, depth)
                if cached is None:
                    g[depth], digest = self.capture_generated(split, depth, geometry)
                    reports[key] = self.save_geometry("generation", split, depth, g[depth], self.generate_data[split], digest)
                else:
                    reports[key], g[depth] = cached
                print("A1", split, f"k{depth}", len(c[depth]["keys"]), len(g[depth]["keys"]), flush=True)
            for family, traces in (("candidate", c), ("generation", g)):
                across[f"{family}/{split}"] = {name: across_depth(traces, name)
                                                for name in ("last_prompt", "continuation", "all")}
        for x, y in combinations(sorted(reports), 2):
            family_x, split_x, depth_x = x.split("/")
            family_y, split_y, depth_y = y.split("/")
            if depth_x != depth_y or (split_x != split_y and family_x != family_y):
                continue
            for name in ("last_prompt", "continuation", "all"):
                u, v = reports[x][name], reports[y][name]
                if not u["n"] or not v["n"]:
                    continue
                u, v = [p["updates"]["clipped_before_cast"]["spectra"]["unit_direction"] for p in (u, v)]
                axes[f"{x}__{y}/{name}"] = {k: float(abs(np.dot(u[k]["axis"], v[k]["axis"])))
                                             for k in u if u[k]["axis"] is not None and v[k]["axis"] is not None}
        self.output("A1_GEOMETRY.json", dict(n_seconds=time.monotonic()-clock, reports=reports,
                    matched_prefix_across_depth_cosines=across, cross_domain_abs_axis_cosine=axes,
                    notes=["Candidate prompts use complete answer-text scoring. Generation prompts list option letters.",
                           "The last-prompt subset uses one predictor per question. Candidate all includes the repeated first predictor in every candidate.",
                           "Generation positions use archived Rank-8 token IDs in a causal teacher-forced pass; no Rank-8 decoding was repeated.",
                           "Raw and clipped FP32 spectra are evaluated in the exact QR span of the frozen output matrix. Applied cast vectors additionally have exact outside-span energy and full-space top-1 bounds.",
                           "Post-cast pair cosines measure the projections into the QR span.",
                           "Candidate indices are question, answer choice and predictor offset. Generation indices are question, zero and predictor offset."]))

    def load_donors(self, family, depth):
        path = self.args.out / f"A1_{family}_arc_validation_k{depth}.npz"
        receipt = read(path.with_suffix(".json"))
        if receipt["coordinates_sha256"] != self.minimal.sha(path):
            raise RuntimeError(f"Geometry donor data changed: {path}")
        with np.load(path, allow_pickle=False) as obj:
            keys, vals = obj["keys"], obj["scalars"][:, 8]
        donor, assignment = shuffled_ratios(keys, vals, f"{family}/arc_validation/k{depth}")
        same = keys[:, 0] == keys[assignment, 0]
        data = {tuple(key): float(value) for key, value in zip(keys, donor, strict=True)}
        provenance = dict(source_sha256=receipt["coordinates_sha256"], donor_n=len(keys), same_question_n=int(same.sum()),
                          same_question_by_offset={str(i): int(np.sum(same & (keys[:, 2] == i))) for i in np.unique(keys[:, 2])})
        return data, provenance

    def action(self, name):
        return Intervention(name, self.net, self.axis, self.random_axis, self.net.bound,
                            self.dtype, learned=self.learned)

    def check_candidate_history(self, info):
        historic = {"native": self.args.archive / "results/validation_predictions.jsonl",
                    "rank8": self.results / f"VAL_{self.method}.jsonl"}
        archive = {m: {r["id"]: r for r in self.minimal.load_jsonl(p)} for m, p in historic.items()}
        for arm, rows in archive.items():
            if rows.keys() != {r["id"] for r in info}:
                raise RuntimeError(f"Historical {arm} candidate IDs differ")
            for row in info:
                other = rows[row["id"]]
                if any(other[key] != row[key] for key in SCORE_KEYS):
                    raise RuntimeError(f"Historical {arm} protocol differs for {row['id']}")
        receipts = {m: self.minimal.sha(p) for m, p in historic.items()}
        axis_protocol = read(self.results / "CAUSAL_ANALYSIS/VALIDATION_COMPRESSION_DIAGNOSTIC.json")
        if receipts["rank8"] != axis_protocol["validation_reference_sha256"]:
            raise RuntimeError("The archived Rank-8 validation results differ from the compression receipt")
        return archive, receipts

    def score_candidate_arm(self, items, info, depth, name):
        policy = self.action(name)
        donors, donor_receipt = self.load_donors("candidate", depth) if name == "shuffled" else (None, None)
        work = 0
        start = time.monotonic()
        for batch in batches(items, self.args.candidate_batch):
            packed = self.old.pack(batch)
            if donors is not None:
                values = [donors[tuple(key)] for key in candidate_keys(batch)]
                policy.donor = torch.as_tensor(values, dtype=torch.float32, device="cuda")
            final, _ = self.minimal.forward(self.model, packed, depth, policy)
            score = self.old.scores(self.model.lm_head(final).float(), packed[1].cpu().numpy(), packed[-1], len(batch))
            self.old.record(info, batch, {name: score})
            work += packed[0].numel() * depth
        return dict(wall_seconds=time.monotonic()-start, padded_token_loops=work, norms=policy.norms(),
                    shuffled_donor=donor_receipt, batch_size=self.args.candidate_batch)

    def artifact_suffix(self):
        return "" if self.learned is None else f"_learned_{self.minimal.sha(self.args.learned)[:12]}"

    def run_factorial(self):
        items, info = self.candidates["arc_validation"]
        archive, source = self.check_candidate_history(info)
        summary_path = self.args.out / f"A2_FACTORIAL{self.artifact_suffix()}.json"
        output = read(summary_path)["depths"] if summary_path.exists() else {}
        comparative = [(name, base) for name in self.args.arms for base in ("native", "rank8")]
        comparative += CONTRASTS
        for depth in self.args.depths:
            k = f"k{depth}"
            rows = [dict(row, scores={m: a[row["id"]]["scores"][f"native_{k}" if m == "native" else k]
                                     for m, a in archive.items()}) for row in info]
            costs = {}
            for name in self.args.arms:
                fragment = self.args.out / f"A2_scores_{k}_{name}{self.artifact_suffix()}.jsonl"
                receipt = fragment.with_suffix(".json")
                if receipt.exists():
                    costs[name] = read(receipt)
                    if costs[name]["scores_sha256"] != self.minimal.sha(fragment):
                        raise RuntimeError(f"Corrupt candidate score fragment: {fragment}")
                    predictions = self.minimal.load_jsonl(fragment)
                    for row, cached in zip(rows, predictions, strict=True):
                        if any(row[key] != cached[key] for key in SCORE_KEYS):
                            raise RuntimeError(f"Candidate score fragment changed protocol: {fragment}")
                        row["scores"][name] = cached["scores"][name]
                else:
                    costs[name] = self.score_candidate_arm(items, rows, depth, name)
                    self.generation.write_jsonl(fragment, [dict(row, scores={name: row["scores"][name]}) for row in rows])
                    costs[name]["scores_sha256"] = self.minimal.sha(fragment)
                    self.minimal.dump(receipt, costs[name])
                print("A2 candidate", k, name, round(costs[name]["wall_seconds"], 1), flush=True)
            scores = self.args.out / f"A2_SCORES_{k}{self.artifact_suffix()}.jsonl"
            self.generation.write_jsonl(scores, rows)
            output[k] = dict(n=len(rows), candidate_scores_sha256=self.minimal.sha(scores), costs=costs,
                             metrics=score_summary(rows, comparative, SEED + depth))
            self.minimal.dump(summary_path, dict(historical_sources=source, depths=output))

    def shuffle_generation(self, data, depth):
        donors, receipt = self.load_donors("generation", depth)
        prediction_ratios = np.full((len(data), self.max_new), np.nan, dtype=np.float32)
        for (i, _, step), ratio in donors.items():
            prediction_ratios[i, step] = ratio
        # New trajectories can continue after the archived recipient stopped.
        # Use a deterministic draw from the same archived step in this case.
        # If no archived trajectory reached that step, reuse the last supported
        # step, count every draw, and retain the recipient's own state RMS.
        rng = np.random.default_rng(SEED + depth)
        fallback = np.zeros_like(prediction_ratios, dtype=bool)
        for step in range(self.max_new):
            available = np.flatnonzero(np.isfinite(prediction_ratios[:, step]))
            for i in np.flatnonzero(~np.isfinite(prediction_ratios[:, step])):
                pool = available[available != i]
                value = prediction_ratios[rng.choice(pool), step] if len(pool) else prediction_ratios[i, step - 1]
                prediction_ratios[i, step] = value
                fallback[i, step] = True
        return prediction_ratios, fallback, receipt

    def generate_arm(self, split, data, depth, name):
        policy = self.action(name)
        prompts = [row["prompt_ids"] for row in data]
        if name == "shuffled":
            ratios, fallback, donor_receipt = self.shuffle_generation(data, depth)
        else:
            ratios, fallback, donor_receipt = None, None, None
        calls = padded = fallback_n = 0
        result = []
        start = time.monotonic()
        for indices in batches(list(range(len(data))), self.args.generation_batch):
            local = [prompts[i] for i in indices]
            # The common generation runner compacts active examples. Recover
            # exact local active IDs through the same stop rule and record each
            # recipient's amplitude at every eligible predictor position.
            active = list(range(len(indices)))
            generated = [[] for _ in indices]

            def predict(packed, last):
                nonlocal active, fallback_n
                if ratios is not None:
                    positions = [np.arange(len(generated[i])+1) for i in active]
                    values = np.concatenate([ratios[indices[i], p] for i, p in zip(active, positions, strict=True)])
                    fallback_n += sum(int(np.sum(fallback[indices[i], p])) for i, p in zip(active, positions, strict=True))
                    policy.donor = torch.as_tensor(values, device="cuda")
                policy.final_indices = last.cpu().numpy()
                terminal, _ = self.minimal.forward(self.model, packed, depth, policy)
                tokens = self.model.lm_head(terminal[last]).argmax(-1).tolist()
                if ratios is not None:
                    for i, token in zip(active, tokens, strict=True):
                        generated[i].append(token)
                    active = [i for i, token in zip(active, tokens, strict=True)
                              if token != self.tokenizer.eos_token_id and "\n" not in self.tokenizer.decode(
                                  generated[i], skip_special_tokens=True).strip("\n ")]
                return tokens

            tokens, stops, num_calls, work = self.generation.generate_batch(self.tokenizer, local, predict,
                                                                            self.max_new, "cuda")
            for i, seq, stop in zip(indices, tokens, stops, strict=True):
                row = data[i]
                text = self.tokenizer.decode(seq, skip_special_tokens=True)
                option, category, strict = self.generation.classify(text, row["options"])
                result.append(dict(split=split, method=name, depth=depth, id=row["id"], prompt=row["prompt"],
                                   prompt_token_count=len(prompts[i]), options=row["options"], official_labels=row["official_labels"],
                                   gold=row["gold"], tokens=seq, text=text, stop=stop, option=option,
                                   valid=option is not None, correct=option == row["gold"],
                                   strict_format=strict, invalid_category=category, predictor_calls=len(seq)))
            calls += num_calls
            padded += work
        return result, dict(wall_seconds=time.monotonic()-start, padded_token_loops=padded*depth, model_calls=calls,
                            norms=policy.norms(), last_predictor_norms=policy.norms(final_only=True),
                            shuffled_donor=donor_receipt, shuffled_extrapolated_predictors=fallback_n,
                            generated_tokens=sum(len(r["tokens"]) for r in result), batch_size=self.args.generation_batch)

    def run_generation(self):
        summary_path = self.args.out / f"A2_GENERATION{self.artifact_suffix()}.json"
        report = read(summary_path) if summary_path.exists() else {}
        for split in self.args.splits:
            data = self.generate_data[split]
            if split != "arc_validation" and "shuffled" in self.args.arms:
                raise ValueError("The shuffled factorial is restricted to ARC Validation")
            for depth in self.args.depths:
                label = f"{split}_k{depth}"
                baseline, source = {}, {}
                for name in ("native", self.method, "linear8_t2"):
                    baseline[name], source[name] = generation_archive(self.minimal, data, self.generation_dir, split,
                                                                      depth, name, self.max_new)
                cells, predictions = {}, {}
                for name in self.args.arms:
                    stem = self.args.out / f"A2_generation_{label}_{name}{self.artifact_suffix()}.jsonl"
                    receipt = stem.with_suffix(".json")
                    if receipt.exists():
                        cost = read(receipt)
                        if cost["sha256"] != self.minimal.sha(stem):
                            raise RuntimeError(f"Corrupt generation score fragment: {stem}")
                        rows = self.minimal.load_jsonl(stem)
                    else:
                        rows, cost = self.generate_arm(split, data, depth, name)
                        self.generation.write_jsonl(stem, rows)
                        cost["sha256"] = self.minimal.sha(stem)
                        self.minimal.dump(receipt, cost)
                    cells[name] = dict(cost=cost, vs={base: self.generation.summarize(rows, values)
                                                     for base, values in baseline.items()})
                    predictions[name] = rows
                    print("A2 generation", label, name, cells[name]["vs"]["native"]["correct"], flush=True)
                contrasts = {f"{name}_vs_{other}": self.generation.summarize(predictions[name], predictions[other])
                             for name, other in CONTRASTS if name in predictions and other in predictions}
                report[label] = dict(historical_sources=source, methods=cells, planned_contrasts=contrasts)
                self.minimal.dump(summary_path, report)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("geometry", "factorial", "generation", "learned"), required=True)
    for name in ("archive", "previous", "previous-study", "model", "corpus", "external", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", choices=DEPTHS, default=DEPTHS)
    parser.add_argument("--splits", nargs="+", choices=SPLITS)
    parser.add_argument("--arms", nargs="+", choices=(*ARMS, "learned"))
    parser.add_argument("--learned", type=Path)
    parser.add_argument("--candidate-batch", type=int, default=32)
    parser.add_argument("--generation-batch", type=int, default=32)
    args = parser.parse_args()
    if args.splits is None:
        args.splits = SPLITS if args.phase == "geometry" else ("arc_validation",)
    if args.arms is None:
        args.arms = ("learned",) if args.phase == "learned" else ARMS
    if (args.candidate_batch < 1 or args.generation_batch < 1 or len(set(args.arms)) != len(args.arms)
            or len(set(args.depths)) != len(args.depths) or len(set(args.splits)) != len(args.splits)):
        parser.error("Provide positive batch sizes and unique depths, splits and arms")
    if ("learned" in args.arms) != bool(args.learned):
        parser.error("Select the learned arm with a --learned checkpoint")
    args.out.mkdir(parents=True, exist_ok=True)
    run = Experiment(args)
    if args.phase == "geometry":
        run.run_geometry()
    if args.phase in ("factorial", "learned"):
        run.run_factorial()
    if args.phase in ("generation", "learned"):
        run.run_generation()


if __name__ == "__main__":
    main()
