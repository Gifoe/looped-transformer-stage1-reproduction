"""Compare finite-dose early transport and late Rank-1 at the same raw Ouro boundary."""

import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from dataset_transfer import load_cohort
from forward_control import forward_control, scope_indices

T1 = "91d4919308bf9060ce0c44e4046e4851e7e59f65b9692c77045535f96534efd2"
SEED = 261123088
BRANCHES = {"attention": "input_layernorm_2", "mlp": "post_attention_layernorm_2"}
METRICS = ("acc", "acc_norm", "margin", "margin_norm", "utility")
DOSE_TOLERANCE = .01


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def atom(path, value):
    target = path.with_suffix(".partial")
    target.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    target.replace(path)


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def token_output(path, value):
    target = path.with_suffix(".partial")
    with target.open("wb") as binary, gzip.GzipFile(fileobj=binary, mode="wb", mtime=0) as compressed:
        compressed.write(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")).encode("utf-8"))
    target.replace(path)


def rows_write(path, rows):
    target = path.with_suffix(".partial")
    with target.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    target.replace(path)


def rms(vector):
    return vector.float().square().mean(-1).sqrt()


def grouped_sum(values, owners, n):
    return torch.zeros(n, dtype=values.dtype, device=values.device).index_add_(0, owners, values)


def seq_rms(delta, owners, lengths):
    squared = delta.float().square().mean(-1)
    return (grouped_sum(squared, owners, len(lengths)) / lengths).sqrt()


def cosine(left, right):
    sizes = left.norm(dim=-1) * right.norm(dim=-1)
    return ((left * right).sum(-1) / sizes.clamp_min(1e-30)).clamp(-1., 1.)


def sequence_cosine(left, right, group, n):
    dot = grouped_sum((left * right).sum(-1), group, n)
    sizes = (grouped_sum(left.square().sum(-1), group, n).sqrt() *
             grouped_sum(right.square().sum(-1), group, n).sqrt())
    return (dot / sizes.clamp_min(1e-30)).clamp(-1., 1.)


def quantized_delta(anchor, delta, scale, owners, dtype):
    return (anchor + delta * scale[owners, None]).to(dtype).float() - anchor


def calibrate(anchor, displacement, owners, lengths, dtype, target):
    """Numerically match realized sequence RMS after BF16, with per-candidate scalars."""
    nominal = seq_rms(displacement, owners, lengths)
    attainable = (nominal > 0) | (target == 0)
    suggestion = torch.where(nominal > 0, target / nominal.clamp_min(1e-30), torch.zeros_like(target))
    low, high = torch.zeros_like(target), 2 * suggestion
    best = suggestion.clone()
    best_err = (seq_rms(quantized_delta(anchor, displacement, best, owners, dtype), owners, lengths) -
                target).abs()
    for _ in range(15):
        scale = (low + high) / 2
        actual = seq_rms(quantized_delta(anchor, displacement, scale, owners, dtype), owners, lengths)
        error = (actual - target).abs()
        best = torch.where(error < best_err, scale, best)
        best_err = torch.minimum(error, best_err)
        low = torch.where(actual < target, scale, low)
        high = torch.where(actual < target, high, scale)
    return displacement * best[owners, None], best, attainable


@contextmanager
def additive_writes(model, loops, reference=None, clamp=None):
    """Capture native additive writes or restore one real branch during transport."""
    core = model.model
    turn = 1
    handles = []
    captured = {}

    def norm_hook(_module, _inputs, _output):
        nonlocal turn
        turn += 1

    def branch_hook(name, index):
        def record(_module, _inputs, output):
            if turn not in loops:
                return None
            key = (turn, index, name)
            if reference is None:
                captured[key] = output.detach()
            elif name == clamp:
                return reference[key]
            return None
        return record

    if loops:
        handles.append(core.norm.register_forward_hook(norm_hook))
        for layer, decoder in enumerate(core.layers, 1):
            for name, module in BRANCHES.items():
                if reference is None or clamp == name:
                    handles.append(getattr(decoder, module).register_forward_hook(branch_hook(name, layer)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def logits(archive, model, hidden, packed, n):
    return archive.scores(model.lm_head(hidden).float(), packed[1].cpu().numpy(), packed[-1], n).tolist()


def utility_gradient(model, packed, items, meta, depth, boundary):
    """Differentiate smooth normalized gold log odds through the common native suffix."""
    with torch.enable_grad():
        states, trace = forward_control(model, packed, depth, anchor_loop=boundary,
                                        token_scope="all_valid", leaf=True)
        lp = model.lm_head(states).float().log_softmax(-1).gather(1, packed[-1][:, None]).squeeze(1)
        candidate_scores = grouped_sum(lp, packed[1], len(items))
        objective = []
        i = 0
        for question in meta:
            n = len(question["labels"])
            score = candidate_scores[i:i+n] / candidate_scores.new_tensor(question["characters"])
            gold = question["gold"]
            objective.append(score[gold] - torch.cat((score[:gold], score[gold+1:])).logsumexp(0))
            i += n
        if i != len(items):
            raise ValueError("Gradient scores require complete question groups in archive request order")
        gradient, = torch.autograd.grad(torch.stack(objective).sum(), trace["leaf"])
    if not torch.isfinite(gradient).all():
        raise RuntimeError("Nonfinite native suffix gradient")
    return gradient.detach()


def observed(trace, reference, boundary, group, pos, kind="pre"):
    return trace[kind][boundary][group, pos].float() - reference[kind][boundary][group, pos].float()


def measured_trace(trace, native, delta, grad, group, pos, lengths, injection, end, edit_scope):
    grad_product = (grad * delta).sum(-1)
    product = grouped_sum(grad_product, group, len(lengths))
    norms = {
        "sequence_delta_rms": seq_rms(delta, group, lengths),
        "initial_gradient_dot": product,
        "initial_gradient_cos": sequence_cosine(grad, delta, group, len(lengths)),
        "sequence_offsite_rms": seq_rms(delta.masked_fill(edit_scope[:, None], 0), group, lengths),
    }
    token = dict(delta_rms=rms(delta), local_grad_dot=grad_product,
                 local_grad_cos=cosine(grad, delta))
    for boundary in range(injection, end + 1):
        raw = observed(trace, native, boundary, group, pos)
        post = observed(trace, native, boundary, group, pos, "post")
        for name, vector in (("raw", raw), ("post", post)):
            token[f"loop{boundary}_{name}_rms"] = rms(vector)
            token[f"loop{boundary}_{name}_cos_from_common_write"] = cosine(vector, delta)
            norms[f"loop{boundary}_{name}_seq_rms"] = seq_rms(vector, group, lengths)
    return norms, token


def execute_batch(archive, model, net, packed, items, meta, args, verify_replay):
    all_loops = tuple(range(1, args.depth + 1))
    group, pos = scope_indices(packed, "all_valid")
    n = len(items)
    lengths = torch.bincount(group, minlength=n)
    site_group, site_pos = scope_indices(packed, args.scope)
    site = torch.zeros_like(packed[0], dtype=torch.bool).index_put(
        (site_group, site_pos), torch.ones_like(site_pos, dtype=torch.bool))
    active_site = site[group, pos]
    capture_writes = {k+1 for k in args.early_loops} if args.branch_controls else set()
    with additive_writes(model, capture_writes) as writes:
        native_hidden, reference = forward_control(model, packed, args.depth, anchor_loop=1, capture=all_loops)
    scores = {"native": logits(archive, model, native_hidden, packed, n)}
    result = {}
    token = dict(row=group.tolist(), position=pos.tolist(), injected_scope=active_site.tolist(),
                 token_id=packed[0][group, pos].tolist(), comparisons={})
    model_calls = 1
    for step in args.early_loops:
        boundary = step + 1
        tag = f"k{step}_to_k{boundary}"
        dtype = reference["pre"][boundary].dtype
        anchor = reference["pre"][boundary][group, pos].float()
        early, tr_early = forward_control(model, packed, args.depth, anchor_loop=step, controller=net,
                                          token_scope=args.scope, capture=all_loops)
        late, tr_late = forward_control(model, packed, args.depth, anchor_loop=boundary, controller=net,
                                        token_scope=args.scope, capture=all_loops)
        model_calls += 2
        raw = {"early": observed(tr_early, reference, boundary, group, pos),
               "late": observed(tr_late, reference, boundary, group, pos)}
        grad = utility_gradient(model, packed, items, meta, args.depth, boundary)
        model_calls += 1
        conditions = {"early": (early, tr_early), "late": (late, tr_late)}
        doses = {name: seq_rms(delta, group, lengths) for name, delta in raw.items()}
        target = torch.minimum(doses["early"], doses["late"])
        diagnostics = {}
        if verify_replay:
            exact_delta = tr_early["pre"][boundary][group, pos].double() - anchor.double()
            hidden, replay = forward_control(model, packed, args.depth, anchor_loop=boundary,
                                              token_scope="all_valid", edit=lambda _: exact_delta,
                                              capture=(boundary,))
            model_calls += 1
            raw_error = (replay["pre"][boundary][group, pos].float() -
                         tr_early["pre"][boundary][group, pos].float()).abs().max()
            h_error = (hidden.float() - early.float()).abs().max()
            diagnostics["unscaled_replay"] = dict(common_raw_max_abs=float(raw_error),
                                                  scored_hidden_max_abs=float(h_error),
                                                  exact_hidden=bool(h_error == 0))
            if raw_error != 0 or h_error != 0:
                raise RuntimeError("Unscaled transport replay is not exact at the common state and scored predictions")
        if args.branch_controls:
            for branch in BRANCHES:
                with additive_writes(model, {boundary}, reference=writes, clamp=branch):
                    held, trace = forward_control(model, packed, args.depth, anchor_loop=step, controller=net,
                                                  token_scope=args.scope, capture=all_loops)
                if not torch.equal(tr_early["delta"], trace["delta"]):
                    raise RuntimeError("Intervening within the transport loop changed the original control write")
                name = f"early_{branch}_restored"
                model_calls += 1
                raw[name] = observed(trace, reference, boundary, group, pos)
                conditions[name] = (held, trace)
        scales = {}
        supports = {}
        for name, displacement in raw.items():
            proposal, scale, attainable = calibrate(anchor, displacement, group, lengths, dtype, target)
            hidden, trace = forward_control(model, packed, args.depth, anchor_loop=boundary,
                                            token_scope="all_valid", edit=lambda _: proposal, capture=all_loops)
            if not torch.equal(trace["anchor"], anchor):
                raise RuntimeError("Matched arms started from different raw states")
            model_calls += 1
            matched = name + "_matched"
            conditions[matched] = (hidden, trace)
            scales[matched] = scale
            supports[matched] = attainable
        candidates = {}
        token_rows = dict(anchor_rms=rms(anchor).tolist(), task_gradient_rms=rms(grad).tolist(),
                          early_late_raw_cos=cosine(raw["early"], raw["late"]).tolist())
        for name, (hidden, trace) in conditions.items():
            delta = observed(trace, reference, boundary, group, pos)
            case = f"{tag}/{name}"
            scores[case] = logits(archive, model, hidden, packed, n)
            totals, profiles = measured_trace(trace, reference, delta, grad, group, pos, lengths,
                                              boundary, args.depth, active_site)
            totals["initial_cos_to_real_late"] = sequence_cosine(delta, raw["late"], group, n)
            if name.endswith("_matched"):
                realized = totals["sequence_delta_rms"]
                error = (realized - target).abs() / target.clamp_min(1e-30)
                totals.update(dose_target=target, dose_factor=scales[name], dose_relative_error=error,
                              within_dose_tolerance=((error <= DOSE_TOLERANCE) & supports[name]).float())
            candidates[name] = {key: value.tolist() for key, value in totals.items()}
            token_rows[name] = {key: value.tolist() for key, value in profiles.items()}
        result[tag] = dict(candidate=candidates, diagnostics=diagnostics)
        token["comparisons"][tag] = token_rows
    return scores, result, token, model_calls


def metric(scores, chars, gold):
    raw = np.asarray(scores, dtype=np.float64)
    norm = raw / chars
    rest = np.arange(len(raw)) != gold
    return np.array([float(raw.argmax() == gold), float(norm.argmax() == gold),
                     raw[gold] - raw[rest].max(), norm[gold] - norm[rest].max(),
                     norm[gold] - np.logaddexp.reduce(norm[rest])])


def statistics(records, sample_n):
    rng = np.random.default_rng(SEED)
    sample = rng.integers(len(records), size=(sample_n, len(records)))
    native = np.stack([metric(row["scores"]["native"], row["characters"], row["gold"]) for row in records])

    def paired(d):
        low, high = np.quantile(d[sample].mean(1), (.025, .975))
        return dict(delta=float(d.mean()), ci95=[float(low), float(high)])

    def score_for(label):
        return np.stack([metric(row["scores"][label], row["characters"], row["gold"]) for row in records])

    def correlation(left, right):
        return float(np.corrcoef(left, right)[0, 1]) if np.std(left) and np.std(right) else None

    def values(key, arm, variable):
        return np.asarray([np.mean(row["transport"][key][arm][variable]) for row in records])

    reports = {}
    for key, arms in records[0]["transport"].items():
        table = {}
        for arm in arms:
            after = score_for(f"{key}/{arm}")
            delta = after - native
            measured = {name: paired(delta[:, i]) for i, name in enumerate(METRICS)}
            for i, name in enumerate(("acc", "acc_norm")):
                measured[name].update(correct=int(after[:, i].sum()), rescue=int((delta[:, i] > 0).sum()),
                                      damage=int((delta[:, i] < 0).sum()))
            dynamics = {var: paired(values(key, arm, var)) for var in arms[arm]}
            # Gradient dots sum over the candidate sets because utility couples their log scores.
            grad = np.array([sum(row["transport"][key][arm]["initial_gradient_dot"]) for row in records])
            observed = delta[:, -1]
            table[arm] = dict(metrics=measured, dynamics=dynamics,
                              gradient_vs_observed=dict(pearson=correlation(grad, observed),
                                                        norm_pearson=correlation(
                                                            values(key, arm, "sequence_delta_rms"), observed),
                                                        mae=float(np.abs(grad-observed).mean()),
                                                        sign_matches=int((np.sign(grad) == np.sign(observed)).sum())))
        comparisons = {}
        arms_to_test = [("late", "early"), ("late_matched", "early_matched")]
        for branch in BRANCHES:
            for suffix in ("", "_matched"):
                candidate = f"early_{branch}_restored{suffix}"
                if candidate in arms:
                    arms_to_test.append((f"early{suffix}", candidate))
        for right, left in arms_to_test:
            delta = score_for(f"{key}/{right}") - score_for(f"{key}/{left}")
            comparisons[f"{right}_minus_{left}"] = {name: paired(delta[:, i]) for i, name in enumerate(METRICS)}
        paired_match = np.array([
            all(min(row["transport"][key][name]["within_dose_tolerance"]) == 1
                for name in ("early_matched", "late_matched")) for row in records])
        reports[key] = dict(conditions=table, contrasts=comparisons,
                            questions_with_all_candidates_dose_matched=int(paired_match.sum()))
    return dict(n=len(records), native=dict(acc=int(native[:, 0].sum()), acc_norm=int(native[:, 1].sum()),
                                            margin=float(native[:, 2].mean()),
                                            margin_norm=float(native[:, 3].mean()), utility=float(native[:, 4].mean())),
                results=reports)


def chunks(args, archive, model, tokenizer, net, rows):
    save = args.out / "chunks"
    save.mkdir(exist_ok=True, parents=True)
    records, timing, replay = [], Counter(), []
    for i in range(0, len(rows), args.batch_questions):
        current = rows[i:i+args.batch_questions]
        name = save / f"{i//args.batch_questions:04d}.json"
        raw = name.with_suffix(".tokens.json.gz")
        ids = [row["id"] for row in current]
        if name.exists():
            blob = json_read(name)
            if ids != blob["ids"] or digest(raw) != blob["tokens_sha256"]:
                raise RuntimeError(f"Resume token or question identity mismatch at {name}")
        else:
            start = time.perf_counter()
            items, meta = archive.encode(tokenizer, current)
            packed = archive.pack(items)
            scores, results, token, calls = execute_batch(
                archive, model, net, packed, items, meta, args, verify_replay=i == 0)
            start_index = 0
            for row in meta:
                options = len(row["labels"])
                boundary = start_index + options
                row["scores"] = {arm: values[start_index:boundary] for arm, values in scores.items()}
                row["transport"] = {tag: {
                    arm: {key: values[start_index:boundary] for key, values in trace.items()}
                    for arm, trace in rec["candidate"].items()} for tag, rec in results.items()}
                start_index = boundary
            token["candidate_identity"] = [[meta[q]["id"], choice] for _, _, q, choice in items]
            token_output(raw, token)
            blob = dict(ids=ids, predictions=meta, tokens_sha256=digest(raw),
                        replay={key: data["diagnostics"] for key, data in results.items()},
                        work=dict(candidates=len(items), model_calls=calls, native_loop_calls=calls*args.depth,
                                  padded_tokens=packed[0].numel()*calls, elapsed_seconds=time.perf_counter()-start))
            atom(name, blob)
            print(args.cohort, "transport", min(i+len(ids), len(rows)), "/", len(rows), flush=True)
        records.extend(blob["predictions"])
        timing.update(blob["work"])
        replay.append(blob["replay"])
    return records, dict(timing), replay


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "previous", "minimal", "model", "rank1", "cohorts", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cohort", choices=("arc_discovery", "arc_confirm"), required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--early-loops", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--batch-questions", type=int, default=8)
    parser.add_argument("--scope", choices=("predictors", "first_predictor"), default="predictors")
    parser.add_argument("--branch-controls", action="store_true")
    args = parser.parse_args()
    if (min(args.early_loops) < 1 or max(args.early_loops) + 1 >= args.depth or args.batch_questions < 1
            or len(set(args.early_loops)) != len(args.early_loops)):
        parser.error("Every common boundary must retain at least one native suffix loop")
    if digest(args.rank1) != T1:
        raise RuntimeError("Expected the locked historical Rank-1 T1")
    sys.path.insert(0, str(args.minimal.resolve()))
    import minimal

    payload = torch.load(args.rank1, map_location="cpu", weights_only=True)
    archive, model, tokenizer, runtime = minimal.runtime(args.archive, args.previous, args.model)
    model.requires_grad_(False)
    net = minimal.Minimal("linear", 1, .45).to(next(model.parameters()).device).eval().requires_grad_(False)
    net.load_state_dict(payload["state"])
    rows = load_cohort(args.cohorts, args.cohort)
    args.out.mkdir(exist_ok=True, parents=True)
    manifest = dict(
        cohort=args.cohort, cohorts_sha256=digest(args.cohorts), cohort_manifest=json_read(args.cohorts)["cohorts"][args.cohort],
        checkpoint_sha256=T1, depth=args.depth, early_loops=list(args.early_loops),
        token_scope=args.scope, batch_questions=args.batch_questions, branch_controls=args.branch_controls,
        runtime=json.loads(json.dumps(runtime)),
        source_sha256=digest(__file__), forward_source_sha256=digest(Path(__file__).with_name("forward_control.py")),
        utility="Smooth character-normalized gold log odds; exact archived token likelihoods",
        matching="Per-candidate all-valid sequence RMS, common target min(real early, real late), calibrated after BF16 cast.",
        branches="Restore one native additive branch during the transport loop only, then repeat matched replay.",
        controlled_comparison_max_relative_dose_error=DOSE_TOLERANCE)
    lock = args.out / "MANIFEST.json"
    if lock.exists():
        if json_read(lock) != manifest:
            raise RuntimeError("Transport manifest mismatch")
    else:
        atom(lock, manifest)
    records, cost, replay = chunks(args, archive, model, tokenizer, net, rows)
    rows_write(args.out / "PREDICTIONS.jsonl", records)
    atom(args.out / "TRANSPORT_SUMMARY.json", dict(**statistics(records, 6000), cost=cost, replay=replay))
    print("Transport complete", args.cohort, len(records), flush=True)


if __name__ == "__main__":
    main()
