"""Direction-by-path factorial and injection-position experiments for Ouro feedback."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

DEPTHS = (1, 2, 3, 4)
BRANCHES = {"attention": "input_layernorm_2", "mlp": "post_attention_layernorm_2"}
PHYSICAL = "layer21"
REV = 261123082


def fingerprint(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def order_key(tag, question_id):
    return hashlib.sha256(f"{REV}/{tag}/{question_id}".encode()).digest()


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def rms(x):
    return x.float().square().mean(-1).sqrt()


def same_rms(proposed, learned):
    source = rms(proposed)
    target = rms(learned)
    if bool(((source == 0) & (target != 0)).any()):
        raise RuntimeError("Control proposed zero update for a nonzero learned direction")
    return proposed * (target / source.clamp_min(1e-20)).unsqueeze(-1)


def anchor(depth):
    return PHYSICAL if depth == 1 else depth - 1


def is_suffix(turn, layer, start):
    if start == PHYSICAL:
        return turn == 1 and layer > 20
    return start is not None and turn > start


def forward(model, packed, depth, policies=None, inspect=(), branch_start=None,
            native=None, clamp=False, keep_trace=True):
    """Write to the native raw state before pass RMSNorm or after physical layer 21."""
    policies = policies or {}
    selected = set(inspect).union(policies)
    inp, group, pos, _ = packed
    core = model.model
    core.total_ut_steps = depth
    turn = 1
    history = None
    hooks = []
    branch = {}
    changes = defaultdict(list)
    captures, raw, edits, realized = {}, {}, {}, {}

    def apply(tensor, location, before):
        if location not in selected:
            return tensor
        h = tensor[group, pos].float()
        if location in inspect:
            captures[location] = (h, before)
        action = policies.get(location)
        if action is None:
            return tensor
        update = action(h, before)
        adjusted = (h + update).to(tensor.dtype)
        realized[location] = rms(adjusted.float() - h) / rms(h).clamp_min(1e-20)
        if keep_trace:
            edits[location] = adjusted.float() - h
        return tensor.index_put((group, pos), adjusted)

    def physical(_, args, tensor):
        return apply(tensor, PHYSICAL, None) if turn == 1 else tensor

    def pass_norm(_, args):
        nonlocal turn, history
        tensor = args[0]
        before = tensor[group, pos].float()
        if keep_trace:
            raw[turn] = before
        changed = apply(tensor, turn, history)
        history = before
        turn += 1
        return (changed, *args[1:]) if changed is not tensor else None

    def branch_hook(name, layer):
        def monitor(_, args, output):
            if not is_suffix(turn, layer, branch_start):
                return None
            key = (turn, layer, name)
            if native is None:
                branch[key] = output.detach()
                return None
            stock = native["branch"][key]
            active = stock if clamp else output
            if keep_trace:
                delta_rms = rms(active[group, pos].float() - stock[group, pos].float())
                changes[(turn, name)].append(delta_rms)
            return stock if clamp else None
        return monitor

    hooks.append(core.norm.register_forward_pre_hook(pass_norm))
    if PHYSICAL in selected:
        hooks.append(core.layers[20].register_forward_hook(physical))
    if branch_start is not None:
        for layer, block in enumerate(core.layers):
            if branch_start == PHYSICAL and layer <= 20:
                continue
            for name, norm in BRANCHES.items():
                hooks.append(getattr(block, norm).register_forward_hook(branch_hook(name, layer)))
    try:
        _, normalized, _ = core(input_ids=inp, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    if turn != depth + 1 or len(normalized) != depth:
        raise RuntimeError(f"Actual native passes differed from K{depth}")
    normed = ({i: tensor[group, pos].float() for i, tensor in enumerate(normalized, 1)}
              if keep_trace else {})
    return normalized[-1][group, pos], dict(capture=captures, raw=raw, normed=normed,
        edits=edits, realized=realized, branch=branch, branch_response={
            key: torch.stack(v).mean(0) for key, v in changes.items()})


def read_data(old, args, tokenizer, split):
    rows = old.read_data(args.corpus, split)
    items, info = old.encode(tokenizer, rows)
    if split == "validation" and args.questions < len(info):
        ids = sorted(range(len(info)), key=lambda i: order_key("subset", info[i]["id"]))[:args.questions]
        ids.sort()
        remap = {old_id: new for new, old_id in enumerate(ids)}
        items = [(seq,n,remap[q],a) for seq,n,q,a in items if q in remap]
        info = [info[i] for i in ids]
    items.sort(key=lambda v: len(v[0]))
    return items, info


def batches(items, size):
    for lo in range(0, len(items), size):
        yield items[lo:lo + size]


def collect_donors(old, args, model, items, info, net):
    """Each recipient uses one fixed different question's first listed answer prefix."""
    it = [x for x in items if x[3] == 0]
    assert len(it) == len(info)
    table = {k: {} for k in DEPTHS}
    for subset in batches(it, args.batch_size):
        _, rec = forward(model, old.pack(subset), 3, inspect=(PHYSICAL, 1, 2, 3), keep_trace=False)
        for k in DEPTHS:
            h, prev = rec["capture"][anchor(k)]
            outputs = net.delta(h, prev)
            start = 0
            for _, n, q, _ in subset:
                table[k][q] = outputs[start:start + n]
                start += n
    ordering = sorted(range(len(info)), key=lambda i: order_key("donors", info[i]["id"]))
    donor_q = {recipient: ordering[(j+1) % len(ordering)] for j, recipient in enumerate(ordering)}
    assert all(x != y for x, y in donor_q.items())
    return table, donor_q


def donor_rows(table, assignment, items):
    rows = []
    for _, n, q, _ in items:
        d = table[assignment[q]]
        indices = torch.arange(n, device=d.device).clamp_max(len(d)-1)
        rows.append(d.index_select(0, indices))
    return torch.cat(rows)


def get_constant(args, old, model, tok, net, dest):
    if args.constant:
        obj = torch.load(args.constant, weights_only=True, map_location="cpu")
        if torch.is_tensor(obj):
            return obj.float().cuda(), dict(kind="single supplied vector", checkpoint_sha256=fingerprint(args.constant))
        from minimal import load_net
        const = load_net(args.constant)
        return const, dict(kind="supplied task-trained module", checkpoint_sha256=fingerprint(args.constant))
    items, _ = read_data(old, args, tok, "train")
    means = defaultdict(list)
    for subset in batches(items, args.batch_size):
        _, state = forward(model, old.pack(subset), 3, inspect=(PHYSICAL, 1, 2, 3), keep_trace=False)
        for loc, (h, prev) in state["capture"].items():
            d = net.delta(h, prev)
            means[loc].append((d / rms(d).clamp_min(1e-20).unsqueeze(-1)).sum(0))
    direction = torch.stack([torch.stack(x).sum(0) for x in means.values()]).mean(0)
    path = dest / "TRAIN_MEAN_DIRECTION.pt"
    torch.save(direction.cpu(), path)
    return direction, dict(kind="Train imitation mean of tokenwise unit-RMS directions",
        checkpoint_sha256=fingerprint(path), train_sha256=fingerprint(args.corpus / "train.parquet"),
        model_calls=math.ceil(len(items)/args.batch_size), native_passes_per_call=3)


def score(old, model, packed, items, terminal):
    return old.scores(model.lm_head(terminal).float(), packed[1].cpu().numpy(), packed[-1], len(items))


def predict(tag, old, info, items, output):
    old.record(info, items, {tag: output})


def save_scores(old, path, rows):
    tmp = path.with_suffix(".tmp")
    old.write_predictions(tmp, rows)
    tmp.replace(path)


def profile(info, key, metric):
    result = []
    for row in info:
        values = np.asarray(row["scores"][key], dtype=np.float64)
        if metric.endswith("_norm"):
            values = values / row["characters"]
        gold = row["gold"]
        if metric.startswith("acc"):
            result.append(float(values.argmax() == gold))
        else:
            result.append(values[gold] - np.delete(values, gold).max())
    return np.asarray(result)


def paired_table(info, contrasts, seed):
    metrics = ("acc", "acc_norm", "margin", "margin_norm")
    rng = np.random.default_rng(seed)
    i = rng.integers(len(info), size=(20_000, len(info)), dtype=np.int32)
    fa = .05 / (len(contrasts) * len(metrics))
    q = (.025, .975, fa/2, 1-fa/2)
    result = []
    for name, weights in contrasts.items():
        for metric in metrics:
            effect = sum(sign * profile(info, tag, metric) for tag, sign in weights.items())
            ci = np.quantile(effect[i].mean(1), q).tolist()
            stats = dict(test=name, metric=metric, difference=float(effect.mean()),
                         paired_bootstrap_95=ci[:2], bonferroni_bootstrap_95=ci[2:],
                         scores={tag: float(profile(info, tag, metric).mean()) for tag in weights})
            if metric.startswith("acc") and len(weights) == 2:
                stats.update(rescued=int((effect > 0).sum()), degraded=int((effect < 0).sum()))
            result.append(stats)
    return result


class Probe:
    def __init__(self):
        self.sum = defaultdict(float)
        self.n = defaultdict(int)

    def add(self, key, values):
        self.sum[key] += float(values.double().sum())
        self.n[key] += values.numel()

    def summary(self):
        return {k: self.sum[k] / v for k, v in self.n.items()}


def test_invariance(old, model, items, net, dose):
    standard = next(x for x in items if x[1] >= 3)
    sequence, n, q, a = standard
    future = list(sequence)
    changed = len(future)-n+1
    future[changed] = 2 if future[changed] == 1 else 1
    packed = old.pack([standard, (future, n, q, a)])
    early = int((packed[2][:n] < changed).sum())
    result = {}
    for k in DEPTHS:
        learn = lambda h, prev: dose*net.delta(h, prev)
        zero = lambda h, _prev: torch.zeros_like(h)
        args = (model, packed, k)
        output, _ = forward(*args, {anchor(k): learn}, keep_trace=False)
        stock, _ = forward(*args, keep_trace=False)
        dose_zero, _ = forward(*args, {anchor(k): zero}, keep_trace=False)
        future_error = float((output[:early].float()-output[n:n+early].float()).abs().max())
        zero_error = float((stock.float()-dose_zero.float()).abs().max())
        if future_error or zero_error:
            raise RuntimeError(f"Failed K{k} causality/parity check ({future_error}, {zero_error})")
        result[f"k{k}"] = dict(prediction_positions_checked=early,
                               future_token_max_abs_hidden_difference=future_error,
                               zero_dose_max_abs_hidden_difference=zero_error)
    return result


def run_b(args, old, model, tokenizer, net):
    from minimal import load_net
    items, info = read_data(old, args, tokenizer, "validation")
    checkpoint = load_net(args.distill) if args.distill else None
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    constant, constant_manifest = get_constant(args, old, model, tokenizer, net, args.out)
    donors, mapping = collect_donors(old, args, model, items, info, net)
    parity = test_invariance(old, model, items, net, args.dose)
    generator = torch.Generator().manual_seed(REV)
    width = model.config.hidden_size
    permutation = torch.randperm(width, generator=generator).cuda()
    signs = (2 * torch.randint(2, (width,), generator=generator) - 1).cuda()
    did, specificity, probes, timings = {}, {}, {}, {}
    for k in DEPTHS:
        begin = time.perf_counter()
        depth_probe = Probe()
        suffix = anchor(k)
        arms = dict(native=None, learned_open="learned", learned_held="learned",
                    rotated_open="rotation", rotated_held="rotation",
                    exchange_open="exchange", constant_open="constant")
        if checkpoint is not None:
            arms["distilled_open"] = "distilled"
        for subset in batches(items, args.batch_size):
            packed = old.pack(subset)
            before, reference = forward(model, packed, k, inspect=(suffix,),
                                        branch_start=suffix, keep_trace=False)
            h, prev = reference["capture"][suffix]
            true = args.dose*net.delta(h, prev)
            exchange = donor_rows(donors[k], mapping, subset)
            vec = constant.delta(h, prev) if hasattr(constant, "delta") else constant.expand_as(h)
            deltas = dict(learned=true, rotation=true[:, permutation]*signs,
                          exchange=same_rms(exchange, true), constant=same_rms(vec, true))
            if checkpoint is not None:
                deltas["distilled"] = same_rms(checkpoint.delta(h, prev), true)
            for name, control in arms.items():
                if control is None:
                    after = before
                else:
                    held = name.endswith("_held")
                    check = name in ("learned_open", "learned_held", "rotated_open", "rotated_held")
                    after, trace = forward(model, packed, k, {suffix: lambda _h,_p: deltas[control]},
                                           branch_start=suffix if check else None,
                                           native=reference, clamp=held, keep_trace=check)
                    depth_probe.add(f"{name}/requested_relative_rms", rms(deltas[control])/rms(h))
                    depth_probe.add(f"{name}/realized_relative_rms", trace["realized"][suffix])
                    for (turn, branch), value in trace["branch_response"].items():
                        depth_probe.add(f"{name}/loop{turn}/{branch}_response_rms", value)
                predict(f"{name}_k{k}", old, info, subset, score(old, model, packed, subset, after))
        tag = lambda label: f"{label}_k{k}"
        did[f"k{k}/direction_x_computation"] = {
            tag("learned_open"): 1, tag("learned_held"): -1,
            tag("rotated_open"): -1, tag("rotated_held"): 1}
        for control in ("native", "rotated_open", "exchange_open", "constant_open", "distilled_open"):
            if control in arms:
                specificity[f"k{k}/learned_vs_{control}"] = {tag("learned_open"): 1, tag(control): -1}
        probes[f"k{k}"] = depth_probe.summary()
        timings[f"k{k}"] = dict(seconds=time.perf_counter()-begin,
                               model_calls=math.ceil(len(items)/args.batch_size)*len(arms),
                               passes_per_call=k)
        print("B complete", k, {name: int(profile(info, tag(name), "acc_norm").sum()) for name in arms},
              flush=True)
    save_scores(old, args.out/"B_PREDICTIONS.jsonl", info)
    write_json(args.out/"B_FACTORIAL.json", dict(
        question_count=len(info), candidate_count=len(items), ids=[x["id"] for x in info],
        donor_question={info[q]["id"]: info[other]["id"] for q,other in mapping.items()},
        subset_rule="First ids by seeded SHA-256 when --questions is smaller than 299",
        donor_rule="Seeded cyclic derangement of questions; first listed donor answer; matched prefix step, "
                   "then carried forward from final available position; all deltas scaled to recipient RMS",
        branch_rule="Clamp both additive attention and MLP writes at every suffix layer and all positions, "
                    "with matched native references from the same causal candidate sequence. Keep direct residual.",
        rotation_rule="One seeded signed coordinate permutation shared across K1–K4.",
        constant=constant_manifest, dose=args.dose, parity=parity, timings=timings,
        donor_model_calls=math.ceil(len(info)/args.batch_size), donor_passes_per_call=3,
        parity_model_calls=12, peak_bytes=torch.cuda.max_memory_allocated(),
        wall_seconds=time.perf_counter()-start,
        factorial=paired_table(info, did, REV+1), direction_controls=paired_table(info, specificity, REV+2),
        update_and_branch_response=probes))


def displacement(probe, trace, reference, arm, first):
    delta = trace["edits"][first]
    for source in trace["edits"]:
        probe.add(f"{arm}/loop{source}/realized_edit_relative_rms", trace["realized"][source])
    for turn in range(first, 5):
        stock = reference["normed"][turn]
        diff = trace["normed"][turn]-stock
        size = rms(diff)
        cosine = (diff*delta).mean(-1) / (size*rms(delta)).clamp_min(1e-20)
        valid = (size > 0) & (rms(delta) > 0)
        probe.add(f"{arm}/loop{turn}/normalized_displacement_relative_rms", size/rms(stock))
        if valid.any():
            degrees = cosine[valid].clamp(-1, 1).acos()*180/math.pi
            probe.add(f"{arm}/loop{turn}/direction_angle_degrees_to_first_raw_edit", degrees)
        if turn > first:
            raw_diff = trace["raw"][turn]-reference["raw"][turn]
            denominator = rms(reference["raw"][turn]).clamp_min(1e-20)
            probe.add(f"{arm}/loop{turn}/raw_before_next_edit_displacement_relative_rms", rms(raw_diff)/denominator)
    for (turn, name), v in trace["branch_response"].items():
        probe.add(f"{arm}/loop{turn}/{name}_response_rms", v)


def conditioned(info, options):
    folds = [[], []]
    ordered = sorted(info, key=lambda row: order_key("crossfit_conditional", row["id"]))
    for i, row in enumerate(ordered):
        folds[i % len(folds)].append(row)
    selectors = []
    for train_side in range(len(folds)):
        tune = folds[train_side]
        choose = max(options, key=lambda tag: (
            profile(tune, tag, "acc").sum()+profile(tune, tag, "acc_norm").sum(),
            tag == "repeat_budget_matched"))
        test_side = 1-train_side
        for row in folds[test_side]:
            row["scores"]["length_conditioned_crossfit"] = row["scores"][choose]
        selectors.append(dict(tuning_side=train_side, evaluation_side=test_side,
                              selected_schedule=choose, evaluation_questions=len(folds[test_side])))
    return selectors


def run_c(args, old, model, tokenizer, net):
    items, info = read_data(old, args, tokenizer, "validation")
    schedules = {
        "native": {},
        "early_1_to_2": {1: 1.},
        "middle_2_to_3": {2: 1.},
        "late_3_to_4": {3: 1.},
        "repeat_same_dose": {1: 1., 2: 1., 3: 1.},
        "repeat_budget_matched": {1: 1/3, 2: 1/3, 3: 1/3},
        "repeat_early_emphasis": {1: .5, 2: 1/3, 3: 1/6},
        "repeat_late_emphasis": {1: 1/6, 2: 1/3, 3: .5},
    }
    probe = Probe()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for subset in batches(items, args.batch_size):
        packed = old.pack(subset)
        stock, reference = forward(model, packed, 4, branch_start=1)
        for arm, schedule in schedules.items():
            if not schedule:
                output = stock
            else:
                def scaled(amount):
                    return lambda h, prev: args.dose*amount*net.delta(h, prev)
                first = min(schedule)
                policies = {turn: scaled(amount) for turn, amount in schedule.items()}
                output, trace = forward(model, packed, 4, policies, branch_start=first, native=reference)
                displacement(probe, trace, reference, arm, first)
            predict(arm, old, info, subset, score(old, model, packed, subset, output))
    exploratory = conditioned(info, ("repeat_budget_matched", "repeat_early_emphasis", "repeat_late_emphasis"))
    contrasts = {f"{tag}_vs_native": {tag: 1, "native": -1}
                 for tag in schedules if tag != "native"}
    for comparator in ("early_1_to_2", "middle_2_to_3", "repeat_same_dose", "repeat_budget_matched"):
        contrasts[f"{comparator}_vs_late_3_to_4"] = {comparator: 1, "late_3_to_4": -1}
    save_scores(old, args.out/"C_PREDICTIONS.jsonl", info)
    write_json(args.out/"C_POSITIONS.json", dict(
        question_count=len(info), candidate_count=len(items), ids=[x["id"] for x in info],
        total_passes_per_call=4, schedules=schedules, locked_single_dose=args.dose,
        cost=dict(seconds=time.perf_counter()-start, peak_bytes=torch.cuda.max_memory_allocated(),
                  calls=math.ceil(len(items)/args.batch_size)*len(schedules)),
        trajectories=probe.summary(), paired=paired_table(info, contrasts, REV+3),
        excitation="State angles are per-token angles between realized first raw update and "
                   "normalized displacement from the matching native trajectory. Angles can also reflect the carry RMSNorm.",
        amplitude_matching="The dose-matched schedules fix the sum of coefficients, with actual per-boundary "
                           "update RMS recorded because directions and states can change after earlier writes.",
        exploring_length_condition=dict(
            split="Deterministic two-fold cross-fit on Validation for choosing early/flat/late dose schedule by raw+normalized accuracy.",
            folds=exploratory, paired=paired_table(info, {
                "length_conditioned_vs_fixed_repeat": {"length_conditioned_crossfit": 1, "repeat_budget_matched": -1},
                "length_conditioned_vs_late": {"length_conditioned_crossfit": 1, "late_3_to_4": -1},
            }, REV+4))))
    print("C complete", {tag: int(profile(info, tag, "acc_norm").sum()) for tag in schedules}, flush=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("b", "c", "both"))
    for name in ("model", "archive", "corpus", "checkpoint", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    for name in ("distill", "constant"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--dose", required=True, type=float)
    parser.add_argument("--questions", type=int, default=299)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not 2 <= args.questions <= 299 or args.batch_size < 1 or args.dose <= 0 or not math.isfinite(args.dose):
        parser.error("Expected 2–299 questions, a positive batch size, and a finite positive locked dose")
    from minimal import load_net
    sys.path.insert(0, str(args.archive.resolve()))
    import run_arc as old
    args.out.mkdir(exist_ok=True, parents=True)
    model, tok, runtime = old.load_runtime(args.model)
    model.requires_grad_(False)
    net = load_net(args.checkpoint)
    write_json(args.out/f"MANIFEST_{args.stage.upper()}.json", dict(
        validation_sha256=fingerprint(args.corpus/"validation.parquet"),
        checkpoint_sha256=fingerprint(args.checkpoint), script_sha256=fingerprint(Path(__file__)),
        distill_sha256=fingerprint(args.distill) if args.distill else None,
        constant_sha256=fingerprint(args.constant) if args.constant else None, dose=args.dose,
        stage=args.stage, seed=REV, batch_size=args.batch_size, runtime=runtime))
    if args.stage in ("b", "both"):
        run_b(args, old, model, tok, net)
    if args.stage in ("c", "both"):
        run_c(args, old, model, tok, net)


if __name__ == "__main__":
    main()

