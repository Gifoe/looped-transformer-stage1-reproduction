"""Causal recurrent writeback into the frozen Ouro stack."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEED = 261122921
DEPTHS = (1, 2, 3, 4)
FIRST = 18
ANCHOR = 20  # zero-based; K1 still executes physical layers 21 through 23
BRANCHES = {
    "attention": "input_layernorm_2",
    "mlp": "post_attention_layernorm_2",
}
CHANNELS = ("attention_first", "attention_last", "mlp_first", "mlp_last")


def configuration(args):
    sys.path.insert(0, str(args.prior.resolve()))
    from amplify import group_choices, operator, orthogonal, rms
    from arc_features import archive, atomic_save, sha
    globals().update(group_choices=group_choices, operator=operator, orthogonal=orthogonal,
                     rms=rms, atomic_save=atomic_save, sha=sha)
    return archive(args.archive)


def update(net, state, k, mode):
    """Run inherited state-only operator, keeping its edit in float32."""
    raw = state["raw"][:, k - 1].float()
    _, (gain, shift, wr, w, hr) = net(state, k)
    if mode == "rotate":
        # One seeded orthogonal signed permutation, then preserve native write
        # orthogonality and the exact recipient direction norm.
        generator = torch.Generator().manual_seed(SEED)
        size = shift.shape[-1]
        permutation = torch.randperm(size, generator=generator).to(shift.device)
        signs = (2 * torch.randint(2, (size,), generator=generator) - 1).to(shift.device)
        rotated = orthogonal(shift[:, permutation] * signs, w, wr)
        shift = rotated * (rms(shift) / rms(rotated))
    if mode == "direction":
        gain = torch.zeros_like(gain)
    return raw, gain * w + shift, hr


def replace(tensor, group, pos, rows):
    return tensor.index_put((group, pos), rows.to(tensor.dtype))


@contextmanager
def intervention(model, depth, group, pos, net, dose, mode="full",
                 collect=False, reference=None, held=(), recording=False, anchor_trace=False):
    """Track raw states across native normalization and write a single boundary."""
    core = model.model
    current = 0
    past = []
    writes = {}
    handles = []
    outputs = {"branch": {}, "residual": {}, "delta": None, "features": None,
               "edit_relative_rms": None}
    tracking_suffix = recording or reference is not None

    def features(raw):
        state = {"raw": torch.stack([*past, raw], dim=1)}
        for name in CHANNELS:
            state[name] = torch.stack(
                [torch.zeros_like(raw)] * len(past) +
                [writes.get(name, torch.zeros_like(raw)).to(raw.dtype)], dim=1)
        return state

    def edited(tensor):
        if net is None and not anchor_trace:
            return tensor
        state = features(tensor[group, pos])
        if anchor_trace:
            outputs["anchor"] = state
        if net is None or not dose:
            return tensor
        h, delta, hr = update(net, state, len(past) + 1, mode)
        if tracking_suffix:
            outputs["delta"] = dose * delta.detach()
        outputs["edit_relative_rms"] = (dose * rms(delta.detach()) / hr.detach()).mean()
        return replace(tensor, group, pos, h + dose * delta)

    def finish(_, args):
        nonlocal current
        tensor = args[0]
        raw = tensor[group, pos]
        if current == depth - 1 and collect:
            outputs["features"] = features(raw)
        if current == depth - 2 and depth > 1:
            tensor = edited(tensor)
        past.append(raw)
        writes.clear()
        current += 1
        if tensor is not args[0]:
            return (tensor, *args[1:])

    def branch(name, layer):
        def handle(_, args, tensor):
            is_suffix = current == depth - 1 and (depth > 1 or layer > ANCHOR)
            key = (name, layer)
            if is_suffix:
                if recording:
                    outputs["branch"][key] = tensor.detach()
                if reference is not None and name in held:
                    tensor = reference["branch"][key]
                if reference is not None:
                    native = reference["branch"][key][group, pos]
                    outputs["branch"][key] = rms(tensor[group, pos].float() - native.float()).mean()
            if layer >= FIRST:
                band = "last" if layer > ANCHOR else "first"
                channel = f"{name}_{band}"
                value = tensor[group, pos].float()
                writes[channel] = writes.get(channel, 0) + value
            return tensor
        return handle

    def physical(layer):
        def handle(_, args, tensor):
            if depth == 1 and layer == ANCHOR:
                tensor = edited(tensor)
            if current == depth - 1 and (depth > 1 or layer > ANCHOR):
                if recording:
                    outputs["residual"][layer] = tensor[group, pos].detach()
                if reference is not None:
                    native = reference["residual"][layer]
                    outputs["residual"][layer] = rms(tensor[group, pos].float() - native.float()).mean()
            return tensor
        return handle

    handles.append(core.norm.register_forward_pre_hook(finish))
    for i, layer in enumerate(core.layers):
        if tracking_suffix or i >= FIRST:
            for name, attr in BRANCHES.items():
                handles.append(getattr(layer, attr).register_forward_hook(branch(name, i)))
        if tracking_suffix or (depth == 1 and i == ANCHOR):
            handles.append(layer.register_forward_hook(physical(i)))
    try:
        yield outputs
        if current != depth:
            raise RuntimeError(f"Expected {depth} native passes, observed {current}")
    finally:
        for handle in handles:
            handle.remove()


def forward(model, packed, depth, net, dose, readout, readout_dose, mode="full", **audit):
    inp, group, pos, _ = packed
    model.model.total_ut_steps = depth
    with intervention(model, depth, group, pos, net, dose, mode,
                      collect=readout is not None, **audit) as trace:
        _, normalized, _ = model.model(input_ids=inp, use_cache=False)
    end = normalized[-1][group, pos]
    if readout is not None:
        raw, delta, _ = update(readout, trace["features"], depth, "full")
        adjusted = model.model.norm((raw + readout_dose * delta).to(end.dtype))
    else:
        adjusted = None
    return end, adjusted, trace


def dump_predictions(old, path, rows):
    tmp = path.with_suffix(".tmp")
    old.write_predictions(tmp, rows)
    tmp.replace(path)


def read_split(args, old, tokenizer, split):
    if split == "external":
        import pyarrow.parquet as pq
        source = args.external_data
        rows = pq.read_table(source).to_pylist()
    else:
        source = args.corpus / f"{split}.parquet"
        rows = old.read_data(args.corpus, split)
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("Evaluation ids must be unique")
    items, info = old.encode(tokenizer, rows)
    items.sort(key=lambda item: len(item[0]))
    return items, info, sha(source)


def write_weights(net, path, history):
    state = {key: value.detach().cpu() for key, value in net.state_dict().items()}
    atomic_save(dict(architecture=net.method, state=state, history=history), path)


@torch.enable_grad()
def fit(args, old, model, tokenizer, initial, distill=False):
    items, info, split_sha = read_split(args, old, tokenizer, "train")
    by_question = defaultdict(list)
    for item in items:
        by_question[item[2]].append(item)
    net = operator(initial).train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=.03)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rng = np.random.default_rng(SEED)
    history = []
    output = "distill" if distill else "feedback"
    for epoch in range(args.epochs):
        order = rng.permutation(len(info)).tolist()
        if args.fit_questions:
            order = order[:args.fit_questions]
        # Process windows by maximum length to limit padding without shrinking
        # the final train cohort.
        for lo in range(0, len(order), 64):
            order[lo:lo+64] = sorted(
                order[lo:lo+64], key=lambda q: max(len(x[0]) for x in by_question[q]))
        losses, grad_norms, batches, passes, padded_tokens = [], [], 0, 0, 0
        for step, lo in enumerate(range(0, len(order), args.train_batch)):
            questions = order[lo:lo+args.train_batch]
            subset = [x for q in questions for x in sorted(by_question[q], key=lambda x: x[3])]
            k = DEPTHS[(step + epoch) % len(DEPTHS)]
            packed = old.pack(subset)
            if distill:
                # Predict the next native normalized state using the same
                # bounded update, with teacher values available only for fit.
                with torch.no_grad():
                    end, _, trace = forward(model, packed, k, None, 0., None, 0.,
                                            anchor_trace=True)
                state = trace["anchor"]
                source = state["raw"].shape[1]
                h, delta, _ = update(net, state, source, "full")
                predicted = model.model.norm((h + delta).to(end.dtype))
                loss = F.mse_loss(predicted.float(), end.float())
            else:
                end, _, trace = forward(model, packed, k, net, 1., None, 0.)
                logits = model.lm_head(end).float()
                logp = -F.cross_entropy(logits, packed[-1], reduction="none")
                sums = logp.new_zeros(len(subset)).index_add(0, packed[1], logp)
                lengths = sums.new_tensor([c for q in questions for c in info[q]["characters"]])
                options = [len(info[q]["labels"]) for q in questions]
                gold = [info[q]["gold"] for q in questions]
                loss = .5 * group_choices(sums / 3., options, gold)
                loss = loss + .5 * group_choices(16. * sums / lengths, options, gold)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = nn.utils.clip_grad_norm_(net.parameters(), 1.)
            if batches == 0 and not (torch.isfinite(grad) and grad > 0):
                raise RuntimeError("Expected a nonzero finite gradient through the frozen Ouro suffix")
            opt.step()
            losses.append(loss.detach())
            grad_norms.append(grad.detach())
            batches += 1
            passes += k
            padded_tokens += packed[0].numel() * k
            if (step + 1) % 50 == 0:
                print(output, "fit", epoch + 1, step + 1, len(order),
                      round(time.perf_counter() - started, 2), flush=True)
        history.append(dict(epoch=epoch + 1, questions=len(order), batches=batches,
                            mean_loss=float(torch.stack(losses).mean()),
                            mean_preclip_grad_norm=float(torch.stack(grad_norms).mean()),
                            loop_passes_across_calls=passes, padded_token_uses=padded_tokens))
        write_weights(net, args.out / f"{output}_fit.pt", history)
    old.dump(args.out / f"{output.upper()}_FIT.json", dict(history=history, train_sha256=split_sha,
             duration_s=time.perf_counter() - started, seed=SEED, learning_rate=args.lr,
             initial_sha256=sha(initial), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
             train_question_batch=args.train_batch, frozen_parameters=sum(
                 p.numel() for p in model.parameters()),
             trainable_parameters=sum(p.numel() for p in net.parameters())))


def counts(info, key):
    raw_good = norm_good = 0
    for row in info:
        raw = np.array(row["scores"][key])
        raw_good += int(raw.argmax() == row["gold"])
        norm_good += int((raw / row["characters"]).argmax() == row["gold"])
    return dict(acc=raw_good, acc_norm=norm_good)


@torch.no_grad()
def score_arm(args, old, model, items, info, k, net, dose, name, readout=None,
              readout_dose=0., mode="full"):
    elapsed, edit = [], []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    tokens = padded = 0
    arm = f"{name}_k{k}"
    both = f"readout_plus_feedback_k{k}"
    for lo in range(0, len(items), args.batch_size):
        begin = time.perf_counter()
        subset = items[lo:lo+args.batch_size]
        packed = old.pack(subset)
        end, adjusted, trace = forward(model, packed, k, net, dose, readout, readout_dose, mode)
        group = packed[1].cpu().numpy()
        output = {arm: old.scores(model.lm_head(end).float(), group, packed[-1], len(subset))}
        if adjusted is not None:
            output[both] = old.scores(model.lm_head(adjusted).float(), group, packed[-1], len(subset))
        elapsed.append(time.perf_counter() - begin)
        old.record(info, subset, output)
        if trace["edit_relative_rms"] is not None:
            edit.append((float(trace["edit_relative_rms"]), packed[-1].numel()))
        tokens += sum(len(x[0]) - 1 for x in subset)
        padded += packed[0].numel()
    return dict(seconds_full_split=sum(elapsed), median_batch_ms=1000 * np.median(elapsed),
                p95_batch_ms=1000 * np.quantile(elapsed,.95), batch_size=args.batch_size,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), batches=len(elapsed),
                exact_loop_passes_per_forward=k, candidates=len(items), token_uses=tokens*k,
                padded_token_uses=padded*k, operator_parameters=sum(p.numel() for p in net.parameters()),
                readout_parameters=sum(p.numel() for p in readout.parameters()) if readout else 0,
                relative_write_rms=sum(mean*n for mean,n in edit)/sum(n for _,n in edit) if edit else 0.,
                joint_score_heads=1 + int(readout is not None))


def select(args, old, model, tokenizer, initial):
    items, info, data_sha = read_split(args, old, tokenizer, "validation")
    trials, cost, best = {}, {}, None
    candidates = [(initial, dose, "initial") for dose in args.doses]
    trained = args.out / "feedback_fit.pt"
    if trained.exists():
        candidates.extend((trained, dose, "trained") for dose in args.doses)
    for path, dose, kind in candidates:
        name = f"{kind}_{dose:g}"
        net = operator(path)
        trials[name] = {}
        for k in DEPTHS:
            key = f"{name}_k{k}"
            cost[key] = score_arm(args, old, model, items, info, k, net, dose, name)
            trials[name][f"k{k}"] = counts(info, key)
        totals = [sum(x[metric] for x in trials[name].values()) for metric in ("acc","acc_norm")]
        score = (sum(totals), totals[1])
        if best is None or score > best[0]:
            best = (score, path, dose, name)
        print("feedback validation", name, totals, round(sum(cost[f"{name}_k{k}"]["seconds_full_split"]
                                                            for k in DEPTHS), 1), flush=True)
    _, path, dose, name = best
    freeze = dict(checkpoint=str(path.resolve()), checkpoint_sha256=sha(path),
                  validation_sha256=data_sha, chosen=name, dose=dose,
                  criterion="sum of acc and acc_norm counts across K1–K4, ties by acc_norm",
                  code_sha256=sha(Path(__file__)), trials=trials)
    if args.readout:
        freeze["readout"] = dict(checkpoint=str(args.readout.resolve()),
                                 checkpoint_sha256=sha(args.readout), dose=args.readout_dose)
    dump_predictions(old, args.out / "VALIDATION_PREDICTIONS.jsonl", info)
    old.dump(args.out / "VALIDATION_COST.json", cost)
    old.dump(args.out / "FREEZE.json", freeze)


def frozen(args):
    freeze = json.loads((args.out / "FREEZE.json").read_text())
    path = Path(freeze["checkpoint"])
    if sha(path) != freeze["checkpoint_sha256"]:
        raise RuntimeError("Frozen feedback checkpoint hash mismatch")
    if sha(Path(__file__)) != freeze["code_sha256"]:
        raise RuntimeError("Frozen feedback source hash mismatch")
    readout = freeze.get("readout")
    if readout is not None:
        checkpoint = Path(readout["checkpoint"])
        if sha(checkpoint) != readout["checkpoint_sha256"]:
            raise RuntimeError("Frozen readout checkpoint hash mismatch")
        return operator(path), freeze["dose"], operator(checkpoint), readout["dose"]
    return operator(path), freeze["dose"], None, 0.


def run_eval(args, old, model, tokenizer, split):
    net, dose, readout, readout_dose = frozen(args)
    items, info, data_sha = read_split(args, old, tokenizer, split)
    cost = {}
    for k in DEPTHS:
        cost[f"k{k}"] = score_arm(args, old, model, items, info, k, net, dose, "feedback_only",
                                  readout, readout_dose)
        print("feedback", split, k, counts(info, f"feedback_only_k{k}"),
              round(cost[f"k{k}"]["seconds_full_split"], 2), flush=True)
        # Save each completed full depth, including on budget interruption.
        dump_predictions(old, args.out / f"{split.upper()}_PREDICTIONS.jsonl", info)
        old.dump(args.out / f"{split.upper()}_COST.json", dict(data_sha256=data_sha, cost=cost))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "distill-fit", "validation", "test", "external", "mechanism", "controls", "smoke"))
    for key in ("model", "corpus", "archive", "prior", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    for key in ("readout", "external-data", "initial", "distill", "shuffled"):
        parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--readout-dose", type=float, default=1.)
    parser.add_argument("--doses", type=float, nargs="+", default=(.5, 1.))
    parser.add_argument("--train-batch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--fit-questions", type=int, default=0)
    parser.add_argument("--mechanism-questions", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=.00025)
    args = parser.parse_args()
    if args.stage == "external" and args.external_data is None:
        parser.error("--external-data is required for external")
    args.out.mkdir(parents=True, exist_ok=True)
    old = configuration(args)
    model, tokenizer, runtime = old.load_runtime(args.model)
    old.dump(args.out / "RUNTIME.json", runtime)
    model.eval().requires_grad_(False)
    initial = args.initial or args.prior / "results" / "direction_round1.pt"
    if args.stage in ("fit", "distill-fit", "smoke"):
        if args.stage == "smoke":
            args.fit_questions = min(args.fit_questions or 16, 16)
        fit(args, old, model, tokenizer, initial, distill=args.stage == "distill-fit")
    elif args.stage == "validation":
        select(args, old, model, tokenizer, initial)
    elif args.stage in ("mechanism", "controls"):
        from feedback_paths import full_controls, mechanism
        run = mechanism if args.stage == "mechanism" else full_controls
        run(sys.modules[__name__], args, old, model, tokenizer)
    else:
        run_eval(args, old, model, tokenizer, args.stage)


if __name__ == "__main__":
    main()



