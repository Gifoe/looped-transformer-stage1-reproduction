"""Downstream causal interventions following Ouro recurrent writeback."""
from collections import defaultdict
from math import comb
import time

import numpy as np
import torch


def controls(fb, args):
    saved = args.out / "distill_fit.pt"
    prior = args.prior / "results"
    paths = {
        "feedback_rotation": None,
        "feedback_previous_direction": args.initial or prior / "direction_round1.pt",
        "feedback_shuffled_teacher": args.shuffled or prior / "shuffled_round1.pt",
    }
    if args.distill is not None or saved.exists():
        paths["feedback_state_distillation"] = args.distill or saved
    else:
        paths["feedback_prior_write_distillation"] = prior / "completion_round1.pt"
    return paths


@torch.no_grad()
def assert_causal(fb, old, model, items, net, dose):
    eligible = next(item for item in items if item[1] >= 3)
    sequence, n, q, a = eligible
    comparison = [eligible, (list(sequence), n, q, a)]
    changed_index = len(sequence) - n + 1
    sequence_b = comparison[1][0]
    sequence_b[changed_index] = 1 if sequence_b[changed_index] != 1 else 2
    packed = old.pack(comparison)
    out = {}
    for k in fb.DEPTHS:
        actual, _, _ = fb.forward(model, packed, k, net, dose, None, 0.)
        native, _, _ = fb.forward(model, packed, k, net, 0., None, 0.)
        baseline, _, _ = fb.forward(model, packed, k, None, 0., None, 0.)
        same = int((packed[2][:n] < changed_index).sum())
        causal_error = float((actual[:same].float() - actual[n:n+same].float()).abs().max())
        zero_error = float((native.float() - baseline.float()).abs().max())
        out[f"k{k}"] = dict(prior_prediction_positions=same,
                           max_prior_hidden_abs_error=causal_error,
                           max_zero_dose_hidden_abs_error=zero_error)
        if causal_error != 0 or zero_error != 0:
            raise RuntimeError(f"Causal or zero-dose parity check failed at K{k}: {out[f'k{k}']}")
    return out


def path_statistics(fb, info):
    """Paired margin and decision contrasts for the pre-specified interventions."""
    comparisons = {
        "natural_effect": ("feedback_full", "local_native_reference"),
        "computed_suffix_over_direct": ("feedback_full", "feedback_direct_residual"),
        "restoring_attention": ("feedback_restore_attention", "feedback_direct_residual"),
        "restoring_mlp": ("feedback_restore_mlp", "feedback_direct_residual"),
        "recurrent_increment_over_readout": ("matched_readout_plus_feedback", "matched_readout_only"),
        "direction_specificity": ("feedback_full", "feedback_rotation"),
    }
    rng = np.random.default_rng(fb.SEED + 2)
    samples = rng.integers(0, len(info), size=(10_000, len(info)))
    results = []

    def interval(difference):
        return np.quantile(difference[samples].mean(1), [.025, .975]).tolist()

    def profile(key, normalized):
        decisions, margins = [], []
        for row in info:
            values = np.array(row["scores"][key])
            if normalized:
                values = values / row["characters"]
            gold = row["gold"]
            decisions.append(int(values.argmax() == gold))
            margins.append(values[gold] - np.delete(values, gold).max())
        return np.array(decisions), np.array(margins)

    for k in fb.DEPTHS:
        for contrast, (left, right) in comparisons.items():
            i, j = f"{left}_k{k}", f"{right}_k{k}"
            if i not in info[0]["scores"] or j not in info[0]["scores"]:
                continue
            for metric in ("acc", "acc_norm"):
                correct_a, margins_a = profile(i, metric == "acc_norm")
                correct_b, margins_b = profile(j, metric == "acc_norm")
                corrected = correct_a - correct_b
                gain = int((corrected == 1).sum())
                loss = int((corrected == -1).sum())
                total = gain + loss
                p_value = min(1., 2 * sum(comb(total, n) for n in range(min(gain,loss)+1)) /
                              2**total)
                margin_shift = margins_a - margins_b
                results.append(dict(depth=k, contrast=contrast, method=i, comparator=j,
                                    metric=metric, rescued=gain, degraded=loss,
                                    accuracy_difference=float(corrected.mean()),
                                    accuracy_paired_bootstrap_95=interval(corrected),
                                    margin_difference=float(margin_shift.mean()),
                                    margin_paired_bootstrap_95=interval(margin_shift),
                                    exact_mcnemar_p=p_value))
    adjusted = 0.
    for index, row in enumerate(sorted(results, key=lambda x: x["exact_mcnemar_p"])):
        adjusted = max(adjusted, min(1., (len(results)-index)*row["exact_mcnemar_p"]))
        row["holm_p_across_depth_metric_and_contrast"] = adjusted
    return results


@torch.no_grad()
def mechanism(fb, args, old, model, tokenizer):
    net, dose, readout, readout_dose = fb.frozen(args)
    items, all_info, split_sha = fb.read_split(args, old, tokenizer, "validation")
    parity = assert_causal(fb, old, model, items, net, dose)
    control_paths = controls(fb, args)
    control_nets = {name: fb.operator(path) if path else net
                    for name, path in control_paths.items()}
    rng = np.random.default_rng(fb.SEED + 1)
    chosen = sorted(rng.choice(len(all_info), size=min(args.mechanism_questions, len(all_info)),
                               replace=False).tolist())
    remap = {old_id: new for new, old_id in enumerate(chosen)}
    items = [(seq,n,remap[q],a) for seq,n,q,a in items if q in remap]
    info = [all_info[i] for i in chosen]
    costs, changes, metrics = {}, {}, {}
    held_conditions = {
        "feedback_full": (),
        "feedback_direct_residual": ("attention", "mlp"),
        "feedback_restore_attention": ("mlp",),
        "feedback_restore_mlp": ("attention",),
    }
    for k in fb.DEPTHS:
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        sums = defaultdict(lambda: defaultdict(float))
        count = 0
        for lo in range(0, len(items), args.batch_size):
            subset = items[lo:lo+args.batch_size]
            packed = old.pack(subset)
            group, target = packed[1].cpu().numpy(), packed[-1]
            n = target.numel()
            count += n

            def score(label, state):
                logits = model.lm_head(state).float()
                return {f"{label}_k{k}": old.scores(logits, group, target, len(subset))}

            end, readout_end, reference = fb.forward(
                model, packed, k, None, 0., readout, readout_dose, recording=True)
            out = score("local_native_reference", end)
            if readout_end is not None:
                out.update(score("matched_readout_only", readout_end))
            for label, held in held_conditions.items():
                paired_readout = readout if label == "feedback_full" else None
                altered, joint, trace = fb.forward(
                    model, packed, k, net, dose, paired_readout, readout_dose,
                    reference=reference, held=held)
                out.update(score(label, altered))
                if joint is not None:
                    out.update(score("matched_readout_plus_feedback", joint))
                for (branch, layer), value in trace["branch"].items():
                    sums[label][f"{branch}_after_layer_{layer+1}_delta_rms"] += value*n
                for layer, value in trace["residual"].items():
                    sums[label][f"residual_after_layer_{layer+1}_delta_rms"] += value*n
                sums[label]["ending_normed_delta_rms"] += fb.rms(altered.float()-end.float()).mean()*n
                sums[label]["anchor_delta_rms"] += fb.rms(trace["delta"]).mean()*n
            for label, control in control_nets.items():
                mode = "rotate" if label == "feedback_rotation" else "full"
                altered, _, trace = fb.forward(model, packed, k, control, dose, None, 0., mode)
                out.update(score(label, altered))
                sums[label]["ending_normed_delta_rms"] += fb.rms(altered.float()-end.float()).mean()*n
                sums[label]["relative_anchor_delta_rms"] += trace["edit_relative_rms"]*n
            old.record(info, subset, out)
        changes[f"k{k}"] = {label: {key: float(value)/count for key, value in row.items()}
                           for label, row in sums.items()}
        metrics[f"k{k}"] = {label: fb.counts(info, label) for label in info[0]["scores"]
                           if label.endswith(f"_k{k}")}
        costs[f"k{k}"] = dict(seconds=time.perf_counter()-start, answer_positions=count,
                             calls=((len(items)+args.batch_size-1)//args.batch_size)*
                                   (1+len(held_conditions)+len(control_nets)),
                             candidates=len(items), layers_after_anchor=3 if k == 1 else 24,
                             batch_size=args.batch_size,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated())
        print("feedback mechanism", k, metrics[f"k{k}"], flush=True)
    fb.dump_predictions(old, args.out / "MECHANISM_PREDICTIONS.jsonl", info)
    old.dump(args.out / "MECHANISM.json", dict(
        split="validation", split_sha256=split_sha, sample_indices=chosen,
        ids=[row["id"] for row in info], subset_seed=fb.SEED+1,
        method="Clamp each specified attention/MLP addition after its output RMSNorm "
               "to its matching native activation at every position of the actual suffix. "
               "All other branch computations run on the evolving edited residual.",
        cost=costs, parity=parity, changes=changes, counts=metrics,
        paired_contrasts=path_statistics(fb, info),
        controls={name: dict(path=str(path), sha256=fb.sha(path)) if path else
                  dict(seed=fb.SEED, geometry="shift norm and native write angle matched")
                  for name, path in control_paths.items()}))


def full_controls(fb, args, old, model, tokenizer):
    net, dose, _, _ = fb.frozen(args)
    items, info, data_sha = fb.read_split(args, old, tokenizer, "test")
    paths = controls(fb, args)
    costs, metrics = {}, {}
    for label, path in paths.items():
        control = fb.operator(path) if path else net
        mode = "rotate" if label == "feedback_rotation" else "full"
        for k in fb.DEPTHS:
            key = f"{label}_k{k}"
            costs[key] = fb.score_arm(args, old, model, items, info, k, control, dose, label,
                                      mode=mode)
            metrics[key] = fb.counts(info, key)
            print("feedback test control", key, metrics[key], flush=True)
            fb.dump_predictions(old, args.out / "TEST_CONTROL_PREDICTIONS.jsonl", info)
            old.dump(args.out / "TEST_CONTROL.json", dict(
                data_sha256=data_sha, cost=costs, counts=metrics,
                checkpoints={name: dict(path=str(path), sha256=fb.sha(path)) if path else
                             dict(seed=fb.SEED) for name, path in paths.items()}))

