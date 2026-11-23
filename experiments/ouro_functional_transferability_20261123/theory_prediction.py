"""Seal out-of-condition predictions before any confirmation-label forward pass."""

import argparse
from itertools import product
import json
import math
from pathlib import Path

import torch
import numpy as np

import minimal
from forward_control import forward_control
from functional_operator import CELLS, MOMENT_CELLS, batches, cell_key, selected_rows
from normalized_geometry import norm_tangent, rms


TARGETS = ("arc", "csqa")


def select_single_controller(records, predictions):
    """Preselect one source using two target Train matrices and a fixed seed."""
    sources = {"arc", "csqa", "obqa", "qasc", "mixed"}
    values = {}
    for name, record in records.items():
        if name.endswith("/k234") and record["seed"] == 261123137 and record["source"] in sources:
            values[name] = {target: float(np.mean([
                predictions[name][target][f"h{k}_m1"]["target_train_clipped_proxy"] for k in (1, 2, 3)
            ])) for target in TARGETS}
    if len(values) != 5:
        raise ValueError("Require all five fixed-seed candidates to preselect a single checkpoint")
    scale = {target: max(1e-12, float(np.median([abs(s[target]) for s in values.values()]))) for target in TARGETS}
    score = {name:min(raw[target]/scale[target] for target in TARGETS) for name,raw in values.items()}
    winner = max(score, key=lambda name: (score[name], name))
    return dict(single_checkpoint=True, source=records[winner]["source"], candidate=winner,
                checkpoint="results/HOLDOUT/"+winner+"/controller.pt",
                checkpoint_sha256=records[winner]["checkpoint_sha256"], parameters=4096,
                native_backbone_forwards_per_answer=1, cap_rms=.45, seed=261123137,
                target_train_depths=[2,3,4], target_train_scales=scale,
                criterion="Maximize the worse of two domain-rescaled source-averaged target-Train moment projections",
                scores=score, raw_per_domain=values, selection_uses_confirmation=False)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def controls(root, historical):
    records = {}
    for file in sorted((root / "HOLDOUT").glob("*/*/s*/*/TRAIN.json")):
        receipt = load_json(file)
        checkpoint = file.parent / "controller.pt"
        if minimal.sha(checkpoint) != receipt["checkpoint_sha256"]:
            raise RuntimeError(f"Corrupt checkpoint: {checkpoint}")
        name = "/".join(str(file.relative_to(root / "HOLDOUT")).split("/")[:-1])
        records[name] = dict(checkpoint=checkpoint, depths=receipt["trained_depths"],
                             anchor=receipt["anchor_loop"], source=receipt["source"], suite=receipt["suite"],
                             seed=receipt["seed"])
    records["historical_t1"] = dict(checkpoint=historical, depths=[2, 3, 4], anchor=None,
                                    source="arc", suite="state_horizon")
    return records


def native_features(archive, model, tokenizer, source_path, n):
    # Do not expose the discovery labels to the model or fit. The archive requires
    # a legal choice index solely to build metadata, which is then discarded.
    examples = [dict(id=row["id"], question=row["question"], choices=row["choices"],
                     answerKey=row["choices"]["label"][0]) for row in selected_rows(source_path, n, 261123317)]
    _, work = batches(archive, tokenizer, examples, 8)
    receiving, native_shift = {i: [] for i in range(1, 6)}, {i: [] for i in range(1, 6)}
    with torch.no_grad():
        for items in work:
            packed = archive.pack(items)
            _, trace = forward_control(model, packed, 6, anchor_loop=1, capture=tuple(range(1, 7)))
            for anchor in receiving:
                h = trace["pre"][anchor][packed[1], packed[2]].float()
                following = trace["pre"][anchor + 1][packed[1], packed[2]].float()
                receiving[anchor].append(h)
                native_shift[anchor].append(following - h)
    return {i: (torch.cat(receiving[i]), torch.cat(native_shift[i])) for i in receiving}


def moment_files(root, source):
    doc = load_json(root / "moments" / source / "MOMENTS.json")
    tables = {}
    for k, m in MOMENT_CELLS:
        key = cell_key(k, m)
        path = root / "moments" / source / f"{key}.pt"
        if minimal.sha(path) != doc["conditions"][key]["matrix_sha256"]:
            raise RuntimeError(f"Corrupt moment file: {path}")
        tables[(k, m)] = torch.load(path, weights_only=True, map_location="cpu")
    return tables


def feasible(k, m, trained, anchor, suite, tables):
    if suite == "state_horizon":
        valid = list(tables)
    elif suite == "position":
        valid = [(anchor, 4 - anchor)]
    else:
        valid = [(depth - 1, 1) for depth in trained]
    if (k, m) in valid:
        return [(k, m)]
    return valid


def get_weights(k, m, allowed, tables, mean_h):
    raw = []
    for src_k, src_m in allowed:
        h0 = tables[(src_k, src_m)]["mean_state"].to(mean_h)
        cos = float(torch.nn.functional.cosine_similarity(h0, mean_h, dim=0))
        distance = abs(k-src_k) + abs(m-src_m) + 2. * max(0., 1.-cos)
        raw.append(math.exp(-distance))
    return [x / sum(raw) for x in raw]


def predictions_for(model, net, record, cells, target, features, source_tables):
    u, w = net.out.weight[:, 0].detach(), net.encode.weight[0].detach()
    norm_weight = model.model.norm.weight.detach()
    epsilon = model.model.norm.variance_epsilon
    output = {}
    for k, m in cells:
        h, natural = features[k]
        valid = feasible(k, m, record["depths"], record["anchor"], record["suite"], source_tables)
        blend = get_weights(k, m, valid, source_tables, h.mean(0))
        source_slopes, pca, mean_direction = [], [], []
        for neighbor, fraction in zip(valid, blend, strict=True):
            matrix = source_tables[neighbor]
            weighted = matrix["matrix"].to(u) @ w
            source_slopes.append(fraction * float(u @ weighted))
            grad_basis = matrix["gradient_pca"].to(u)
            pca.append(fraction * float((u @ grad_basis).norm() / u.norm().clamp_min(1e-12)))
            gradient = matrix["mean_gradient"].to(u)
            mean_direction.append(fraction * float(gradient @ u))
        unconstrained = (h @ w)[:, None] * u
        deployed = net.delta(h)
        ratio = (deployed.norm() / unconstrained.norm().clamp_min(1e-12)).item()
        avg_slope = sum(source_slopes)
        expected = norm_tangent(h, deployed, norm_weight, epsilon)
        state_score = float((h.mean(0) @ w) / (h.mean(0).norm() * w.norm()).clamp_min(1e-12))
        nat_align = float(torch.nn.functional.cosine_similarity(deployed, natural, dim=-1).mean())
        output[cell_key(k, m)] = dict(
            target=target, anchor=k, remaining=m, total_depth=k+m, source_neighbors=[list(x) for x in valid],
            source_weights=blend, signed_source_moment=avg_slope,
            clipped_proxy=avg_slope * ratio, clip_rms_ratio=ratio,
            state_read_cosine=state_score, unsigned_gradient_pca_alignment=sum(pca),
            unsupervised_native_displacement_cosine=nat_align, deployed_write_rms=float(rms(deployed).mean()),
            linearized_postnorm_rms=float(rms(expected).mean()), signed_mean_gradient=sum(mean_direction)*float((h@w).mean()),
            signed_source_predictions_use_no_target_labels=True, trained_depths=record["depths"],
        )
    return output


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "previous", "model", "historical-rank1", "data-root", "results", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--round2", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    artifact, lock_file = args.out / "PREDICTIONS.json", args.out / "FORECAST_LOCK.json"
    if artifact.exists() or lock_file.exists():
        raise FileExistsError("Forecast already exists. Archive it under a distinct round before a new fit.")
    controllers = controls(args.results, args.historical_rank1)
    for source in ("arc", "csqa"):
        count = sum(entry["source"] == source for entry in controllers.values())
        if count < (9 if source == "arc" else 1):
            raise RuntimeError("Core scratch checkpoints must finish before sealing")
    archive, model, tokenizer, runtime = minimal.runtime(args.archive, args.previous, args.model)
    device = next(model.parameters()).device
    pools = {}
    for target in TARGETS:
        pools[target] = native_features(archive, model, tokenizer, args.data_root / f"{target}_discovery.jsonl", args.n)
    source_domains = (*TARGETS, "obqa", "qasc", "mixed") if args.round2 else TARGETS
    moments = {source: moment_files(args.results, source) for source in source_domains}
    rows = {}
    evaluation = {}
    for name, record in controllers.items():
        if record["source"] not in moments:
            continue
        net = minimal.Minimal("linear", 1, .45).to(device).eval().requires_grad_(False)
        net.load_state_dict(torch.load(record["checkpoint"], weights_only=True, map_location="cpu")["state"])
        if record["suite"] == "state_horizon":
            cells = CELLS
        elif record["suite"] == "position":
            cells = [(k, 4-k) for k in range(1, 4)]
        else:
            cells = [(k, 1) for k in range(1, 6)]
        evaluation[name] = {}
        for target in TARGETS:
            entries = predictions_for(model, net, record, cells, target, pools[target], moments[record["source"]])
            if args.round2:
                target_train_reference = predictions_for(model, net, record, cells, target, pools[target],
                                                         moments[target])
                for key in entries:
                    entries[key]["target_train_moment"] = target_train_reference[key]["signed_source_moment"]
                    entries[key]["target_train_clipped_proxy"] = target_train_reference[key]["clipped_proxy"]
            evaluation[name][target] = entries
        rows[name] = {k: v for k, v in record.items() if k != "checkpoint"}
        rows[name]["checkpoint_sha256"] = minimal.sha(record["checkpoint"])
        print("FORECAST", name, flush=True)
    protocol = dict(
        source_label_access="Only fitted source moments at the training controller's actual depths; K5/K6 absent.",
        target_label_access="None: target discovery questions are encoded using a discarded placeholder choice index.",
        transfer="Use the source moment at the exact admitted condition, else exponential interpolation of admitted "
                 "source moments by anchor/remaining distance and no-label mean state cosine.",
        finite_dose_proxy="Multiply the signed first-order projection by the target no-label clipped/raw RMS ratio; "
                          "ratio is nonnegative and cannot fix sign errors. No learned label calibration.",
        audit_hypotheses=["Signed projection will outperform unsigned gradient PCA and write magnitude for ranking",
                          "The finite-dose law may break sign or ordering even with a positive local projection",
                          "At least one scratch controller shows positive utility at an unseen native depth"],
        target_train_reference="Target-domain labeled Train moment sensitivity, separate from zero-target-label predictions."
                               if args.round2 else None,
        source_moments_sha256={source: minimal.sha(args.results / "moments" / source / "MOMENTS.json")
                               for source in source_domains},
        unsupervised_discovery_sha256={target: minimal.sha(args.data_root / f"{target}_discovery.jsonl") for target in TARGETS},
        source_code_sha256=minimal.sha(Path(__file__)), runtime=runtime,
    )
    minimal.dump(artifact, dict(controllers=rows, predictions=evaluation, protocol=protocol))
    if args.round2:
        minimal.dump(args.out / "SINGLE_CONTROLLER.json", select_single_controller(rows, evaluation))
    data_paths = ({f"{target}_round2_confirm": args.data_root/"optional"/f"{target}_round2_confirm.jsonl"
                   for target in TARGETS} if args.round2 else {x: args.data_root/f"{x}.jsonl" for x in
                   ("arc_confirm", "csqa_confirm", "arc_challenge_validation", "arc_challenge_test", "csqa_validation")})
    minimal.dump(lock_file, dict(controller_sha256s=sorted({r["checkpoint_sha256"] for r in rows.values()}),
                                 evaluation_sha256s={x: minimal.sha(path) for x, path in data_paths.items()},
                                 predictions_sha256=minimal.sha(artifact)))


if __name__ == "__main__":
    main()
