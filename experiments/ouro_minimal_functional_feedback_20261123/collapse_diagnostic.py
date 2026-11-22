"""Validate a Train-derived one-direction approximation without changing the locked arm."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize

from benchmark import holm, paired
from minimal import DEPTHS, WIDTH, Minimal, dump, load_jsonl, load_net, runtime, save_net, sha
from train import evaluate


def predict(rows, depth, metric):
    result = []
    for r in rows:
        scores = np.asarray(r["scores"][f"k{depth}"])
        if metric == "acc_norm":
            scores = scores / r["characters"]
        result.append(int(scores.argmax()) == r["gold"])
    return np.asarray(result)


def verify(a, b):
    for x, y in zip(a, b, strict=True):
        if any(x[k] != y[k] for k in ("id", "labels", "gold", "characters", "answer_tokens")):
            raise ValueError("Archived Validation protocol differs")


@torch.no_grad()
def run(args):
    root = args.out
    cfg = json.loads((root / "FREEZE.json").read_text(encoding="utf-8"))
    choice = cfg["best"]
    arm = cfg["arms"][choice]
    checkpoint = root / arm["file"]
    assert sha(checkpoint) == arm["sha256"]
    reference = root / f"VAL_{choice}.jsonl"
    archived = load_jsonl(reference)
    old, model, tok, metadata = runtime(args.archive, args.previous, args.model)
    validation = old.read_data(args.corpus, "validation")
    net = load_net(checkpoint)
    assert net.kind == "linear"
    source = root / "SUBSPACE_ANALYSIS"
    basis_lock = sha(source / "ARC_GRADIENT_LOCK.json")
    replay = {}
    checkpoint_inputs = {}
    for k in DEPTHS:
        path = source / f"train_k{k}.pt"
        obj = torch.load(path, map_location="cpu", weights_only=True)
        assert obj["cohort_lock_sha256"] == basis_lock
        raw = obj["values"]["hidden"].to(device="cuda", dtype=torch.float32)
        replay[k] = normalize(net.out(net.encode(raw)))
        checkpoint_inputs[path.name] = sha(path)
    w = normalize(torch.cat(list(replay.values())).mean(0), dim=0)
    rank1 = Minimal("linear", 1, net.bound).cuda().eval()
    rank1.encode.weight[0].copy_((w @ net.out.weight) @ net.encode.weight)
    rank1.out.weight[:, 0].copy_(w)
    cap = Minimal("constant", 1, net.bound).cuda().eval()
    cap.vector.copy_(w * (net.bound * WIDTH**.5))
    controls = {"mean_direction_rank1": rank1, "mean_direction_at_rms_cap": cap}
    dest = root / "CAUSAL_ANALYSIS"
    statistics = {}
    for name, control in controls.items():
        rows, timings = evaluate(old, model, tok, validation, control, arm["dose"])
        verify(archived, rows)
        metrics = {}
        for metric in ("acc", "acc_norm"):
            cells = []
            for k in DEPTHS:
                a, b = (predict(data, k, metric) for data in (archived, rows))
                cells.append(dict(k=k, correct=int(b.sum()), reference=int(a.sum()), **paired(a, b)))
            metrics[metric] = cells
        holm([cell for part in metrics.values() for cell in part])
        output = dest / f"VAL_{name}.jsonl"
        weight = dest / f"{name}.pt"
        old.write_predictions(output, rows)
        save_net(weight, control, dict(parameters=sum(p.numel() for p in control.parameters()),
                                      source_checkpoint=arm["sha256"], train_capture_hashes=checkpoint_inputs))
        statistics[name] = dict(metrics=metrics, timing=timings, params=sum(p.numel() for p in control.parameters()),
                                score_sha256=sha(output), checkpoint_sha256=sha(weight))
        print(name, {m: [x["correct"] for x in cells] for m, cells in metrics.items()}, flush=True)
    dump(dest / "VALIDATION_COMPRESSION_DIAGNOSTIC.json", dict(
        selected_method=choice, selected_checkpoint_sha256=arm["sha256"], n=len(validation),
        train_source_sha256=checkpoint_inputs, validation_reference_sha256=sha(reference), model=metadata,
        definition="Direction is the normalized mean of the frozen module's unit edits at train prompts over K1–K4. "
                   "Rank-one uses the original operator's projection onto that single direction. "
                   "The fixed-dose arm follows the same direction at the original RMS cap. "
                   "Both are Validation-only diagnostics outside the confirmed model selection.",
        split="arc_validation", mean_cosine_vs_shared_direction={str(k): float((x @ w).mean()) for k, x in replay.items()},
        comparisons=statistics,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "archive", "previous", "corpus", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    run(parser.parse_args())
