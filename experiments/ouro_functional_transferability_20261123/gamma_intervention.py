"""Test a shared, train-only spectral write at both signs and two finite doses."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import minimal
from forward_control import forward_control
from functional_operator import batches
from normalized_geometry import rms
from state_horizon import score_arrays


DOSES = {"positive": 1., "negative": -1., "strong_positive": 3.}


def bounded(h, net, dose):
    change = dose * net.out(net.encode(h))
    return change * (.45 * minimal.rms(h) / minimal.rms(change)).clamp_max(1.)


def prepare(root, data, out):
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "PROBE_LOCK.json"
    if lock.exists():
        raise FileExistsError(lock)
    source = root / "SPECTRUM/TRAIN_ONLY_GAMMA_DIRECTIONS.pt"
    gamma = torch.load(source, weights_only=True)["joint"]["relative"]
    u, w = gamma["u"].float(), gamma["w"].float()
    u = u / rms(u)
    observed = []
    cells = {}
    for domain in ("arc", "csqa"):
        cells[domain] = {}
        for k in (1, 2, 3):
            moment = torch.load(root / "moments" / domain / f"h{k}_m1.pt", weights_only=True)
            eig = moment["state_eigenvalues"].clamp_min(0)
            approx = torch.sqrt(len(w)*torch.sum(eig * torch.square(w @ moment["state_pca"])) / eig.sum())
            observed.append(float(approx))
            cells[domain][str(k+1)] = float(u @ (moment["matrix"] @ w))
    target = .08
    scale = target / max(float(np.median(observed)), .01)
    net = minimal.Minimal("linear", 1, .45)
    with torch.no_grad():
        net.out.weight[:, 0].copy_(u)
        net.encode.weight[0].copy_(w * scale)
    checkpoint = out / "GAMMA.pt"
    minimal.save_net(checkpoint, net, dict(form="shared_relative_gamma", source_sha256=minimal.sha(source)))
    datasets = {p.stem: minimal.sha(p) for p in data}
    minimal.dump(lock, dict(controller_sha256=minimal.sha(checkpoint), directions_sha256=minimal.sha(source),
                            code_sha256=minimal.sha(Path(__file__)), reference_label_access="Train only",
                            data_sha256=datasets, requested_small_rms_reference=target, scale=scale,
                            train_quantities=observed, doses=DOSES, train_projection=cells,
                            forecasts={"k2_to_k6_positive": 1, "k2_to_k6_negative": -1,
                                       "strong_positive": "Finite-dose stress. Sign and size cannot be assumed.",
                                       "k1": "Physical-layer geometry is distinct. No precommitted sign."}))
    print("LOCKED", minimal.sha(lock), "read scale", scale, flush=True)


@torch.no_grad()
def score(archive, model, tokenizer, net, path, depth, fresh):
    rows, work = batches(archive, tokenizer, minimal.load_jsonl(path), 12)
    score_doses = {"native": 0., **DOSES}
    start = time.monotonic()
    calls, loop_tokens = 0, 0
    for items in work:
        packed = archive.pack(items)
        for label, dose in score_doses.items():
            options = {}
            if label != "native":
                options["edit"] = lambda h: bounded(h, net, dose)
            score_hidden, trace = forward_control(model, packed, depth, **options)
            candidate = archive.scores(model.lm_head(score_hidden).float(), packed[1].cpu().numpy(), packed[-1], len(items))
            archive.record(rows, items, {label: candidate})
            if label != "native":
                factor = (rms(trace["delta"]) / rms(trace["anchor"]).clamp_min(1e-6)).flatten()
                for (_, count, q, a), value in zip(items, factor.split([item[1] for item in items]), strict=True):
                    bucket = rows[q].setdefault("actual_doses", {}).setdefault(label, [None] * len(rows[q]["labels"]))
                    bucket[a] = float(value.mean())
            calls += 1
            loop_tokens += packed[0].numel() * depth
    out = {}
    rng = np.random.default_rng(261123572)
    draw = rng.integers(len(rows), size=(4000, len(rows)))
    for label in DOSES:
        compare = [[score_arrays(row["scores"][role], row["gold"], row["characters"])
                    for row in rows] for role in ("native", label)]
        out[label] = {}
        for name in ("acc", "acc_norm", "utility", "margin"):
            a, b = (np.array([entry[name] for entry in r]) for r in compare)
            change = b.astype(float) - a.astype(float)
            out[label][name] = dict(average=float(change.mean()), ci95=np.quantile(change[draw].mean(1), [.025,.975]).tolist(),
                                    native=float(a.mean()), treatment=float(b.mean()),
                                    rescue=int((change>0).sum()), damage=int((change<0).sum()))
    out_dir = fresh / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"K{depth}.jsonl"
    archive.write_predictions(raw, rows)
    summary = dict(results=out, n=len(rows), model_calls=calls, padded_token_loops=loop_tokens,
                   wall_seconds=time.monotonic()-start, sha256=minimal.sha(raw))
    print(path.stem, depth, {name: x["utility"]["average"] for name,x in out.items()}, flush=True)
    minimal.dump(raw.with_suffix(".json"), summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "score"))
    for name in ("results", "out"):
        p.add_argument(f"--{name}", required=True, type=Path)
    for name in ("archive", "previous", "model"):
        p.add_argument(f"--{name}", type=Path)
    p.add_argument("--data", type=Path, action="append", required=True)
    args = p.parse_args()
    if args.phase == "freeze":
        prepare(args.results, args.data, args.out)
        return
    lock_file = args.out/"PROBE_LOCK.json"
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    ckpt = args.out/"GAMMA.pt"
    if (lock["controller_sha256"] != minimal.sha(ckpt) or
        lock["data_sha256"] != {x.stem:minimal.sha(x) for x in args.data} or
        lock["code_sha256"] != minimal.sha(Path(__file__))):
        raise ValueError("The finite-dose probe inputs changed after locking")
    archive, model, tokenizer, runtime = minimal.runtime(args.archive, args.previous, args.model)
    net = minimal.load_net(ckpt).requires_grad_(False)
    results = {x.stem: {f"K{k}": score(archive, model, tokenizer, net, x, k, args.out)
                        for k in range(1, 7)} for x in args.data}
    minimal.dump(args.out/"SUMMARY.json", dict(results=results, lock_sha256=minimal.sha(lock_file), runtime=runtime))


if __name__ == "__main__":
    main()
