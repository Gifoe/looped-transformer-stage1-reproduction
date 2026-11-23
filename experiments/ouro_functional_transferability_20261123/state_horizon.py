"""Cross a natural recipient boundary with independent durations of frozen suffix."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import minimal
from forward_control import forward_control, scope_indices
from functional_operator import CELLS, MOMENT_CELLS, batches, cell_key, selected_rows
from normalized_geometry import cosine, geometry_rows, radial_tangent, rms


def unit_rms(x):
    return x / rms(x).clamp_min(1e-12)


def resolve_moment(k, m):
    return min(MOMENT_CELLS, key=lambda c: (abs(c[1] - m) + abs(c[0] - k), abs(c[1] - m), c))


def per_question(items):
    current = None
    offset = 0
    for _, count, qid, _ in items:
        if current != qid:
            if current is not None:
                yield current, slice(first, offset)
            current, first = qid, offset
        offset += count
    if current is not None:
        yield current, slice(first, offset)


@torch.no_grad()
def run_cell(archive, model, net, info, work, moment, k, m, seed):
    device = next(model.parameters()).device
    records = {i: dict(row, scores={}, geometry={}) for i, row in enumerate(info)}
    prefix_width = model.config.hidden_size
    rng = torch.Generator(device=device).manual_seed(seed + k * 10 + m)
    random = unit_rms(torch.randn(prefix_width, generator=rng, device=device))
    functional_axis = unit_rms(moment["u"][:, 0].to(device))
    functional_read = moment["w"][:, 0].to(device)
    loop = k + m
    calls = 0

    def write_scores(arm, hidden, packed, items):
        _, groups, _, targets = packed
        values = archive.scores(model.lm_head(hidden).float(), groups.cpu().numpy(), targets, len(items))
        for (_, _, q, a), value in zip(items, values, strict=True):
            scores = records[q]["scores"].setdefault(arm, [None] * len(info[q]["labels"]))
            scores[a] = float(value)

    for items in work:
        packed = archive.pack(items)
        groups, at = scope_indices(packed, "predictors")
        _, ordinary = forward_control(model, packed, max(loop, k + 1), anchor_loop=k,
                                      capture=tuple({k, k + 1, loop}))
        native = ordinary["post"][loop][groups, at]
        write_scores("native", native, packed, items)
        h = ordinary["anchor"]
        law = net.delta(h)
        radial, tangent = radial_tangent(h, law)
        scale = rms(law)
        read_sign = torch.where((h @ functional_read) >= 0, 1., -1.)[:, None]
        natural = ordinary["pre"][k + 1][groups, at].float() - h
        delta_norm = (rms(h) / rms(ordinary["post"][k][groups, at]).clamp_min(1e-6)).clamp_min(1e-6)
        arms = {
            "learned": (law, False),
            "learned_small": (.10 * law, False),
            "natural_matched": (unit_rms(natural) * scale, False),
            "functional_matched": (functional_axis * read_sign * scale, False),
            "random_matched": (random * scale, False),
            "radial_component": (radial, False),
            "tangent_component": (tangent, False),
            "postnorm_same_relative_rms": (law / delta_norm, True),
        }
        calls += 1
        for arm, (write, post) in arms.items():
            edited, trace = forward_control(model, packed, loop, anchor_loop=k, after_norm=post, edit=lambda _: write,
                                            capture=(k, loop))
            expected_anchor = ordinary["post"][k][groups, at].float() if post else h
            if not torch.equal(trace["anchor"], expected_anchor):
                raise RuntimeError(f"Recipient state changed at {k=} {m=} {arm}")
            write_scores(arm, edited, packed, items)
            named = dict(relative=(rms(trace["delta"]) / rms(expected_anchor).clamp_min(1e-6)).flatten(),
                         measured_rms=rms(trace["delta"]).flatten(), requested_cosine=cosine(trace["delta"], write),
                         local_norm_delta_rms=rms(trace["post"][k][groups, at].float() -
                                                  ordinary["post"][k][groups, at].float()).flatten(),
                         terminal_delta_rms=rms(edited.float() - native.float()).flatten())
            if arm == "learned":
                named.update(geometry_rows(h, trace["delta"], ordinary["post"][k][groups, at],
                                           trace["post"][k][groups, at], model.model.norm))
            for q, span in per_question(items):
                records[q]["geometry"][arm] = {name: float(values[span].mean()) for name, values in named.items()}
            calls += 1
    return list(records.values()), calls


def score_arrays(values, gold, characters):
    raw = np.asarray(values, dtype=np.float64)
    z = raw / np.asarray(characters, dtype=np.float64)
    wrong = np.delete(z, gold)
    return dict(acc=bool(np.argmax(raw) == gold), acc_norm=bool(np.argmax(z) == gold),
                utility=float(z[gold] - np.logaddexp.reduce(wrong)), margin=float(z[gold] - wrong.max()))


def summarize(rows, seed):
    arms = tuple(rows[0]["scores"])
    values = {arm: [score_arrays(row["scores"][arm], row["gold"], row["characters"]) for row in rows] for arm in arms}
    native = values["native"]
    rng = np.random.default_rng(seed)
    resample = rng.integers(len(rows), size=(4000, len(rows)))
    out = {}
    for arm, score in values.items():
        result = {}
        for name in ("acc", "acc_norm", "utility", "margin"):
            after, before = np.array([s[name] for s in score]), np.array([b[name] for b in native])
            delta = after.astype(float) - before.astype(float)
            stat = dict(mean=float(after.mean()), native=float(before.mean()), delta=float(delta.mean()),
                        ci95=np.quantile(delta[resample].mean(1), [.025, .975]).tolist())
            if name in ("acc", "acc_norm"):
                stat.update(correct=int(after.sum()), rescued=int((delta > 0).sum()), damaged=int((delta < 0).sum()))
            result[name] = stat
        if arm != "native":
            fields = rows[0]["geometry"][arm]
            result["geometry_medians"] = {name: float(np.median([row["geometry"][arm][name] for row in rows]))
                                          for name in fields}
        out[arm] = result
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "previous", "model", "rank1", "data", "moment-root", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--seed", type=int, default=261123115)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--forecast-lock", type=Path)
    parser.add_argument("--evaluation-name")
    args = parser.parse_args()
    if "confirm" in args.data.name.lower():
        if args.forecast_lock is None or not args.evaluation_name:
            parser.error("Independent confirmation requires the stored forecast lock")
        receipt = json.loads(args.forecast_lock.read_text(encoding="utf-8"))
        if receipt["evaluation_sha256s"].get(args.evaluation_name) != minimal.sha(args.data):
            raise ValueError("Confirmation data differs from the forecast lock")
    args.out.mkdir(exist_ok=True, parents=True)
    archive, model, tokenizer, runtime = minimal.runtime(args.archive, args.previous, args.model)
    payload = torch.load(args.rank1, map_location="cpu", weights_only=True)
    net = minimal.Minimal("linear", 1, .45).to(next(model.parameters()).device).eval().requires_grad_(False)
    net.load_state_dict(payload["state"])
    rows = selected_rows(args.data, args.n, args.seed)
    info, work = batches(archive, tokenizer, rows, args.batch)
    manifest = dict(ids=[x["id"] for x in rows], data_sha256=minimal.sha(args.data), runtime=runtime,
                    controller_sha256=minimal.sha(args.rank1), n=args.n, seed=args.seed, batch=args.batch,
                    forecast_sha256=minimal.sha(args.forecast_lock) if args.forecast_lock else None)
    manifest = json.loads(json.dumps(manifest))
    lock_path = args.out / "INPUT_LOCK.json"
    if lock_path.is_file() and json.loads(lock_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("State × horizon work directory contains a different run")
    minimal.dump(lock_path, manifest)
    summary = {}
    for k, m in CELLS:
        key = cell_key(k, m)
        moment_key = cell_key(*resolve_moment(k, m))
        moment_path = args.moment_root / f"{moment_key}.pt"
        detail = args.out / f"{key}.json"
        raw_path = args.out / f"{key}.jsonl"
        if detail.exists():
            row = json.loads(detail.read_text(encoding="utf-8"))
            if row["moment_sha256"] != minimal.sha(moment_path) or row["rows_sha256"] != minimal.sha(raw_path):
                raise ValueError(f"Hash mismatch at {detail}")
        else:
            start = time.monotonic()
            moments = torch.load(moment_path, map_location="cpu", weights_only=True)
            predictions, calls = run_cell(archive, model, net, info, work, moments, k, m, args.seed)
            temp = raw_path.with_suffix(".partial")
            archive.write_predictions(temp, predictions)
            temp.replace(raw_path)
            row = dict(result=summarize(predictions, args.seed), measured_seconds=time.monotonic()-start,
                       n=len(predictions), calls=calls, sampled_moment=moment_key,
                       moment_sha256=minimal.sha(moment_path), rows_sha256=minimal.sha(raw_path))
            minimal.dump(detail, row)
        summary[key] = row
        print(key, row["result"]["learned"]["utility"]["delta"], row["measured_seconds"], flush=True)
    minimal.dump(args.out / "STATE_HORIZON.json", dict(manifest=manifest, conditions=summary))


if __name__ == "__main__":
    main()
