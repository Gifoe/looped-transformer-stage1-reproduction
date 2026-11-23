"""Test the signed first-order forecast against real controlled forward passes."""

import argparse
import json
import math
from pathlib import Path
import time

import torch

import minimal
from advantages import MULTIPLE_CHOICE, choice_batch, token_batch, utilities
from collect_moments import DATASETS, batches
from forward_control import forward_control
from task_data import load, save


DOSES = (.10, .50, 1.)


def axes_controller(pair, amplitude):
    net = minimal.Minimal("linear", 1, .45).cuda().eval()
    with torch.no_grad():
        net.out.weight.copy_(pair["u"][:, None] * math.sqrt(amplitude))
        net.encode.weight.copy_(pair["w"][None, :] * math.sqrt(amplitude))
    return net.requires_grad_(False)


def question_indices(packet):
    candidate = packet.packed[1]
    if packet.metadata is None:
        return candidate
    counts = torch.tensor([len(row["labels"]) for row in packet.metadata], device=candidate.device)
    return torch.repeat_interleave(torch.arange(len(counts), device=candidate.device), counts)[candidate]


def sum_question(values, index, n):
    return values.new_zeros(n).index_add_(0, index, values)


def measurements(archive, model, tok, rows, mode, depth, boundary, arms, batch):
    for block in batches(rows, batch):
        packet = (choice_batch if mode == "candidate" else token_batch)(archive, tok, block)
        idx = question_indices(packet)
        count = sum_question(torch.ones_like(idx, dtype=torch.float32), idx, len(block))
        with torch.enable_grad():
            z, trace = forward_control(model, packet.packed, depth, anchor_loop=boundary, leaf=True)
            h = trace["leaf"]
            q0 = utilities(model.lm_head(z).float(), packet, tok.eos_token_id)
            derivative = {}
            for i, (key, value) in enumerate(q0.items()):
                g, = torch.autograd.grad(value.sum(), h, retain_graph=i + 1 < len(q0))
                derivative[key] = g.detach()
        with torch.no_grad():
            h = h.detach()
            q0 = {k: q.detach() for k, q in q0.items()}
            scale = h.norm(dim=-1).clamp_min(1e-6)
            for arm, method in arms.items():
                write = method.delta(h)
                slope = {name: sum_question((g * write).sum(-1), idx, len(block)) for name, g in derivative.items()}
                for dose in DOSES:
                    z_edit, record = forward_control(model, packet.packed, depth, anchor_loop=boundary,
                                                    controller=method, dose=dose)
                    q1 = utilities(model.lm_head(z_edit).float(), packet, tok.eos_token_id)
                    ratio = sum_question(record["delta"].norm(dim=-1) / scale, idx, len(block)) / count
                    for name, result in q1.items():
                        predicted = (slope[name] * dose).tolist()
                        natural = q0[name].tolist()
                        for row, start, end, forecast, physical in zip(block, natural, result.tolist(), predicted,
                                                                    ratio.tolist(), strict=True):
                            yield dict(id=row["sample_id"], group=name, depth=depth, boundary=boundary, arm=arm,
                                       dose=dose, q_native=start, q_edited=end, linear=forecast, write_ratio=physical)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "previous", "model", "data", "shared", "candidate", "rank_only", "retention", "multitask", "out"):
        p.add_argument(f"--{arg}", type=Path, required=True)
    p.add_argument("--depths", nargs="+", type=int, default=(2, 3, 4, 5, 6))
    p.add_argument("--earlier-write", action="store_true")
    p.add_argument("--batch", type=int, default=4)
    a = p.parse_args()
    old, model, tok, runtime = minimal.runtime(a.archive, a.previous, a.model)
    history = {name: minimal.load_net(getattr(a, name)).requires_grad_(False) for name in ("rank_only", "retention", "multitask")}
    ruler = history["rank_only"]
    amplitude = float((ruler.encode.weight.norm() * ruler.out.weight.norm()).detach())
    shared = torch.load(a.shared, map_location="cpu", weights_only=True)
    candidates = torch.load(a.candidate, map_location="cpu", weights_only=True)
    controllers = {**history, **{name: axes_controller(pair, amplitude) for name, pair in shared.items()},
                   "candidate_direction": axes_controller(candidates["worst_group"], amplitude)}
    a.out.mkdir(parents=True, exist_ok=True)
    lock = json.loads((a.data / "DATA_LOCK.json").read_text())
    for depth in a.depths:
        boundary = depth - (2 if a.earlier_write else 1)
        if boundary < 1:
            continue
        for dataset in DATASETS:
            path = a.data / f"{dataset}_moment_holdout.jsonl"
            if minimal.sha(path) != lock["files"][path.name]["sha256"]:
                raise RuntimeError("Changed moment holdout data")
            rows = load(path)
            for mode in (("candidate", "text") if dataset in MULTIPLE_CHOICE else ("text",)):
                out = a.out / f"{dataset}__{mode}__k{depth}__h{boundary}.jsonl"
                start = time.perf_counter()
                records = list(measurements(old, model, tok, rows, mode, depth, boundary, controllers, a.batch))
                save(out, records)
                minimal.dump(out.with_suffix(".json"), dict(n=len(records), seconds=time.perf_counter()-start,
                    rows_sha256=minimal.sha(out), model_sha256=runtime["model_sha256"],
                    common_operator_norm=amplitude, dataset_sha256=minimal.sha(path), doses=DOSES))
                print(out.stem, len(records), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
