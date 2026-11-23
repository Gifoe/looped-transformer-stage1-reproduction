"""Measure exact grouped functional matrices at frozen native write boundaries."""

import argparse
import json
from pathlib import Path
import time

import torch

import minimal
from advantages import MULTIPLE_CHOICE, choice_batch, forward, token_batch, utilities
from forward_control import forward_control
from task_data import load


DATASETS = ("arc_challenge", "csqa", "squad", "gsm8k")


def allocate(width, device):
    return dict(matrix=torch.zeros((width, width), device=device), gradient=torch.zeros(width, device=device),
                state=torch.zeros(width, device=device), q0=[], positions=0)


def move(model, batch, depth, anchor):
    if anchor == depth - 1:
        return forward(model, batch, depth, leaf=True)
    hidden, trace = forward_control(model, batch.packed, depth, anchor_loop=anchor, leaf=True)
    return model.lm_head(hidden).float(), trace["leaf"]


def batches(rows, size):
    ordered = sorted(rows, key=lambda row: (row["input_tokens"], row["sample_id"]))
    for start in range(0, len(rows), size):
        yield ordered[start:start + size]


def collect(model, archive, tok, rows, depth, anchor, style, size, data_sig):
    device = next(model.parameters()).device
    moments = {}
    start = time.perf_counter()
    for group in batches(rows, size):
        packed = (choice_batch if style == "candidate" else token_batch)(archive, tok, group)
        with torch.enable_grad():
            logits, h = move(model, packed, depth, anchor)
            measures = list(utilities(logits, packed, tok.eos_token_id).items())
            for i, (name, q) in enumerate(measures):
                g, = torch.autograd.grad(q.sum(), h, retain_graph=i + 1 < len(measures))
                state = h.detach().float()
                g = g.detach().float()
                if name not in moments:
                    moments[name] = allocate(model.config.hidden_size, device)
                obj = moments[name]
                obj["matrix"].addmm_(g.T, state)
                obj["gradient"] += g.sum(0)
                obj["state"] += state.sum(0)
                obj["positions"] += len(state)
                obj["q0"].extend(dict(id=row["sample_id"], value=v) for row, v in zip(group, q.detach().tolist(), strict=True))
    result = {}
    for name, obj in moments.items():
        n = len(obj["q0"])
        target = torch.tensor([r["value"] for r in obj["q0"]], dtype=torch.float64)
        matrix = obj["matrix"].cpu() / n
        mu_g = obj["gradient"].cpu() / n
        mu_h = obj["state"].cpu() / obj["positions"]
        outer = mu_g[:, None] * mu_h[None, :]
        residual = matrix - outer
        result[name] = dict(matrix=matrix, mean_state=mu_h, mean_gradient=mu_g, n=n,
                            frobenius=float(matrix.norm()), mean_outer_frobenius=float(outer.norm()),
                            centered_frobenius=float(residual.norm()), mean_centered_dot=float((outer * residual).sum()),
                            sigma=max(.25, float(target.std(unbiased=False))), native_mean=float(target.mean()),
                            q0=obj["q0"], data_sha256=data_sig, depth=depth, anchor=anchor, seconds=time.perf_counter()-start)
    torch.cuda.synchronize()
    return result


def save(output, name, value, model_hash):
    path = output / (name + ".pt")
    temporary = output / (name + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)
    receipt = {key: v for key, v in value.items() if not isinstance(v, torch.Tensor)}
    receipt.update(model_sha256=model_hash, matrix_sha256=minimal.sha(path))
    minimal.dump(output / (name + ".json"), receipt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "previous", "model", "data", "out"):
        p.add_argument(f"--{arg}", type=Path, required=True)
    p.add_argument("--fold", choices=("moment_fit", "moment_holdout"), required=True)
    p.add_argument("--depths", nargs="+", type=int, default=(2, 3, 4))
    p.add_argument("--earlier-write", action="store_true")
    p.add_argument("--candidate-only", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch", type=int, default=4)
    a = p.parse_args()
    old, model, tok, runtime = minimal.runtime(a.archive, a.previous, a.model)
    a.out.mkdir(parents=True, exist_ok=True)
    lock = json.loads((a.data / "DATA_LOCK.json").read_text())
    for ds in DATASETS:
        if a.candidate_only and ds not in MULTIPLE_CHOICE:
            continue
        path = a.data / f"{ds}_{a.fold}.jsonl"
        fingerprint = minimal.sha(path)
        if fingerprint != lock["files"][path.name]["sha256"]:
            raise ValueError(f"Changed frozen data {path}")
        rows = load(path)
        if a.limit is not None:
            rows = rows[:a.limit]
        styles = ("candidate",) if a.candidate_only else (("candidate", "generation") if ds in MULTIPLE_CHOICE else ("generation",))
        for style in styles:
            for depth in a.depths:
                anchor = depth - (2 if a.earlier_write else 1)
                if anchor < 1:
                    continue
                result = collect(model, old, tok, rows, depth, anchor, style, a.batch, fingerprint)
                for key, value in result.items():
                    name = key.replace("/", "__") + f"__k{depth}__h{anchor}"
                    save(a.out, name, value, runtime["model_sha256"])
                    print(name, value["n"], round(value["frobenius"], 4), round(value["sigma"], 4), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
