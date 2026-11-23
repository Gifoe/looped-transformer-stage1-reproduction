"""Fit task-conditioned moments from authorized training labels only."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import minimal
from forward_control import forward_control
from native_to_rank1 import gradient_utility
from normalized_geometry import cosine, rms


# Leave both whole native depths K5 and K6 out of every moment fit.
MOMENT_CELLS = tuple((k, m) for k in (1, 2, 3) for m in range(4) if k + m <= 4)
CELLS = tuple((k, m) for k in (1, 2, 3) for m in range(4))


def cell_key(k, m):
    return f"h{k}_m{m}"


def batches(archive, tokenizer, rows, batch):
    items, info = archive.encode(tokenizer, rows)
    groups = defaultdict(list)
    for item in items:
        groups[item[2]].append(item)
    order = sorted(groups, key=lambda n: (max(len(item[0]) for item in groups[n]), info[n]["id"]))
    return info, [[item for n in order[i:i + batch] for item in groups[n]]
                  for i in range(0, len(order), batch)]


def selected_rows(path, n, seed):
    rows = minimal.load_jsonl(path)
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids) or len(rows) < n:
        raise ValueError("Expected sufficient unique questions")
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}/{row['id']}".encode()).digest())[:n]


def atomic_pt(path, payload):
    temp = path.with_suffix(".partial")
    torch.save(payload, temp)
    temp.replace(path)


def spectral(matrix, seed, rank):
    with torch.random.fork_rng(devices=[matrix.device.index] if matrix.is_cuda else []):
        torch.manual_seed(seed)
        u, s, v = torch.svd_lowrank(matrix, q=min(*matrix.shape, rank + 12), niter=6)
    return u[:, :rank].cpu().contiguous(), s[:rank].cpu().contiguous(), v[:, :rank].cpu().contiguous()


def summarize_moment(M, g, h, gh, seed, n):
    U, S, V = spectral(M, seed, 12)
    grad_U, grad_S, _ = spectral(g, seed + 1, 8)
    state_U, state_S, _ = spectral(h, seed + 2, 8)
    frob = float(M.norm())
    return dict(u=U, s=S, w=V, gradient_pca=grad_U, gradient_eigenvalues=grad_S,
                state_pca=state_U, state_eigenvalues=state_S, matrix=M.cpu().contiguous(),
                mean_state=gh["state"].cpu() / gh["token_count"],
                mean_gradient=gh["gradient"].cpu() / n,
                stable_rank=frob ** 2 / max(float(S[0].square()), 1e-20),
                frobenius=frob, token_count=gh["token_count"], n=n)


def fit_moment(model, archive, info, work, k, m, seed):
    device = next(model.parameters()).device
    width = model.config.hidden_size
    M, G, H = [torch.zeros(width, width, device=device) for _ in range(3)]
    totals = {"state": torch.zeros(width, device=device), "gradient": torch.zeros(width, device=device), "token_count": 0}
    observations = []
    for items in work:
        packed = archive.pack(items)
        with torch.enable_grad():
            hidden, trace = forward_control(model, packed, k + m, anchor_loop=k, leaf=True)
            q, counts = gradient_utility(model, hidden, packed, items, info)
            derivative, = torch.autograd.grad(q.sum(), trace["leaf"])
        h, g = trace["anchor"].float(), derivative.float().detach()
        M.addmm_(g.T, h)
        G.addmm_(g.T, g)
        H.addmm_(h.T, h)
        totals["token_count"] += len(h)
        totals["state"] += h.sum(0)
        totals["gradient"] += g.sum(0)
        row_span = np.cumsum([0, *(item[1] for item in items)])
        offset = 0
        for group_size, value in zip(counts, q.detach().tolist(), strict=True):
            qid = info[items[offset][2]]["id"]
            left, right = row_span[offset], row_span[offset + group_size]
            observations.append(dict(id=qid, q=value, scored_positions=int(right-left),
                                     state_rms=float(rms(h[left:right]).mean()),
                                     gradient_rms=float(rms(g[left:right]).mean()),
                                     gradient_state_cosine=float(cosine(g[left:right], h[left:right]).mean())))
            offset += group_size
    n = len(info)
    return summarize_moment(M / n, G / n, H / n, totals, seed, n), observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "previous", "model", "train-data", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=26112343)
    args = parser.parse_args()
    rows = selected_rows(args.train_data, args.n, args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    lock = dict(input_sha256=minimal.sha(args.train_data), ids=[row["id"] for row in rows],
                cells=[list(cell) for cell in MOMENT_CELLS], n=args.n, seed=args.seed, batch=args.batch,
                code_sha256={name: minimal.sha(Path(__file__).with_name(name)) for name in
                             ("functional_operator.py", "forward_control.py", "normalized_geometry.py")})
    lock_path = out / "INPUT_LOCK.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != lock:
        raise RuntimeError("Existing moment fit does not match input lock")
    minimal.dump(lock_path, lock)
    archive, model, tokenizer, runtime = minimal.runtime(args.archive, args.previous, args.model)
    info, work = batches(archive, tokenizer, rows, args.batch)
    index = {}
    for k, m in MOMENT_CELLS:
        key = cell_key(k, m)
        payload, detail = out / f"{key}.pt", out / f"{key}.json"
        if detail.is_file():
            receipt = json.loads(detail.read_text(encoding="utf-8"))
            if receipt["matrix_sha256"] != minimal.sha(payload):
                raise RuntimeError(f"Invalid matrix receipt {detail}")
        else:
            start = time.monotonic()
            moment, trace = fit_moment(model, archive, info, work, k, m, args.seed)
            atomic_pt(payload, moment)
            receipt = dict(runtime_seconds=time.monotonic() - start, condition=key, n=args.n,
                           stable_rank=moment["stable_rank"], singular_values=moment["s"].tolist(),
                           matrix_sha256=minimal.sha(payload), observations=trace)
            minimal.dump(detail, receipt)
        index[key] = {k: v for k, v in receipt.items() if k != "observations"}
        print(key, receipt["runtime_seconds"], flush=True)
    minimal.dump(out / "MOMENTS.json", dict(protocol=lock, runtime=runtime, conditions=index))


if __name__ == "__main__":
    main()
