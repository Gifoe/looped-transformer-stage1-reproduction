"""Estimate and bound the normalized worst-group value of a shared rank-one direction."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def scores(matrices, u, w):
    return torch.einsum("s i,g i j,s j->s g", u, matrices, w)


def projected_problem(matrices, rank):
    rank = min(rank, *matrices.shape[-2:])
    left_axes, right_axes = [], []
    for matrix in matrices:
        u, _, w = torch.svd_lowrank(matrix, q=rank, niter=5)
        left_axes.append(u)
        right_axes.append(w)
    left = torch.linalg.qr(torch.cat(left_axes, dim=1), mode="reduced").Q
    right = torch.linalg.qr(torch.cat(right_axes, dim=1), mode="reduced").Q
    compressed = left.T @ matrices @ right
    # Use explicit full-space residuals to keep small reconstruction errors from
    # being lost to subtraction of the two squared Frobenius norms.
    residual = torch.stack([(full - left @ reduced @ right.T).norm()
                            for full, reduced in zip(matrices, compressed, strict=True)])
    return left, right, compressed, residual


def best_direction(compressed, seed, starts=16, steps=500):
    u0, _, vh0 = torch.linalg.svd(compressed.mean(0), full_matrices=False)
    rng = torch.Generator(device=compressed.device).manual_seed(seed)
    u = torch.randn((starts, compressed.shape[1]), dtype=compressed.dtype, device=compressed.device, generator=rng)
    w = torch.randn((starts, compressed.shape[2]), dtype=compressed.dtype, device=compressed.device, generator=rng)
    u[0], w[0] = u0[:, 0], vh0[0]
    u, w = F.normalize(u).requires_grad_(), F.normalize(w).requires_grad_()
    optimizer = torch.optim.Adam((u, w), lr=.035)
    best, best_pair = float("-inf"), None
    norm = float(compressed.flatten(1).norm(dim=-1).median())
    for step in range(steps + 1):
        values = scores(compressed, F.normalize(u), F.normalize(w))
        minimum, start = values.detach().min(1).values.max(0)
        if float(minimum) > best:
            best = float(minimum)
            best_pair = (F.normalize(u[start], dim=0).detach().clone(), F.normalize(w[start], dim=0).detach().clone())
        temperature = max(1e-8, norm * (.09 * (.04 ** (step / max(1, steps)))))
        objective = -temperature * torch.logsumexp(-values / temperature, dim=1)
        optimizer.zero_grad(set_to_none=True)
        (-objective.sum()).backward()
        optimizer.step()
    return best_pair, (u0[:, 0], vh0[0])


def certificate(compressed, residual, steps=200):
    """Select a simplex mixture, then return a full-space spectral upper bound."""
    weights = torch.zeros(len(compressed), device=compressed.device, dtype=compressed.dtype, requires_grad=True)
    optimizer = torch.optim.Adam((weights,), lr=.08)
    best = (float("inf"), None)
    for _ in range(steps + 1):
        distribution = weights.softmax(0)
        mixture = torch.einsum("g,gij->ij", distribution, compressed)
        bound = torch.linalg.svdvals(mixture)[0] + distribution @ residual
        if float(bound.detach()) < best[0]:
            best = (float(bound.detach()), distribution.detach().clone())
        optimizer.zero_grad(set_to_none=True)
        bound.backward()
        optimizer.step()
    return best


def analyze(data, rank, seed, starts, steps):
    matrices = torch.stack([item["matrix"] / item["sigma"] for item in data.values()]).to(torch.float64)
    # Factorization treats the fitting groups symmetrically before testing them on
    # a separate cohort. The certificate is only for the finite fitted matrices.
    torch.manual_seed(seed)
    left, right, compressed, residual = projected_problem(matrices, rank)
    (u, w), (mean_u, mean_w) = best_direction(compressed, seed, starts, steps)
    upper, mixture = certificate(compressed, residual)
    axes = {"worst_group": (left @ u, right @ w), "mean_group": (left @ mean_u, right @ mean_w)}
    matrices_flat = F.normalize(matrices.flatten(1), dim=-1)
    report = dict(groups=list(data), upper_bound=upper, dual_mixture=mixture.tolist(),
                  group_matrix_cosine=(matrices_flat @ matrices_flat.T).tolist(), controls={})
    for name, (a, b) in axes.items():
        values = scores(matrices, a[None], b[None]).flatten()
        report["controls"][name] = dict(per_group=values.tolist(), minimum=float(values.min()), mean=float(values.mean()))
    if report["controls"]["worst_group"]["minimum"] > upper + 1e-8:
        raise RuntimeError("Inconsistent feasible value and spectral bound")
    return report, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subspace-rank", type=int, default=6)
    parser.add_argument("--seed", type=int, default=261124173)
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    a = parser.parse_args()
    data = ({path.stem: torch.load(path, map_location="cpu", weights_only=True)
             for path in sorted(a.fit.glob("*.pt"))} if a.fit.is_dir()
            else torch.load(a.fit, map_location="cpu", weights_only=True))
    report, axes = analyze(data, a.subspace_rank, a.seed, a.starts, a.steps)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "SHARED_DIRECTION.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({name: {"u": u.float(), "w": w.float()} for name, (u, w) in axes.items()}, a.out / "DIRECTIONS.pt")
    print(report["controls"], "upper", report["upper_bound"], flush=True)


if __name__ == "__main__":
    main()
