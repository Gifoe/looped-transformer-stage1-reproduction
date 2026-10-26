"""Backend-independent loop strategies.

The functions in this module only need a tensor and a callable implementing one
application of the looped block. They are intentionally independent from
Transformers so the numerical recipe can be tested without loading a model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import torch

LoopStrategy = Literal[
    "naive",
    "residual_scaled",
    "ema",
    "euler",
    "midpoint",
    "heun",
    "rk4",
    "ema_sched",
    "heavy_ball",
    "anderson",
    "aitken",
    "norm_stab",
    "poly_blend",
    "input_anchor",
    "block_anchored",
]


def anchored_euler(
    x_init: torch.Tensor,
    apply_map: Callable[[torch.Tensor], torch.Tensor],
    *,
    K: int,
    beta: float,
) -> torch.Tensor:
    """Anchored damped Euler used by the official Qwen3 reproduction script.

    Let ``g`` be one pass through the selected layer window.

    ``natural = g(x_0)``
    ``x_t = x_{t-1} + (1/K) * (g(x_{t-1}) - x_{t-1})``
    ``out = beta * natural + (1 - beta) * x_K``
    """

    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    beta = float(beta)
    natural = apply_map(x_init)
    step = 1.0 / float(K)
    x_iter = x_init + step * (natural - x_init)
    for _ in range(1, K):
        mapped = apply_map(x_iter)
        x_iter = x_iter + step * (mapped - x_iter)
    return beta * natural + (1.0 - beta) * x_iter


def iterate_map(
    x_init: torch.Tensor,
    apply_map: Callable[[torch.Tensor], torch.Tensor],
    *,
    strategy: LoopStrategy,
    K: int,
    ema_alpha: float = 0.5,
    momentum_beta: float = 0.3,
    anderson_m: int = 2,
    anderson_beta: float = 1.0,
    per_step_alphas: Sequence[float] | None = None,
    halt_tau: float | None = None,
    anchor_beta: float = 0.0,
) -> torch.Tensor:
    """Apply a fixed-point loop strategy to ``apply_map``.

    ``apply_map`` should return a tensor with the same shape and dtype as
    ``x_init``. ``strategy='block_anchored'`` is the recipe used in the
    upstream reproduction script for the two Qwen3 runs.
    """

    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    eps = 1e-6

    def halted(x_new: torch.Tensor, x_old: torch.Tensor) -> bool:
        if halt_tau is None:
            return False
        with torch.no_grad():
            delta = (x_new - x_old).float().norm()
            denom = x_old.float().norm().clamp_min(eps)
            return float(delta / denom) < float(halt_tau)

    if strategy == "block_anchored":
        return anchored_euler(x_init, apply_map, K=K, beta=anchor_beta)

    if strategy == "naive":
        x = x_init
        for _ in range(K):
            x = apply_map(x)
        return x

    if strategy == "residual_scaled":
        y = x_init
        for _ in range(K):
            y = apply_map(y)
        return x_init + (y - x_init) / float(K)

    if strategy == "ema":
        x = x_init
        alpha = float(ema_alpha)
        for _ in range(K):
            y = apply_map(x)
            x_new = alpha * y + (1.0 - alpha) * x
            if halted(x_new, x):
                return x_new
            x = x_new
        return x

    if strategy == "ema_sched":
        if per_step_alphas is None or len(per_step_alphas) != K:
            got = None if per_step_alphas is None else len(per_step_alphas)
            raise ValueError(f"ema_sched needs {K} alphas, got {got}")
        x = x_init
        for alpha in per_step_alphas:
            y = apply_map(x)
            x_new = float(alpha) * y + (1.0 - float(alpha)) * x
            if halted(x_new, x):
                return x_new
            x = x_new
        return x

    if strategy == "heavy_ball":
        alpha = float(ema_alpha)
        beta = float(momentum_beta)
        x = x_init
        x_prev = x_init
        for _ in range(K):
            mapped = apply_map(x)
            x_new = x + alpha * (mapped - x) + beta * (x - x_prev)
            if halted(x_new, x):
                return x_new
            x_prev = x
            x = x_new
        return x

    if strategy == "anderson":
        return _anderson(
            x_init,
            apply_map,
            K=K,
            m_max=anderson_m,
            beta=anderson_beta,
            halted=halted,
        )

    if strategy == "aitken":
        if K % 2 != 0:
            raise ValueError("aitken requires an even K")
        x = x_init
        for _ in range(K // 2):
            g1 = apply_map(x)
            g2 = apply_map(g1)
            d1 = (g1 - x).float()
            d2 = (g2 - 2.0 * g1 + x).float()
            mask = d2.abs() > (1e-3 * d1.abs() + 1e-4)
            update = torch.where(mask, (d1 * d1) / d2.where(mask, torch.ones_like(d2)), -d1)
            update = update.clamp(min=-d1.abs(), max=d1.abs())
            x_new = (x.float() - update).to(x.dtype)
            if halted(x_new, x):
                return x_new
            x = x_new
        return x

    if strategy == "norm_stab":
        alpha = float(ema_alpha)
        x = x_init
        target_norm = x_init.float().norm(dim=-1, keepdim=True).clamp_min(eps)
        for _ in range(K):
            y = apply_map(x)
            proposal = alpha * y + (1.0 - alpha) * x
            cur_norm = proposal.float().norm(dim=-1, keepdim=True).clamp_min(eps)
            x_new = (proposal.float() * (target_norm / cur_norm)).to(x.dtype)
            if halted(x_new, x):
                return x_new
            x = x_new
        return x

    if strategy == "poly_blend":
        if per_step_alphas is None or len(per_step_alphas) != K + 1:
            got = None if per_step_alphas is None else len(per_step_alphas)
            raise ValueError(f"poly_blend needs {K + 1} weights, got {got}")
        weights = [float(w) for w in per_step_alphas]
        x_power = x_init
        out = weights[0] * x_power
        for weight in weights[1:]:
            x_power = apply_map(x_power)
            out = out + weight * x_power
        return out

    if strategy == "input_anchor":
        alpha = float(ema_alpha)
        x = x_init
        for _ in range(K):
            y = apply_map(x)
            x_new = alpha * y + (1.0 - alpha) * x_init
            if halted(x_new, x):
                return x_new
            x = x_new
        return x

    if strategy in {"euler", "midpoint", "heun", "rk4"}:
        return _runge_kutta(x_init, apply_map, strategy=strategy, K=K, halted=halted)

    raise ValueError(f"unknown loop strategy: {strategy!r}")


def _runge_kutta(
    x_init: torch.Tensor,
    apply_map: Callable[[torch.Tensor], torch.Tensor],
    *,
    strategy: str,
    K: int,
    halted: Callable[[torch.Tensor, torch.Tensor], bool],
) -> torch.Tensor:
    step = 1.0 / float(K)

    def field(z: torch.Tensor) -> torch.Tensor:
        return apply_map(z) - z

    x = x_init
    if strategy == "euler":
        for _ in range(K):
            x_new = x + step * field(x)
            if halted(x_new, x):
                return x_new
            x = x_new
        return x

    if strategy == "midpoint":
        for _ in range(K):
            k1 = field(x)
            k2 = field(x + 0.5 * step * k1)
            x = x + step * k2
        return x

    if strategy == "heun":
        for _ in range(K):
            k1 = field(x)
            k2 = field(x + step * k1)
            x = x + 0.5 * step * (k1 + k2)
        return x

    if strategy == "rk4":
        for _ in range(K):
            k1 = field(x)
            k2 = field(x + 0.5 * step * k1)
            k3 = field(x + 0.5 * step * k2)
            k4 = field(x + step * k3)
            x = x + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x

    raise AssertionError(f"unreachable strategy {strategy!r}")


def _anderson(
    x_init: torch.Tensor,
    apply_map: Callable[[torch.Tensor], torch.Tensor],
    *,
    K: int,
    m_max: int,
    beta: float,
    halted: Callable[[torch.Tensor, torch.Tensor], bool],
) -> torch.Tensor:
    reg = 1e-6
    m_max = max(1, int(m_max))
    beta = float(beta)
    x = x_init
    hist_x: list[torch.Tensor] = []
    hist_g: list[torch.Tensor] = []

    for _ in range(K):
        mapped = apply_map(x)
        hist_x.append(x)
        hist_g.append(mapped)
        if len(hist_x) > m_max + 1:
            hist_x = hist_x[-(m_max + 1) :]
            hist_g = hist_g[-(m_max + 1) :]

        m_eff = len(hist_x) - 1
        if m_eff == 0:
            x_new = (1.0 - beta) * x + beta * mapped
        else:
            dF_list = [
                (hist_g[i + 1] - hist_x[i + 1]) - (hist_g[i] - hist_x[i])
                for i in range(m_eff)
            ]
            dX_list = [hist_x[i + 1] - hist_x[i] for i in range(m_eff)]
            dF = torch.stack(dF_list, dim=-1).float()
            dX = torch.stack(dX_list, dim=-1).float()
            f_last = (mapped - x).float()
            dF_flat = dF.reshape(-1, m_eff)
            f_flat = f_last.reshape(-1)
            lhs = dF_flat.T @ dF_flat
            diag_max = lhs.diagonal().abs().max().clamp_min(1e-8)
            lhs = lhs + reg * diag_max * torch.eye(m_eff, device=lhs.device, dtype=lhs.dtype)
            rhs = dF_flat.T @ f_flat
            try:
                gamma = torch.linalg.solve(lhs, rhs)
            except RuntimeError:
                gamma = torch.zeros(m_eff, device=lhs.device, dtype=lhs.dtype)
            gamma_view = gamma.view(*([1] * (dX.ndim - 1)), m_eff)
            upd_x = (dX * gamma_view).sum(dim=-1).to(x.dtype)
            upd_f = (dF * gamma_view).sum(dim=-1).to(x.dtype)
            x_new = (1.0 - beta) * (x - upd_x) + beta * (mapped - upd_f)

        if halted(x_new, x):
            return x_new
        x = x_new
    return x

