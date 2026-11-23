"""Audit weighted Ouro RMSNorm and the deployed finite, clipped Rank-1 law."""

import torch


def rms(x):
    return x.float().square().mean(dim=-1, keepdim=True).sqrt()


def relative_write(h, u, w, cap):
    """Exact deployed expression in FP32, including the implementation's RMS floors."""
    h = h.float()
    scale = rms(h).clamp_min(1e-6)
    axis_scale = rms(u)
    if float(axis_scale) == 0.0:
        return torch.zeros_like(h)
    signed = (h / scale) @ w.float()
    signed = signed[:, None] * axis_scale
    # A simple clamp is exact outside the floor of the deployed update norm.
    denom = (scale * signed.abs()).clamp_min(1e-6)
    signed = signed * (cap * scale / denom).clamp_max(1.)
    return scale * signed * (u.float() / axis_scale)


def norm_formula(h, weight, epsilon):
    h = h.double()
    return weight.double() * h * (h.square().mean(-1, keepdim=True) + epsilon).rsqrt()


def norm_tangent(h, d, weight, epsilon):
    h, d = h.double(), d.double()
    r2 = h.square().mean(-1, keepdim=True) + epsilon
    radial_cov = (h * d).mean(-1, keepdim=True)
    return weight.double() * (d - h * radial_cov / r2) * r2.rsqrt()


def radial_tangent(h, d):
    r2 = h.square().mean(-1, keepdim=True).clamp_min(1e-30)
    radial = h * (h * d).mean(-1, keepdim=True) / r2
    return radial, d - radial


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-12)


def geometry_rows(h, delta, original_norm, edited_norm, mod):
    """Compare actual worker BF16 values to real-arithmetic and local-normalization models."""
    weight, epsilon = mod.weight.detach(), mod.variance_epsilon
    radial, tangent = radial_tangent(h, delta)
    exact = norm_formula(h + delta, weight, epsilon) - norm_formula(h, weight, epsilon)
    linear = norm_tangent(h, delta, weight, epsilon)
    actual = (edited_norm.float() - original_norm.float())
    radial_linear = norm_tangent(h, radial, weight, epsilon)
    return dict(
        state_rms=rms(h).flatten(), injection_rms=rms(delta).flatten(),
        radial_injection_rms=rms(radial).flatten(), tangent_injection_rms=rms(tangent).flatten(),
        normalized_rms=rms(actual).flatten(),
        exact_error_rms=rms(actual - exact).flatten(), linear_error_rms=rms(actual - linear).flatten(),
        cosine_exact=cosine(actual, exact), cosine_linear=cosine(actual, linear),
        radial_linear_rms=rms(radial_linear).flatten(),
    )
