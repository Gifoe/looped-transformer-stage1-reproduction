"""Read the two verified Rank-1 checkpoint encodings without changing the factors."""

import torch

import minimal


def controller(path, device):
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("form") == "rank1":
        net = minimal.Minimal("linear", 1, .45)
    else:
        net = minimal.Minimal(**payload["arch"])
    net.load_state_dict(payload["state"], strict=True)
    if net.kind != "linear" or net.rank != 1 or net.bound != .45 or sum(p.numel() for p in net.parameters()) != 4096:
        raise ValueError(f"Unexpected Rank-1 control structure: {path}")
    return net.to(device).requires_grad_(False).eval()
