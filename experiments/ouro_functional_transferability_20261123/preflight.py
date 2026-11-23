"""Small real-model forward parity and RMSNorm algebra audit."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

import minimal
from forward_control import forward_control
from normalized_geometry import geometry_rows, relative_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "previous", "model", "corpus", "rank1", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    archive, model, tokenizer, runtime = minimal.runtime(args.archive, args.previous, args.model)
    net = minimal.Minimal("linear", 1, .45).to(next(model.parameters()).device).eval().requires_grad_(False)
    net.load_state_dict(torch.load(args.rank1, map_location="cpu", weights_only=True)["state"])
    corpus = archive.read_data(args.corpus, "train")
    rows = sorted(corpus, key=lambda x: hashlib.sha256(f"transfer-parity/{x['id']}".encode()).digest())[:4]
    examples, _ = archive.encode(tokenizer, rows)
    packed = archive.pack(examples)
    records = []
    with torch.no_grad():
        _, base = forward_control(model, packed, 6, capture=tuple(range(1, 7)))
        for depth in range(1, 7):
            new, state = forward_control(model, packed, depth, controller=net, capture=(depth,))
            anchor = state["anchor"]
            reparam = relative_write(anchor, net.out.weight[:, 0], net.encode.weight[0], .45)
            ordinary = net.delta(anchor)
            torch.testing.assert_close(reparam, ordinary, rtol=1e-5, atol=3e-6)
            if depth <= 4:
                old, old_anchor = minimal.forward(model, packed, depth, net)
                assert torch.equal(old, new) and torch.equal(old_anchor, anchor)
            h = base["pre"][depth]
            if depth >= 2:
                k = depth - 1
                _, edited = forward_control(model, packed, depth, controller=net, capture=(k, depth))
                radial = geometry_rows(base["pre"][k], edited["pre"][k].float() - base["pre"][k].float(),
                                       base["post"][k], edited["post"][k], model.model.norm)
                med = {name: float(x.flatten().median()) for name, x in radial.items()}
            else:
                med = {}
            records.append(dict(depth=depth, parity=depth <= 4, pre_boundary_rms=float(h.float().square().mean().sqrt()),
                                maximum_reparam_error=float((reparam-ordinary).abs().max()), geometry_medians=med))
    with torch.enable_grad():
        new, trace = forward_control(model, packed, 3, anchor_loop=1, leaf=True)
        grad, = torch.autograd.grad(new.float().square().mean(), trace["leaf"])
    if not torch.isfinite(grad).all() or float(grad.norm()) <= 0:
        raise RuntimeError("Frozen suffix did not return a valid leaf gradient")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    minimal.dump(args.out, dict(ids=[row["id"] for row in rows], runtime=runtime, cells=records,
                                gradient_norm=float(grad.norm()), seconds=time.monotonic()-start))
    print(json.dumps({"cells": records, "gradient_norm": float(grad.norm())}), flush=True)


if __name__ == "__main__":
    main()
