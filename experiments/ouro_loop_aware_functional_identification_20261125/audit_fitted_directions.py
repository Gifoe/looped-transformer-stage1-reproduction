"""Compare learned matrices with T1 on untouched fold-3 gradient moments."""

import argparse
import json
from pathlib import Path

import torch

import minimal


def factors(path):
    state = torch.load(path, map_location="cpu", weights_only=True)["state"]
    return state["out.weight"].flatten().float(), state["encode.weight"].flatten().float()


def score(m, u, w):
    return float(u @ (m @ w))


def run(a):
    torch.set_num_threads(4)
    plan = json.loads(a.forecast.read_text(encoding="utf-8"))
    slots = dict(a.checkpoint)
    controls = {name: factors(path) for name, path in slots.items()}
    template_u, template_w = controls["t1"]
    fit = {}
    moments = {}
    for source in plan["source_order"]:
        f = a.moments / source / f"{plan['sources'][source]['mode']}/k4/fold_3.pt"
        part = torch.load(f, map_location="cpu", weights_only=True)
        scale = plan["sources"][source]["fit_q_robust_scale"]
        divisor = part["count"] * scale
        loop = part["moments"]["loop"] / divisor
        direct = part["moments"]["direct"] / divisor
        moments[source] = {"n": part["count"], "sha256": minimal.sha(f),
                           "ordinary": loop, "excess": loop - direct}
    for name, (u, w) in controls.items():
        strength = u.norm() * w.norm()
        if strength == 0:
            raise ValueError(f"Zero-write fitted control {name}")
        u0, w0 = u / u.norm(), w / w.norm()
        fit[name] = dict(checkpoint_sha256=minimal.sha(slots[name]), matrix_norm=float(strength),
                         cosine_t1=float((u0 @ (template_u / template_u.norm()))
                                         * (w0 @ (template_w / template_w.norm()))),
                         fold3_projections={source: {k: score(fields[k], u0, w0) for k in ("ordinary", "excess")}
                                            for source, fields in moments.items()})
    meta = {source: {"n": fields["n"], "sha256": fields["sha256"]} for source, fields in moments.items()}
    minimal.dump(a.out, dict(metric="unit-Frobenius uncapped local projection", heldout=meta, controls=fit))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("forecast", "moments", "out"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--checkpoint", action="append", required=True)
    a = p.parse_args()
    a.checkpoint = [(name, Path(path)) for name, path in (arg.split("=", 1) for arg in a.checkpoint)]
    run(a)
