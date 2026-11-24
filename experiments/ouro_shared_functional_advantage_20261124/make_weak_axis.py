"""Build the weak-dose fitted direction as an ordinary 4,096-parameter checkpoint."""

import argparse
import math
from pathlib import Path

import torch

import minimal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("axis", "reference", "out"):
        p.add_argument(f"--{key}", type=Path, required=True)
    a = p.parse_args()
    reference = torch.load(a.reference, map_location="cpu", weights_only=True)["state"]
    pair = torch.load(a.axis, map_location="cpu", weights_only=True)["worst_group"]
    dose = .10
    parameter_product = float(reference["out.weight"].norm() * reference["encode.weight"].norm())
    mult = math.sqrt(dose * parameter_product)
    net = minimal.Minimal("linear", 1, .45 * dose).eval()
    with torch.no_grad():
        net.out.weight.copy_(pair["u"][:, None] * mult)
        net.encode.weight.copy_(pair["w"][None, :] * mult)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    minimal.save_net(a.out, net, dict(dose=dose, source_sha256=minimal.sha(a.axis),
                                    norm_reference_sha256=minimal.sha(a.reference), parameters=4096))
    print(a.out, minimal.sha(a.out), flush=True)


if __name__ == "__main__":
    main()
