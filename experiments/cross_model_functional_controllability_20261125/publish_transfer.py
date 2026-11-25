"""Publish the verified three-Ouro release on the cross-model branch."""

import argparse
import json
from pathlib import Path
import sys


def run(a):
    sys.path.insert(0, str(a.publisher))
    import publish_research as api

    manifest = json.loads((a.directory / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["complete_ouro_models"] != 3 or manifest["measurement_arms"] != ["native", "t1"]:
        raise ValueError("The release is missing a complete model or a comparison arm")
    api.EXPECTED_BENCHMARKS = 24
    api.FEATURE = "experiments/cross_model_functional_controllability_20261125"
    api.BRANCH = "research/cross-model-functional-controllability"
    api.COMMIT_MESSAGE = "Publish complete frozen T1 transfer benchmarks on the three Ouro checkpoints"
    api.run(a, "" if a.verify_only else sys.stdin.readline().strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--expected-head")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--publisher", type=Path,
                        default=Path("/root/ouro-loop-aware-functional-identification-20261125"))
    args = parser.parse_args()
    if not args.verify_only and not args.expected_head:
        parser.error("--expected-head is required when publishing")
    run(args)
