"""Run the archived experiment from a checkout and an exact local checkpoint."""

import argparse
import os
from pathlib import Path
import sys


ARCHIVES = (
    "lookup_branches", "lookup_residual", "lookup_query", "value_compose",
    "boundary_factor", "borrow_paths", "borrow_edge", "borrow_origin",
    "native_state_computation", "open_carry", "bidirectional", "rule_switch",
    "borrow_query", "borrow_transport", "borrow_factorial",
    "native_borrow_interaction", "borrow_block",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify-deck", action="store_true")
    args = parser.parse_args()
    model = args.checkpoint.resolve()
    output = args.output.resolve()
    experiment = Path(__file__).resolve().parent
    archives = {name: experiment.parent / f"ouro_{name}_20261121" for name in ARCHIVES}
    for p in archives.values():
        if not p.is_dir():
            parser.error(f"missing archived dependency {p}")
    sys.path[1:1] = [str(p) for p in archives.values()]
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

    import branches_deck
    from native import MODEL_HASH, load, save, sha
    import onset_deck
    import onset_test
    import query_deck
    import report
    import residual_deck
    import torch

    onset_deck.PARENT = archives["lookup_branches"] / "results/PAIR_NATIVE.json"
    branches_deck.PARENT = archives["lookup_residual"] / "results"
    residual_deck.PARENT = archives["lookup_query"] / "results/DECK.json"
    query_deck.PARENT = archives["value_compose"] / "results/LOOKUP_DECK.json"
    deck = load(experiment / "results/DECK.json")
    assert onset_deck.build() == deck
    if args.verify_deck:
        print("PASS archived deck and prompt history")
        return

    checkpoint_hash = sha(model / "model.safetensors")
    if checkpoint_hash != MODEL_HASH:
        parser.error("checkpoint SHA-256 differs from the frozen model")
    if output == experiment:
        parser.error("use a writable output directory outside the archived experiment")
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    manifest = results / "DECK.json"
    if manifest.exists():
        assert load(manifest) == deck
    else:
        manifest.write_bytes((experiment / "results/DECK.json").read_bytes())
    onset_deck.run(model, results)
    if load(results / "NATIVE_SUMMARY.json")["gate"]:
        onset_test.run(model, results)
    os.environ["OURO_ACTUAL_SHA256"] = checkpoint_hash
    os.environ["OURO_GPU"] = torch.cuda.get_device_name()
    report.PARENT = onset_deck.PARENT
    v = report.verify()
    report.render(v)
    save(results / "VERIFICATION.json", v)


if __name__ == "__main__":
    main()
