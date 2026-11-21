"""Run the lookup attention-factor replay from a checkout and the exact local checkpoint."""

import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify-deck", action="store_true")
    args = parser.parse_args()
    output, model = args.output.resolve(), args.checkpoint.resolve()
    experiment = Path(__file__).resolve().parent
    archives = experiment.parent
    onset = archives / "ouro_lookup_onset_20261121"
    middle = archives / "ouro_lookup_midwrites_20261121"
    inherited = runpy.run_path(str(onset/"reproduce.py"))["ARCHIVES"]
    paths = {k:archives/f"ouro_{k}_20261121" for k in inherited}
    for p in paths.values():
        if not p.is_dir():
            parser.error(f"missing archived dependency {p}")
    sys.path[1:1] = [str(middle),str(onset),*[str(p) for p in paths.values()]]
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

    import av_deck
    import av_test
    import branches_deck
    import mid_deck
    from native import MODEL_HASH,load,save,sha
    import onset_deck
    import query_deck
    import report
    import residual_deck
    import torch

    av_deck.PARENT = middle/"results"
    mid_deck.PARENT = onset/"results"
    onset_deck.PARENT = paths["lookup_branches"]/"results/PAIR_NATIVE.json"
    branches_deck.PARENT = paths["lookup_residual"]/"results"
    residual_deck.PARENT = paths["lookup_query"]/"results/DECK.json"
    query_deck.PARENT = paths["value_compose"]/"results/LOOKUP_DECK.json"
    deck = experiment/"results/DECK.json"
    expected = av_deck.build()
    assert expected == load(deck)
    if args.verify_deck:
        print("PASS archived deck and prompt history")
        return
    actual = sha(model/"model.safetensors")
    if actual != MODEL_HASH:
        parser.error("checkpoint SHA-256 differs from the frozen model")
    if output == experiment:
        parser.error("use a writable output directory outside the archived experiment")
    root = output/"results"
    root.mkdir(parents=True,exist_ok=True)
    copy = root/"DECK.json"
    if copy.exists():
        assert load(copy)==expected
    else:
        copy.write_bytes(deck.read_bytes())
    os.chdir(output)
    av_deck.run(model,root)
    if load(root/"NATIVE_SUMMARY.json")["gate"]:
        av_test.run(model,root)
    os.environ["OURO_ACTUAL_SHA256"] = actual
    os.environ["OURO_GPU"] = torch.cuda.get_device_name()
    v = report.verify()
    report.render(v)
    save(root/"VERIFICATION.json",v)


if __name__ == "__main__":
    main()
