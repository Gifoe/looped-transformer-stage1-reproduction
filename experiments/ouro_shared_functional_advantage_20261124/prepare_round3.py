"""Lock a paired continuation arm from development data before new fitting."""

import argparse
import json
from pathlib import Path

from freeze_eval import ADVANTAGE, BASELINES, select
import minimal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    a = p.parse_args()
    root = a.results
    selected, evidence, frontier = select(root / "development", BASELINES, ADVANTAGE)
    source = root / "fits" / selected / "controller.pt"
    receipt = root / "PREFIX_CAUSALITY" / "RECEIPT.json"
    if json.loads(receipt.read_text())["n_replays"] < 24:
        raise ValueError("Expected the paired native and controlled prefix diagnostic")
    result = dict(initial_arm=selected, initial_file=str(source), initial_sha256=minimal.sha(source),
                  diagnostic_sha256=minimal.sha(receipt), common_steps=336, common_learning_rate=.0005,
                  development_exclusions_sha256=minimal.sha(root/"DEVELOPMENT_EXCLUSIONS.json"),
                  policy_fit_limit=168, seed=261124173,
                  changed_factor="replace half of the affected sample-stage q by continuation log-odds at its generated premature-EOS prefix",
                  development_evidence=evidence, development_frontier=frontier)
    output = root / "ROUND3_LOCK.json"
    if output.exists() and json.loads(output.read_text()) != result:
        raise RuntimeError("Dev-only continuation was already decided")
    minimal.dump(output, result)
    print(selected, evidence[selected]["worst_delta"], flush=True)


if __name__ == "__main__":
    main()
