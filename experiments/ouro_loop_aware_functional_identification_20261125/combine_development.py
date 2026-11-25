"""Choose a checkpoint from identical independently evaluated panels."""

import argparse
import json
from pathlib import Path

import minimal


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(a):
    fields = ("cohorts", "model_sha256", "depth", "accuracy_rank")
    comparison = None
    candidates, scores, fingerprints = {}, {}, {}
    for path in a.development:
        selected = read(path / "SELECTION.json")
        receipt = path / "DEV_PROTOCOL.json"
        seal = minimal.sha(receipt)
        if selected["development_protocol_sha256"] != seal:
            raise ValueError(f"Changed development protocol in {path}")
        protocol = read(receipt)
        config = {key: protocol[key] for key in fields}
        if comparison is not None and config != comparison:
            raise ValueError(f"Different development protocol in {path}")
        comparison = config
        for name, summary in selected["summary"].items():
            if name in {"native", "t1", "t2", "mixed"} or name.startswith("source_") or name.endswith("_low"):
                continue
            key = protocol["checkpoint_sha256"][name]
            weights = (a.fits / name / "controller.pt").resolve()
            if key != minimal.sha(weights):
                raise ValueError(f"Changed development checkpoint {name}")
            if name in candidates and (fingerprints[name] != key or scores[name] != summary):
                raise ValueError(f"Independent panel repeats give different results for {name}")
            candidates[name], scores[name], fingerprints[name] = str(weights), summary, key
    winner = max(candidates, key=lambda name: (scores[name]["score"], name))
    lock = {**comparison, "checkpoint_sha256": fingerprints,
            "input_development_sha256": {str(p): minimal.sha(p / "SELECTION.json") for p in a.development}}
    a.out.mkdir(parents=True, exist_ok=True)
    protocol = a.out / "DEV_PROTOCOL.json"
    if protocol.exists() and read(protocol) != lock:
        raise ValueError("Combined development choice was already sealed")
    minimal.dump(protocol, lock)
    minimal.dump(a.out / "SELECTION.json", dict(best=winner, path=candidates[winner],
                 checkpoint_sha256=fingerprints[winner], development_protocol_sha256=minimal.sha(protocol),
                 summary=scores))
    print("SELECTED", winner, scores[winner]["score"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development", nargs="+", required=True, type=Path)
    p.add_argument("--fits", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    run(p.parse_args())
