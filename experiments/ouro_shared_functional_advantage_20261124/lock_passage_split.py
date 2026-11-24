"""Lock context-disjoint SQuAD selection IDs without reading any model predictions."""

import argparse
from pathlib import Path

import minimal
from task_data import hash_order, identity, load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("current_fit", "historical_fit", "development", "confirmation", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    fit = [x for f in (a.current_fit, a.historical_fit) for x in load(f) if x["dataset"] == "squad"]
    passages = {identity(x["context"]) for x in fit}
    dev = sorted(load(a.development), key=lambda x: hash_order(x["sample_id"], "advantage/squad/evaluation"))[:64]
    ex = [x["sample_id"] for x in dev if identity(x["context"]) in passages]
    final = [x["sample_id"] for x in load(a.confirmation) if identity(x["context"]) in passages]
    if len(dev) == len(ex) or final:
        raise RuntimeError("No passage-independent development rows or overlapping confirmation passages")
    receipt = dict(squad=dict(excluded_ids=ex, original_n=len(dev), independent_n=len(dev)-len(ex),
                             confirmation_overlap_ids=final),
                   source_sha256={name: minimal.sha(getattr(a, name))
                                  for name in ("current_fit", "historical_fit", "development", "confirmation")})
    if a.out.exists():
        raise FileExistsError(a.out)
    minimal.dump(a.out, receipt)
    print(receipt["squad"], flush=True)


if __name__ == "__main__":
    main()
