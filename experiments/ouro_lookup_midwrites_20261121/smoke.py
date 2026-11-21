"""Exercise every replay arm on archived examples before the independent deck."""

import sys
from pathlib import Path

import torch

from mid_deck import DOMAINS, PARENT, repair
from mid_test import audit_case, decode, prepare
from native import Forward, load, setup


@torch.inference_mode()
def main():
    m,tok = setup(Path(sys.argv[1]))
    f = Forward(m)
    archived = load(PARENT / "NATIVE.json")["rows"]
    rows = [prepare(tok,next(r for r in archived if r["domain"]==d and repair(r))) for d in DOMAINS]
    out, fallback, n = decode(f,tok,rows)
    assert len(out)==len(DOMAINS) and 0<fallback<n
    tests = [audit_case(f,tok,r) for r in rows]
    assert all(v["checks"]==72 and v["suffix"]<.001 and v["identity"]<.001 for v in tests)
    print("PASS replay arms, controls, full native answers, and direct suffix computation",flush=True)


if __name__ == "__main__":
    main()
