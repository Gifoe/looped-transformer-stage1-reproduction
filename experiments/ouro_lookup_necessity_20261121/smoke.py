"""Exercise native deletion and margin extraction on prior archived questions."""

import sys
from pathlib import Path

import torch

from native import Forward,load,setup
from nec_deck import DOMAINS,PARENT,prompt
from nec_match import enroll,measure
from nec_test import audit_case,decode


@torch.inference_mode()
def main():
    m,tok = setup(Path(sys.argv[1]))
    f = Forward(m)
    candidates = enroll(load(PARENT/"NATIVE.json")["rows"])
    examples = [next(r for r in candidates if (r["domain"],r["kind"])==(d,k)) for d in DOMAINS for k in ("repair","stable")]
    margins = measure(f,tok,examples)
    assert all(r["selected"] is not None for r in margins)
    examples = [dict(r,ids=tok.encode(prompt(r,"original"),add_special_tokens=False)) for r in examples]
    assert len(decode(f,tok,examples))==4
    for row in examples:
        audit = audit_case(f,tok,row)
        assert audit["checks"]==18 and all(x<.001 for k,x in audit.items() if k!="checks")
    print("PASS clean-margin extraction, live subtraction, identity generation, and direct suffix",flush=True)


if __name__ == "__main__":
    main()
