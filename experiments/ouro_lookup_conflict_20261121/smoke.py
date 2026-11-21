"""Check both conflict directions and the live readout on previously archived worlds."""

import sys
from pathlib import Path

import torch

from conf_deck import DOMAINS
from conf_pair import SIDES,qualified,spans
from conf_test import audit_case,decode
from native import Forward,load,setup


@torch.inference_mode()
def main():
    m,tok = setup(Path(sys.argv[1]))
    f = Forward(m)
    source = load(Path(sys.argv[2]))["rows"]
    rows = []
    for d in DOMAINS:
        for r in source:
            if r["domain"]!=d:
                continue
            w = dict(r,span=spans(tok,r),pair_native={s:r["native"][s][1] for s in SIDES})
            if qualified(w):
                rows.append(w)
                break
    assert len(rows)==len(DOMAINS)
    ans,_ = decode(f,tok,rows)
    assert len(ans)==len(DOMAINS)
    for r in rows:
        a = audit_case(f,r)
        assert a["v_sites"]==240 and a["carry_sites"]==10 and a["tokens"]==16
        assert all(a[k]<.001 for k in ("cache_causal","attention"))
    print("PASS exact token span, live V projection, carry isolation, dynamic AV, clean integer",flush=True)


if __name__ == "__main__":
    main()
