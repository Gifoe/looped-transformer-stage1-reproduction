"""Verify sealed predictions, cohort denominators, parity and rendered exports."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from analyze import END, collect_manifest, load, phase_rows, sha
import json


def verify(root):
    manifest, lookup = collect_manifest(root)
    receipts, populations, nonfinite = {}, {}, {}
    for phase,n in (("pilot",4),("development",16),("validation",96),("confirmation",192)):
        rows,pred,native=phase_rows(root,phase,lookup)
        assert len(rows)==len({r["group"] for r in rows})==len(native["rows"])==n
        assert sum(r["family"]=="tableau" for r in rows)==n//2
        assert all(set(r["endpoints"])==set(END) for r in rows)
        ids={r["group"] for r in rows}
        assert not any(ids.intersection(pop) for pop in populations.values())
        populations[phase]=ids
        nonfinite[phase]=pred["failures"]+native["failures"]
        receipts[phase]=native["prediction_sha256"]
    source=Path(__file__).resolve().parent
    for p in ("DEVELOPMENT_LOCK.sha256","CONFIRMATION_LOCK.sha256"):
        for line in (root/p).read_text(encoding="utf-8").splitlines():
            digest,path=line.split(maxsplit=1)
            name=Path(path)
            resolved=(root if name.is_absolute() else source)/name.name
            assert sha(resolved)==digest,resolved
    for label, stem, pred, receipt, truth, n in (
        ("local_development","LOCAL_","PREDICTIONS_development","PREDICTION_SHA_development","NATIVE_development",48),
        ("local_validation","LOCAL_","PREDICTIONS_validation","PREDICTION_SHA_validation","NATIVE_validation",96),
        ("arc","ARC_","PREDICTIONS","PREDICTION_SHA","NATIVE",96),
        ("bf16","BF16_","PREDICTIONS","PREDICTION_SHA","NATIVE",16),
    ):
        p=root/f"{stem}{pred}.json"
        q=load(root/f"{stem}{receipt}.json")
        t=load(root/f"{stem}{truth}.json")
        assert sha(p)==q["sha256"]==t["prediction_sha256"]
        assert len(t["rows"])==len(load(p)["rows"])==n
        receipts[label]=q["sha256"]
    native=load(root/"NATIVE_confirmation_BF16.json")
    assert native["prediction_sha256"]==receipts["confirmation"]
    assert len(native["rows"])==192 and all(q["initial"]["mixed_l2"]==0 for q in native["rows"])
    ca=load(root/"CONTROLS.json")
    assert ca["prediction_sha256"]==receipts["confirmation"] and len(ca["rows"])==192
    assert sha(root/"PREDICTIONS_development.json")==load(root/"FREE_METHOD.json")["validation_receipt"]
    assert sha(root/"LOCAL_NATIVE_development.json")==load(root/"LOCAL_FREEZE.json")["development_native_sha256"]
    assert all(g["targets"][3] not in g["targets"][:3] and not set(g["source"]) & set(g["rule"])
               for g in lookup.values())
    a=load(root/"AUDIT.json")
    for dtype,payload in a["results"].items():
        assert len(payload["parity"])==len(payload["norm"])==3, dtype
    figs=sorted((root/"figures").glob("*.png"))
    assert len(figs)==6
    for p in figs:
        with Image.open(p) as im:
            im.verify()
        assert p.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
    states=sorted(root.glob("STATE_*.npz"))
    assert len(states)==18
    for p in states:
        with np.load(p) as z:
            h=z["states"]
            assert h.ndim==4 and h.shape[-1]==2048 and h.shape[:2]==(7,4)
            assert np.isfinite(h).all()
    assert len(load(root/"STATE_AUDIT.json")["rows"])==2
    return dict(status="PASS", model=a["runtime"]["model_revision"], weights_sha256=a["runtime"]["model_sha256"],
                phase_world_groups={p:len(z) for p,z in populations.items()}, prediction_hashes=receipts,
                failed_batches=nonfinite, parity_batches_per_dtype=3, feature_png=6, feature_pdf=6,
                state_shards=len(states), arc_n=96)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    print(json.dumps(verify(p.parse_args().out),indent=2))

