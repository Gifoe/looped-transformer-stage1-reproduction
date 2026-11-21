"""Validate the frozen artifacts and inspect the release's boundaries."""
import argparse
from pathlib import Path

from PIL import Image

from native import load, save, sha


def verify(root):
    r=root/"results"
    model=load(r/"RUNTIME.json")
    assert len(model["parity"])==6 and all(x["loop_outputs_bitwise"] and x["operator_same_input_loop_0_to_3_bitwise"] for x in model["parity"])
    for row in (r/"CONFIRMATION_LOCK.sha256").read_text().splitlines():
        digest,name=row.split(maxsplit=1)
        assert sha(root/name)==digest,name
    atlas={"pilot":24,"dev":96,"confirm_arc":96,"confirm_external":96,"controlled":48}
    for n,count in atlas.items():
        a=load(r/f"ATLAS_{n}.json")
        assert len(a)==count and all(len(x["measures"])==72 for x in a),n
        if n!="pilot":
            b=load(r/f"BEHAVIOR_{n}.json")
            assert len(b["rows"])==count and {x["id"] for x in a}=={x["id"] for x in b["rows"]},n
            c=load(r/f"CAUSAL_{n}.json")
            assert len(c["rows"])==(48 if n=="dev" else count),n
            if n!="dev":
                assert c["prediction_sha256"]==sha(r/"PREDICTIONS.json")
                assert all(len(x["tests"])==6 for x in c["rows"])
    pred=load(r/"PREDICTIONS.json")
    assert pred["candidates_sha256"]==sha(r/"CANDIDATES.json")
    c=load(r/"CANDIDATES.json")
    assert c["atlas_sha256"]==sha(r/"ATLAS_dev.json")
    assert c["discovery_manifest_sha256"]==sha(root/"ARC_BLIND_INPUTS.json")
    blind=load(root/"BLIND.json")
    assert all("gold" not in q for p in atlas for q in blind[p])
    assert set(x["id"] for n in atlas for x in blind[n]).__len__()==sum(atlas.values())
    names={p.name for p in r.glob("TRACE_K*.npz")}
    assert len(names)==16
    assert (r/"TRACE_VOCAB_LOGITS.npz").exists()
    for key in ("confirm_arc","confirm_external"):
        f=load(r/f"FULL_PATCH_{key}.json")
        assert len(f["rows"])==24 and f["prediction_sha256"]==sha(r/"PREDICTIONS.json")
    assert len(load(r/"SEMANTIC_NATIVE.json")["rows"])==48
    assert load(r/"SEMANTIC_PREDICTIONS.json")["design_sha256"]==sha(root/"STRUCTURE.json")
    assert load(r/"SEMANTIC_NATIVE.json")["prediction_sha256"]==sha(r/"SEMANTIC_PREDICTIONS.json")
    assert load(r/"RESPONSE_CONTROLS.json")["prediction_sha256"]==sha(r/"RESPONSE_PREDICTIONS.json")
    d=r/"formation"
    p=d/"ORIGIN_PREDICTIONS.json"
    lock=load(p)
    assert lock["selected_sha256"]==sha(d/"ORIGIN_LOCK.json")
    assert lock["protocol_sha256"]==sha(root/"FORMATION_PROTOCOL.md")
    assert lock["candidate_sha256"]==sha(r/"CANDIDATES.json")
    for name,value in lock["code"].items():
        assert sha(root/name)==value
    assert len(load(d/"ORIGIN_SCREEN.json")["rows"])==16
    populations={"confirm_arc":48,"confirm_external":48,"controlled":24,"confirm_external_later":48}
    for phase,n in populations.items():
        record=load(d/f"ORIGIN_{phase}.json")
        assert len(record["rows"])==n and record["prediction_sha256"]==sha(p)
    ff=sorted((r/"figures").glob("*.png"))
    assert len(ff)==7
    for one in ff:
        with Image.open(one) as im:
            im.verify()
        assert one.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
    save(r/"RELEASE_VERIFICATION.json",dict(status="PASS",weights_sha256=model["model_sha256"],
         atlas=atlas,prediction_sha256=sha(r/"PREDICTIONS.json"),origin_sha256=sha(p),origin_counts=populations,
         activation_shards=len(names),figures_png=len(ff),figures_pdf=len(ff),native_parity_checks=6))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("release",type=Path)
    verify(p.parse_args().release)

