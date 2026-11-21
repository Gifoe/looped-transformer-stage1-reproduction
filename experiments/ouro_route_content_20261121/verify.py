"""Validate frozen policies, cohort exclusions and complete raw evidence."""
import argparse
from pathlib import Path

from PIL import Image

from native import load, save, sha


def verify(work,first):
    r=work/"results"
    m=load(r/"NATURAL_MANIFEST.json")
    exposure=load(first/"openbook_exclusions.json")
    old=load(first/"INPUTS.json")
    exposed=set(exposure["exclude_ids"])
    earlier={q["id"] for phase in ("pilot","dev","confirm_arc","confirm_external") for q in old[phase]}
    assert len(exposed)==432==m["historical_id_count"]
    q=load(work/"selection_qa/NATURAL_MANIFEST.json")
    before={x["id"] for x in q["cohorts"]["openbookqa"]["rows"]}
    current={x["id"] for x in m["cohorts"]["openbookqa"]["rows"]}
    assert before&exposed
    assert not current&(exposed|earlier)
    save(r/"SELECTION_AUDIT.json",dict(preliminary_obqa_overlap=len(before&exposed),
             current_overlap=0,corrected_openbook_n=320,pinned_exposure_sha256=sha(first/"openbook_exclusions.json"),
             preliminary_manifest_sha256=sha(work/"selection_qa/NATURAL_MANIFEST.json"),
             canonical_manifest_sha256=sha(r/"NATURAL_MANIFEST.json")))
    for f,n in (("openbookqa",320),("arc_challenge",192)):
        ids=[x["id"] for x in m["cohorts"][f]["rows"]]
        assert len(ids)==len(set(ids))==n and not set(ids)&earlier
        for phase,count in (("NATIVE",n),("FACTORIAL",n),("WRITER",128),("BILINEAR",128)):
            z=load(r/f"{phase}_{f}.json")
            assert len(z["rows"])==z["expected"]==count
            assert [q["id"] for q in z["rows"]]==ids[:count]
            assert z["manifest_sha256"]==sha(r/"NATURAL_MANIFEST.json")
    sm=load(r/"STRUCTURE_MANIFEST.json")
    p=load(r/"PREDICTIONS.json")
    assert p["manifest_sha256"]==sha(r/"STRUCTURE_MANIFEST.json")
    assert p["prereg_sha256"]==sha(work/"PREREGISTRATION.md")
    assert len({x["id"] for x in sm["rows"]})==152
    for phase,n in (("pilot",24),("confirmation",128)):
        z=load(r/f"STRUCTURE_{phase}.json")
        assert z["prediction_sha256"]==sha(r/"PREDICTIONS.json")
        rows=[x for x in sm["rows"] if x["phase"]==phase]
        assert [x["id"] for x in rows]==[x["id"] for x in z["rows"]]
        assert len(rows)==len(z["rows"])==n
        for a,b in zip(rows,z["rows"],strict=True):
            assert len(set((*a["targets"][0],*a["targets"][1],*a["donor_targets"])))==6
            assert b["max_row_sum_error"]<1e-5
            assert abs(b["scramble_dose"]-b["dose"])<1e-4
            for corners in b["conditions"].values():
                assert all(len(y)==2 for c in corners for v in c for y in v)
    for p in ("NATURAL","STRUCTURE","FACTORIAL","WRITER","BILINEAR"):
        for line in (r/f"{p}_LOCK.sha256").read_text().splitlines():
            expect,name=line.split(maxsplit=1)
            assert sha(work/name)==expect,(p,name)
    if (r/"figures").exists():
        figs=list((r/"figures").glob("*.png"))
        assert len(figs)==5
        for p in figs:
            with Image.open(p) as im:
                im.verify()
            assert p.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
        save(r/"RELEASE_VERIFICATION.json",dict(status="PASS",corrected_natural=512,
            structured_confirmation=128,structured_pilot=24,complete_native_factorials=512,
            writer_support=256,bilinear_replays=256,figure_pairs=5,manifest_sha256=sha(r/"NATURAL_MANIFEST.json"),
            structured_prediction_sha256=sha(r/"PREDICTIONS.json")))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("work",type=Path)
    p.add_argument("previous",type=Path)
    a=p.parse_args()
    verify(a.work,a.previous)

