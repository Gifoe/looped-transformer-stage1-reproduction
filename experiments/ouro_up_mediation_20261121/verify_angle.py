"""Extend the release receipt with the disjoint matched-orientation experiment."""
import argparse
from pathlib import Path
from PIL import Image

from native import load,save,sha


def ids(m):
    return {q["id"] for k in load(m)["cohorts"].values() for q in k["rows"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("baseline",type=Path)
    p.add_argument("route",type=Path)
    a=p.parse_args()
    r=a.root/"results"
    m=r/"ANGLE_MANIFEST.json"
    prior=load(a.baseline/"INPUTS.json")
    past={q["id"] for v in ("dev","pilot","confirm_arc","confirm_external") for q in prior[v]}
    past.update(load(a.baseline/"openbook_exclusions.json")["exclude_ids"])
    for file in (a.route/"results/NATURAL_MANIFEST.json",a.route/"selection_qa/NATURAL_MANIFEST.json",
                 r/"MANIFEST.json",r/"BODY_MANIFEST.json"):
        past.update(ids(file))
    assert not ids(m)&past
    assert len(ids(m))==288
    for f in (r/"ANGLE_LOCK.sha256").read_text().splitlines():
        h,path=f.split(maxsplit=1)
        assert sha(a.root/path)==h
    z=load(r/"ANGLE_SUMMARY.json")
    h=sum(q["audit"]["held_k1"] for q in z["families"].values())
    err=max(q["audit"]["max_token_norm_error"] for q in z["families"].values())
    dd=max(q["audit"]["max_per_token_dose_error"] for q in z["families"].values())
    assert h==2304 and err<2e-5 and dd<3e-5
    paths=list((r/"figures").glob("*.png")); assert len(paths)==6
    for f in paths:
        with Image.open(f) as p:
            p.verify()
        assert f.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
    proof=load(r/"RELEASE_VERIFICATION.json")
    assert proof["status"]=="PASS"
    save(r/"RELEASE_VERIFICATION.json",dict(proof,angular_n=288,total_distinct_questions=1440,
               angular_manifest_sha256=sha(m),angular_held_k1_readouts=h,max_norm_error=err,max_angle_dose_error=dd,figure_pairs=6))


if __name__=="__main__":
    main()

