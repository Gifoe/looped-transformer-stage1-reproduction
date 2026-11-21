"""Reserve disjoint new questions for a norm-and-angle-matched transfer."""
import argparse
from pathlib import Path

from data_protocol import choose,collect_gold,exclusion_lists,multiple_choice_rows,normalized_stem,serialize
from native import load,save,sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument("pre",type=Path)
    p.add_argument("route",type=Path)
    p.add_argument("up",type=Path)
    p.add_argument("book",type=Path)
    p.add_argument("arc",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    exp=a.pre/"openbook_exclusions.json"
    decks=[a.route/"results/NATURAL_MANIFEST.json",a.route/"selection_qa/NATURAL_MANIFEST.json",
           a.up/"MANIFEST.json",a.up/"BODY_MANIFEST.json"]
    e,stems=exclusion_lists([exp],None)
    old=load(a.pre/"INPUTS.json")
    items=[q for k in ("pilot","dev","confirm_arc","confirm_external") for q in old[k]]
    items.extend(q for p in decks for c in load(p)["cohorts"].values() for q in c["rows"])
    for q in items:
        e.add(q["id"])
        stems.add(normalized_stem(q["prompt"].removeprefix("Question: ").split("\nAnswer:")[0]))
    y={}
    for family,path,n in (("openbookqa",a.book,192),("arc_challenge",a.arc,96)):
        q,j=choose(multiple_choice_rows(path,family),f"orientation-only-locked-{family}-211121",stems,e,720)
        assert len(q)>=n,(family,len(q),n)
        q=q[:n]
        assert not e & {x.identifier for x in q}
        gold=collect_gold(path,q,"answerKey")
        y[family]=dict(n=n,rows=[serialize(x,gold[x.identifier]) for x in q],source_sha256=sha(path),rejected=j)
    save(a.out,dict(cohorts=y,old_manifests={p.name:sha(p) for p in decks if p.parent==a.up or p.parent.name=="results"},
                    historical_exclusions_sha256=sha(exp),excluded_ids=len(e)))


if __name__=="__main__":
    main()

