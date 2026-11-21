"""Freeze new question decks before any evaluation under the target mechanism."""
import argparse
from pathlib import Path

from data_protocol import choose, collect_gold, exclusion_lists, multiple_choice_rows, normalized_stem, serialize
from native import load,save,sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument("prior_native",type=Path)
    p.add_argument("prior_route",type=Path)
    p.add_argument("arc",type=Path)
    p.add_argument("openbook",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    exclusions=a.prior_native/"openbook_exclusions.json"
    first=a.prior_native/"INPUTS.json"
    route=a.prior_route/"results/NATURAL_MANIFEST.json"
    ids,stems=exclusion_lists([exclusions],None)
    assert len(ids)==432
    earlier1=load(first)
    earlier2=load(route)["cohorts"]
    previous=[q for key in ("pilot","dev","confirm_arc","confirm_external") for q in earlier1[key]]
    previous.extend(q for z in earlier2.values() for q in z["rows"])
    for q in previous:
        ids.add(q["id"])
        stems.add(normalized_stem(q["prompt"].removeprefix("Question: ").split("\nAnswer:")[0]))
    rows={}
    for name,path in (("openbookqa",a.openbook),("arc_challenge",a.arc)):
        elig,drop=choose(multiple_choice_rows(path,name),f"corrective-up-{name}-211121",stems,ids,720)
        n=384
        chosen=elig[:n]
        assert len(chosen)==n
        assert not set(q.identifier for q in chosen)&ids
        gold=collect_gold(path,chosen,"answerKey")
        rows[name]=dict(n=n,rows=[serialize(q,gold[q.identifier]) for q in chosen],
                         source_sha256=sha(path),rejects=drop,eligible=len(elig))
    keys=[q["id"] for v in rows.values() for q in v["rows"]]
    assert len(set(keys))==len(keys)==768
    save(a.out,dict(protocol="ordered by frozen hash; no scoring used for admission",
                  exclusions={str(x.name):sha(x) for x in (first,route,exclusions)},
                  n_excluded_ids=len(ids),n_old_stems=len(stems),cohorts=rows))


if __name__=="__main__":
    main()

