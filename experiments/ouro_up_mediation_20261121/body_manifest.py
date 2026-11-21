"""Build an untouched confirmation deck for the context-transfer experiment."""
import argparse
from pathlib import Path

from data_protocol import choose,collect_gold,exclusion_lists,multiple_choice_rows,normalized_stem,serialize
from native import load,save,sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument("baseline",type=Path)
    p.add_argument("route",type=Path)
    p.add_argument("up",type=Path)
    p.add_argument("openbook",type=Path)
    p.add_argument("arc",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    e=a.baseline/"openbook_exclusions.json"
    b=a.baseline/"INPUTS.json"
    ids,stems=exclusion_lists([e],None)
    base=load(b)
    decks=[load(a.route),load(a.up)]
    items=[q for n in ("pilot","dev","confirm_arc","confirm_external") for q in base[n]]
    items.extend(q for deck in decks for c in deck["cohorts"].values() for q in c["rows"])
    for q in items:
        ids.add(q["id"])
        stems.add(normalized_stem(q["prompt"].removeprefix("Question: ").split("\nAnswer:")[0]))
    result={}
    for f,path in (("openbookqa",a.openbook),("arc_challenge",a.arc)):
        population,failures=choose(multiple_choice_rows(path,f),f"frozen-context-only-384-{f}",stems,ids,720)
        take=population[:192]
        assert len(take)==192 and not ids&{q.identifier for q in take}
        gold=collect_gold(path,take,"answerKey")
        result[f]=dict(rows=[serialize(q,gold[q.identifier]) for q in take],
                       n=len(take),eligible=len(population),source_sha=sha(path),rejected=failures)
    save(a.out,dict(cohorts=result,primary_prior_sha256=sha(a.up),route_prior_sha256=sha(a.route),
                    baseline_sha256=sha(b),exposure_sha256=sha(e),excluded_ids=len(ids)))


if __name__=="__main__":
    main()

