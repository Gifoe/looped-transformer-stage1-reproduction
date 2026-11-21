"""Reserve consecutive label-blind items after excluding historical discovery pools."""
import argparse
from pathlib import Path

from data_protocol import choose, collect_gold, exclusion_lists, multiple_choice_rows, normalized_stem, serialize
from native import load, save, sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument("previous",type=Path)
    p.add_argument("arc",type=Path)
    p.add_argument("openbook",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    prev=load(a.previous/"INPUTS.json")
    history=a.previous/"openbook_exclusions.json"
    ids,stems=exclusion_lists([history],None)
    historical_ids=set(ids)
    assert len(historical_ids)==432
    for phase in ("pilot","dev","confirm_arc","confirm_external"):
        for q in prev[phase]:
            ids.add(q["id"])
            stems.add(normalized_stem(q["prompt"].removeprefix("Question: ").split("\nAnswer:")[0]))
    output={}
    for family,path,n in (("openbookqa",a.openbook,320),("arc_challenge",a.arc,192)):
        rows=multiple_choice_rows(path,family)
        eligible,rejected=choose(rows,f"route-content-20261121-{family}",stems,ids,720)
        selected=eligible[:n]
        assert len(selected)==n
        assert not set(x.identifier for x in selected).intersection(ids)
        gold=collect_gold(path,selected,"answerKey")
        output[family]=dict(n=n, source_sha256=sha(path),rejected=rejected,
                          rows=[serialize(x,gold[x.identifier]) for x in selected])
    save(a.out,dict(previous_sha256=sha(a.previous/"INPUTS.json"),historical_exclusions_sha256=sha(a.previous/"openbook_exclusions.json"),
                     excluded_id_count=len(ids),historical_id_count=len(historical_ids),cohorts=output))


if __name__=="__main__":
    main()

