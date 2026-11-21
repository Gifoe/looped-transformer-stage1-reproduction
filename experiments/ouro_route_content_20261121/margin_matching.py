"""Descriptive sensitivity to proximity to the K2 answer boundary."""
import argparse
from pathlib import Path

from analysis import metric
from native import load, save


def matched(cases,meta):
    size={q["id"]:len(q["prompt"]) for q in meta}
    early=sorted((q for q in cases if q["stratum"]=="WC"),key=lambda q:q["id"])
    pool={q["id"]:q for q in cases if q["stratum"]=="CC"}
    pairs=[]
    for response in early:
        a=response["native_margin"][1]
        n=size[response["id"]]
        options=[q for q in pool.values() if abs(q["native_margin"][1]-a)<=.15 and abs(size[q["id"]]-n)<=100]
        if not options:
            continue
        partner=min(options,key=lambda q:(abs(q["native_margin"][1]-a)+.001*abs(size[q["id"]]-n),q["id"]))
        del pool[partner["id"]]
        pairs.append((response,partner))
    result={}
    for op in ("drop","rollback","position0","rest"):
        ref="matched_writer" if op=="drop" else ("other_heads" if op=="rollback" else None)
        def delta(q):
            y=q["native_margin"][1] if ref is None else q["arm"][ref]["score"]
            return q["arm"][op]["score"]-y
        result[op]=dict(control=ref or "native",wc=metric([delta(x) for x,y in pairs]),
                        cc=metric([delta(y) for x,y in pairs]),
                        wc_minus_cc=metric([delta(x)-delta(y) for x,y in pairs]))
    return dict(n=len(pairs),available_wc=len(early),available_cc=sum(q["stratum"]=="CC" for q in cases),
       assignment="greedy in fixed WC ID order, no replacement, K2 margin caliper 0.15, character count caliper 100",
       pairs=[dict(wc=x["id"],cc=y["id"],margin_delta=x["native_margin"][1]-y["native_margin"][1],
          chars_delta=size[x["id"]]-size[y["id"]]) for x,y in pairs],summary=result)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    a=p.parse_args().out
    s=load(a/"SUMMARY.json")["natural"]
    m=load(a/"NATURAL_MANIFEST.json")["cohorts"]
    y={f:matched(s[f]["mean"]["cases"],m[f]["rows"]) for f in m}
    save(a/"MARGIN_MATCHING.json",y)


if __name__=="__main__":
    main()

