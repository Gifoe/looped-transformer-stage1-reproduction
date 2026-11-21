"""Validate completed cohorts, unchanged readouts and dose matching."""
import argparse
from pathlib import Path

from PIL import Image

from native import load,save,sha


def keys(path):
    d=load(path)["cohorts"]
    return [q["id"] for f in d.values() for q in f["rows"]]


def verify(root,previous,route):
    r=root/"results"
    source=r/"MANIFEST.json"
    context=r/"BODY_MANIFEST.json"
    old=load(previous/"INPUTS.json")
    past={q["id"] for k in ("dev","pilot","confirm_arc","confirm_external") for q in old[k]}
    past.update(keys(route/"results/NATURAL_MANIFEST.json"))
    past.update(keys(route/"selection_qa/NATURAL_MANIFEST.json"))
    past.update(load(previous/"openbook_exclusions.json")["exclude_ids"])
    a,b=set(keys(source)),set(keys(context))
    assert len(a)==768 and len(b)==384 and not a&b
    assert not (a|b)&past
    for name in ("CONFIRMATION","GENERATION","SUPPORT","BODY"):
        seal=r/f"{name}_LOCK.sha256"
        for line in seal.read_text().splitlines():
            digest,path=line.split(maxsplit=1)
            assert sha(root/path)==digest,(name,path)
    n=0
    max_parity=0.
    max_raw_dose=0.
    max_product_dose=0.
    for p in ("openbookqa","arc_challenge"):
        source_ids=[q["id"] for q in load(source)["cohorts"][p]["rows"]]
        context_ids=[q["id"] for q in load(context)["cohorts"][p]["rows"]]
        for kind,spec,nq,ids in (("UP",source,384,source_ids),("DECODE",source,128,source_ids),
                                  ("SUPPORT",source,128,source_ids),("BODY",context,192,context_ids)):
            q=load(r/f"{kind}_{p}.json")
            assert len(q["rows"])==q["expected"]==nq
            assert [x["id"] for x in q["rows"]]==ids[:nq]
            assert q["manifest_sha256"]==sha(spec)
            if kind=="UP":
                for item in q["rows"]:
                    for z in item["support"].values():
                        max_parity=max(max_parity,z["noop_error"])
                        for k,a1,a2 in (("raw","actual_raw_up","matched_raw_up"),
                                         ("product","actual_product","matched_product")):
                            e=abs(z[a1]-z[a2])/max(z[a1],1e-12)
                            if k=="raw":
                                max_raw_dose=max(max_raw_dose,e)
                            else:
                                max_product_dose=max(max_product_dose,e)
            if kind=="BODY":
                for row in q["rows"]:
                    for side,arms in row["arms"].items():
                        for name,v in arms.items():
                            if name.startswith("body"):
                                assert v["k1_readout_bitwise_held"]
                                assert v["k1_margin_input"]==row["native"][side][0]
                                n+=1
    assert max_parity<5e-5 and max_raw_dose<1e-5 and max_product_dose<1e-5
    figs=list((r/"figures").glob("*.png"))
    assert len(figs)==5
    for p in figs:
        with Image.open(p) as im:
            im.verify()
        assert p.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
    save(r/"RELEASE_VERIFICATION.json",dict(status="PASS",up_n=768,free_generation_n=256,token_support_n=256,
         independent_context_n=384,held_k1_readout_arms=n,figure_pairs=len(figs),
         max_noop_error=max_parity,max_raw_up_dose_error=max_raw_dose,max_product_dose_error=max_product_dose,
         no_previous_exact_id_overlap=True,up_manifest_sha256=sha(source),body_manifest_sha256=sha(context)))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("previous",type=Path)
    p.add_argument("route",type=Path)
    args=p.parse_args()
    verify(args.root,args.previous,args.route)

