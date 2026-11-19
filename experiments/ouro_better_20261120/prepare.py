"""Deterministic eligible-population and disjoint split manifest; read labels only in runners."""
import argparse
from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pyarrow.parquet as pq

SALT = "ouro-better-v1"
ARC_SHA = "fdb98c79578d61bdf502e2ade3253c71154dd11f"
OPENBOOK_REVISION = "388097ea7776314e93a529163e0fea805b8a6454"
OPENBOOK_SHA = "98148f8a54e62eb862346a75192d5fb824d6cbb68f2f59aecd793d39ecb5cd8b"


def key(*fields):
    return hashlib.sha256("\0".join((SALT,*fields)).encode("utf8")).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf8")


def canon(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[^\W_]+",text))


def load(path, source, labels=False, ids=None):
    question = "question" if source=="arc" else "question_stem"
    columns = ["id",question,"choices"]+ (["answerKey"] if labels else [])
    filters = None if ids is None else [("id","in",sorted(ids))]
    rows = pq.read_table(path,columns=columns,filters=filters).to_pylist()
    seen = {r["id"] for r in rows}
    assert len(seen)==len(rows)
    for r in rows:
        r["question"]=r[question]
        opts = r["choices"]["text"]
        assert len(opts)==len(r["choices"]["label"]) and 3<=len(opts)<=5
        assert all(text.strip() for text in opts) and r["question"].strip()
        r["content_key"]=canon(r["question"])
        r["opt_key"]=tuple(sorted(canon(text) for text in opts))
        r["full_key"]=key(r["content_key"],*r["opt_key"])
        if labels:
            r["gold"]=r["choices"]["label"].index(r["answerKey"])
    return rows


def clusters(records):
    root = list(range(len(records)))
    def find(i):
        while root[i]!=i:
            root[i]=root[root[i]]
            i=root[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        root[max(a,b)] = min(a,b)
    stems, optsets = {}, defaultdict(list)
    for i,r in enumerate(records):
        stem = r["content_key"]
        if stem in stems:
            union(i,stems[stem])
        else:
            stems[stem] = i
        o = r["opt_key"]
        for j in optsets[o]:
            other=records[j]["content_key"]
            if min(len(stem),len(other))<24 or abs(len(stem)-len(other)) > .12*max(len(stem),len(other)):
                continue
            if SequenceMatcher(None,stem,other,autojunk=False).ratio()>=.90:
                union(i,j)
        optsets[o].append(i)
    groups=defaultdict(list)
    for i,r in enumerate(records):
        groups[find(i)].append(r)
    return list(groups.values())


def main():
    p=argparse.ArgumentParser()
    for n in ("arc","openbook","prior_freeze","demos","prior_openbook_exclusions","prior_arc_targeted","decision","out"):
        p.add_argument(n,type=Path)
    args=p.parse_args()
    arc,transfer=(load(args.arc,"arc"),load(args.openbook,"openbook"))
    assert len(arc)==1119 and len(transfer)==4957 and sha(args.openbook)==OPENBOOK_SHA
    for r in arc:
        r["source"]="arc"
    for r in transfer:
        r["source"]="openbook"
    demo=json.loads(args.demos.read_bytes())
    past=json.loads(args.prior_freeze.read_bytes())
    old_ids=set(past["selection"]["train"]["main"])
    demo_stems={canon(x["stem"]) for x in demo}
    targeted=json.loads(args.prior_arc_targeted.read_bytes())
    targeted_ids=set(targeted["exclude_ids"])
    assert len(targeted_ids)==351 and old_ids <= targeted_ids
    assert targeted_ids <= {r["id"] for r in arc}
    old_stems=demo_stems | {r["content_key"] for r in arc if r["id"] in targeted_ids}
    assert len(old_ids)==64 and len(demo)==25
    prior_ob=json.loads(args.prior_openbook_exclusions.read_bytes())
    prior_ob_ids=set(prior_ob["exclude_ids"])
    assert len(prior_ob_ids)==35 and prior_ob["parquet_sha256"]==OPENBOOK_SHA
    additional_ob_near_duplicate_ids={"12-297","14-595"}
    assert (prior_ob_ids | additional_ob_near_duplicate_ids) <= {r["id"] for r in transfer}
    combined=clusters(arc+transfer)
    arc_eligible, ob_eligible, rejected= [], [], defaultdict(list)
    arc_fit_only_clusters=set()
    prior_demo_ids=[r["id"] for r in arc if r["content_key"] in {canon(x["stem"]) for x in demo}]
    assert len(prior_demo_ids)==25
    for g in combined:
        names={r["source"] for r in g}
        if len(names)>1:
            rejected["cross_source_overlap"].extend(f'{r["source"]}:{r["id"]}' for r in g)
            continue
        if any(r["source"]=="openbook" and r["id"] in (prior_ob_ids | additional_ob_near_duplicate_ids) for r in g):
            rejected["historical_openbook_eval_stem_match"].extend(f'{r["source"]}:{r["id"]}' for r in g)
            continue
        previous_target=any(r["id"] in targeted_ids or r["content_key"] in old_stems for r in g)
        if any(r["content_key"] in demo_stems for r in g):
            rejected["prior_demonstration"].extend(f'{r["source"]}:{r["id"]}' for r in g)
            continue
        # One canonical representative per duplicate cluster in every new split.
        sorted_group=sorted(g,key=lambda r:key("cluster_member",r["source"],r["id"]))
        representative=sorted_group[0]
        dataset=representative["source"]
        data_key=key(*(r["source"]+":"+r["id"] for r in sorted_group))
        rejected["extra_group_members"].extend(f'{r["source"]}:{r["id"]}' for r in sorted_group[1:])
        if dataset=="arc" and previous_target:
            arc_fit_only_clusters.add(data_key)
        (arc_eligible if dataset=="arc" else ob_eligible).append((data_key,representative))
    arc_eligible.sort(key=lambda a:a[0]); ob_eligible.sort(key=lambda a:a[0])
    nfit=512
    confirming=[(c,r) for c,r in arc_eligible[nfit:] if c not in arc_fit_only_clusters]
    if len(confirming)<384:
        confirming=[(c,r) for c,r in arc_eligible if c not in arc_fit_only_clusters]
        reserved={c for c,r in confirming[:384]}
        fit=[(c,r) for c,r in arc_eligible if c not in reserved][:nfit]
    else:
        fit=arc_eligible[:nfit]
    assert len(confirming)>=384 and len(fit)==nfit and len(ob_eligible)>=384
    batches={
        "fit":fit,"dev":confirming[:128],
        "diagnostic":confirming[128:384],"transfer":ob_eligible[:384],
    }
    assert [len(batches[k]) for k in ("fit","dev","diagnostic","transfer")]==[nfit,128,256,384]
    result={
        "source": {"arc":{"revision":ARC_SHA,"split":"train","rows":len(arc),"sha256":sha(args.arc)},
                   "openbookqa":{"revision":OPENBOOK_REVISION,"config":"main","split":"train",
                                 "rows":len(transfer),"sha256":sha(args.openbook)}},
        "historical_exposure":"ARC Train previously scored in full; supports within-run dev/diagnostic only. "
                              "OpenBookQA main Train is sole transfer confirmation.",
        "available_eligible_arc":len(arc_eligible),"available_eligible_openbookqa":len(ob_eligible),
        "near_duplicate":"Exact canonical stem across choice sets, or canonical stem min-length 24 with "
                         "SequenceMatcher >= .90 and identical unordered canonical options.",
        "salt":SALT, "prior_development_ids":sorted(old_ids),"prior_demonstration_ids":sorted(prior_demo_ids),
        "rejected":dict(rejected),
        "cohorts":{name:[{"id":r["id"],"cluster":c,"content_sha256":r["full_key"]} for c,r in values]
                   for name,values in batches.items()},
        "implementation_sha256":sha(Path(__file__)),"decision_sha256":sha(args.decision),
        "prior_freeze_sha256":sha(args.prior_freeze), "demos_sha256":sha(args.demos),
        "prior_openbook_exclusions_sha256":sha(args.prior_openbook_exclusions),
        "prior_openbook_exclusions":sorted(prior_ob_ids),
        "additional_openbook_near_duplicates":sorted(additional_ob_near_duplicate_ids),
        "prior_arc_targeted_sha256":sha(args.prior_arc_targeted),
        "prior_arc_targeted_fit_only_ids":sorted(targeted_ids),
        "fit_only_population_clusters":sorted(arc_fit_only_clusters),
        "label_data_read":False,
    }
    assert len({r["id"] for a in result["cohorts"].values() for r in a})==sum(map(len,batches.values()))
    assert all(r["id"] not in targeted_ids and r["cluster"] not in arc_fit_only_clusters
               for name in ("dev","diagnostic") for r in result["cohorts"][name])
    dump(args.out,result)
    print("eligible",len(arc_eligible),len(ob_eligible),"split",[(k,len(v)) for k,v in batches.items()],
          "rejected",[(k,len(v)) for k,v in rejected.items()],flush=True)


if __name__=="__main__":
    main()

