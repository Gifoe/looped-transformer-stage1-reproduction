"""Frozen complete-problem cohorts and semantic counterfactual annotations."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"ouro_expanded_causal_20261118"))
from collect import aligned, digest, dump_line, get_encoding, load_lines, write_json
from factory import benchmark, choose, group, proof_rows, rng_for, TASKS, NAMES
from transformers import AutoTokenizer

GOALS={"relation":1000,"logic":1000,"arithmetic":1000,
       "prontoqa":500,"arc":299,"hellaswag":500}
SPLITS={"fit":300,"discover":100,"confirm":300,"transfer":300}


def generated(data,prior,tok):
    classes=("lempus","gorpus","vumpus","yumpus","sterpus","jompus","wumpus","rompus")
    signatures={(r["task"],r["family"],json.dumps(r["meta"]["signature"]))
                for path in prior for r in load_lines(path) if r["task"] in TASKS}

    def accept(r):
        sig=(r["task"],r["family"],json.dumps(r["meta"]["signature"]))
        if sig in signatures or not aligned(tok,r):
            return False
        signatures.add(sig)
        return True

    for task in TASKS:
        for split,n in SPLITS.items():
            for j in range(n):
                i=3*j+2 if split=="transfer" else 3*(j//2)+j%2
                source,proof=None,None
                if task=="logic":
                    rng=rng_for(f"latent-proof/{split}/{j}")
                    name=rng.choice(NAMES)
                    start,end=rng.sample(classes,2)
                    proof={"chain_of_thought":[f"{name} is a {start}.",f"Every {start} is a {end}.",
                                              f"{name} is a {end}."]}
                    source={"proof_origin":"synthetic PrOntoQA-style proof seed",**proof}
                mode="context" if task=="relation" and (j//2)%2 else "lexical"
                r=group(task,f"org-{split}",i,source,proof,mode,accept)
                r["split"]=split
                r["capture_worlds"]=["r","x","y","d"]+(["n"] if j%3==0 else [])
                for n,w in r["worlds"].items():
                    w["semantic"]={"intermediate":r["intermediate_options"][w["mid"]],
                                   "final":r["options"][w["gold"]],
                                   "first_factor":int(n in ("x","d","m")),
                                   "map_factor":int(n in ("y","d","m")),
                                   "nuisance":int(n in ("n","m"))}
                yield r


def official(data):
    import pyarrow.parquet as pq
    with ZipFile(data/"prontoqa.zip") as z:
        members=sorted({n for n in z.namelist() if "ProofsOnly" in n and n.endswith(".json")})
    seen=set()
    pool=[]
    for member in members:
        for key,p in proof_rows(data,member):
            statement=p["query"].removeprefix("Prove: ").removesuffix(".")
            sig=p["question"],statement
            if sig in seen or " is " not in statement:
                continue
            seen.add(sig)
            negate=(statement.replace(" is not "," is ",1) if " is not " in statement
                    else statement.replace(" is "," is not ",1))
            options=[statement,negate]
            rng_for(f"org/{key}").shuffle(options)
            pool.append(benchmark("prontoqa",key,p["question"]+"\nWhich statement follows?",
                                  options,options.index(statement),p["chain_of_thought"]))
    assert len(pool)>=500
    for row in choose(pool,500,"org-prontoqa-scene-selection-v1"):
        yield row
    arc=pq.read_table(data/"arc.parquet").to_pylist()
    assert len({r["id"] for r in arc})==len(arc)==299
    for r in arc:
        yield benchmark("arc",r["id"],"Question: "+r["question"],r["choices"]["text"],
                        r["choices"]["label"].index(r["answerKey"]))
    h=pq.read_table(data/"hellaswag.parquet").to_pylist()
    unique={str(r["ind"]):r for r in h}
    for r in choose(unique.values(),500,"org-hellaswag-v1"):
        yield benchmark("hellaswag",str(r["ind"]),"Choose the most likely continuation.\n"+r["ctx"],
                       r["endings"],int(r["label"]))


def balance_slots(groups, tok, study):
    """Assign emitted answer slots after the semantic scene already exists."""
    old=Counter()
    new=Counter()
    for g in groups:
        if g["task"] not in TASKS:
            continue
        options=g["options"]
        order=list(range(len(options)))
        rng_for("independent-output-slot/org-v2/"+g["id"]).shuffle(order)
        balanced=[options[i] for i in order]
        for name,w in g["worlds"].items():
            old[g["task"],name,w["gold"]]+=1
            start=w["roles"]["option"][0][0]
            end=w["roles"]["option"][-1][1]
            prefix=w["prompt"][:start]
            suffix=w["prompt"][end:]
            intervals=[]
            block=""
            for i,option in enumerate(balanced):
                line=f"{chr(65+i)}. {option}\n"
                a=start+len(block)
                block+=line
                intervals.append((a,a+len(line)))
            assert len(block)==end-start
            w["prompt"]=prefix+block+suffix
            w["roles"]["option"]=intervals
            w["gold"]=order.index(w["gold"])
            assert balanced[w["gold"]]==w["semantic"]["final"]
            new[g["task"],name,w["gold"]]+=1
        g["options"]=balanced
        g["emission_slot_permutation"]=order
        assert aligned(tok,g)
    counts=lambda c:{"|".join(map(str,k)):v for k,v in sorted(c.items())}
    assert all(sum(1 for k,v in new.items() if k[:2]==(t,"r") and v>0)>=2 for t in TASKS)
    write_json(study/"slot_balance_audit.json",{"previous_counts":counts(old),"balanced_counts":counts(new),
        "operator":"fixed semantic scenes; independently permute emitted choice text and remap gold for all worlds"})
    return groups


def main():
    p=argparse.ArgumentParser()
    p.add_argument("study",type=Path)
    p.add_argument("--sources",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--prior",type=Path,nargs="+",required=True)
    p.add_argument("--rebalance-existing",type=Path)
    a=p.parse_args()
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
    original=(load_lines(a.rebalance_existing) if a.rebalance_existing else
              list(generated(a.sources,a.prior,tok))+list(official(a.sources)))
    groups=balance_slots(original,tok,a.study)
    for g in groups:
        if "capture_worlds" not in g:
            g["capture_worlds"]=["r"]
    assert Counter(g["task"] for g in groups)==Counter(GOALS)
    ids=Counter(g["id"] for g in groups)
    duplicates=[key for key,n in ids.items() if n>1]
    assert not duplicates, duplicates[:10]
    path=a.study/"groups.jsonl"
    with path.open("x",encoding="utf-8") as f:
        for g in groups:
            dump_line(f,g)
    count=Counter(f"{g['task']}/{g['split']}" for g in groups)
    write_json(a.study/"manifest.json",{"source_base":"5502634381e8a0c532f0a91a9b5cafe8bd7617b4",
              "group_sha256":digest(path),"generator_sha256":digest(Path(__file__)),
              "upstream_sources":{n:digest(a.sources/n) for n in ("prontoqa.zip","arc.parquet","hellaswag.parquet")},
              "prior_manifest_sha256":[digest(f) for f in a.prior],"groups":dict(count),
              "prior_unbalanced_manifest_sha256":digest(a.rebalance_existing) if a.rebalance_existing else None,
              "total_forwards":sum(len(g["capture_worlds"]) for g in groups),
              "template_transfer":"family 2 entirely absent from fit/discover/confirm"})
    print(dict(count),flush=True)


if __name__=="__main__":
    main()
