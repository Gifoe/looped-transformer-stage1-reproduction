"""Generate structurally stratified, answer-balanced groups with disjoint signatures."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"ouro_latent_organization_20261118"))
from manifest_org import digest, dump_line, load_lines, write_json
from capture_composition import TASKS, WORLDS, matching
from collect import get_encoding
import factory
from transformers import AutoTokenizer

NAMES=("Adam","Anna","Carol","Claire","Daniel","Emma","Ethan","Eva",
       "Helen","James","Jane","John","Julia","Mary","Peter","Sarah")
LOGIC=("lempus","gorpus","vumpus","yumpus","sterpus","jompus","wumpus","rompus")
ALIAS=("west","east","north","south","central")


def numeric(rng,family,extended):
    a=rng.randint(34,67)
    x=rng.sample(range(1,7),2)
    nominal=rng.randrange(8,15)
    s=(1,-1,1)[family]
    t=1 if family!=2 else -1
    v=[a+s*b for b in x]
    steps=[nominal,nominal+t*(v[0]-v[1])]
    def solve(i,j):
        return str(v[i]+t*steps[j])
    options=sorted({solve(i,j) for i in (0,1) for j in (0,1)})
    def render(i,j,n,mode):
        p=factory.Text()
        p.add(f"A separate example uses the number {6+n}.\n","noise")
        opener=("Start with","A tray has","Start at")[family]
        prefix=f"{opener} {a}"+(" coins. " if family==1 else ". ")
        p.add(prefix+("Add ","Remove ","Move ")[family])
        p.add(str(x[i]),"signal")
        p.add((".\n"," coins.\n"," steps forward.\n")[family],"bridge")
        boundary=len(p.value)
        if extended:
            p.add(f"Next move 2 steps forward.\nThen move {steps[j]+2} steps backward.\n","map")
        else:
            line=(f"Next add {steps[j]}.\n",f"Next put {steps[j]} coins on the tray.\n",
                  f"Next move {steps[j]} steps backward.\n")[family]
            p.add(line,"map")
        p.add("What is the final number?","query")
        return p,boundary
    return render,options,list(map(str,v)),solve,lambda i:str(v[i]),"After the first step?",{
        "cardinality":3,"signature":[a,sorted(x),nominal,sorted(steps)],"depth":3 if extended else 2}


def add_depth(render,task,meta):
    def extended(i,j,n,mode):
        text,start=render(i,j,n,mode)
        stop=text.regions["map"][-1][1]
        lines=text.value[start:stop].splitlines(keepends=True)
        replacement=factory.Text()
        replacement.value=text.value[:start]
        replacement.regions={k:v for k,v in text.regions.items() if v[-1][1]<=start}
        aliases=dict(zip(meta["signature"][4 if task=="relation" else 2],ALIAS))
        for line in lines:
            if task=="relation":
                match=re.fullmatch(r"The (\w+) depot serves (\w+)\.\n",line)
                if match:
                    depot,city=match.groups()
                    alias=aliases[depot]
                    line=f"The {depot} depot refers to the {alias} district.\nThe {alias} district serves {city}.\n"
            else:
                match=re.fullmatch(r"Every (\w+) gets (a \w+)\.\n",line)
                assert match
                cat,outcome=match.groups()
                alias=aliases[cat]
                line=f"Every {cat} holds the {alias} badge.\nEveryone with the {alias} badge gets {outcome}.\n"
            replacement.add(line,"map")
        replacement.add(text.value[stop:],"query")
        return replacement,start
    return extended


def build(tok,task,family,devset,index,signatures):
    split="fresh_same" if devset else "fresh_structure"
    depth=not devset and index>=40
    mode="context" if task=="relation" and index%2 else "lexical"
    for j in range(150):
        gid=f"{task}/{split}/{index:03}"
        rng=factory.rng_for(f"composition-independent-v3/{gid}/{j}")
        if task=="relation":
            render,opts,mids,solve,mid,question,meta=factory.relation(rng,family,None)
        elif task=="logic":
            n=rng.choice(NAMES)
            a,b=rng.sample(LOGIC,2)
            proof={"chain_of_thought":[f"{n} is a {a}.",f"Every {a} is a {b}.",f"{n} is a {b}."]}
            render,opts,mids,solve,mid,question,meta=factory.logic(rng,family,proof)
        else:
            render,opts,mids,solve,mid,question,meta=numeric(rng,family,depth)
        sig=(task,json.dumps(meta["signature"],sort_keys=True))
        if sig in signatures:
            continue
        if depth and task!="arithmetic":
            render=add_depth(render,task,meta)
        golds=tuple(solve(a,b) for a,b in ((0,0),(1,0),(0,1),(1,1)))
        intermediates=(mid(0),mid(1))
        options=list(opts)
        mid_options=list(mids)
        factory.rng_for(f"unbiased-emission/{gid}").shuffle(options)
        assert golds[0]==golds[3] and golds[0] not in golds[1:3]
        worlds={}
        for w,a,b,n in (("r",0,0,0),("x",1,0,0),("y",0,1,0),("d",1,1,0),("n",0,0,1)):
            body,start=render(a,b,n,mode)
            pre=body.value[:start]
            body.choose(options)
            probe=factory.Text()
            probe.value=pre+question
            probe.choose(mid_options)
            worlds[w]={"prompt":body.value,"roles":body.regions,"prefix_chars":start,
                       "gold":options.index(solve(a,b)),"mid":mid_options.index(mid(a)),"probe":probe.value,
                       "semantic":{"intermediate":mid(a),"final":solve(a,b),"first_factor":a,"map_factor":b,
                                   "nuisance":n}}
        g=dict(id=gid,task=task,split=split,family=family,mode=mode,
               difficulty="extra_hop" if depth else ("same_family" if devset else "unseen_family"),
               options=options,intermediate_options=mid_options,worlds=worlds,meta=meta,attempt=j)
        try:
            encoded={w:get_encoding(tok,g,worlds[w]) for w in worlds}
            matching(g,encoded)
        except (AssertionError,ValueError,RuntimeError):
            continue
        signatures.add(sig)
        return g
    raise RuntimeError(f"No aligned and distinct realization: {task} {split} {index}")


def run(checkpoint,out,prior):
    out.mkdir(parents=True,exist_ok=True)
    factory.NAMES=NAMES
    tok=AutoTokenizer.from_pretrained(checkpoint,trust_remote_code=True,local_files_only=True)
    signatures={(g["task"],json.dumps(g["meta"]["signature"],sort_keys=True)) for p in prior
                for g in load_lines(p) if g["task"] in TASKS and "signature" in g.get("meta",{})}
    original=len(signatures)
    rows=[]
    for task in TASKS:
        for same in (True,False):
            for i in range(80):
                rows.append(build(tok,task,i%2 if same else 2,same,i,signatures))
    assert len(signatures)-original==len(rows)==480
    file=out/"fresh_groups.jsonl"
    with file.open("x",encoding="utf-8") as f:
        for g in rows:
            dump_line(f,g)
    def counts(values):
        return dict(Counter("/".join(map(str,x)) for x in values))
    audit={"group_sha256":digest(file),"generator_sha256":digest(Path(__file__)),
           "prior_hashes":[digest(p) for p in prior],"old_signatures":original,"overlap":0,
           "new_groups":counts((g["task"],g["split"],g["difficulty"]) for g in rows),
           "choice_distribution":counts((g["task"],g["split"],w,g["worlds"][w]["gold"])
                                       for g in rows for w in WORLDS)}
    write_json(out/"generation.json",audit)
    print(audit["new_groups"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("prior",type=Path,nargs="+")
    a=p.parse_args()
    run(a.checkpoint,a.out,a.prior)
