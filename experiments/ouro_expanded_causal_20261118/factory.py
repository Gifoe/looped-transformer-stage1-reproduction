"""Deterministic, structurally varied tasks and bound counterfactual worlds."""
import hashlib
import json
from pathlib import Path
import random
import re
from zipfile import ZipFile

TASKS = ("relation", "logic", "arithmetic")
NAMES = ("Alice", "Ben", "Clara", "Dan", "Eve", "Frank", "Grace", "Hank",
         "Iris", "Jack", "Kate", "Luke", "Maya", "Nick", "Olivia", "Paul")
COLORS = ("red", "blue", "green", "black", "white", "pink", "gold", "gray", "brown")
CITIES = ("Rome", "Paris", "London", "Boston", "Dallas", "Miami", "Denver", "Berlin", "Tokyo")
PROOFS = ("1hop_ProofsOnly_2testhops_random_noadj.json",
          "2hop_ProofsOnly_random_noadj.json", "4hop_ProofsOnly_random_noadj.json")
LOGIC_SOURCES = (
    "2hop_ProofsOnly_4shot_3testhops_random_noadj.json",
    "2hop_ProofsOnly_4shot_4testhops_random_noadj.json",
    "2hop_ProofsOnly_4shot_5testhops_random_noadj.json",
    "2hop_ProofsOnly_4shot_random_noadj.json",
)


def rng_for(s):
    return random.Random(int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big"))


class Text:
    def __init__(self):
        self.value = ""
        self.regions = {}

    def add(self, text, role=None):
        start = len(self.value)
        self.value += text
        if role is not None:
            self.regions.setdefault(role, []).append((start, len(self.value)))

    def choose(self, options):
        self.add("\n")
        for i, option in enumerate(options):
            self.add(f"{chr(65+i)}. {option}\n", "option")
        self.add("Answer:", "read")
        return self


def relation(rng, family, source):
    n = (3, 4, 5)[family]
    people, colors, cities = rng.sample(NAMES, n), rng.sample(COLORS, n), rng.sample(CITIES, n)
    distractors = rng.sample([c for c in COLORS if c not in colors],2)
    depots = rng.sample(("Oak","Pine","Maple","Elm","Ash","Birch","Cedar"), n)
    j = rng.randrange(1, n)
    order1, order2 = rng.sample(range(n), n), rng.sample(range(n), n)
    kind = ("box", "folder", "tag")[family]
    target = people[0]
    raw = list(range(n))
    switched = raw.copy()
    switched[0], switched[j] = switched[j], switched[0]

    def render(a,b,noise,mode="lexical"):
        t = Text()
        t.add(f"Find the city for {target}.\n", "early_query")
        for i in order1:
            indexes = switched if a else raw
            owner = people[indexes[i] if mode == "context" else i]
            value = colors[indexes[i] if mode == "lexical" else i]
            to_edit = i in (0,j)
            if family == 0:
                t.add(owner, "bind" if to_edit else "unrelated")
                t.add(" received the ")
                t.add(value, f"payload_{i}")
                t.add(" box.\n")
            elif family == 1:
                t.add("Dispatch: person ")
                t.add(owner, "bind" if to_edit else "unrelated")
                t.add(" filed under folder ")
                t.add(value, f"payload_{i}")
                t.add(".\n")
            else:
                t.add("The clerk assigned ")
                t.add(owner, "bind" if to_edit else "unrelated")
                t.add(" a ")
                t.add(value, f"payload_{i}")
                t.add(" tag.\n")
        t.add(f"An empty sample {kind} is {distractors[noise]}.\n", "noise")
        prefix = len(t.value)
        for i in order2:
            slot = switched[i] if b else i
            if family == 0:
                t.add(f"The {colors[i]} {kind} routes to {cities[slot]}.\n", "map")
            elif family == 1:
                t.add(f"{cities[slot]} receives the {colors[i]} folder.\n", "map")
            else:
                t.add(f"The {colors[i]} tag routes to the {depots[slot]} depot.\n", "map")
        if family == 2:
            for i in order1:
                t.add(f"The {depots[i]} depot serves {cities[i]}.\n", "map")
        t.add(f"Which city should {target} go to?", "query")
        return t,prefix

    return (render, cities, colors, lambda a,b:cities[(switched if b else raw)[j if a else 0]],
            lambda a:colors[j if a else 0],
            f"Which color {kind} was assigned to {target}?", {"queried":0,"other":j,"cardinality":n,"signature":[people,colors,cities,j,depots if family==2 else []]})


def logic(rng, family, source):
    first = re.fullmatch(r"(\w+) is an? (\w+)\.",source["chain_of_thought"][0])
    second = re.fullmatch(r"(\w+) is an? (\w+)\.",source["chain_of_thought"][2])
    name, start = first.groups()
    assert second[1] == name
    bridge = second[2]
    alternative = rng.choice(sorted(set(("lempus","gorpus","vumpus","yumpus","sterpus","jompus",
                                        "wumpus","rompus","brimpus","shumpus"))-{start,bridge}))
    category = [bridge, alternative]
    outcomes = rng.sample(("a key","a medal","a flag","a coin","a ring","a cup"), 2)
    tags = rng.sample(COLORS, 4)
    if rng.randrange(2):
        category.reverse()
    order = rng.sample(range(2),2)

    def render(a,b,noise,mode="lexical"):
        t=Text()
        t.add(f"Apply the rules to {name}.\n", "early_query")
        t.add(f"{name} has a ")
        t.add(tags[a], "signal")
        t.add(" card.\n")
        t.add(f"A spare card on the desk is {tags[2+noise]}.\n", "noise")
        if family == 0:
            t.add(f"{name} is a {start}.\n")
        elif family == 1:
            t.add(f"{name} is a {start} enrolled in the club.\n")
        else:
            t.add(f"{name} is a {start} on duty.\n")
        for i in order:
            if family == 0:
                s=f"Every {start} with a {tags[i]} card is a "
            elif family == 1:
                s=f"An enrolled {start} with a {tags[i]} card qualifies as a "
            else:
                s=f"If a {start} on duty carries a {tags[i]} card, that person is a "
            t.add(s)
            t.add(category[i], "bridge")
            t.add(".\n")
        prefix=len(t.value)
        for i in order:
            s=(f"Every {category[i]} gets {outcomes[i^b]}.\n" if family != 1
               else f"Being a {category[i]} means one receives {outcomes[i^b]}.\n")
            t.add(s, "map")
        t.add(f"What does {name} get?", "query")
        return t,prefix

    return (render,outcomes,category,lambda a,b:outcomes[a^b],lambda a:category[a],
            f"What role does {name} qualify for?", {"cardinality":2,"signature":[name,start,category,outcomes,tags[:2]]})


def arithmetic(rng, family, source):
    a = rng.randint(4,18)
    if family == 0:
        operands = sorted(rng.sample(range(1,6),2))
        values = [a+x for x in operands]
        delta = rng.randint(5,10)
        op1 = lambda x:f"Start with {a}. Add {x}.\n"
        op2 = lambda d:f"Next add {d}.\n"
        f=lambda x,d:x+d
        factor=1
    elif family == 1:
        operands = sorted(rng.sample(range(1,6),2))
        values=[a+4-x for x in operands]
        delta=rng.randint(5,10)
        op1=lambda x:f"A tray has {a+4} coins. Remove {x} coins.\n"
        op2=lambda d:f"Next put {d} coins on the tray.\n"
        f=lambda x,d:x+d
        factor=1
    else:
        operands = sorted(rng.sample(range(1,5),2))
        values=[a+x for x in operands]
        delta=rng.randint(3,6)
        op1=lambda x:f"Start at {a} on a number line. Move {x} steps forward.\n"
        op2=lambda d:f"Next move {d} steps backward.\n"
        f=lambda x,d:x-d
        factor=-1
    if rng.randrange(2):
        operands.reverse()
        values.reverse()
    deltas=(delta,delta+factor*(values[0]-values[1]))
    targets=sorted({str(f(v,d)) for v in values for d in deltas})
    assert len(targets)==3

    def render(w,b,noise,mode="lexical"):
        t=Text()
        t.add(f"A separate example uses the number {6+noise}.\n","noise")
        text=op1(operands[w])
        # The operand occurs in the second clause; the two constants can coincide.
        m=list(re.finditer(r"\d+",text))[-1]
        t.add(text[:m.start()]); t.add(m[0],"signal"); t.add(text[m.end():], "bridge")
        prefix=len(t.value)
        t.add(op2(deltas[b]),"map")
        t.add("What is the final number?", "query")
        return t,prefix

    return (render,targets,list(map(str,values)),lambda a,b:str(f(values[a], deltas[b])),
            lambda a:str(values[a]),"What is the number after the first step?", {"cardinality":3,"signature":[a,sorted(operands),deltas[0],sorted(deltas)]})


def group(task, split, i, source_key, source, mode, accept):
    family=i%3
    generator={"relation":relation, "logic":logic, "arithmetic":arithmetic}[task]
    for attempt in range(120):
        key=f"expanded-v1/{task}/{split}/{i}/{attempt}"
        rng=rng_for(key)
        render,opts,med,gold,mid,question,meta=generator(rng,family,source)
        rng.shuffle(opts); rng.shuffle(med)
        worlds={}
        for name,a,b,n in (("r",0,0,0),("x",1,0,0),("y",0,1,0),("d",1,1,0),
                           ("n",0,0,1),("m",1,1,1)):
            body, end=render(a,b,n,mode)
            probe=Text()
            probe.value=body.value[:end]+question
            probe.choose(med)
            body.choose(opts)
            worlds[name]={"prompt":body.value, "roles":body.regions,
                          "prefix_chars":end, "gold":opts.index(gold(a,b)),
                          "mid":med.index(mid(a)), "probe":probe.value}
        assert worlds["r"]["gold"]==worlds["d"]["gold"]
        assert worlds["r"]["gold"]!=worlds["x"]["gold"] and worlds["d"]["gold"]!=worlds["y"]["gold"]
        assert worlds["r"]["mid"]!=worlds["d"]["mid"]
        result={"id":f"{task}/{split}/{i:03}","task":task,"split":split,"family":family,
                "mode":mode,"options":opts,"intermediate_options":med,"source":source_key,
                "generation_attempt":attempt,"meta":meta,"worlds":worlds}
        if accept(result):
            return result
    raise RuntimeError(f"No aligned realization for {key}")


def proof_rows(data, member):
    with ZipFile(data/"prontoqa.zip") as z:
        datafile=json.loads(z.read(member))
    return [(f"{member}:{k}",v["test_example"]) for k,v in datafile.items()]


def choose(rows, count, seed):
    picked=list(rows)
    rng_for(seed).shuffle(picked)
    return picked[:count]


def controlled(data, accept):
    logic=list(row for member in LOGIC_SOURCES for row in proof_rows(data,member))
    # Canonical deduplication by premises and query before splitting the source proofs.
    seen=set()
    eligible=[]
    for k,v in logic:
        proof=v["chain_of_thought"]
        if len(proof)<5:
            continue
        s=[re.fullmatch(r"(\w+) is an? (\w+)\.",proof[i]) for i in (0,2)]
        signature=(v["question"],v["query"])
        if all(s) and s[0][1]==s[1][1] and s[0][2]!=s[1][2] and signature not in seen:
            seen.add(signature)
            eligible.append((k,v))
    logic=eligible
    logic=iter(choose(logic,len(logic),"expanded-logic-sources"))
    signatures=set()

    def new(g):
        signature=(g["task"],g["family"],json.dumps(g["meta"]["signature"]))
        if signature in signatures or not accept(g):
            return False
        signatures.add(signature)
        return True

    for task,total in (("relation",240),("logic",180),("arithmetic",240)):
        splits=[("dev",30),("confirm",120),("natural",total-150)]
        if task=="relation":
            splits.extend((("A",90),("B",90)))
        else:
            splits.append(("C",90))
        for split,n in splits:
            mode="context" if split=="A" else "lexical"
            for i in range(n):
                source_key,source=next(logic) if task=="logic" else (None,None)
                yield group(task,split,i,source_key,source,mode,new)


def benchmarks(data, prior):
    import pyarrow.parquet as pq

    seen=set()
    for member in PROOFS:
        pool=[]
        for k,v in proof_rows(data,member):
            query=v["query"].removeprefix("Prove: ").removesuffix(".")
            sig=(v["question"],query)
            if sig in seen or " is " not in query:
                continue
            seen.add(sig)
            alt=(query.replace(" is not "," is ",1) if " is not " in query
                 else query.replace(" is "," is not ",1))
            options=[query,alt]
            rng_for("options/"+k).shuffle(options)
            pool.append((k,v["question"]+"\nWhich statement follows?",options,options.index(query),v["chain_of_thought"]))
        if len(pool)<90:
            raise RuntimeError(f"Proof cohort too small: {member}")
        for k,text,opts,gold,proof in choose(pool,90,member):
            yield benchmark("prontoqa", k,text,opts,gold,proof)
    arc=pq.read_table(data/"arc.parquet").to_pylist()
    assert len({r["id"] for r in arc})==len(arc)==299
    for r in arc:
        yield benchmark("arc",r["id"],"Question: "+r["question"],r["choices"]["text"],
                        r["choices"]["label"].index(r["answerKey"]))
    hella=pq.read_table(data/"hellaswag.parquet").to_pylist()
    exclusion={str(x["source"]) for x in prior if x["task"]=="hellaswag"}
    for r in choose([r for r in hella if str(r["ind"]) not in exclusion],150,"expanded-hellaswag"):
        yield benchmark("hellaswag",str(r["ind"]),"Choose the most likely continuation.\n"+r["ctx"],
                        r["endings"],int(r["label"]))


def benchmark(task, source, prompt, options, gold, proof=None):
    t=Text()
    t.add(prompt,"query")
    t.choose(options)
    return {"id":f"{task}/{source}","task":task,"split":"benchmark","source":source,
            "options":options,"proof":proof,"worlds":{"r":{"prompt":t.value,"roles":t.regions,
                     "prefix_chars":0,"gold":gold}}}
