"""Competence cohorts using fresh proof seeds and two-step addition."""
import argparse
import json
from pathlib import Path
import re

from transformers import AutoTokenizer

from collect import aligned, dump_line, load_lines, write_json, digest
from factory import COLORS, Text, choose, group, proof_rows, rng_for
from intervene import SITES

MEMBERS=(
 "1hop_ProofsOnly_3testhops_random_noadj.json",
 "1hop_ProofsOnly_4testhops_random_noadj.json",
 "1hop_ProofsOnly_5testhops_random_noadj.json",
 "2hop_ProofsOnly_testrandom_nodistractor_testdistractor_noadj.json",
 "2hop_ProofsOnly_testrandom_irrelevantdistractor_testdistractor_noadj.json",
)
SPLITS=(("dev",30),("confirm",90),("C",90))


def source_proofs(data):
    seen=set()
    out=[]
    for member in MEMBERS:
        for key,p in proof_rows(data,member):
            proof=p["chain_of_thought"]
            regex=r"(\w+) is an? (\w+)\."
            if len(proof)<5:
                continue
            a,b=(re.fullmatch(regex,proof[i]) for i in (0,2))
            identity=(p["question"],p["query"])
            if a and b and a[1]==b[1] and a[2]!=b[2] and identity not in seen:
                seen.add(identity)
                out.append((key,a[1],a[2],b[2],proof))
    return choose(out,len(out),"fresh-logic-plus-proof-sources")


def deduction(id_, split, format_, item, tok, seen):
    source, person, original, intermediate, chain=item
    for attempt in range(100):
        gen=rng_for(f"competence-chain/{split}/{id_}/{attempt}")
        candidates=sorted(set(("grimpus","wumpus","numpus","jompus","lempus","rompus",
                               "vumpus","sterpus","tumpus","yumpus","gorpus","dumpus"))
                          -{original,intermediate})
        alt_origin,alt_bridge=gen.sample(candidates,2)
        origins=[original,alt_origin]
        bridge=[intermediate,alt_bridge]
        end=gen.sample(("red","blue","green","white","black","cold","warm","dry","round","tall"),2)
        spoil=gen.sample([x for x in COLORS if x not in end],2)
        order=gen.sample(range(2),2)
        swap=gen.sample(range(2),2)
        options=[f"{person} is {end[i]}." if format_==0 else end[i] for i in swap]
        intermediate_options=[bridge[i] for i in order[::-1]]
        worlds={}
        for n,a,b,u in (("r",0,0,0),("x",1,0,0),("y",0,1,0),("d",1,1,0),("n",0,0,1),("m",1,1,1)):
            t=Text()
            t.add(f"Use the premises about {person}.\n","early_query")
            t.add(f"An unrelated sign is {spoil[u]}.\n","noise")
            for j in order:
                t.add(f"Every {origins[j]} is a ")
                t.add(bridge[j],"bridge")
                t.add(".\n")
            t.add(f"{person} is a ")
            t.add(origins[a],"signal")
            t.add(".\n")
            prefix=len(t.value)
            for j in swap[::-1]:
                t.add(f"Every {bridge[j]} is {end[j^b]}.\n","map")
            q="Which conclusion follows?" if format_==0 else f"What property must {person} have?"
            t.add(q,"query")
            p=Text()
            p.value=t.value[:prefix]+f"Which derived category applies to {person}?"
            p.choose(intermediate_options)
            t.choose(options)
            worlds[n]={"prompt":t.value,"roles":t.regions,"prefix_chars":prefix,"gold":swap.index(a^b),
                       "probe":p.value,"mid":intermediate_options.index(bridge[a])}
        sig=(person,tuple(origins),tuple(bridge),tuple(end))
        row={"id":f"logic/competence_{split}/{id_:03}","task":"logic","split":split,"family":format_,
             "mode":"lexical","source":source,"source_proof":chain,"options":options,
             "intermediate_options":intermediate_options,"worlds":worlds,"meta":{"signature":sig}}
        if sig not in seen and aligned(tok,row):
            seen.add(sig)
            return row
    raise ValueError(f"No aligned proof variant for {source}")


def prepare(data,model,base,out):
    out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True)
    original=load_lines(base/"inputs.jsonl")
    previous={json.dumps(g["meta"]["signature"]) for g in original
              if g["task"]=="arithmetic" and g["family"]==0}
    seen=set()
    sources=iter(source_proofs(data))
    rows=[]
    for split,n in SPLITS:
        for i in range(n):
            rows.append(deduction(i,split,i%2,next(sources),tok,seen))
            def fresh(g):
                sig=json.dumps(g["meta"]["signature"])
                if sig in previous or not aligned(tok,g):
                    return False
                previous.add(sig)
                return True
            a=group("arithmetic",f"competence_{split}",i*3,None,None,"lexical",fresh)
            a["split"]=split
            rows.append(a)
    # Development and future samples remain in separate files before calibration.
    for split,_ in SPLITS:
        with (out/f"{split}_input.jsonl").open("x",encoding="utf-8") as f:
            for r in rows:
                if r["split"]==split:
                    dump_line(f,r)
    (out/"inputs.jsonl").write_bytes((out/"dev_input.jsonl").read_bytes())


def freeze(out,code):
    calibration=load_lines(out/"calibration"/"rows.jsonl")
    rates={}
    for task,family in (("logic",0),("logic",1),("arithmetic",0)):
        subset=[r for r in calibration if (r["task"],r["family"])==(task,family)]
        by={}
        for r in subset:
            by.setdefault(r["id"],[]).append(r)
        scores={k:sum(all(r["pred"][k]==r["gold"] for r in v) for v in by.values())/len(by) for k in range(4)}
        rates[f"{task}:{family}"]={"n":len(by),"joint_by_K":scores,"k4_joint":scores[3]}
    # Equal representation of both measured logic layouts remains frozen in the
    # same 90-group sample. Report them individually and state competence at each.
    files=[out/f"{s}_input.jsonl" for s,_ in SPLITS]
    path=out/"inputs.jsonl"
    path.write_bytes(b"".join(f.read_bytes() for f in files))
    protocol={"input_sha256":digest(path),"intervention_sha256":digest(code/"intervene.py"),
              "calibration_sha256":digest(out/"calibration"/"rows.jsonl"),"natural_competence_gate":0.60,
              "gate_evaluation":"fraction of groups with all four natural K4 outcomes correct; fresh set, per task/family",
              "development":rates,"sites":SITES,"inputs":{f.name:digest(f) for f in files}}
    write_json(out/"frozen_protocol.json",protocol)
    print(json.dumps(protocol,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("prepare","freeze"))
    p.add_argument("--study",type=Path,required=True)
    p.add_argument("--base",type=Path,required=True)
    p.add_argument("--data",type=Path)
    p.add_argument("--model",type=Path)
    a=p.parse_args()
    if a.action=="prepare":
        prepare(a.data,a.model,a.base,a.study)
    else:
        freeze(a.study,Path(__file__).parent)


if __name__=="__main__":
    main()
