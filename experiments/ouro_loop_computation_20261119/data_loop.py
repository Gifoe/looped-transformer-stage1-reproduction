"""Balanced independent four-outcome worlds with identical open/choice stems."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

WORLDS=("r","x","y","d","u","n")
TASKS=("relation","logic","arithmetic")
SPLITS={"fit":112,"discover":48,"confirm":96,"transfer":96}
SITES=("source","bridge","mapping","query","readout")
NAMES=("Alice","Ben","Clara","Dan","Eve","Frank","Grace","Hank","Iris","Jack","Kate","Luke")
NEW=("Adam","Anna","Carol","Claire","Daniel","Emma","Ethan","Eva","Helen","James","Jane","John")
KEYS=("red","blue","green","black","white","pink","gold","gray","brown")
PLACES=("Rome","Paris","London","Boston","Dallas","Miami","Denver","Berlin","Tokyo")
TYPES=("lion","fox","wolf","bear","tiger","eagle","horse","rabbit")
OBJECTS=("lamp","cup","coin","book","key","ring","flag","bell")


def hash_key(key):
    return hashlib.sha256(key.encode()).digest()


def dump(f,obj):
    f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+"\n")


def rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def jsonfile(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


class Text:
    def __init__(self):
        self.text=""
        self.spans={}

    def add(self,kind,line):
        l=len(self.text)
        self.text+=line
        self.spans.setdefault(kind,[]).append((l,len(self.text)))


def scene(task,split,repl,style):
    held=split=="transfer"
    name=repl.choice(NEW if held else NAMES)
    noise=repl.sample(KEYS,2)
    if task=="arithmetic":
        base=repl.randint(32,48) if held else repl.randint(12,27)
        u,v=repl.randint(0,4),repl.randint(2,6)
        a,b=repl.sample((10+u,20+u,30+u,40+u),2)
        m,q=repl.sample((10+v,20+v,30+v,40+v),2)
        mid=(base+a,base+b)
        sign=-1 if held and style==0 else 1
        final=[mid[s]+sign*val for val in (m,q) for s in (0,1)]
        if len(set(final))!=4 or not all(10<=x<=99 for x in final):
            return
        same=m+sign*(mid[0]-mid[1])
        if same<10:
            return
        sig=(base,a,b,m,q,sign,held,style)
        def render(i,j,match,n):
            t=Text()
            t.add("noise",f"A separate card is {noise[n]}.\n")
            t.add("source",(f"Start with {base}. Add {(a,b)[i]}.\n" if not held
                            else f"A display starts at {base}. It increases by {(a,b)[i]}.\n"))
            t.add("bridge","Keep the result of the first calculation.\n")
            val=same if match else (m,q)[j]
            if held:
                t.add("mapping",f"Increase by 2. Then {'decrease' if sign<0 else 'increase'} by {val+2 if sign<0 else val-2}.\n")
            else:
                t.add("mapping",f"Now {'add' if style==0 else 'increase by'} {val}.\n")
            t.add("query","What is the final number?")
            return t,str(mid[i]),str(val),str(mid[i]+sign*val)
        return sig,render,list(map(str,final)),list(map(str,mid)),list(map(str,(m,q,same)))
    keys=repl.sample(KEYS if task=="relation" else TYPES,2)
    outputs=repl.sample(PLACES if task=="relation" else OBJECTS,4)
    synonyms=repl.sample(("west","east","north","south"),2)
    rule_order=repl.sample(range(2),2)
    sig=(name,keys,outputs,held,style)
    def render(i,j,match,n):
        t=Text()
        t.add("noise",f"A separate card is {noise[n]}.\n")
        word=keys[i]
        if task=="relation":
            t.add("source",f"{name} received the {word} box.\n" if not held else f"The clerk issued a {word} tag to {name}.\n")
            t.add("bridge","Use the assigned color and follow its route.\n")
        else:
            t.add("source",f"{name} is a {word} in this story.\n")
            t.add("bridge","Use only the rules listed below.\n")
        target=outputs[2*j:2*j+2]
        if match:
            target=outputs[:2][::-1]
        for k in rule_order:
            f=keys[k]
            if task=="relation":
                if held:
                    t.add("mapping",f"The {f} tag goes to the {synonyms[k]} depot.\n")
                    t.add("mapping",f"The {synonyms[k]} depot serves {target[k]}.\n")
                else:
                    t.add("mapping",(f"The {f} box routes to {target[k]}.\n" if style==0
                                      else f"{target[k]} receives the {f} box.\n"))
            elif held:
                t.add("mapping",f"Every {f} has the {synonyms[k]} badge.\n")
                t.add("mapping",f"Everyone with the {synonyms[k]} badge gets a {target[k]}.\n")
            else:
                t.add("mapping",(f"Every {f} gets a {target[k]}.\n" if style==0
                                  else f"A {target[k]} is given to every {f}.\n"))
        t.add("query",f"Which city should {name} go to?" if task=="relation" else f"What object does {name} get?")
        return t,word,target[0],target[i]
    return sig,render,outputs,keys,outputs[:1]+outputs[2:3]+outputs[1:2]


def encoding(tok,g,w,protocol):
    e=g["worlds"][w]
    stem=e["stem"]
    sep=" "
    if protocol=="choice":
        prompt=stem+"\n"+"\n".join(f"{chr(65+i)}. {label}" for i,label in enumerate(g["options"]))+"\nAnswer:"
        options=[chr(65+i) for i in range(len(g["options"]))]
    else:
        prompt=stem+"\nAnswer:"
        options=g["options"]
        if g["task"]=="arithmetic":
            prompt+=" "
            sep=""
    t=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
    ids=t["input_ids"]
    tokens=[]
    for value in options:
        seq=tok.encode(prompt+sep+value,add_special_tokens=False)
        assert seq[:len(ids)]==ids
        tail=seq[len(ids):]
        assert len(tail)==(2 if g["task"]=="arithmetic" and protocol=="open" else 1)
        tokens.append(tail)
    targets=[r[0] for r in tokens]
    assert len(set(targets))==len(targets)
    roles={k:[i for i,(x,y) in enumerate(t["offset_mapping"]) if any(x<b and y>a for a,b in v)]
           for k,v in e["spans"].items()}
    roles["readout"]=[len(ids)-1]
    assert all(roles.values())
    return {"ids":ids,"target":targets,"roles":roles,"read":len(ids)-1,"prompt":prompt,"answer_tokens":tokens}


def build(tok,task,split,i,seen):
    gid=f"{task}/{split}/{i:03}"
    for attempt in range(1500):
        rng=random.Random(int.from_bytes(hash_key(f"loop-computation-v1/{gid}/{attempt}"),"big"))
        setup=scene(task,split,rng,i%2)
        if setup is None:
            continue
        sig,render,options,mids,maps=setup
        key=(task,json.dumps(sig))
        if key in seen:
            continue
        options=list(options)
        rng.shuffle(options)
        worlds={}
        for name,a,b,match,n in (("r",0,0,False,0),("x",1,0,False,0),("y",0,1,False,0),
                                 ("d",1,1,False,0),("u",1,0,True,0),("n",0,0,False,1)):
            prompt,s,m,y=render(a,b,match,n)
            worlds[name]=dict(stem=prompt.text,spans=prompt.spans,S=s,M=m,Y=y,gold=options.index(y))
        assert len({worlds[w]["Y"] for w in WORLDS[:4]})==4
        assert worlds["r"]["Y"]==worlds["u"]["Y"] and worlds["r"]["S"]!=worlds["u"]["S"]
        g=dict(id=gid,task=task,split=split,style=i%2,structure=split=="transfer",
               signature=sig,worlds=worlds,options=options,S_options=mids,M_options=list(dict.fromkeys(maps)))
        try:
            for p in ("choice","open"):
                es={w:encoding(tok,g,w,p) for w in WORLDS}
                z=es["r"]
                assert all(e["roles"]==z["roles"] and e["target"]==z["target"] and
                           len(e["ids"])==len(z["ids"]) for e in es.values())
                for w,role in (("x","source"),("y","mapping"),("n","noise")):
                    changed={l for l,(a,b) in enumerate(zip(z["ids"],es[w]["ids"])) if a!=b}
                    assert changed and changed<=set(z["roles"][role])
                assert not set(z["roles"]["source"])&set(z["roles"]["mapping"])
        except AssertionError:
            continue
        seen.add(key)
        return g
    raise RuntimeError(f"No aligned realization for {gid}")


def prepare(model,out):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True)
    out.mkdir(parents=True,exist_ok=True)
    seen=set()
    all_rows=[build(tok,t,s,i,seen) for t in TASKS for s,n in SPLITS.items() for i in range(n)]
    p=out/"groups.jsonl"
    with p.open("x",encoding="utf-8") as f:
        for r in all_rows:
            dump(f,r)
    counts=Counter(f"{r['task']}/{r['split']}/{r['worlds']['r']['gold']}" for r in all_rows)
    jsonfile(out/"manifest.json",{"groups":len(all_rows),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
             "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "r_answer_slots":dict(counts),"worlds":WORLDS,"four_distinct_outcomes":True})
    print("generated",len(all_rows),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    prepare(a.model,a.out)
