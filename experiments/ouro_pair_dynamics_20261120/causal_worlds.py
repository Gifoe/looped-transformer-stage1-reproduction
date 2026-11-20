"""Freeze new 3×3 source–rule groups and a separate ARC natural reference."""
import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

from pair_fields import ROOT, EMBED, write_json

DESIGN=ROOT.parent/"ouro_hidden_computation_20261120"
CONT=ROOT.parent/"ouro_continuous_latent_20261120"
sys.path.insert(0,str(DESIGN))
from mechanism_design import encode, group
from study import twohop, role_tokenize
import study


SHAPES={"pilot":8,"development":48,"validation":48,"confirmation":96}
LEVELS={name:6000+1000*i for i,name in enumerate(SHAPES)}
VOCABULARIES={
    "pilot":(("Anna","Ben","Jane"),("red","green","white","blue","black"),("circle","square","star","cross","diamond"),
             ("apple","peach","pear","orange","plum","melon")),
    "development":(("Alex","Laura","Peter"),("violet","gray","gold","silver","yellow"),("moon","heart","arrow","bell","triangle"),
                   ("horse","wolf","lion","rabbit","tiger","zebra")),
    "validation":(("Kevin","Helen","Tom"),("brown","pink","purple","orange","bronze"),("sun","flower","leaf","shell","flag"),
                  ("carrot","onion","potato","garlic","pepper","cabbage")),
    "confirmation":(("Lucy","Mark","Paul"),("navy","cream","pearl","teal","tan"),("cloud","tree","ring","crown","shield"),
                    ("eagle","whale","dolphin","shark","falcon","sparrow")),
}


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def keyed(x):
    return hashlib.sha256(("pair-order/Ouro/"+x).encode()).hexdigest()


def masked(world):
    """Only token IDs, positions and lookup token IDs reach the predictor."""
    return {k:world[k] for k in ("ids","read","content")}


def create(tokenizer,phase,family,index,attempt):
    if family=="tableau":
        config=group("new_entities" if phase!="confirmation" else "new_structure",index,attempt)
        def get(s,m,r=0):
            q=encode(tokenizer,config,s,m,"open")
            return dict(q,group=config["id"],item=f"{config['id']}:{s}{m}",family=family,b=s)
        signature=tuple(config["keys"]+config["answers"])
    else:
        original=study.VOCAB["new_structure"]
        study.VOCAB["new_structure"]=VOCABULARIES[phase]
        try:
            values=twohop(tokenizer,"new_structure",index+10000,attempt)
        finally:
            study.VOCAB["new_structure"]=original
        look={(w["s"],w["m"],w["r"]):w for w in values}
        def get(s,m,r=0):
            w=look[s,m,r]
            if phase!="confirmation":
                return w
            t=w["text"].removesuffix("\nAnswer:")+"\nOptions: "+", ".join(w["choices"])+"."
            z=role_tokenize(tokenizer,[(None,t),("readout","\nAnswer:")],w["choices"],w)
            if z["ids"][:w["read"]-1]!=w["ids"][:w["read"]-1]:
                raise ValueError("Context and roles shifted")
            z["roles"]={**w["roles"],**z["roles"]}
            return z
        signature=tuple(tuple(sorted(v)) for v in values[0]["within"])
    worlds=[get(s,m) for s,m in ((0,0),(1,0),(0,1),(1,1))]
    ref=worlds[0]
    rule="mapping" if family=="tableau" else "bridge"
    src,mp=ref["roles"]["source"],ref["roles"][rule]
    if any(len(w["ids"])!=len(ref["ids"]) or w["content"]!=ref["content"] or
           w["roles"]["source"]!=src or w["roles"][rule]!=mp for w in worlds):
        raise ValueError("Unaligned roles")
    if set(src)&set(mp) or len(src)!=1:
        raise ValueError("Overlapping donor roles")
    targets=[w["y"] for w in worlds]
    assert targets[-1] not in targets[:3]
    row=dict(group=ref["group"],family=family,phase=phase,worlds=worlds,source=src,rule=mp,
             targets=targets,bridge=worlds[-1]["b"])
    if family=="twohop":
        row["extra"]={"same_bridge_other_answer":get(0,0,1),"same_answer_other_bridge":get(1,0,2)}
    return row,signature


def build(model,out):
    out.mkdir(exist_ok=True,parents=True)
    tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True)
    arc=[]
    with (CONT/"results"/"data_manifest.jsonl").open(encoding="utf-8") as f:
        arc=[w for w in map(json.loads,f) if w["family"]=="arc" and len(w["content"])==4]
    before=json.loads((EMBED/"results"/"MANIFEST.json").read_text())
    omit={w["item"] for k in ("dev","test") for w in before["arc"][k]}
    omit.update(g[0]["item"] for g in json.loads((EMBED/"results"/"ARC_CAUSAL_MANIFEST.json").read_text()))
    arc=sorted((w for w in arc if w["item"] not in omit),key=lambda w:keyed(w["item"]))
    groups={}
    seen=set()
    for phase,size in SHAPES.items():
        groups[phase]={}
        for family in ("tableau","twohop"):
            items=[]
            for i in range(LEVELS[phase],LEVELS[phase]+500):
                for attempt in range(30):
                    try:
                        row,signature=create(tok,phase,family,i,attempt)
                    except ValueError:
                        continue
                    if signature in seen:
                        continue
                    seen.add(signature)
                    items.append(row)
                    break
                if len(items)==size:
                    break
            assert len(items)==size,(phase,family,len(items))
            groups[phase][family]=items
    write_json(out/"MANIFEST.json",dict(model_revision="574fa66cb8bf5abdc979642d01cf2b79b16bfab1",
        groups=groups,arc_pilot=arc[:8],arc_confirmation=arc[8:104],
        arc_scope="ARC validation previously scored in this project, no embedding development or confirmation IDs",
        source_sha256={n:sha(ROOT/n) for n in ("causal_worlds.py","pair_fields.py","norm_flow.py")}))


if __name__=="__main__":
    assert len(sys.argv)==3
    build(Path(sys.argv[1]),Path(sys.argv[2]))
