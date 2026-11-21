"""Generate fresh structurally specified world groups for post-freeze validation."""
import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer
from native import load, save, sha


def convert(row,tok,family):
    text=row["text"].removesuffix("\nAnswer:")+"\nGive only the conclusion.\nAnswer:"
    if family=="tableau":
        choices=row["options"]
        if choices and isinstance(choices[0],dict):
            choices=[x.get("text",x.get("value")) for x in choices]
    else:
        choices=row["choices"]
    if len(choices)!=len(row["content"]) or any(not isinstance(x,str) for x in choices):
        raise ValueError(("choices", choices))
    ids=tok.encode(text,add_special_tokens=False)
    assert all(ids[i]==row["ids"][i] for name,role in row["roles"].items() if name!="readout"
               for i in role if i<len(row["ids"])-3)
    return dict(id=row["item"],group=row["group"],prompt=text,rewrite=text.replace("Give only the conclusion.","Write only the conclusion."),
                choices=choices,gold=row["y"],family="structured_"+family,bridge=row.get("b"),
                source=row.get("s"),rule=row.get("r"),role_tokens={k:v for k,v in row["roles"].items() if k!="readout"},
                candidate_ids=row["content"])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("pair",type=Path)
    p.add_argument("manifest",type=Path)
    p.add_argument("blind",type=Path)
    p.add_argument("full",type=Path)
    p.add_argument("blind_out",type=Path)
    p.add_argument("groups",type=Path)
    a=p.parse_args()
    sys.path.insert(0,str(a.pair))
    from causal_worlds import create
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
    manifest,blind=load(a.manifest),load(a.blind)
    selected=[]
    signatures=set()
    for family in ("tableau","twohop"):
        family_count=0
        for ix in range(30000,31000):
            for attempt in range(15):
                try:
                    g,signature=create(tok,"confirmation",family,ix,attempt)
                    if signature in signatures:
                        continue
                    z=[convert(r,tok,family) for r in g["worlds"]]
                    aux={k:convert(r,tok,family) for k,r in g.get("extra",{}).items()}
                    assert len({len(tok.encode(r["prompt"],add_special_tokens=False)) for r in z+list(aux.values())})==1
                except (ValueError, AssertionError):
                    continue
                selected.append(dict(group=g["group"],family=family,worlds=z,extra=aux))
                signatures.add(signature)
                family_count+=1
                break
            if family_count>=24:
                break
        assert family_count==24,family
    manifest["controlled"]=[q["worlds"][0] for q in selected]
    blind["controlled"]=[{k:v for k,v in q.items() if k!="gold"} for q in manifest["controlled"]]
    manifest["structured_provenance"]={"pair_generator_sha256":sha(a.pair/"causal_worlds.py"),
       "seed_index_start":30000,"variant":"new aligned output instruction; all core world identities newly generated"}
    save(a.full,manifest)
    save(a.blind_out,blind)
    save(a.groups,selected)


if __name__=="__main__":
    main()

