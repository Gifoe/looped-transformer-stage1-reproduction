"""Create a matched novel-source/same-bridge donor in an already frozen structure group."""
import argparse
import re
from pathlib import Path

from transformers import AutoTokenizer
from native import load, save, sha

MAP=re.compile(r"(?m)^The ([^\n]+?) key opens the ([^\n]+?) cabinet\.$")
SOURCE=re.compile(r"holds the ([^\n]+?) key\.")


def match(group,tok):
    live,other=group["worlds"][:2]
    raw=other["prompt"]
    names=[SOURCE.search(x["prompt"]).group(1) for x in (live,other)]
    patterns={m.group(1):m.group(2) for m in MAP.finditer(raw)}
    assert names[0]!=names[1] and patterns[names[0]]!=patterns[names[1]]
    override={names[0]:patterns[names[1]],names[1]:patterns[names[0]]}
    changed=MAP.sub(lambda m:f"The {m.group(1)} key opens the {override.get(m.group(1),m.group(2))} cabinet.",raw)
    encode=lambda text:tok.encode(text,add_special_tokens=False)
    expected=encode(raw)
    altered=encode(changed)
    if len(expected)!=len(altered):
        return None
    delta=[i for i,(a,b) in enumerate(zip(expected,altered,strict=True)) if a!=b]
    assert delta and set(delta)<=set(live["role_tokens"]["bridge"])
    return dict(other,id=other["id"]+":match_bridge",prompt=changed,
               rewrite=changed.replace("Give only the conclusion.","Write only the conclusion."),
               bridge=live["bridge"],gold=live["gold"])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("src",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("receipt",type=Path)
    a=p.parse_args()
    original=sha(a.src)
    rows=load(a.src)
    t=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=True)
    recorded=[]
    for group in rows:
        if group["family"]=="twohop":
            item=match(group,t)
            if item is not None:
                group["extra"]["same_bridge_different_source"]=item
            recorded.append(dict(group=group["group"],added=item is not None))
    save(a.out,rows)
    save(a.receipt,dict(original_sha256=original,augmented_sha256=sha(a.out),rows=recorded))


if __name__=="__main__":
    main()

