"""Decode aligned table-digit sites and freeze the matched exact-answer cohort."""

from nec_match import matches
from route_deck import DOMAINS,prompt


def prepare(tok,row):
    text = prompt(row,"original")
    v = row["example"]["original"]
    arr = v["values"]
    base = 1 if row["domain"]=="standard" else 5
    target,foil = arr.index(row["target"])+base,arr.index(row["foil"])+base
    keys = [target,foil,*[k for k in v["order"] if k not in (target,foil)]]
    tokenized = tok(text,add_special_tokens=False,return_offsets_mapping=True)
    sites = []
    for k in keys:
        value = str(arr[k-base])
        target_text = f"{k}: {value}"
        end = text.rindex(target_text)+len(target_text)
        start = end-len(value)
        offsets = tokenized["offset_mapping"]
        p = [j for j,(a,b) in enumerate(offsets) if a<end and b>start]
        if len(p)!=3 or offsets[p[-1]][1]!=end or text[offsets[p[0]][0]:end].strip()!=value:
            return None
        sites.append(p)
    assert len({x for q in sites for x in q})==12 and max(map(max,sites))<len(tokenized["input_ids"])-1
    return dict(row,ids=tokenized["input_ids"],sites=sites)


def freeze(tok,rows,margin):
    examples = []
    eligible = []
    for r,w in zip(rows,margin,strict=True):
        assert r["id"]==w["id"]
        if r["kind"]=="stable":
            if w["selected"] is None:
                continue
            r = dict(r,foil=w["selected"]["foil"])
        q = prepare(tok,r)
        if q is not None:
            examples.append(q)
            eligible.append(w)
    paired = matches(eligible)
    stable = {p["stable"] for q in paired.values() for p in q}
    chosen = [r for r in examples if r["kind"]=="repair" or r["id"] in stable]
    return chosen,paired,{d:{k:sum((r["domain"],r["kind"])==(d,k) for r in examples) for k in ("repair","stable")} for d in DOMAINS}
