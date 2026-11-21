"""Freeze repair and stable cases, and match their clean K2 decision margins."""

import re

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from native import pad
from nec_deck import DOMAINS, prompt, repair

CALIPER = 2.0
MIN_MATCHES = 48


def enroll(rows):
    out = []
    for r in rows:
        native = r["native"]["original"]
        if repair(r):
            out.append(dict(r,kind="repair",foil=native[0]["number"]))
        elif all(x["number"]==r["target"] for x in native):
            out.append(dict(r,kind="stable",foil=None))
    return out


def comparison(tok,text,ids,native,foil):
    m = re.search(r"(?m)^[ \t]*([0-9]{1,4})[ \t]*\r?$",native["text"])
    assert m and int(m.group(1))==native["number"]
    tokens = tok.encode(text+native["text"][:m.start(1)]+str(foil)+native["text"][m.end(1):],add_special_tokens=False)
    if tokens[:len(ids)]!=ids:
        return None
    for i,(a,b) in enumerate(zip(native["tokens"],tokens[len(ids):])):
        if a!=b:
            left = tok.decode(native["tokens"][:i+1],skip_special_tokens=True)
            right = tok.decode(tokens[len(ids):len(ids)+i+1],skip_special_tokens=True)
            if not re.search(r"[0-9]",left) or not re.search(r"[0-9]",right):
                return None
            return len(ids)+i-1,a,b
    return None


def choose(candidates):
    if not candidates:
        return None
    return min(candidates,key=lambda r:(r["margin"],r["foil"]))


@torch.inference_mode()
def measure(f,tok,cases):
    records = []
    for first in range(0,len(cases),32):
        batch = cases[first:first+32]
        tasks = []
        seq = []
        for j,r in enumerate(batch):
            text = prompt(r,"original")
            ids = tok.encode(text,add_special_tokens=False)
            native = r["native"]["original"][1]
            seq.append(ids+native["tokens"])
            alternatives = [r["foil"]] if r["kind"]=="repair" else [x for x in r["example"]["original"]["values"] if x!=r["target"]]
            for foil in alternatives:
                picked = comparison(tok,text,ids,native,foil)
                if picked:
                    where,a,b = picked
                    tasks.append((j,foil,where,a,b))
        margins = [[] for _ in batch]
        if tasks:
            ids,_ = pad(seq)
            hidden = f.run(ids,2)[-1]
            pick = torch.tensor([(j,p) for j,_,p,_,_ in tasks],device=ids.device)
            logits = f.model.lm_head(hidden[pick[:,0],pick[:,1]]).float()
            k = torch.arange(len(tasks),device=ids.device)
            a,b = [torch.tensor([x[i] for x in tasks],device=ids.device) for i in (3,4)]
            assert torch.equal(logits.argmax(-1),a)
            gaps = (logits[k,a]-logits[k,b]).tolist()
            for (j,foil,where,a,b),gap in zip(tasks,gaps,strict=True):
                assert gap>=0
                margins[j].append(dict(foil=foil,margin=gap,position=where,target_token=a,foil_token=b))
        for r,candidates in zip(batch,margins,strict=True):
            candidates.sort(key=lambda x:x["foil"])
            records.append(dict(id=r["id"],domain=r["domain"],kind=r["kind"],choices=candidates,
                                selected=choose(candidates)))
        print("necessity margin",len(records),len(cases),flush=True)
    return records


def matches(margins):
    out = {}
    for d in DOMAINS:
        repairs,stable = [[r for r in margins if r["domain"]==d and r["selected"] is not None and r["kind"]==which]
                          for which in ("repair","stable")]
        if not repairs or not stable:
            out[d] = []
            continue
        a,b = [np.array([r["selected"]["margin"] for r in q]) for q in (repairs,stable)]
        weight = np.abs(a[:,None]-b[None,:])
        allowed = weight <= CALIPER
        # The unmatched cost is greater than the maximum total feasible matching cost.
        penalty = (len(repairs)+1)*(CALIPER+1)
        cost = np.full((len(repairs),len(stable)+len(repairs)),penalty)
        cost[:,:len(stable)] = np.where(allowed,weight,2*penalty)
        i,j = linear_sum_assignment(cost)
        out[d] = [dict(repair=repairs[x]["id"],stable=stable[y]["id"],repair_margin=float(a[x]),
                       stable_margin=float(b[y]),absolute_gap=float(weight[x,y])) for x,y in zip(i,j,strict=True) if y<len(stable)]
    return out
