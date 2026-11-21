"""Subtract native route–content interaction from the running K2 attention output."""

import argparse
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn.functional import linear,pad as pad_tensor
from torch.nn.utils.rnn import pad_sequence

from av_test import START, STOP, LAST, SITES, Trace
from edge import short_hash
from interchange import terminated
from mid_test import suffix_context
from native import Forward, load, output, pad, save, setup, sha, timer
from nec_deck import DOMAINS, prompt
from nec_match import enroll, matches, measure
from nec_stats import ARMS, summarize
from query_test import whitespace
from rule import MAX_NEW, text_results
from value_test import shuffle


class Deletion:
    def __init__(self,positions,delta,audit):
        self.positions = positions
        self.delta = delta
        self.audit = audit
        self.checks = 0
        self.err = []

    @contextmanager
    def apply(self,f):
        def patch(_,args,live,l):
            delta = self.delta[l]
            valid = torch.arange(live.size(1),device=live.device)[None,:] <= self.positions[:,None]
            changed = torch.where(valid[...,None],live-delta,live)
            if self.audit:
                rounded = (live-changed)[valid]
                torch.testing.assert_close(rounded,delta[valid],rtol=9e-5,atol=2e-5)
                self.err.append(((live-changed)-delta).abs().amax(dim=(1,2)))
                self.checks += len(live)
            return changed

        hooks = [f.layers[l].self_attn.o_proj.register_forward_hook(lambda m,a,x,l=l: patch(m,a,x,l)) for l in SITES]
        try:
            yield
        finally:
            for h in hooks:
                h.remove()


def subtractions(f,trace,l,audit):
    block = f.layers[l].self_attn
    a0,a1 = [trace.values[k,l,"a"] for k in (0,1)]
    v0,v1 = [f.module.repeat_kv(trace.values[k,l,"v"],block.num_key_value_groups) for k in (0,1)]

    def head(a,v):
        return (a@v).transpose(1,2).contiguous().flatten(2)

    base = trace.values[1,l,"att_projected"]
    d = linear(head(a1-a0,v1-v0),block.o_proj.weight)
    g = base-shuffle(base,-d)
    err = 0.
    if audit:
        c00 = trace.values[0,l,"att_projected"]
        cv,av = [block.o_proj(head(a,v)) for a,v in ((a1,v0),(a0,v1))]
        reconstructed = (base-av)+(c00-cv)
        torch.testing.assert_close(reconstructed,d,rtol=1e-4,atol=4e-5)
        torch.testing.assert_close(g.norm(dim=-1),d.norm(dim=-1),rtol=5e-5,atol=6e-6)
        err = (reconstructed-d).abs().max().item()
    return dict(true=d,geometry=g,identity=torch.zeros_like(d)),err


def snapshots(f,examples,keys,audit):
    bank = {}
    for offset in range(0,len(keys),16):
        group = keys[offset:offset+16]
        seq = [examples[i]["ids"]+list(p) for i,p in group]
        ids,pos = pad(seq)
        tr = Trace(True,audit)
        k1,k2 = f.run(ids,2,record=tr)
        native = output(f.model,k2,pos).argmax(-1).tolist() if any(not p for _,p in group) else None
        site,errors = zip(*(subtractions(f,tr,l,audit) for l in SITES),strict=True)
        for j,(i,p) in enumerate(group):
            n = len(seq[j])
            delta = {a:{l:x[a][j,:n] for l,x in zip(SITES,site,strict=True)} for a in ARMS}
            bank[i,p] = dict(ids=seq[j],start=k1[j,:n],delta=delta,states={key:x[j,:n] for key,x in tr.states.items()},factorization=max(errors))
            if not p:
                assert native[j]==examples[i]["native"]["original"][1]["tokens"][0]
    return bank


def step(f,examples,requests,audit=False):
    bank = snapshots(f,examples,list(dict.fromkeys((i,p) for i,_,p in requests)),audit)
    out,diag = {},{}
    for offset in range(0,len(requests),24):
        group = requests[offset:offset+24]
        batch = [bank[i,p] for i,_,p in group]
        ids,pos = pad([r["ids"] for r in batch])
        start = pad_sequence([r["start"] for r in batch],batch_first=True)
        writes = {}
        for l in SITES:
            d = pad_sequence([r["delta"][a][l] for r,(_,a,_) in zip(batch,group,strict=True)],batch_first=True)
            writes[l] = pad_tensor(d,(0,0,0,ids.size(1)-d.size(1)))
        deletion = Deletion(pos,writes,audit)
        edited = Trace(False,audit) if audit else None
        with deletion.apply(f),suffix_context(f,audit) as params:
            h = f.run(ids,1,start=start,record=edited)[0]
        nxt = output(f.model,h,pos).argmax(-1)
        out.update((k[:-1],y) for k,y in zip(group,nxt.tolist(),strict=True))
        if not audit:
            continue
        suffix = edited.states[0,STOP]
        for l in range(STOP,LAST+1):
            suffix = f.layers[l](suffix,**params[l])
        assert deletion.checks==len(group)*len(SITES)
        assert torch.equal(nxt,output(f.model,f.core.norm(suffix),pos).argmax(-1))
        error = torch.stack(deletion.err).amax(0).tolist()
        for j,(r,k) in enumerate(zip(batch,group,strict=True)):
            n = len(r["ids"])
            clean = r["states"]
            torch.testing.assert_close(edited.states[0,START][j,:n],clean[1,START],rtol=8e-5,atol=2e-4)
            x = edited.states[0,LAST][j,:n]
            y = suffix[j,:n]
            torch.testing.assert_close(x,y,rtol=8e-5,atol=2e-4)
            identity = 0.
            if k[1]=="identity":
                torch.testing.assert_close(x,clean[1,LAST],rtol=8e-5,atol=3e-4)
                identity = (x-clean[1,LAST]).abs().max().item()
            diag[k[:-1]] = dict(checks=len(SITES),factorization=r["factorization"],suffix=(x-y).abs().max().item(),
                                subtraction=error[j],identity=identity)
    return out,diag


def decode(f,tok,examples):
    active = {(i,a):[] for i in range(len(examples)) for a in ARMS}
    done = {}
    for _ in range(MAX_NEW):
        if not active:
            break
        nxt,_ = step(f,examples,[(*k,tuple(p)) for k,p in active.items()])
        for k,p in list(active.items()):
            p.append(nxt[k])
            if terminated(tok,p):
                done[k] = active.pop(k)
    done.update(active)
    values = dict(zip(done,text_results(tok,list(done.values())),strict=True))
    out = [{a:values[i,a] for a in ARMS} for i in range(len(examples))]
    for r,z in zip(examples,out,strict=True):
        assert z["identity"]==r["native"]["original"][1]
    return out


def audit_case(f,tok,row):
    prefix = tuple(whitespace(tok,row["native"]["original"]))
    _,diag = step(f,[row],[(0,a,prefix) for a in ARMS],audit=True)
    return {k:(sum if k=="checks" else max)(q[k] for q in diag.values()) for k in next(iter(diag.values()))}


def auditors(cases):
    return {d:{g:sorted((r["id"] for r in cases if (r["domain"],r["kind"])==(d,g)),
                        key=lambda x:short_hash("necessity audit",x))[:8] for g in ("repair","stable")} for d in DOMAINS}


@torch.inference_mode()
def run(model,root):
    m,tok = setup(model)
    f = Forward(m)
    start = timer()
    native = root/"NATIVE.json"
    digest = sha(native)
    rows = enroll(load(native)["rows"])
    mfile = root/"MARGINS.json"
    if not mfile.exists():
        z = measure(f,tok,rows)
        save(mfile,dict(native_sha256=digest,rows=z))
    recorded = load(mfile)
    assert recorded["native_sha256"]==digest and [r["id"] for r in rows]==[v["id"] for v in recorded["rows"]]
    match = matches(recorded["rows"])
    audit = auditors(rows)
    inspected = {i for q in audit.values() for g in q.values() for i in g}
    lock = root/"COHORT.json"
    assignment = dict(native_sha256=digest,margins_sha256=sha(mfile),cases=[r["id"] for r in rows],matched=match,audit=audit)
    if lock.exists():
        assert load(lock)==assignment
    else:
        save(lock,assignment)
    print("necessity frozen pairs",{d:len(match[d]) for d in DOMAINS},flush=True)
    examples = []
    for r,z in zip(rows,recorded["rows"],strict=True):
        if r["kind"]=="stable" and z["selected"]:
            r["foil"] = z["selected"]["foil"]
        examples.append(dict(r,ids=tok.encode(prompt(r,"original"),add_special_tokens=False)))
    target = root/"NEC_RAW.json"
    data = load(target) if target.exists() else dict(rows=[],cohort_sha256=sha(lock),elapsed=0.)
    out = data["rows"]
    assert data["cohort_sha256"]==sha(lock) and [r["id"] for r in out]==assignment["cases"][:len(out)]
    elapsed = data["elapsed"]
    for off in range(len(out),len(examples),16):
        batch = examples[off:off+16]
        for r,z in zip(batch,decode(f,tok,batch),strict=True):
            item = dict({k:v for k,v in r.items() if k!="ids"},pred=z)
            if r["id"] in inspected:
                item["audit"] = audit_case(f,tok,r)
            out.append(item)
        data.update(elapsed=elapsed+timer()-start,peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,1))
        save(target,data)
        print("necessity replay",len(out),len(examples),flush=True)
    z = summarize(out,match)
    save(root/"NEC_SUMMARY.json",z)
    print({k:v for k,v in z.items() if k!="domains"},flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a = p.parse_args()
    run(a.model,a.root)
