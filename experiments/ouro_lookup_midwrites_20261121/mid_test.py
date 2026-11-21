"""Replay native layer 12–17 branch writes and recompute the receiving suffix."""

import argparse
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from edge import short_hash
from interchange import Patch, terminated
from mid_deck import DOMAINS, control, prompt, repair
from mid_stats import ARMS, DIRECTIONS, MODES, summarize
from native import Forward, load, output, pad, save, setup, sha, timer
from query_test import whitespace
from rule import MAX_NEW, text_results
from value_test import shuffle

START, STOP, LAST = 11, 17, 23
BRANCHES = MODES["both"]
SITES = tuple((l, b) for l in range(START, STOP) for b in BRANCHES)


class Trace:
    def __init__(self, audit):
        self.audit = audit
        self.values = {}
        self.states = {}

    def put(self, key, name, x):
        phase, l = key
        if (l, name) in SITES and not self.audit == "edited":
            self.values[phase, l, name] = x.detach()
        if self.audit and (l, name) in ((START, "input"), (STOP, "input"), (LAST, "output")):
            self.states[phase, l] = x.detach()


class Replay(Patch):
    def __init__(self, at, inject, audit):
        super().__init__(at)
        self.inject = inject
        self.audit = audit
        self.audit_checks = 0

    def write(self, key, name, x):
        item = self.inject.get((key[1], name))
        if item is None:
            return x
        idx, z = item
        active = torch.arange(x.size(1), device=x.device)[None, :] <= self.pos[idx, None]
        y = x.clone()
        y[idx] = torch.where(active[..., None], z, y[idx])
        if self.audit:
            assert torch.equal(y[idx][active], z[active])
            self.audit_checks += len(idx)
        return y

@contextmanager
def suffix_context(f, enabled):
    kwargs = {}
    if not enabled:
        yield kwargs
        return

    def capture(i, named):
        assert named["past_key_value"] is None and not named["use_cache"]
        kwargs[i] = named.copy()

    hooks = [f.layers[l].register_forward_pre_hook(lambda _, a, k, l=l: capture(l,k), with_kwargs=True)
             for l in range(STOP, LAST+1)]
    try:
        yield kwargs
    finally:
        for h in hooks:
            h.remove()


def prepare(tok, row):
    r = control(row)
    u,v = [tok(prompt(r,k), add_special_tokens=False, return_offsets_mapping=True)
           for k in ("original", "unrelated_original")]
    assert u["offset_mapping"] == v["offset_mapping"], f"unaligned control tokens {row['id']}"
    return dict(r, ids=u["input_ids"], unrelated=v["input_ids"])


def snapshots(f, examples, needed, audit):
    result = {}
    for first in range(0, len(needed), 24):
        jobs = needed[first:first+24]
        sequences = [examples[i][v] + list(p) for i,p,v in jobs]
        ids, at = pad(sequences)
        tr = Trace(audit)
        h1,h2 = f.run(ids, 2, record=tr)
        token = [output(f.model,h,at).argmax(-1).tolist() for h in (h1,h2)] if any(not p and v=="ids" for i,p,v in jobs) else None
        for j, (i,p,v) in enumerate(jobs):
            n = len(sequences[j])
            values = {k:x[j,:n] for k,x in tr.values.items()}
            result[i,p,v] = dict(ids=sequences[j], start=h1[j,:n], values=values,
                                 states={k:x[j,:n] for k,x in tr.states.items()})
            if not p and v == "ids":
                assert [t[j] for t in token] == [r["tokens"][0] for r in examples[i]["native"]["original"]]
    return result


def match(group, bank, own, pos, key, phase, width):
    l,b = key
    selected = [j for j,k in enumerate(group) if b in MODES[k[1]]]
    if not selected:
        return None, 0, 0

    def pack(values):
        x = pad_sequence(values, batch_first=True)
        return torch.nn.functional.pad(x,(0,0,0,width-x.size(1)))

    r = pack([own[j]["values"][phase,l,b] for j in selected])
    d = pack([own[j]["values"][1-phase,l,b] for j in selected])
    z = d.clone()
    backup,total = 0,0
    for treatment in ("geometry","unrelated","identity"):
        ids = [i for i,j in enumerate(selected) if group[j][3] == treatment]
        if not ids:
            continue
        given = r[ids]
        if treatment == "identity":
            z[ids] = given
            continue
        delta = d[ids]-given
        geo = shuffle(given,delta)
        if treatment == "geometry":
            z[ids] = geo
            continue
        requests = [group[selected[i]] for i in ids]
        unrelated = pack([bank[i,p,"unrelated"]["values"][1-phase,l,b] for i,_,_,_,p in requests])
        u = unrelated-d[ids]
        length = u.norm(dim=-1,keepdim=True)
        fallback = length <= 1e-8
        change = torch.where(fallback, geo-given, u*(delta.norm(dim=-1,keepdim=True)/length.clamp_min(1e-8)))
        torch.testing.assert_close(change.norm(dim=-1),delta.norm(dim=-1),rtol=5e-5,atol=5e-6)
        z[ids] = given+change
        valid = torch.arange(width,device=pos.device)[None,:] <= pos[[selected[i] for i in ids],None]
        backup = int((fallback.squeeze(-1) & valid).sum())
        total = sum(len(own[selected[i]]["ids"]) for i in ids)
    return (torch.tensor(selected,device=pos.device),z), backup, total


def step(f, examples, requests, audit=False):
    jobs = dict.fromkeys((i,p,"ids") for i,_,_,_,p in requests)
    jobs.update(dict.fromkeys((i,p,"unrelated") for i,_,_,a,p in requests if a=="unrelated"))
    bank = snapshots(f, examples, list(jobs), audit)
    tokens, diagnostics = {}, {}
    fallback, positions = 0, 0
    for phase,direction in enumerate(DIRECTIONS):
        queued = [k for k in requests if k[2] == direction]
        for at in range(0,len(queued),24):
            group = queued[at:at+24]
            own = [bank[i,p,"ids"] for i,_,_,_,p in group]
            ids,pos = pad([x["ids"] for x in own])
            start = pad_sequence([b["start"] for b in own],batch_first=True) if phase else None
            inject = {}
            for l,b in SITES:
                value,b0,t0 = match(group,bank,own,pos,(l,b),phase,ids.size(1))
                fallback += b0
                positions += t0
                if value is not None:
                    inject[l,b] = value
            replay = Replay(pos,inject,audit)
            trace = Trace("edited") if audit else None
            with suffix_context(f,audit) as kwargs:
                after = f.run(ids,1,start=start,edit=replay,record=trace)[0]
            pred = output(f.model,after,pos).argmax(-1)
            if audit:
                physical = trace.states[0,STOP]
                for l in range(STOP,LAST+1):
                    physical = f.layers[l](physical,**kwargs[l])
                valid = torch.arange(ids.size(1),device=ids.device)[None,:] <= pos[:,None]
                torch.testing.assert_close(physical[valid],trace.states[0,LAST][valid],rtol=8e-5,atol=2e-4)
                assert torch.equal(output(f.model,f.core.norm(physical),pos).argmax(-1),pred)
                assert replay.audit_checks == sum(len(z[0]) for z in inject.values())
                for j,k in enumerate(group):
                    n = len(own[j]["ids"])
                    natural = own[j]["states"]
                    torch.testing.assert_close(trace.states[0,START][j,:n],natural[phase,START],rtol=8e-5,atol=2e-4)
                    err = (physical[j,:n]-trace.states[0,LAST][j,:n]).abs().max().item()
                    identity = 0.
                    if k[3] == "identity":
                        z = trace.states[0,LAST][j,:n]
                        torch.testing.assert_close(z,natural[phase,LAST],rtol=8e-5,atol=2e-4)
                        identity = (z-natural[phase,LAST]).abs().max().item()
                    diagnostics[k[:-1]] = (err,identity,len(MODES[k[1]])*(STOP-START))
            tokens.update((k[:-1],z) for k,z in zip(group,pred.tolist(),strict=True))
    return tokens, diagnostics, fallback, positions


def decode(f,tok,examples):
    active = {(i,m,d,a):[] for i in range(len(examples)) for m in MODES for d in DIRECTIONS for a in ARMS}
    active.update({(i,"both",d,"identity"):[] for i in range(len(examples)) for d in DIRECTIONS})
    done = {}
    backup,total = 0,0
    for _ in range(MAX_NEW):
        if not active:
            break
        tokens,_,b,t = step(f,examples,[(*k,tuple(p)) for k,p in active.items()])
        backup += b
        total += t
        for k,p in list(active.items()):
            p.append(tokens[k])
            if terminated(tok,p):
                done[k] = active.pop(k)
    done.update(active)
    decoded = dict(zip(done,text_results(tok,list(done.values())),strict=True))
    result = []
    for i,r in enumerate(examples):
        identity = {d:decoded[i,"both",d,"identity"] for d in DIRECTIONS}
        assert list(identity.values()) == r["native"]["original"]
        pred = {m:{d:{a:decoded[i,m,d,a] for a in ARMS} for d in DIRECTIONS} for m in MODES}
        result.append(dict(identity=identity,pred=pred))
    return result,backup,total


@torch.inference_mode()
def audit_case(f,tok,row):
    prefix = tuple(whitespace(tok,row["native"]["original"]))
    req = [(0,m,d,"authentic",prefix) for d in DIRECTIONS for m in MODES]
    req += [(0,"both",d,"identity",prefix) for d in DIRECTIONS]
    _,diag,_,_ = step(f,[row],req,audit=True)
    return dict(suffix=max(x[0] for x in diag.values()),identity=max(x[1] for x in diag.values()),checks=sum(x[2] for x in diag.values()))


@torch.inference_mode()
def run(model,root):
    m,tok = setup(model)
    f = Forward(m)
    start = timer()
    src = root/"NATIVE.json"
    examples = [prepare(tok,r) for r in load(src)["rows"] if repair(r)]
    audits = {d:sorted((r["id"] for r in examples if r["domain"]==d),key=lambda x:short_hash("midwrites audit",x))[:8] for d in DOMAINS}
    audited = {i for d in audits.values() for i in d}
    c = root/"COHORT.json"
    plan = dict(native_sha256=sha(src),repair=[r["id"] for r in examples],audit=audits)
    if c.exists():
        assert load(c) == plan
    else:
        save(c,plan)
    p = root/"MID_RAW.json"
    record = load(p) if p.exists() else dict(rows=[],cohort_sha256=sha(c),elapsed=0.,fallback=0,positions=0)
    rows = record["rows"]
    assert record["cohort_sha256"] == sha(c) and [r["id"] for r in rows] == plan["repair"][:len(rows)]
    elapsed = record["elapsed"]
    for j in range(len(rows),len(examples),8):
        batch = examples[j:j+8]
        out,b,t = decode(f,tok,batch)
        for r,z in zip(batch,out,strict=True):
            item = dict({k:v for k,v in r.items() if k not in ("ids","unrelated")},**z)
            if r["id"] in audited:
                item["audit"] = audit_case(f,tok,r)
            rows.append(item)
        record.update(elapsed=elapsed+timer()-start,peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,1),
                      fallback=record["fallback"]+b,positions=record["positions"]+t)
        save(p,record)
        print("midwrite replay",len(rows),len(examples),flush=True)
    summary = summarize(rows)
    save(root/"MID_SUMMARY.json",summary)
    print({k:v for k,v in summary.items() if k!="domains"},flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a = p.parse_args()
    run(a.model,a.root)
