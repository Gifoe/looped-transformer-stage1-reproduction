"""Replay factored native attention and independently recompute the suffix."""

import argparse
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn.functional import linear, pad as pad_tensor
from torch.nn.utils.rnn import pad_sequence

from av_deck import DOMAINS, prompt, repair
from av_stats import ARMS, DIRECTIONS, summarize
from edge import short_hash
from interchange import terminated
from mid_test import suffix_context
from native import Forward, load, output, pad, save, setup, sha, timer
from query_test import whitespace
from rule import MAX_NEW, text_results
from value_test import shuffle

START, STOP, LAST = 11, 17, 23
SITES = range(START,STOP)


class Trace:
    def __init__(self, operands, audit):
        self.values = {}
        self.states = {}
        self.operands = operands
        self.audit = audit

    def put(self,key,name,value):
        k,l = key
        if self.operands and l in SITES and name in ("a","v","att_projected"):
            self.values[k,l,name] = value.detach()
        if self.audit and (l,name) in ((START,"input"),(STOP,"input"),(LAST,"output")):
            self.states[k,l] = value.detach()


class Replay:
    def __init__(self, positions, projected, audit):
        self.positions = positions
        self.values = projected
        self.audit = audit
        self.checks = 0

    @contextmanager
    def apply(self,f):
        def patch(_,args,x,l):
            n = torch.arange(x.size(1),device=x.device)[None,:] <= self.positions[:,None]
            y = torch.where(n[...,None],self.values[l],x)
            if self.audit:
                assert torch.equal(y[n],self.values[l][n])
                self.checks += x.size(0)
            return y

        hooks = [f.layers[l].self_attn.o_proj.register_forward_hook(lambda m,a,x,l=l: patch(m,a,x,l)) for l in SITES]
        try:
            yield
        finally:
            for h in hooks:
                h.remove()


def prepare(tok,row):
    old,k2 = row["native"]["original"]
    assert k2["number"]==row["target"] and old["number"] in row["example"]["original"]["values"]
    return dict(row,old=old["number"],ids=tok.encode(prompt(row,"original"),add_special_tokens=False))


def projected(f,trace,l,audit):
    block = f.layers[l].self_attn
    A0,A1 = [trace.values[k,l,"a"] for k in (0,1)]
    V0,V1 = [f.module.repeat_kv(trace.values[k,l,"v"],block.num_key_value_groups) for k in (0,1)]
    clean = [trace.values[k,l,"att_projected"] for k in (0,1)]

    def packed(a,v):
        return (a@v).transpose(1,2).contiguous().flatten(2)

    c10,c01 = [block.o_proj(packed(a,v)) for a,v in ((A1,V0),(A0,V1))]
    cross = linear(packed(A1-A0,V1-V0),block.o_proj.weight)
    check_native,check_factor = 0.,0.
    if audit:
        rebuilt = [block.o_proj(packed(a,v)) for a,v in ((A0,V0),(A1,V1))]
        for real,replay in zip(clean,rebuilt,strict=True):
            torch.testing.assert_close(real,replay,rtol=6e-5,atol=1e-5)
            check_native = max(check_native,(real-replay).abs().max().item())
        decomposition = (clean[0]-c10)+(clean[1]-c01)
        torch.testing.assert_close(cross,decomposition,rtol=1e-4,atol=4e-5)
        check_factor = (cross-decomposition).abs().max().item()
    result = []
    for k,(route,value) in enumerate(((c10,c01),(c01,c10))):
        joint = clean[1-k]
        result.append(dict(route=route,value=value,joint=joint,minus_interaction=joint-cross,
                           minus_control=shuffle(joint,-cross),identity=clean[k]))
    return result,(check_native,check_factor)


def snapshots(f,examples,keys,audit):
    bank = {}
    for start in range(0,len(keys),16):
        group = keys[start:start+16]
        seq = [examples[i]["ids"]+list(p) for i,p in group]
        ids,at = pad(seq)
        tr = Trace(True,audit)
        h1,h2 = f.run(ids,2,record=tr)
        observed = [output(f.model,h,at).argmax(-1).tolist() for h in (h1,h2)] if any(not p for _,p in group) else None
        sites,checks = zip(*(projected(f,tr,l,audit) for l in SITES),strict=True)
        native,factor = map(max,zip(*checks,strict=True))
        for j,(i,p) in enumerate(group):
            n = len(seq[j])
            vectors = {k:{a:{l:s[k][a][j,:n] for l,s in zip(SITES,sites,strict=True)} for a in ARMS} for k in (0,1)}
            bank[i,p] = dict(ids=seq[j],start=h1[j,:n],vectors=vectors,states={k:x[j,:n] for k,x in tr.states.items()},
                             factorization=factor,native_projection=native)
            if not p:
                tokens = [x["tokens"][0] for x in examples[i]["native"]["original"]]
                assert [t[j] for t in observed] == tokens
    return bank


def step(f,examples,requests,audit=False):
    bank = snapshots(f,examples,list(dict.fromkeys((i,p) for i,_,_,p in requests)),audit)
    tokens,diag = {},{}
    for phase,direction in enumerate(DIRECTIONS):
        queued = [k for k in requests if k[1]==direction]
        for offset in range(0,len(queued),24):
            group = queued[offset:offset+24]
            own = [bank[i,p] for i,_,_,p in group]
            ids,pos = pad([r["ids"] for r in own])
            h = pad_sequence([r["start"] for r in own],batch_first=True) if phase else None
            writes = {}
            for l in SITES:
                z = pad_sequence([r["vectors"][phase][a][l] for r,(_,_,a,_) in zip(own,group,strict=True)],batch_first=True)
                writes[l] = pad_tensor(z,(0,0,0,ids.size(1)-z.size(1)))
            edit = Replay(pos,writes,audit)
            tr = Trace(False,audit) if audit else None
            with edit.apply(f),suffix_context(f,audit) as kwargs:
                after = f.run(ids,1,start=h,record=tr)[0]
            nxt = output(f.model,after,pos).argmax(-1)
            tokens.update((k[:-1],v) for k,v in zip(group,nxt.tolist(),strict=True))
            if not audit:
                continue
            assert edit.checks==len(group)*len(SITES)
            physical = tr.states[0,STOP]
            for l in range(STOP,LAST+1):
                physical = f.layers[l](physical,**kwargs[l])
            assert torch.equal(nxt,output(f.model,f.core.norm(physical),pos).argmax(-1))
            for j,(r,k) in enumerate(zip(own,group,strict=True)):
                n = len(r["ids"])
                torch.testing.assert_close(tr.states[0,START][j,:n],r["states"][phase,START],rtol=8e-5,atol=2e-4)
                raw = tr.states[0,LAST][j,:n]
                torch.testing.assert_close(raw,physical[j,:n],rtol=8e-5,atol=2e-4)
                err = (raw-physical[j,:n]).abs().max().item()
                identity = 0.
                if k[2]=="identity":
                    old = r["states"][phase,LAST]
                    torch.testing.assert_close(raw,old,rtol=8e-5,atol=3e-4)
                    identity = (raw-old).abs().max().item()
                diag[k[:-1]] = dict(checks=len(SITES),suffix=err,identity=identity,
                                    factorization=r["factorization"],native_projection=r["native_projection"])
    return tokens,diag


def decode(f,tok,examples):
    active = {(i,d,a):[] for i in range(len(examples)) for d in DIRECTIONS for a in ARMS}
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
    out = [{d:{a:values[i,d,a] for a in ARMS} for d in DIRECTIONS} for i in range(len(examples))]
    for i,r in enumerate(examples):
        assert [out[i][d]["identity"] for d in DIRECTIONS] == r["native"]["original"]
    return out


@torch.inference_mode()
def audit_case(f,tok,row):
    prefix = tuple(whitespace(tok,row["native"]["original"]))
    _,v = step(f,[row],[(0,d,a,prefix) for d in DIRECTIONS for a in ARMS],audit=True)
    return {k:(sum if k=="checks" else max)(x[k] for x in v.values()) for k in next(iter(v.values()))}


def auditors(examples):
    return {d:sorted((r["id"] for r in examples if r["domain"]==d),key=lambda x:short_hash("av replay audit",x))[:8] for d in DOMAINS}


@torch.inference_mode()
def run(model,root):
    m,tok = setup(model)
    f = Forward(m)
    started = timer()
    p = root/"NATIVE.json"
    examples = [prepare(tok,r) for r in load(p)["rows"] if repair(r)]
    audit = auditors(examples)
    audited = set().union(*audit.values())
    cohort = root/"COHORT.json"
    plan = dict(native_sha256=sha(p),repair=[r["id"] for r in examples],audit=audit)
    if cohort.exists():
        assert load(cohort)==plan
    else:
        save(cohort,plan)
    target = root/"AV_RAW.json"
    record = load(target) if target.exists() else dict(rows=[],cohort_sha256=sha(cohort),elapsed=0.)
    rows = record["rows"]
    assert record["cohort_sha256"]==sha(cohort) and [r["id"] for r in rows]==plan["repair"][:len(rows)]
    elapsed = record["elapsed"]
    for j in range(len(rows),len(examples),8):
        batch = examples[j:j+8]
        for r,answers in zip(batch,decode(f,tok,batch),strict=True):
            item = {k:v for k,v in r.items() if k!="ids"}
            item["pred"] = answers
            if r["id"] in audited:
                item["audit"] = audit_case(f,tok,r)
            rows.append(item)
        record.update(elapsed=elapsed+timer()-started,peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,1))
        save(target,record)
        print("av replay",len(rows),len(examples),flush=True)
    z = summarize(rows)
    save(root/"AV_SUMMARY.json",z)
    print({k:v for k,v in z.items() if k!="domains"},flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a = p.parse_args()
    run(a.model,a.root)
