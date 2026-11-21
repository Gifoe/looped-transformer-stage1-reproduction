"""Contrast target-row K2 values with the current incoming K1 residual."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from conf_deck import DOMAINS
from conf_pair import qualified
from conf_stats import ARMS,DIRECTIONS,summarize
from edge import short_hash
from interchange import terminated
from native import Forward,load,output,pad,save,setup,sha,timer
from rule import MAX_NEW,text_results
from value_test import shuffle

LAYERS = range(24)
V_ARMS = ("value","value_geometry","value_unrelated","both","clean")
C_ARMS = ("carry","carry_geometry","carry_unrelated","both","clean")


def spans(device,indices):
    r,c = zip(*((j,p) for j,positions in enumerate(indices) for p in positions),strict=True)
    return torch.tensor(r,device=device),torch.tensor(c,device=device)


@contextmanager
def native_v(f,rows,pos):
    values,count = {},defaultdict(int)

    def pull(_,args,x,l):
        count[l]+=1
        if count[l]==2:
            values[l] = x[rows,pos].detach()

    hooks = [f.layers[l].self_attn.v_proj.register_forward_hook(lambda m,a,x,l=l: pull(m,a,x,l)) for l in LAYERS]
    try:
        yield values
    finally:
        for h in hooks:
            h.remove()


def snapshots(f,examples,keys):
    bank = {}
    for start in range(0,len(keys),20):
        block = keys[start:start+20]
        sequences = [examples[i]["span"]["tokens"][s]+list(p) for i,s,p in block]
        ids,at = pad(sequences)
        sites = [examples[i]["span"]["positions"] for i,_,_ in block]
        rr,pp = spans(ids.device,sites)
        with native_v(f,rr,pp) as captured:
            c,k2 = f.run(ids,2)
        assert set(captured)==set(LAYERS)
        native = output(f.model,k2,at).argmax(-1).tolist() if any(not p for _,_,p in block) else None
        lengths = [len(x) for x in sites]
        by_layer = [captured[l].split(lengths) for l in LAYERS]
        for j,(i,s,p) in enumerate(block):
            bank[i,s,p] = dict(ids=sequences[j],carry=c[j,:len(sequences[j])],target=[x[j] for x in by_layer])
            if not p:
                assert native[j]==examples[i]["pair_native"][s]["tokens"][0]
    return bank


def control(base,donor,other,arm):
    if arm in ("value","carry","both"):
        return donor,0,0
    if arm=="clean":
        return base,0,0
    d = donor-base
    g = shuffle(base,d)
    if arm.endswith("geometry"):
        return g,0,0
    q = other-base
    length = q.norm(dim=-1,keepdim=True)
    zero = length<=1e-8
    e = torch.where(zero,g-base,q*(d.norm(dim=-1,keepdim=True)/length.clamp_min(1e-8)))
    torch.testing.assert_close(e.norm(dim=-1),d.norm(dim=-1),rtol=9e-5,atol=1e-5)
    return base+e, int(zero.sum()),zero.numel()


class Audit:
    def __init__(self,f,pos):
        self.f = f
        self.pos = pos
        self.row = torch.arange(len(pos),device=pos.device)
        self.v = None
        self.a = None
        self.err = None

    def put(self,key,name,x):
        if name=="v":
            self.v = self.f.module.repeat_kv(x,self.f.layers[key[1]].self_attn.num_key_value_groups)
        if name=="a":
            self.a = x[self.row,:,self.pos,:].unsqueeze(-2)
        if name=="av":
            y = (self.a@self.v).squeeze(-2)
            actual = x[self.row,self.pos]
            torch.testing.assert_close(y,actual,rtol=5e-5,atol=2e-5)
            z = (y-actual).abs().amax(dim=(1,2))
            self.err = z if self.err is None else torch.maximum(z,self.err)


@contextmanager
def install(f,rows,pos,patch,baseline,v_mask,audit):
    errors = []

    def edit(_,args,x,l):
        if not audit:
            if not len(rows):
                return x
            y = x.clone()
            y[rows,pos] = patch[l]
            return y
        all_rows,all_pos,reference = baseline
        same = ~v_mask
        diff = (x[all_rows[same],all_pos[same]]-reference[l][same]).abs()
        torch.testing.assert_close(x[all_rows[same],all_pos[same]],reference[l][same],rtol=8e-5,atol=3e-4)
        errors.append(float(diff.max()) if diff.numel() else 0.)
        if not len(rows):
            return x
        y = x.clone()
        y[rows,pos] = patch[l]
        assert torch.equal(y[rows,pos],patch[l])
        mask = torch.ones(x.shape[:2],dtype=torch.bool,device=x.device)
        mask[rows,pos] = False
        assert torch.equal(x[mask],y[mask])
        return y

    handles = [f.layers[l].self_attn.v_proj.register_forward_hook(lambda m,a,x,l=l: edit(m,a,x,l)) for l in LAYERS]
    try:
        yield errors
    finally:
        for h in handles:
            h.remove()


def step(f,examples,requests,audit=False):
    def other(s):
        return "swapped" if s=="original" else "original"

    keys = dict.fromkeys((i,s,p) for i,s,_,p in requests)
    keys.update(dict.fromkeys((i,other(s),p) for i,s,a,p in requests if a!="clean"))
    keys.update(dict.fromkeys((i,"unrelated_"+s,p) for i,s,a,p in requests if a.endswith("unrelated")))
    bank = snapshots(f,examples,list(keys))
    result,diag = {},{}
    fallback = {p:[0,0] for p in ("value","carry")}
    for offset in range(0,len(requests),24):
        group = requests[offset:offset+24]
        rec = [bank[i,s,p] for i,s,_,p in group]
        donors = [bank.get((i,other(s),p),r) for (i,s,_,p),r in zip(group,rec,strict=True)]
        irrelevant = [bank.get((i,"unrelated_"+s,p)) for i,s,_,p in group]
        seq,at = pad([r["ids"] for r in rec])
        state = pad_sequence([r["carry"] for r in rec],batch_first=True)
        raw = state.clone() if audit else None
        carry_rows = [j for j,(_,_,a,_) in enumerate(group) if a in C_ARMS]
        for j in carry_rows:
            a = group[j][2]
            alt = irrelevant[j]
            h,n,total = control(rec[j]["carry"][-1],donors[j]["carry"][-1],None if alt is None else alt["carry"][-1],a)
            state[j,at[j]] = h
            fallback["carry"][0]+=n
            fallback["carry"][1]+=total
            if audit:
                assert torch.equal(state[j,at[j]],h)
        if audit:
            cm = torch.ones(state.shape[:2],device=seq.device,dtype=torch.bool)
            cm[carry_rows,at[carry_rows]] = False
            assert torch.equal(raw[cm],state[cm])
        positions = [examples[i]["span"]["positions"] for i,_,_,_ in group]
        vi = [j for j,(_,_,a,_) in enumerate(group) if a in V_ARMS]
        row = torch.tensor([j for j in vi for _ in positions[j]],device=seq.device,dtype=torch.long)
        cols = torch.tensor([u for j in vi for u in positions[j]],device=seq.device,dtype=torch.long)
        patch = []
        if vi:
            for l in LAYERS:
                values = {}
                for arm in V_ARMS:
                    use = [j for j in vi if group[j][2]==arm]
                    if not use:
                        continue
                    hd = f.layers[l].self_attn.head_dim
                    u,v = [torch.cat([s[j]["target"][l] for j in use]).unflatten(-1,(-1,hd)) for s in (rec,donors)]
                    alt = torch.cat([irrelevant[j]["target"][l] for j in use]).unflatten(-1,(-1,hd)) if arm.endswith("unrelated") else None
                    x,n,t = control(u,v,alt,arm)
                    fallback["value"][0]+=n
                    fallback["value"][1]+=t
                    values.update(zip(use,x.flatten(-2).split([len(positions[j]) for j in use]),strict=True))
                patch.append(torch.cat([values[j] for j in vi]))
        reference,mask = None,None
        if audit:
            m,p = spans(seq.device,positions)
            reference = (m,p,[torch.cat([r["target"][l] for r in rec]) for l in LAYERS])
            mask = torch.tensor([j in vi for j,pp in enumerate(positions) for _ in pp],device=seq.device)
        record = Audit(f,at) if audit else None
        with install(f,row,cols,patch,reference,mask,audit) as errors:
            hidden = f.run(seq,1,start=state,record=record)[0]
        chosen = output(f.model,hidden,at).argmax(-1).tolist()
        result.update((k[:-1],n) for k,n in zip(group,chosen,strict=True))
        if audit:
            for j,k in enumerate(group):
                diag[k[:-1]] = dict(v_sites=len(LAYERS) if j in vi else 0,carry_sites=int(j in carry_rows),tokens=1,
                                    cache_causal=max(errors),attention=float(record.err[j]))
    return result,diag,fallback


def decode(f,tok,examples):
    active = {(i,s,a):[] for i in range(len(examples)) for s in DIRECTIONS for a in ARMS}
    done = {}
    counts = {p:[0,0] for p in ("value","carry")}
    for _ in range(MAX_NEW):
        if not active:
            break
        out,_,changed = step(f,examples,[(*k,tuple(p)) for k,p in active.items()])
        for p in counts:
            counts[p] = [a+b for a,b in zip(counts[p],changed[p],strict=True)]
        for k,p in list(active.items()):
            p.append(out[k])
            if terminated(tok,p):
                done[k] = active.pop(k)
    done.update(active)
    ans = dict(zip(done,text_results(tok,list(done.values())),strict=True))
    results = [{s:{a:ans[i,s,a] for a in ARMS} for s in DIRECTIONS} for i in range(len(examples))]
    for r,ans in zip(examples,results,strict=True):
        assert all(ans[s]["clean"]==r["pair_native"][s] for s in DIRECTIONS)
    return results,counts


def audit_case(f,row):
    _,d,_ = step(f,[row],[(0,s,a,()) for s in DIRECTIONS for a in ARMS],audit=True)
    return {k:(max if k in ("cache_causal","attention") else sum)(x[k] for x in d.values()) for k in next(iter(d.values()))}


def auditors(examples):
    return {d:{g:sorted((r["id"] for r in examples if (r["domain"],r["kind"])==(d,g)),
                        key=lambda x:short_hash("conflict projection audit",x))[:4] for g in ("repair","stable")} for d in DOMAINS}


@torch.inference_mode()
def run(model,root):
    m,tok = setup(model)
    f = Forward(m)
    started = timer()
    pair = root/"PAIRS.json"
    examples = [r for r in load(pair)["rows"] if qualified(r)]
    check = auditors(examples)
    audited = {i for d in check.values() for g in d.values() for i in g}
    lock = root/"COHORT.json"
    plan = dict(pair_sha256=sha(pair),cases=[r["id"] for r in examples],audit=check)
    if lock.exists():
        assert load(lock)==plan
    else:
        save(lock,plan)
    raw = root/"CONFLICT_RAW.json"
    record = load(raw) if raw.exists() else dict(rows=[],fallback={p:[0,0] for p in ("value","carry")},elapsed=0.,cohort_sha256=sha(lock))
    rows = record["rows"]
    assert record["cohort_sha256"]==sha(lock) and [r["id"] for r in rows]==plan["cases"][:len(rows)]
    elapsed = record["elapsed"]
    for offset in range(len(rows),len(examples),10):
        batch = examples[offset:offset+10]
        ans,counts = decode(f,tok,batch)
        for r,p in zip(batch,ans,strict=True):
            q = dict({k:v for k,v in r.items() if k!="span"},pred=p)
            if r["id"] in audited:
                q["audit"] = audit_case(f,r)
            rows.append(q)
        for path in counts:
            record["fallback"][path] = [a+b for a,b in zip(record["fallback"][path],counts[path],strict=True)]
        record.update(elapsed=elapsed+timer()-started,peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,1))
        save(raw,record)
        print("conflict replay",len(rows),len(examples),flush=True)
    z = summarize(rows)
    save(root/"CONFLICT_SUMMARY.json",z)
    print({k:v for k,v in z.items() if k!="domains"},flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a = p.parse_args()
    run(a.model,a.root)
