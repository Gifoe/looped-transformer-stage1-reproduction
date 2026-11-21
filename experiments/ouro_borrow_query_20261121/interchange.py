"""Natural two-way query and residual-write tracing at the first answer divergence."""
import argparse
from collections import Counter
from pathlib import Path

import torch

from borrow import ci, native_event
from native import Forward, load, output, pad, save, setup, sha
from rule import MAX_NEW, greedy, prompt, text_results
from state import rotate_about

LAYERS=tuple(range(24))
WRITES=("a_write","m_write")


class Trace:
    def __init__(self,pos):
        self.pos=pos
        self.row=torch.arange(len(pos),device=pos.device)
        self.data={}

    def put(self,key,name,x):
        if name=="q":
            self.data[(*key,name)]=x[self.row,:,self.pos,:].detach()
        elif name in WRITES:
            self.data[(*key,name)]=x[self.row,self.pos].detach()

    def at(self,k,l,name):
        return self.data[(k,l,name)]


class Patch:
    def __init__(self,pos,query=None,writes=None):
        self.pos=pos
        self.rows=torch.arange(len(pos),device=pos.device)
        self.query=query or {}
        self.writes=writes or {}

    def resid(self,k,l,x):
        return x

    def qk(self,key,q,k):
        if key[1] not in self.query:
            return q,k
        heads,replace=self.query[key[1]]
        patched=q.clone()
        if heads is None:
            patched[self.rows,:,self.pos,:]=replace
        else:
            patched[self.rows,heads,self.pos,:]=replace
        return patched,k

    def attn(self,key,a,v):
        return a,v

    def mlp(self,key,z,record):
        return z

    def write(self,key,label,x):
        values=self.writes.get((key[1],label))
        if values is None:
            return x
        result=x.clone()
        result[self.rows,self.pos]=values
        return result


def text_case(tok,row,t):
    a,b=row["pairs"][t]
    text=prompt(a,b,1)
    ids=tok.encode(text,add_special_tokens=False)
    vals=[tok.encode(text+" "+str(row[k])+"\n",add_special_tokens=False)
          for k in ("target","foil")]
    assert all(x[:len(ids)]==ids for x in vals)
    for j,(target,foil) in enumerate(zip(*vals)):
        if target!=foil:
            prefix=vals[0][len(ids):j]
            given=[z[t] for z in row["pred"]]
            hit=[g["tokens"][:len(prefix)]==prefix and len(g["tokens"])>len(prefix) for g in given]
            if native_event(row,t):
                # Derive the switch step from the native generated continuations.
                q,p=[op["tokens"] for op in given]
                k=next(i for i,(f,w) in enumerate(zip(q,p)) if f!=w)
                assert q[:k]==p[:k]
                verified=(q[k],p[k])==(foil,target) and q[:k]==prefix
                hit=[verified,verified]
            return dict(id=row["id"],target=row["target"],foil=row["foil"],kind=("repair" if t else "safe"),
                        native=given,ids=ids,prefix=prefix,reach=hit,labels=[target,foil])
    raise ValueError((row["id"],row["target"],row["foil"]))


def load_cases(root,tok,split):
    rows=[r for r in load(root/"NATIVE.json")["rows"] if r["split"]==split and native_event(r,1)]
    rows.sort(key=lambda r:r["id"])
    return rows,[text_case(tok,r,1) for r in rows]


def batch(f,cases):
    ids,at=pad([c["ids"]+c["prefix"] for c in cases])
    trace=Trace(at)
    h1,h2=f.run(ids,2,record=trace)
    assert torch.equal(f.run(ids,1,start=h1)[0],h2)
    return ids,at,trace,h1,h2


def fit_margin(f,h,at,labels):
    big=output(f.model,h,at)
    pos=torch.arange(len(big),device=big.device)
    return big[pos,labels[:,0]]-big[pos,labels[:,1]]


def select_scores(f,cases,options):
    record=[]
    for i in range(0,len(cases),12):
        a=cases[i:i+12]
        ids,pos,trace,h1,h2=batch(f,a)
        labels=torch.tensor([c["labels"] for c in a],device=ids.device)
        native=[fit_margin(f,h,pos,labels) for h in (h1,h2)]
        rows=[]
        for name,l,head in options:
            effect=[]
            for k in (0,1):
                other=1-k
                d=trace.at(other,l,name)
                if name=="q":
                    q={l:(head,d if head is None else d[:,head])}
                    edit=Patch(pos,query=q)
                else:
                    edit=Patch(pos,writes={(l,name):d})
                h=f.run(ids,1,start=h1 if k else None,edit=edit)[0]
                current=fit_margin(f,h,pos,labels)
                effect.append((current-native[k])*(1 if k==0 else -1))
            rows.append(torch.stack(effect,dim=-1))
        s=torch.stack(rows,dim=1).tolist()
        for j,c in enumerate(a):
            record.append(dict(id=c["id"],native_margin=[z[j].item() for z in native],effect=s[j]))
    scores=[]
    for i,(label,l,head) in enumerate(options):
        means=[sum(p["effect"][i][o] for p in record)/len(record) if record else 0.0 for o in (0,1)]
        scores.append(dict(site=label,layer=l,head=head,mean_directional_nats=means,rank=min(means)))
    return scores,record


@torch.inference_mode()
def discover(f,tok,root):
    rows,all_items=load_cases(root,tok,"discovery")
    cases=[c for c in all_items if all(c["reach"])]
    layer_options=[("q",l,None) for l in LAYERS]
    write_options=[(w,l,None) for l in LAYERS for w in WRITES]
    scores,raw=select_scores(f,cases,layer_options+write_options)
    layer=max(scores[:24],key=lambda r:r["rank"])["layer"]
    chosen_write=max(scores[24:],key=lambda r:r["rank"])
    by_head,head_data=select_scores(f,cases,[("q",layer,h) for h in range(f.core.config.num_attention_heads)])
    best=max(by_head,key=lambda r:r["rank"])["head"]
    low=min((r for r in by_head if r["head"]!=best),key=lambda r:r["rank"])["head"]
    result=dict(layer=layer,head=best,low_head=low,write=chosen_write,n_discovery=len(cases),
                all_native_repairs=len(rows),layer_scores=scores,head_scores=by_head,
                by_case=raw,head_by_case=head_data,native_sha256=sha(root/"NATIVE.json"))
    save(root/"SELECTION.json",result)
    print("selection",layer,best,low,chosen_write,"n",len(cases),flush=True)
    return result


def independent_like(x,cases,l,n):
    draws=[]
    for c in cases:
        seed=261123100+sum((i+1)*ord(p) for i,p in enumerate(c["id"]+c["kind"]))+l*103+n
        rng=torch.Generator(device=x.device).manual_seed(seed)
        draws.append(torch.randn(x.shape[1:],device=x.device,dtype=x.dtype,generator=rng))
    return torch.stack(draws)


def alt_head(r,d,own,other):
    norm=lambda x:x.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    u=r/norm(r)
    v=own/norm(own)
    delta=d-r
    parallel=(delta*u).sum(dim=-1,keepdim=True)
    dist=(delta-parallel*u).norm(dim=-1,keepdim=True)
    w=other-own
    new=w-(w*v).sum(dim=-1,keepdim=True)*v
    displacement=parallel*v+dist*new/norm(new)
    torch.testing.assert_close(displacement.norm(dim=-1),delta.norm(dim=-1),rtol=1e-5,atol=1e-5)
    return own+displacement


MODES=("clean","head","head_sham","low_head","layer","layer_sham","stack","stack_sham",
       "write","write_sham","head_graphcut")


def edited(s,mode,k,cases,pos,trace):
    l=s["layer"]
    head=s["head"]
    o=1-k
    if mode=="clean":
        return None
    if mode in ("head","head_sham","low_head","head_graphcut"):
        original=trace.at(k,l,"q")[:,head]
        donor=trace.at(o,l,"q")[:,head]
        writes={}
        if mode=="head_sham":
            donor=rotate_about(original,donor,independent_like(donor,cases,l,0))
        if mode=="low_head":
            head=s["low_head"]
            donor=alt_head(original,donor,trace.at(k,l,"q")[:,head],trace.at(o,l,"q")[:,head])
        if mode=="head_graphcut":
            writes={(l,"a_write"):trace.at(k,l,"a_write")}
        return Patch(pos,query={l:(head,donor)},writes=writes)
    if mode.startswith(("layer","stack")):
        across=LAYERS if mode.startswith("stack") else (l,)
        q={}
        for j in across:
            v=trace.at(o,j,"q")
            if mode.endswith("sham"):
                v=rotate_about(trace.at(k,j,"q"),v,independent_like(v,cases,j,1))
            q[j]=(None,v)
        return Patch(pos,query=q)
    w=s["write"]
    j=w["layer"]
    n=w["site"]
    d=trace.at(o,j,n)
    if mode.endswith("sham"):
        d=rotate_about(trace.at(k,j,n),d,independent_like(d,cases,j,2))
    return Patch(pos,writes={(j,n):d})


def terminated(tok,x):
    return x[-1]==tok.eos_token_id or "\n" in tok.decode(x,skip_special_tokens=True).strip()


def continue_greedy(f,tok,cases,pieces,k):
    tokens=[list(x) for x in pieces]
    active=[i for i,x in enumerate(tokens) if not terminated(tok,x) and len(x)<MAX_NEW]
    while active:
        ids,at=pad([cases[j]["ids"]+tokens[j] for j in active])
        new=output(f.model,f.run(ids,k)[-1],at).argmax(-1).tolist()
        left=[]
        for i,v in zip(active,new,strict=True):
            tokens[i].append(v)
            if not terminated(tok,tokens[i]) and len(tokens[i])<MAX_NEW:
                left.append(i)
        active=left
    return text_results(tok,tokens)


def complete(f,tok,cases,pred,k):
    answers=[c["native"][k] for c in cases]
    changed=[]
    parts=[]
    for i,(c,new) in enumerate(zip(cases,pred,strict=True)):
        if not c["reach"][k]:
            continue
        p=len(c["prefix"])
        if new!=c["native"][k]["tokens"][p]:
            changed.append(i)
            parts.append(c["prefix"]+[new])
    if changed:
        y=continue_greedy(f,tok,[cases[i] for i in changed],parts,k+1)
        for i,v in zip(changed,y,strict=True):
            answers[i]=v
    return answers


def decoded(f,tok,cases,s):
    seq,at,tr,h1,h2=batch(f,cases)
    for k,native in enumerate((h1,h2),1):
        actual=greedy(f,tok,[c["ids"] for c in cases],k)
        assert [x["tokens"] for x in actual]==[c["native"][k-1]["tokens"] for c in cases]
        z=output(f.model,native,at).argmax(-1).tolist()
        assert all(not c["reach"][k-1] or z[j]==c["native"][k-1]["tokens"][len(c["prefix"])]
                   for j,c in enumerate(cases))
    result={}
    physical={}
    for mode in MODES:
        yy=[]
        evidence=[]
        for k,native in enumerate((h1,h2)):
            edit=edited(s,mode,k,cases,at,tr)
            h=native if edit is None else f.run(seq,1,start=h1 if k else None,edit=edit)[0]
            if mode=="head_graphcut":
                assert torch.equal(h,native)
            z=output(f.model,h,at).argmax(-1).tolist()
            yy.append(complete(f,tok,cases,z,k))
            evidence.append(z)
        result[mode]=list(zip(*yy,strict=True))
        physical[mode]=list(zip(*evidence,strict=True))
    return result,physical


def summarize(rows):
    agg={}
    for part in ("repair","safe"):
        cases=[p for p in rows if p["kind"]==part]
        arms={}
        for arm in MODES:
            c=Counter()
            bins=set()
            for r in cases:
                p,q=[v["number"] for v in r["pred"][arm]]
                correct=r["target"]
                foil=r["foil"]
                c["K1_correct"]+=p==correct
                c["K2_correct"]+=q==correct
                c["K1_foil"]+=p==foil
                c["K2_foil"]+=q==foil
                c["K1_unparsed"]+=p is None
                c["K2_unparsed"]+=q is None
                c["joint"]+=p==correct and q==foil
                if p==correct and q==foil:
                    bins.add(correct)
                y,z=[v["number"] for v in r["pred"]["clean"]]
                c["wrong_created_K1"]+=y==correct and p!=correct
                c["wrong_created_K2"]+=z==correct and q!=correct
            c["joint_target_bins"]=len(bins)
            arms[arm]=dict(c)
        agg[part]=dict(n=len(cases),both_reach=sum(all(r["reach"]) for r in cases),modes=arms)
    repair=[r for r in rows if r["kind"]=="repair"]
    cmp={}
    for x,y in (("head","head_sham"),("head","low_head"),("layer","layer_sham"),
                ("stack","stack_sham"),("write","write_sham")):
        results={key:{} for key in ("joint","K1","K2")}
        for r in repair:
            t=r["target"]; u=r["foil"]
            def got(s):
                p,q=[e["number"] for e in r["pred"][s]]
                return {"joint":p==t and q==u,"K1":p==t,"K2":q==u}
            a,b=got(x),got(y)
            for key in results:
                tab=results[key]
                tab[t]=tab.get(t,0)+int(a[key])-int(b[key])
        levels=(.025,.975) if x=="head" else (.0125,.9875)
        cmp[x+"_vs_"+y]={s:ci(list(v.values()),levels) for s,v in results.items()}
    g=agg["repair"]["modes"]["head"]
    agg["comparisons"]=cmp
    agg["primary_gate"]=bool(g["joint"]>=8 and g["joint_target_bins"]>=6 and
        all(cmp[f"head_vs_{m}"][key]["interval"] and cmp[f"head_vs_{m}"][key]["interval"][0]>0
            for m in ("head_sham","low_head") for key in ("joint","K1","K2")))
    return agg


@torch.inference_mode()
def run(args):
    root=args.out
    assert load(root/"NATIVE_SUMMARY.json")["gate"]
    m,tok=setup(args.model)
    f=Forward(m)
    lock=root/"SELECTION.json"
    s=load(lock) if lock.exists() else discover(f,tok,root)
    rows,repairs=load_cases(root,tok,"confirmation")
    all_cases=[item for pair in zip(repairs,[text_case(tok,x,0) for x in rows],strict=True) for item in pair]
    path=root/"QUERY_RAW.json"
    all_rows=load(path)["rows"] if path.exists() else []
    assert [(r["id"],r["kind"]) for r in all_rows]==[(c["id"],c["kind"]) for c in all_cases[:len(all_rows)]]
    for i in range(len(all_rows),len(all_cases),12):
        pack=all_cases[i:i+12]
        result,tokens=decoded(f,tok,pack,s)
        for j,c in enumerate(pack):
            z={k:v for k,v in c.items() if k not in ("ids","native")}
            z["pred"]={k:v[j] for k,v in result.items()}
            z["pivot_token"]={k:v[j] for k,v in tokens.items()}
            all_rows.append(z)
        save(path,dict(rows=all_rows,selection_sha256=sha(lock),native_sha256=sha(root/"NATIVE.json")))
        print("confirm query",len(all_rows),len(all_cases),flush=True)
    summary=summarize(all_rows)
    save(root/"QUERY_SUMMARY.json",summary)
    print(summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    run(p.parse_args())

