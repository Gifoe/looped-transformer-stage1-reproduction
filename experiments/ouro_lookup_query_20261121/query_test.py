"""Test one output-query layer for bidirectional lookup and remapped retrieval."""
import argparse
from pathlib import Path

import torch

from edge import short_hash
from interchange import Patch, terminated
from native import Forward, load, output, pad, save, setup, sha, timer
from query_deck import KEYS, PREFIX, VARIANTS, eligible, text
from query_stats import CONTROLS, LATTICE, summarize
from rule import MAX_NEW, text_results
from value_test import shuffle


def encode(tok,r,variant):
    e=r["example"][variant]
    prompt=text(r["key"],**e)
    token=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
    offsets=[tuple(x) for x in token["offset_mapping"]]
    idx=len(PREFIX)+len(f"Query = {r['key']}; Table = ")
    positions={}
    for k in e["order"]:
        positions[k]=[offsets.index((j,j+1)) for j in (idx,idx+3,idx+4,idx+5)]
        idx+=8
    sc=len(prompt)-len("; Answer:")
    location=offsets.index((sc,sc+1))
    assert all(i<location for p in positions.values() for i in p)
    return dict(ids=token["input_ids"],positions=positions,location=location)


def whitespace(tok,both):
    a,b=[v["tokens"] for v in both]
    seq=[]
    for x,y in zip(a,b):
        if x!=y:
            break
        if tok.decode([x],skip_special_tokens=True).strip() or x==tok.eos_token_id:
            break
        seq.append(x)
    return seq


class Trace:
    def __init__(self,at,where,layer=None,fields=None):
        self.at=at
        self.where=where
        self.layer=layer
        self.fields=fields
        self.rows=torch.arange(len(at),device=at.device)
        self.q={}
        self.loc={}
        self.mass={}

    def put(self,key,name,x):
        if self.layer is not None and key[1]!=self.layer:
            return
        if name=="q" and self.layer is not None:
            self.q[key]=x[self.rows,:,self.at,:].detach()
            self.loc[key]=x[self.rows,:,self.where,:].detach()
        if name=="a" and self.fields is not None:
            vals=x[self.rows,:,self.at,:]
            r,h,t=vals.shape
            select=self.fields[:,None].expand(-1,h,-1,-1)
            expand=vals[:,:,None,:].expand(-1,-1,4,-1)
            self.mass[key]=expand.gather(3,select).sum(-1).detach()


def layout(rows,key,ids,at,keep_mass):
    device=ids.device
    where=torch.tensor([r["entry"][key]["location"] for r in rows],device=device)
    fields=None
    if keep_mass:
        fields=torch.tensor([[r["entry"][key]["positions"][k] for k in KEYS] for r in rows],device=device)
    return where,fields


def prepare(tok,row):
    return dict(row,entry={k:encode(tok,row,k) for k in VARIANTS},
                fixed={k:whitespace(tok,row["native"][k]) for k in VARIANTS})


def masses(f,tok,rows):
    ids,at=pad([r["entry"]["original"]["ids"]+r["fixed"]["original"] for r in rows])
    loc,fields=layout(rows,"original",ids,at,True)
    tr=Trace(at,loc,fields=fields)
    h=f.run(ids,2,record=tr)
    for k in (0,1):
        assert output(f.model,h[k],at).argmax(-1).tolist()==[
            r["native"]["original"][k]["tokens"][len(r["fixed"]["original"])] for r in rows]
    correct=torch.tensor([r["key"]-1 for r in rows],device=ids.device)
    errors=torch.tensor([r["example"]["original"]["values"].index(r["native"]["original"][0]["number"])
                         for r in rows],device=ids.device)
    state=torch.stack([torch.stack([tr.mass[k,l] for l in range(24)],dim=1) for k in (0,1)],dim=1)
    ix=torch.arange(len(rows),device=ids.device)
    c=state[ix,:,:,:,correct]
    w=state[ix,:,:,:,errors]
    return (c-w).mean(-1).tolist()


def select(f,tok,root,nobs):
    chosen=[prepare(tok,r) for r in nobs if r["split"]=="development" and eligible(r,"original")]
    assert len(chosen)>=8
    out=[]
    for i in range(0,len(chosen),8):
        s=chosen[i:i+8]
        out.extend(dict(id=r["id"],gap=x) for r,x in zip(s,masses(f,tok,s),strict=True))
    avg=[sum(x["gap"][1][l]-x["gap"][0][l] for x in out)/len(out) for l in range(24)]
    layer=max(range(24),key=lambda l:avg[l])
    lock=dict(layer=layer,layers=24,mean_routing_gain=avg,development=out,
              native_sha256=sha(root/"NATIVE.json"))
    fpath=root/"QUERY_SITE.json"
    if fpath.exists():
        assert load(fpath)==lock
    else:
        save(fpath,lock)
    print("selected physical layer",layer+1,"routing gain",avg[layer],flush=True)
    return layer


def relocate(own,other,delta):
    u=own/own.norm(dim=-1,keepdim=True).clamp_min(1e-13)
    v=other/other.norm(dim=-1,keepdim=True).clamp_min(1e-13)
    parallel=(delta*u).sum(-1,keepdim=True)
    residual=delta-parallel*u
    alternative=residual-(residual*v).sum(-1,keepdim=True)*v
    result=parallel*v+alternative*(residual.norm(dim=-1,keepdim=True)/alternative.norm(dim=-1,keepdim=True).clamp_min(1e-13))
    torch.testing.assert_close(result.norm(dim=-1),delta.norm(dim=-1),rtol=5e-5,atol=4e-6)
    torch.testing.assert_close((result*v).sum(-1,keepdim=True),parallel,rtol=5e-5,atol=4e-6)
    return other+result


def snapshots(f,rows,keys,layer,audit):
    data={}
    for j in range(0,len(keys),16):
        pack=keys[j:j+16]
        examples=[rows[i] for i,_,_ in pack]
        ids,at=pad([r["entry"][v]["ids"]+list(s) for r,(_,v,s) in zip(examples,pack,strict=True)])
        where=torch.tensor([r["entry"][v]["location"] for r,(_,v,_) in zip(examples,pack,strict=True)],device=at.device)
        fields=None
        if audit:
            fields=torch.tensor([[r["entry"][v]["positions"][k] for k in KEYS]
                                 for r,(_,v,_) in zip(examples,pack,strict=True)],device=at.device)
        tr=Trace(at,where,layer,fields)
        h=f.run(ids,2,record=tr)
        native=[output(f.model,d,at).argmax(-1).tolist() for d in h]
        for n,key in enumerate(pack):
            i,v,s=key
            row=rows[i]
            if not s:
                assert [w[n] for w in native]==[r["tokens"][0] for r in row["native"][v]]
            item=dict(ids=row["entry"][v]["ids"]+list(s),where=int(where[n]),
                      start=h[0][n,:int(at[n])+1],tail=[z[n] for z in native],
                      q=[tr.q[k,layer][n] for k in (0,1)],loc=[tr.loc[k,layer][n] for k in (0,1)])
            if audit:
                item["mass"]=[tr.mass[k,layer][n].mean(0).tolist() for k in (0,1)]
            data[key]=item
    return data


def step(f,rows,layer,req,audit=False):
    keys=[]
    for i,k,a,p in req:
        keys.extend(((i,LATTICE[k][0],p),(i,"original",p)))
    bank=snapshots(f,rows,list(dict.fromkeys(keys)),layer,audit)
    ans={}
    stats={}
    selected={0:[],1:[]}
    for i,k,a,p in req:
        shape,phase=LATTICE[k]
        recv=bank[i,shape,p]
        donor=bank[i,"original",p]
        if a=="native":
            ans[i,k,a]=recv["tail"][phase]
            if audit:
                stats[i,k,a]=recv["mass"][phase]
        else:
            selected[phase].append((i,k,a,recv,donor["q"][1-phase]))
    for phase,requests in selected.items():
        for j in range(0,len(requests),22):
            block=requests[j:j+22]
            ids,at=pad([x["ids"] for _,_,_,x,_ in block])
            where=torch.tensor([r["where"] if a=="location" else int(at[w])
                                for w,(_,_,a,r,_) in enumerate(block)],device=ids.device)
            base=torch.stack([r["q"][phase] for _,_,_,r,_ in block])
            v=torch.stack([d for _,_,_,_,d in block])
            q=v.clone()
            geom=[i for i,x in enumerate(block) if x[2]=="geometry"]
            loc=[i for i,x in enumerate(block) if x[2]=="location"]
            identity=[i for i,x in enumerate(block) if x[2]=="identity"]
            if geom:
                q[geom]=shuffle(base[geom],v[geom]-base[geom])
            if loc:
                g=torch.stack([block[x][3]["loc"][phase] for x in loc])
                q[loc]=relocate(base[loc],g,v[loc]-base[loc])
            if identity:
                q[identity]=base[identity]
            start=torch.nn.utils.rnn.pad_sequence([r["start"] for _,_,_,r,_ in block],batch_first=True) if phase else None
            detect=None
            if audit:
                fields=torch.tensor([[rows[i]["entry"][LATTICE[k][0]]["positions"][t] for t in KEYS]
                                     for i,k,_,_,_ in block],device=ids.device)
                detect=Trace(at,where,layer,fields)
            moved=f.run(ids,1,start=start,edit=Patch(where,query={layer:(None,q)}),record=detect)[0]
            future=output(f.model,moved,at).argmax(-1).tolist()
            for z,(i,k,a,r,_) in enumerate(block):
                ans[i,k,a]=future[z]
                if a=="identity":
                    assert future[z]==r["tail"][phase]
                if audit:
                    stats[i,k,a]=detect.mass[0,layer][z].mean(0).tolist()
    return ans,stats


def decode(f,tok,rows,layer):
    active={(i,k,a):[] for i,r in enumerate(rows) for k in r["test"] for a in ("native","authentic",*CONTROLS)}
    out={}
    for _ in range(MAX_NEW):
        if not active:
            break
        nxt,_=step(f,rows,layer,[(i,k,a,tuple(p)) for (i,k,a),p in active.items()])
        for k,s in list(active.items()):
            s.append(nxt[k])
            if terminated(tok,s):
                out[k]=active.pop(k)
    out.update(active)
    code=dict(zip(out,text_results(tok,list(out.values())),strict=True))
    res=[]
    for i,r in enumerate(rows):
        d={}
        for l in r["test"]:
            shape,k=LATTICE[l]
            pred={a:code[i,l,a] for a in ("native","authentic",*CONTROLS)}
            assert pred["native"]==r["native"][shape][k]
            d[l]=pred
        res.append(d)
    return res


def schema(r):
    out=[]
    if eligible(r,"original"):
        out+=["rescue","reversal"]
    if r["native"]["original"][1]["number"]==r["target"]:
        for k in VARIANTS[1:]:
            if eligible(r,k):
                out.append(k)
    return out


@torch.inference_mode()
def run(model,root):
    m,tok=setup(model)
    f=Forward(m)
    started=timer()
    native=root/"NATIVE.json"
    entries=load(native)["rows"]
    layer=select(f,tok,root,entries)
    chosen=[dict(prepare(tok,r),test=schema(r)) for r in entries if r["split"]=="confirmation" and schema(r)]
    diag=sorted((r["id"] for r in chosen if "rescue" in r["test"]),key=lambda s:short_hash("q-causal",s))[:24]
    transfer=sorted((r["id"] for r in chosen if "crossed" in r["test"]),key=lambda s:short_hash("q-transfer",s))[:24]
    planned=dict(native_sha256=sha(native),site_sha256=sha(root/"QUERY_SITE.json"),diagnostic_both=sorted(diag),
                 diagnostic_transfer=sorted(transfer),rows=[dict(id=r["id"],test=r["test"]) for r in chosen])
    lock=root/"COHORT.json"
    if lock.exists():
        assert load(lock)==planned
    else:
        save(lock,planned)
    raw=root/"QUERY_RAW.json"
    prior=load(raw) if raw.exists() else dict(rows=[],elapsed=0)
    rows=prior["rows"]
    if rows:
        assert prior["cohort_sha256"]==sha(lock) and prior["native_sha256"]==sha(native)
        assert [r["id"] for r in rows]==[r["id"] for r in chosen[:len(rows)]]
    for j in range(len(rows),len(chosen),7):
        s=chosen[j:j+7]
        codes=decode(f,tok,s,layer)
        probes=[]
        for i,r in enumerate(s):
            todo=(["rescue","reversal"] if r["id"] in diag else [])+(["crossed"] if r["id"] in transfer else [])
            for kind in todo:
                variant=LATTICE[kind][0]
                prefix=tuple(r["fixed"][variant])
                probes.extend((i,kind,a,prefix) for a in ("native","authentic",*CONTROLS,"identity"))
        _,inspection=step(f,s,layer,probes,audit=True) if probes else ({},{})
        for i,(r,o) in enumerate(zip(s,codes,strict=True)):
            data={k:r[k] for k in ("id","target","new","key","native","example","test")}
            data["pred"]=o
            data["routing"]={k:{a:v for (j,t,a),v in inspection.items() if j==i and t==k}
                             for k in r["test"] if (i,k,"native") in inspection}
            rows.append(data)
        save(raw,dict(rows=rows,cohort_sha256=sha(lock),native_sha256=sha(native),
                      elapsed=prior["elapsed"]+timer()-started,
                      peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,1)))
        print("query swaps",len(rows),len(chosen),flush=True)
    z=summarize(rows)
    save(root/"QUERY_SUMMARY.json",z)
    print("lookup query gates",z["gates"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)
