"""Compare attention projection restoration after cross-prompt H1 recombination."""
import argparse
from pathlib import Path
import random

import numpy as np
import torch

from bidi import event
from carry import MAX_NEW, numeric
from native import Forward, load, output, pad, save, setup, sha
from operand import common_effect, make_case, precompute
from state import rotate_about


MODES=("native","prefix","keys","keys_noise","values","values_noise","both")


def slices(x,ix):
    return [z.index_select(1,pos).detach() for z,pos in zip(x,ix,strict=True)]


class NativeKV:
    def __init__(self,idx):
        self.idx=idx
        self.data={}

    def put(self,key,name,x):
        if name in ("k","v"):
            self.data[key,name]=slices(x,self.idx)


class Restore:
    def __init__(self,trace,positions,seeds,mode):
        self.trace=trace
        self.positions=positions
        self.seeds=seeds
        self.mode=mode
        self.error=0.0

    def resid(self,k,l,x):
        return x

    def qk(self,key,q,k):
        return q,self.place(key,"k",k) if self.mode=="both" or self.mode.startswith("keys") else k

    def attn(self,key,a,v):
        return a,self.place(key,"v",v) if self.mode=="both" or self.mode.startswith("values") else v

    def mlp(self,key,z,record):
        return z

    def place(self,key,kind,x):
        native=self.trace.data[key,kind]
        out=x.clone()
        for i,pos in enumerate(self.positions):
            if self.mode.endswith("_noise"):
                current=x[i].index_select(1,pos)
                lhs=current.transpose(0,1).reshape(len(pos),-1)
                rhs=native[i].transpose(0,1).reshape_as(lhs)
                rng=torch.Generator(device=x.device).manual_seed(self.seeds[i]+101*key[1]+(kind=="v"))
                u=torch.randn(lhs.shape,device=x.device,dtype=x.dtype,generator=rng)
                control=rotate_about(lhs,rhs,u)
                changed=control.reshape(len(pos),current.shape[0],current.shape[2]).transpose(0,1).contiguous()
            else:
                changed=native[i]
            out[i].index_copy_(1,pos,changed)
        return out


def exact_generate(f,tok,cases,source,case_seeds,mode):
    emitted=[[] for _ in cases]
    active=list(range(len(cases)))
    for _ in range(MAX_NEW):
        seqs=[cases[j]["tokens"]+emitted[j] for j in active]
        ids,last=pad(seqs)
        h1=f.run(ids,1)[0]
        mix=h1.clone()
        positions=[torch.tensor(cases[j]["prefix"],device=ids.device) for j in active]
        for i,j in enumerate(active):
            if mode!="native":
                mix[i,positions[i]]=torch.stack([source[j][p] for p in cases[j]["prefix"]])
        idsrow=torch.arange(len(mix),device=mix.device)
        assert torch.equal(mix[idsrow,last],h1[idsrow,last])
        if mode in ("native","prefix"):
            h2=f.run(ids,1,start=mix)[0]
        else:
            tr=NativeKV(positions)
            native=f.run(ids,1,start=h1,record=tr)[0]
            edit=Restore(tr,positions,[case_seeds[j] for j in active],mode)
            h2=f.run(ids,1,start=mix,edit=edit)[0]
            if mode=="both":
                torch.testing.assert_close(h2[idsrow,last],native[idsrow,last],rtol=2e-5,atol=2e-5)
        z=output(f.model,h2,last).argmax(dim=-1).tolist()
        keep=[]
        for j,v in zip(active,z,strict=True):
            emitted[j].append(v)
            if v != tok.eos_token_id and "\n" not in tok.decode(emitted[j],skip_special_tokens=True).strip():
                keep.append(j)
        active=keep
        if not active:
            break
    text=[tok.decode(v,skip_special_tokens=True) for v in emitted]
    return [dict(text=t,number=numeric(t),tokens=z) for t,z in zip(text,emitted,strict=True)]


def summarize(results,root):
    sums={}
    for s in ("discovery","confirmation"):
        pack=[x for x in results if x["split"]==s]
        if not pack:
            sums[s]={"n":0}
            continue
        treatments={}
        for m in MODES:
            count=dict(recipient=0,counterfactual=0,donor=0,other=0,both_recipient=0)
            for x in pack:
                out=[r["number"] for r in x["outputs"][m]]
                same=True
                for i,n in enumerate(out):
                    r=x["case"][i]
                    expected=(r["answer"],r["counterfactual"],r["donor_answer"])
                    name=next((k for k,v in zip(("recipient","counterfactual","donor"),expected,strict=True) if n==v),"other")
                    count[name]+=1
                    same=same and name=="recipient"
                count["both_recipient"]+=same
            treatments[m]=count
        for m in ("keys","values"):
            diff=[]
            for r in pack:
                answer=[z["answer"] for z in r["case"]]
                counts=[int([v["number"] for v in r["outputs"][w]]==answer) for w in (m,m+"_noise")]
                diff.append(counts[0]-counts[1])
            values=np.array(diff)
            r=np.random.default_rng(211134941).integers(0,len(values),size=(20000,len(values)))
            ci=np.quantile(values[r].mean(axis=1),[0.0125,0.9875]).tolist()
            treatments[m]["both_restoration_excess"]=dict(mean=float(values.mean()),ci=ci,total=int(values.sum()))
        sums[s]=dict(n=len(pack),treatments=treatments)
    save(root/"CHANNEL_SUMMARY.json",sums)
    return sums


def pick(root):
    raw=load(root/"OPERAND_RAW.json")["rows"]
    rng=random.Random(211134051)
    gs=[]
    for s,n in (("discovery",24),("confirmation",48)):
        v=sorted((x for x in raw if x["aligned"] and x["split"]==s
                  and [p["number"] for p in x["modes"]["prefix"]]==[c["counterfactual"] for c in x["cases"]]),
                 key=lambda x:x["id"])
        gs.extend(rng.sample(v,min(len(v),n)))
    return gs


@torch.inference_mode()
def run(model,root):
    assert load(root/"OPERAND_SUMMARY.json")["confirmation"]["pass_primary"]
    grants=pick(root)
    native={r["id"]:r for r in load(root/"NATIVE.json")["rows"]}
    model,tok=setup(model)
    f=Forward(model)
    path=root/"CHANNEL_RAW.json"
    rows=load(path)["rows"] if path.exists() else []
    assert [x["id"] for x in rows]==[x["id"] for x in grants[:len(rows)]]
    for i in range(len(rows),len(grants),8):
        pack=grants[i:i+8]
        cases=[make_case(tok,native[g["id"]],k) for g in pack for k in range(2)]
        assert all(c is not None and c["prefix"]==list(range(c["prefix"][0],c["prefix"][-1]+1)) for c in cases)
        acts=precompute(f,cases)
        seeds=[211136000+g["midpoint"]*2+k for g in pack for k in range(2)]
        values={m:exact_generate(f,tok,cases,acts,seeds,m) for m in MODES}
        for j,r in enumerate(pack):
            results={m:columns[2*j:2*j+2] for m,columns in values.items()}
            for name in ("native","prefix","both"):
                ref=r["modes"]["native" if name=="both" else name]
                assert [z["tokens"] for z in results[name]]==[z["tokens"] for z in ref], (name,r["id"])
            rows.append(dict(id=r["id"],split=r["split"],midpoint=r["midpoint"],case=r["cases"],outputs=results))
        save(path,dict(rows=rows,operand_sha256=sha(root/"OPERAND_RAW.json")))
        print("channels",len(rows),len(grants),flush=True)
    print(summarize(rows,root),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)

