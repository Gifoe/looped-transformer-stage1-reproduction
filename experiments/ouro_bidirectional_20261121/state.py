"""Transfer naturally occurring first-loop context with the original readout held fixed."""
import argparse
from pathlib import Path

import numpy as np
import torch

from carry import MAX_NEW, PROMPT, numeric
from native import Forward, load, output, pad, save, setup, sha

from bidi import event


def prompt_ids(tok, pair):
    a,b=pair
    return tok.encode(PROMPT + f"{a} + {b} =",add_special_tokens=False)


def rotate_about(source, target, noise):
    u=source/source.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    parallel=(target*u).sum(-1,keepdim=True)*u
    perp=target-parallel
    randomized=noise-(noise*u).sum(-1,keepdim=True)*u
    answer=parallel+randomized/randomized.norm(dim=-1,keepdim=True).clamp_min(1e-12)*perp.norm(dim=-1,keepdim=True)
    torch.testing.assert_close(answer.norm(dim=-1),target.norm(dim=-1),rtol=1e-5,atol=1e-5)
    torch.testing.assert_close((answer-source).norm(dim=-1),(target-source).norm(dim=-1),rtol=1e-5,atol=1e-5)
    return answer


def aligned_case(tok,row):
    a,b=[prompt_ids(tok,p) for p in row["pairs"]]
    if len(a)!=len(b):
        return None
    changed=[i for i,(u,v) in enumerate(zip(a,b,strict=True)) if u!=v]
    if not changed or changed[-1]>=len(a)-1:
        return None
    return a,b,changed


def multimode(f,tok,cases,mode,seeds):
    sources=[case["recipient"] for case in cases]
    donors=[case["donor"] for case in cases]
    location=[([len(case["recipient"])-1] if mode.startswith("readout") else case["changed"]) for case in cases]
    draws=[]
    if mode.endswith("noise"):
        for p,seed in zip(location,seeds,strict=True):
            random=torch.Generator(device="cuda").manual_seed(seed)
            draws.append(torch.randn((len(p),f.core.config.hidden_size),device="cuda",generator=random))
    result=[[] for _ in cases]
    active=list(range(len(cases)))
    for t in range(MAX_NEW):
        current=[sources[i]+result[i] for i in active]
        other=[donors[i]+result[i] for i in active]
        ids,last=pad(current)
        ids_other,last_other=pad(other)
        assert torch.equal(last,last_other)
        recipient_h1=f.run(ids,1)[0]
        donor_h1=f.run(ids_other,1)[0]
        if t==0:
            assert torch.equal(f.run(ids,1,start=recipient_h1)[0],f.run(ids,2)[1])
            assert torch.equal(f.run(ids,1,start=donor_h1)[0],f.run(ids_other,2)[1])
        mix=recipient_h1.clone()
        if mode=="full_donor":
            mix=donor_h1
        elif mode!="native":
            for k,i in enumerate(active):
                p=location[i]
                base=recipient_h1[k,p]
                donor=donor_h1[k,p]
                mix[k,p]=rotate_about(base,donor,draws[i]) if draws else donor
            if mode.startswith("source"):
                assert all(torch.equal(mix[k,last[k]],recipient_h1[k,last[k]]) for k in range(len(active)))
        nxt=output(f.model,f.run(ids,1,start=mix)[0],last).argmax(dim=-1).tolist()
        keep=[]
        for i,v in zip(active,nxt,strict=True):
            result[i].append(v)
            if v!=tok.eos_token_id and "\n" not in tok.decode(result[i],skip_special_tokens=True).strip():
                keep.append(i)
        active=keep
        if not active:
            break
    text=[tok.decode(z,skip_special_tokens=True) for z in result]
    return [dict(text=t,number=numeric(t),tokens=z) for t,z in zip(text,result,strict=True)]


def conditions(summary, native):
    chosen={p["id"] for p in native if event(p)["direct"]}
    for part in summary.values():
        assert set(part["direct_ids"]).issubset(chosen)
    return [p for p in native if p["id"] in chosen]


@torch.no_grad()
def transfer(model,root):
    summary=load(root/"NATIVE_SUMMARY.json")
    assert summary["confirmation"]["pass_gate"]
    native=load(root/"NATIVE.json")["rows"]
    model,tok=setup(model)
    f=Forward(model)
    selected=conditions(summary,native)
    path=root/"STATE_TRANSFER.json"
    result=load(path)["rows"] if path.exists() else []
    for i in range(len(result),len(selected),12):
        block=selected[i:i+12]
        cases=[]
        map_back=[]
        partial=[]
        for j,row in enumerate(block):
            item=dict(id=row["id"],split=row["split"],midpoint=row["midpoint"],aligned=False)
            partial.append(item)
            aligned=aligned_case(tok,row)
            if aligned is None:
                continue
            a,b,changed=aligned
            item.update(aligned=True,changed=changed,target=row["target"],outputs={})
            for k,ordered in enumerate(((a,b),(b,a))):
                cases.append(dict(recipient=ordered[0],donor=ordered[1],changed=changed))
                map_back.append((j,k))
        seeds=[211121971+(i+j)*2+k for j,k in map_back]
        if cases:
            for mode in ("native","full_donor","source","source_noise","readout","readout_noise"):
                values=multimode(f,tok,cases,mode,seeds)
                for x,(j,k) in zip(values,map_back,strict=True):
                    if mode in ("native","full_donor"):
                        assert x["tokens"]==block[j]["pred"][1][k if mode=="native" else 1-k]["tokens"]
                    partial[j]["outputs"].setdefault(mode,[]).append(x)
        result.extend(partial)
        save(path,dict(native_sha256=sha(root/"NATIVE.json"),rows=result))
        print("state",len(result),len(selected),flush=True)
    stage={}
    for part in ("discovery","confirmation"):
        table=[r for r in result if r["split"]==part]
        bins=load(root/"DECK.json")["midpoint_splits"]
        typed={}
        for family in ("source","readout"):
            cluster={int(m):0 for m in sorted(bins) if bins[m]==part}
            actual=control=actual_direction=control_direction=0
            carrying=set()
            for row in table:
                if not row["aligned"]:
                    continue
                target=row["target"][::-1]
                treated=[x["number"] for x in row["outputs"][family]]
                sham=[x["number"] for x in row["outputs"][family+"_noise"]]
                exact=treated==target
                noise=sham==target
                cluster[row["midpoint"]]+=int(exact)-int(noise)
                actual+=exact
                control+=noise
                actual_direction+=sum(a==b for a,b in zip(treated,target,strict=True))
                control_direction+=sum(a==b for a,b in zip(sham,target,strict=True))
                if exact:
                    carrying.add(row["midpoint"])
            points=np.array(list(cluster.values()))
            rs=np.random.default_rng(211121997).integers(0,len(points),size=(20000,len(points)))
            ci=np.quantile(points[rs].mean(axis=-1),[0.025,0.975]).tolist()
            typed[family]=dict(two_way_native_vectors=actual,two_way_noise=control,
                               native_vector_exact_outputs=actual_direction,noise_exact_outputs=control_direction,
                               bins_two_way=len(carrying),interval=ci,pass_gate=len(carrying)>=6 and ci[0]>0)
        stage[part]=dict(n=len(table),aligned=sum(x["aligned"] for x in table),type=typed)
    save(root/"STATE_SUMMARY.json",stage)
    print(stage,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    transfer(a.model,a.root)

