"""Capture local branch updates and causal supervision at a fixed physical layer."""

import argparse
from pathlib import Path
import re
import time

import torch

from block import Trace,replay
from csc_data import prompt
from native import Forward,load,output,pad,save,setup,sha

LAYER=14
KINDS=("joint","safe")
UPDATES=("after_a","after_m","after_both","formed_a","formed_m","formed_both")


def whites(tok,arg):
    bits=[]
    for t in arg["tokens"]:
        if t==tok.eos_token_id or tok.decode([t],skip_special_tokens=True).strip():
            break
        bits.append(t)
    return tuple(bits)


def fields(tok,text):
    shift="; Left:" in text.split("\n")[-1]
    pattern=(r"Operation: (.+?); Left: (\d+); Right: (\d+); Result:$" if shift else
             r"A = (\d+), B = (\d+), Rule: (.+?), Answer:$")
    match=re.search(pattern,text)
    assert match
    x=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    spans=x["offset_mapping"]

    def gather(i):
        start,end=match.span(i)
        return [j for j,(a,b) in enumerate(spans) if a<end and b>start]

    a,b,rule=[gather(k) for k in ((2,3,1) if shift else (1,2,3))]
    assert all(len(s)==3 for s in (a,b)) and rule and not set(a).intersection(b)
    at=len(x["input_ids"])-1
    context=[j for j in range(max(0,at-3),at)]
    return x["input_ids"],(a[-2:],b[-2:],rule,context)


def metadata(tok,row,kind):
    quad=row["corners"] if kind=="joint" else row["safe"]
    rows=[fields(tok,prompt(row,p)) for p in quad]
    ids,roles=zip(*rows,strict=True)
    assert len({len(q) for q in ids})==1 and all(roles[0]==x for x in roles)
    k=3 if kind=="joint" else 4
    native=[x[k] for x in row["native"]]
    prefixes={()}
    for v in native:
        w=whites(tok,v)
        prefixes.update(w[:i] for i in range(1,len(w)+1))
    return dict(id=row["id"],kind=kind,form=row["form"],split=row["split"],target=row["target"],
                foil=row["foil"],quad=ids,roles=roles[0],native=native,prefixes=sorted(prefixes,key=lambda p:(len(p),p)))


def encode(x,at,roles):
    a,b,rule,context=roles
    return torch.cat((x[a],x[b],x[rule].mean(0,keepdim=True),x[context].mean(0,keepdim=True),x[at,None]),0)


def frozen(x):
    return x.detach().to(device="cpu",dtype=torch.bfloat16).contiguous()


@torch.inference_mode()
def capture(f,memo,requests):
    tokens,at=pad([t+list(p) for i,p in requests for t in memo[i]["quad"]])
    tr=Trace(f)
    with tr.phases():
        h=f.run(tokens,2,record=tr)
    n=len(requests)
    rows=torch.arange(len(at),device=at.device)
    idx=rows[3::4]
    where=at[idx]
    original=[output(f.model,z,at).argmax(-1)[idx].tolist() for z in h]
    x={(k,l):tr.saved[(k,l),"input"] for k in (0,1) for l in range(13,17)}
    w={(k,l):torch.stack([tr.saved[(k,l),a][rows,at] for a in ("a_write","m_write")],dim=1).view(n,4,2,-1)
       for k in (0,1) for l in range(13,17)}
    delta=w[1,LAYER][:,3]-w[0,LAYER][:,3]
    formed=[]
    errors=[]
    for k in (0,1):
        xx=x[k,LAYER].view(n,4,*x[k,LAYER].shape[1:])
        ref=xx[:,1]+xx[:,2]-xx[:,0]
        kw=tr.kw[k,LAYER]
        y,branch=replay(f.layers[LAYER],ref.repeat_interleave(4,dim=0),kw,at)
        mixed=torch.stack([branch[a][idx] for a in ("a","m")],dim=1)-w[k,LAYER][:,1]-w[k,LAYER][:,2]+w[k,LAYER][:,0]
        # Confirm the generated term by an independently evaluated residual identity.
        saved=tr.saved[(k,LAYER),"output"][rows,at].view(n,4,-1)
        new=y[idx,where]-saved[:,1]-saved[:,2]+saved[:,0]
        err=(new-mixed.sum(1)).abs().amax(-1)
        torch.testing.assert_close(new,mixed.sum(1),rtol=9e-4,atol=3e-4)
        formed.append(mixed)
        errors.append(err)
    z={}
    for j,(i,p) in enumerate(requests):
        row=memo[i]
        for k in (0,1):
            if tuple(row["native"][k]["tokens"][:len(p)])==p and len(row["native"][k]["tokens"])>len(p):
                assert original[k][j]==row["native"][k]["tokens"][len(p)]
        a=int(idx[j]); pos=int(where[j])
        motifs=[encode(t[a],pos,row["roles"]) for t in (x[0,LAYER],h[0])]
        updates=torch.stack([delta[j,0],delta[j,1],delta[j].sum(0),formed[1][j,0],formed[1][j,1],formed[1][j].sum(0)])
        profiles={}
        for l in range(13,17):
            phase=w[1,l][j,3]-w[0,l][j,3]
            mixed=[t[j,3]-t[j,1]-t[j,2]+t[j,0] for t in (w[0,l],w[1,l])]
            profiles[str(l+1)]=dict(phase_change=phase.norm(dim=-1).tolist(),
                        native_mixed=[t.norm(dim=-1).tolist() for t in mixed])
        z[i,p]=dict(features=frozen(torch.stack(motifs)),updates=updates.cpu().contiguous(),
                    base=torch.stack([tr.saved[(k,LAYER),"output"][a,pos] for k in (0,1)]).cpu(),
                    profile=profiles,formed_k1=frozen(formed[0][j]),identity_error=max(float(t[j]) for t in errors))
    return z


def load_bank(root):
    m=load(root/"BANK_INDEX.json")
    out={}
    for item in m["files"]:
        p=root/item["file"]
        assert sha(p)==item["sha256"]
        for k,v in torch.load(p,map_location="cpu",weights_only=True).items():
            assert k not in out
            out[k]=v
    return out


@torch.inference_mode()
def run(model,root):
    native=root/"NATIVE.json"
    rows=load(native)["rows"]
    assert len(rows)==len(load(root/"DECK.json")["rows"])
    m,tok=setup(model)
    f=Forward(m)
    path=root/"BANK_INDEX.json"
    index=load(path) if path.exists() else dict(native_sha256=sha(native),rows=0,files=[])
    assert index["native_sha256"]==sha(native)
    for off in range(index["rows"],len(rows),6):
        now=time.monotonic()
        batch=rows[off:off+6]
        meta=[metadata(tok,r,k) for r in batch for k in KINDS]
        jobs=[(j,p) for j,r in enumerate(meta) for p in r["prefixes"]]
        cache={}
        for j in range(0,len(jobs),6):
            cache.update(capture(f,meta,jobs[j:j+6]))
        v={(case["id"],case["kind"],p):t for (j,p),t in cache.items() for case in (meta[j],)}
        file=root/f"bank_{off:04}.pt"
        temporary=file.with_suffix(".tmp")
        torch.save(v,temporary)
        temporary.replace(file)
        index["rows"]+=len(batch)
        index["files"].append(dict(file=file.name,sha256=sha(file),items=len(v)))
        save(path,index)
        print("CSC bank",index["rows"],len(rows),"seconds",round(time.monotonic()-now,1),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("result",type=Path)
    a=p.parse_args()
    run(a.model,a.result)
