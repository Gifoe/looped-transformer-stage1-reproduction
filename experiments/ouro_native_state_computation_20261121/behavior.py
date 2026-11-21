"""Separate natural generation, full answer likelihood and letter diagnostics."""
import argparse
from pathlib import Path

import torch

from data_protocol import parse_generation
from native import Forward, load, output, pad, save, setup, tensors, timer


def answer_tokens(tok,prompt,choice):
    prefix=tok.encode(prompt,add_special_tokens=False)
    full=tok.encode(prompt+" "+choice,add_special_tokens=False)
    if full[:len(prefix)]==prefix and len(full)>len(prefix):
        return full,len(prefix)
    suffix=tok.encode(" "+choice,add_special_tokens=False)
    if not suffix:
        raise ValueError("Empty answer continuation")
    return prefix+suffix,len(prefix)


def likelihood(f,tok,rows):
    encoded=[answer_tokens(tok,r["prompt"],c) for r in rows for c in r["choices"]]
    ids,_=pad([x for x,_ in encoded])
    states=f.run(ids,4)
    pos=[(i,j-1) for i,(seq,start) in enumerate(encoded) for j in range(start,len(seq))]
    pairs=torch.tensor(pos,device=ids.device)
    truth=ids[pairs[:,0],pairs[:,1]+1]
    counts=[len(seq)-start for seq,start in encoded]
    scored=[]
    for h in states:
        logits=f.model.lm_head(h[pairs[:,0],pairs[:,1]]).float()
        values=logits.log_softmax(-1).gather(-1,truth[:,None]).flatten()
        vals=[x.tolist() for x in values.split(counts)]
        scored.append([dict(sum=sum(x),mean=sum(x)/len(x),tokens=x) for x in vals])
    return scored


def letters(f,tok,rows):
    texts=[r.get("letter_prompt") or
        r["prompt"].removesuffix("\nAnswer:")+"\n"+"\n".join(f"{chr(65+i)}) {v}" for i,v in enumerate(r["choices"]))+"\nAnswer:"
        for r in rows]
    ids,pos=tensors(tok,texts)
    letter=[tok.encode(" "+chr(65+i),add_special_tokens=False) for i in range(max(len(r["choices"]) for r in rows))]
    if any(len(a)!=1 for a in letter):
        raise ValueError("Letter tokens do not align")
    li=[a[0] for a in letter]
    return [[z[i,li[:len(r["choices"])]].tolist() for i,r in enumerate(rows)] for z in
                [output(f.model,h,pos) for h in f.run(ids,4)]]


def generate(f,tok,rows,k,max_new):
    prefixes=[tok.encode(r["prompt"],add_special_tokens=False) for r in rows]
    continuations=[[] for _ in rows]
    active=list(range(len(rows)))
    for _ in range(max_new):
        ids,positions=pad([prefixes[j]+continuations[j] for j in active])
        predicted=output(f.model,f.run(ids,k)[-1],positions).argmax(-1).tolist()
        keep=[]
        for i,w in zip(active,predicted,strict=True):
            continuations[i].append(w)
            text=tok.decode(continuations[i],skip_special_tokens=True)
            if w!=tok.eos_token_id and "\n" not in text.strip():
                keep.append(i)
        active=keep
        if not active:
            break
    return [(tok.decode(x,skip_special_tokens=True),x) for x in continuations]


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("manifest",type=Path)
    p.add_argument("phase")
    p.add_argument("--max-new",type=int,default=12)
    a=p.parse_args()
    all_rows=load(a.manifest)[a.phase]
    model,tok=setup(a.model,torch.bfloat16)
    f=Forward(model)
    dest=a.out/f"BEHAVIOR_{a.phase}.json"
    records=load(dest)["rows"] if dest.exists() else []
    times=[]
    for i in range(len(records),len(all_rows),8):
        rows=all_rows[i:i+8]
        start=timer()
        tf=likelihood(f,tok,rows)
        ls=letters(f,tok,rows)
        gens=[generate(f,tok,rows,k,a.max_new) for k in range(1,5)]
        offset=0
        for j,r in enumerate(rows):
            n=len(r["choices"])
            rec=dict(id=r["id"],family=r["family"],gold=r["gold"],k={})
            for k in range(4):
                text,raw=gens[k][j]
                ll=tf[k][offset:offset+n]
                rec["k"][str(k+1)]=dict(text=text,generated_tokens=raw,
                    gen=parse_generation(text,r["choices"]),likelihood=ll,
                    tf=max(range(n),key=lambda m:ll[m]["mean"]),
                    tf_sum=max(range(n),key=lambda m:ll[m]["sum"]),
                    letter=max(range(n),key=lambda m:ls[k][j][m]),letter_logits=ls[k][j])
            records.append(rec)
            offset+=n
        times.append(timer()-start)
        save(dest,dict(rows=records,expected=len(all_rows),dtype="BF16 eager,no-cache",max_new=a.max_new,
                       seconds_batches_this_run=times,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
        print(a.phase,len(records),len(all_rows),flush=True)


if __name__=="__main__":
    main()

