"""Evaluate controllers with unrestricted greedy decoding and original candidate scores."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import string
import time

import numpy as np
import torch
from scipy.stats import binomtest

import generation
import minimal
from task_data import CONTENT, identity, load, multiple_choice, save


FIRST_LOWER=re.compile(r"^\s*(?:\(|\[)?([a-zA-Z])(?:\b|[.)\]])")
NUMBER=r"[-+]?(?:[0-9][0-9,]*\.[0-9]+|[0-9][0-9,]*)"


def normal_answer(answer):
    s=answer.lower().translate(str.maketrans("","",string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b"," ",s).split())


def token_f1(text,ref):
    x,y=normal_answer(text).split(),normal_answer(ref).split()
    common=sum((Counter(x)&Counter(y)).values())
    return 2*common/(len(x)+len(y)) if x or y else 1.


def lcs(a,b):
    table=[0]*(len(b)+1)
    for x in a:
        prev=0
        for j,y in enumerate(b,1):
            old=table[j]
            table[j]=prev+1 if x==y else max(table[j],table[j-1])
            prev=old
    return table[-1]


def constraints(text,instruction):
    clean=instruction.casefold()
    clauses=[]
    words=len(re.findall(r"\b[\w’'-]+\b",text))
    for op,n in re.findall(r"\b(exactly|at least|at most|no more than|no fewer than)\s+(\d+)\s+words\b",clean):
        size=int(n)
        clauses.append(words==size if op=="exactly" else words>=size if op in ("at least","no fewer than") else words<=size)
    if re.search(r"(?:output|return|respond in|format as|provide)\s+(?:a |valid |only )*(?:the )?json(?:\b|[- ])",clean):
        try:
            json.loads(text)
            clauses.append(True)
        except ValueError:
            clauses.append(False)
    return bool(all(clauses)) if clauses else None


def evaluate(row,mode,text,reason):
    stripped=text.strip()
    first=stripped.split("\n",1)[0].strip()
    fmt=False
    valid=bool(stripped)
    if mode=="letter":
        labels=[chr(65+i) for i in range(len(row["options"]))]
        choice,category,fmt=generation.classify(text,labels)
        loose=FIRST_LOWER.match(text)
        normalized=loose[1].upper() if loose and loose[1].upper() in labels else None
        return dict(correct=choice==row["label"],valid=choice is not None,strict=fmt and choice==row["label"],
                    loose_correct=normalized==row["label"],option=choice,failed=category)
    if mode=="content":
        out=identity(first)
        options=[identity(x) for x in row["options"]]
        option=options.index(out) if out in options and options.count(out)==1 else None
        good=identity(row["answer_content"])
        strict=(out==good and identity(stripped)==good)
        return dict(correct=out==good,valid=option is not None,strict=strict,loose_correct=out==good,
                    content_f1=token_f1(first,row["answer_content"]),option=option)
    if mode=="span":
        exact=any(normal_answer(first)==normal_answer(ref) for ref in row["references"])
        return dict(correct=exact,valid=valid,strict=exact and identity(stripped)==identity(first),
                    f1=max(token_f1(first,ref) for ref in row["references"]))
    if mode in ("gsm_final","gsm_process"):
        matches=re.findall(r"####\s*("+NUMBER+")",stripped)
        nums=re.findall(NUMBER,stripped)
        reference=row["references"][0].replace(",","")
        parsed=(matches or nums or [None])[-1]
        def number(value):
            try:
                return float(value.replace(",",""))
            except (ValueError,AttributeError):
                return None
        correct=number(parsed)==number(reference) if parsed is not None else False
        strict=bool(matches) and number(matches[-1])==number(reference)
        if mode=="gsm_final":
            strict=strict and bool(re.fullmatch(r"####\s*"+NUMBER,stripped))
        rationale=bool("\n" in stripped and "####" in stripped and len(stripped.split("####")[0].split())>=8)
        return dict(correct=correct,strict=strict,valid=parsed is not None,parsed=parsed,process=rationale,
                    completed=reason!="token_limit")
    text_words=identity(stripped).split()
    ref=identity(row["references"][0]).split()
    common=lcs(text_words,ref)
    rouge=2*common/max(1,len(text_words)+len(ref))
    checks=constraints(stripped,row["instruction"])
    digits=set(re.findall(NUMBER,row["references"][0]))
    answer_digits=set(re.findall(NUMBER,stripped))
    return dict(valid=valid,completed=reason!="token_limit",rouge_l=rouge,constraint=checks,
                reference_number_recall=len(digits&answer_digits)/len(digits) if digits else None)


def generate(archive,model,tok,net,rows,mode,k,batch,max_new):
    prompts=[tok.encode(row["prompt"],add_special_tokens=False) for row in rows]
    order=sorted(range(len(rows)),key=lambda i:(len(prompts[i]),rows[i]["sample_id"]))
    vector=[None]*len(rows)
    backbone=0
    calls=0
    total_output=0
    device=next(model.parameters()).device
    start=time.monotonic()
    for start_index in range(0,len(order),batch):
        selected=order[start_index:start_index+batch]
        cur=[prompts[i] for i in selected]
        outputs=[[] for _ in selected]
        stop=["token_limit"]*len(selected)
        live=list(range(len(selected)))
        for _ in range(max_new):
            packed,last,width=generation.pack_generation(cur,outputs,live,0,device)
            h,_=minimal.forward(model,packed,k,net)
            tokens=model.lm_head(h[last]).argmax(-1).tolist()
            backbone+=width*len(live)*k
            calls+=1
            total_output+=len(tokens)
            future=[]
            for j,token in zip(live,tokens,strict=True):
                outputs[j].append(token)
                if token==tok.eos_token_id:
                    stop[j]="eos"
                else:
                    decoded=tok.decode(outputs[j],skip_special_tokens=True)
                    if mode not in ("other","gsm_process") and "\n" in decoded.strip("\n "):
                        stop[j]="internal_newline"
                    elif mode in ("other","gsm_process") and re.search(r"\n(?:User|Question):",decoded):
                        stop[j]="next_turn"
                    else:
                        future.append(j)
            live=future
            if not live:
                break
        for j,original in enumerate(selected):
            row=rows[original]
            text=tok.decode(outputs[j],skip_special_tokens=True)
            vector[original]=dict(id=row["sample_id"],prediction=text,generated_tokens=outputs[j],stop=stop[j],
                                  label=row.get("label"),answer_tokens=len(outputs[j]),metrics=evaluate(row,mode,text,stop[j]))
    return vector,dict(seconds=time.monotonic()-start,model_calls=calls,backbone_padded_token_loops=backbone,
                        generated_tokens=total_output,cuda_peak=torch.cuda.max_memory_allocated())


def candidates(archive,model,tok,net,rows,k,batch):
    raw=[x["candidate"] for x in rows]
    groups,meta=archive.encode(tok,raw)
    index={}
    for entry in groups:
        index.setdefault(entry[2],[]).append(entry)
    order=sorted(index,key=lambda j:max(len(r[0]) for r in index[j]))
    start=time.monotonic()
    calls=work=0
    for i in range(0,len(order),batch):
        entries=[entry for j in order[i:i+batch] for entry in index[j]]
        packed=archive.pack(entries)
        h,_=minimal.forward(model,packed,k,net)
        lp=archive.scores(model.lm_head(h).float(),packed[1].cpu().numpy(),packed[-1],len(entries))
        archive.record(meta,entries,dict(model=lp))
        calls+=1
        work+=packed[0].numel()*k
    records=[]
    for row in meta:
        x=np.asarray(row["scores"]["model"])
        measures=dict(acc=int(x.argmax())==row["gold"],acc_norm=int((x/row["characters"]).argmax())==row["gold"])
        records.append(dict(id=row["id"],labels=row["labels"],gold=row["gold"],chars=row["characters"],
                            scores=x.tolist(),metrics=measures))
    return records,dict(seconds=time.monotonic()-start,model_calls=calls,backbone_padded_token_loops=work,
                        cuda_peak=torch.cuda.max_memory_allocated())


def summarize(rows):
    scores={}
    for key in rows[0]["metrics"]:
        if key in ("option","parsed","failed"):
            continue
        values=[x["metrics"][key] for x in rows if isinstance(x["metrics"][key],(bool,int,float))]
        if values:
            scores[key]=dict(value=float(np.mean(values)),n=len(values),count=int(sum(values)) if isinstance(values[0],bool) else None)
    return scores


def data_mode(raw,ds,mode):
    if mode=="letter":
        return [multiple_choice(r["candidate"],ds,"letter") for r in raw]
    if mode=="content" and raw[0]["task_type"]!="mc_content":
        return [multiple_choice(r["candidate"],ds,"content") for r in raw]
    return raw


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("archive","previous","model","out"):
        p.add_argument(f"--{key}",required=True,type=Path)
    p.add_argument("--arm",required=True)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--dataset",required=True)
    p.add_argument("--modes",nargs="+",required=True)
    p.add_argument("--depths",type=int,nargs="+",required=True)
    p.add_argument("--max-new",type=int)
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--limit",type=int)
    p.add_argument("--protocol",type=Path)
    a=p.parse_args()
    scores=load(a.data)
    if a.limit is not None:
        scores=scores[:a.limit]
    if a.protocol is not None and not a.protocol.is_file():
        raise ValueError("Evaluation requires a sealed independent protocol")
    archive,model,tok,runtime=minimal.runtime(a.archive,a.previous,a.model)
    net=minimal.Minimal("linear",1,.45).cuda().eval() if a.checkpoint else None
    if net is not None:
        net.load_state_dict(torch.load(a.checkpoint,map_location="cpu",weights_only=True)["state"])
        net.requires_grad_(False)
    a.out.mkdir(parents=True,exist_ok=True)
    for depth in a.depths:
        for mode in a.modes:
            cap=a.max_new or {"letter":8,"content":24,"span":24,"gsm_final":28,"gsm_process":192,"other":160,"candidate":1}[mode]
            stem=f"{a.dataset}__{a.arm}__k{depth}__{mode}"
            raw=a.out/f"{stem}.jsonl"
            receipt=raw.with_suffix(".json")
            meta=dict(dataset=a.dataset,arm=a.arm,depth=depth,mode=mode,
                checkpoint_sha256=minimal.sha(a.checkpoint) if a.checkpoint else "original_backbone",
                data_sha256=minimal.sha(a.data),protocol_sha256=minimal.sha(a.protocol) if a.protocol else None,
                max_new=cap,batch=a.batch,n=len(scores),method="native_greedy_no_cache_all_response_positions")
            if receipt.is_file():
                ready=json.loads(receipt.read_text())
                if ready["plan"]!=meta or ready["rows_sha256"]!=minimal.sha(raw):
                    raise RuntimeError(f"Existing evaluation receipt differs at {stem}")
                continue
            torch.cuda.reset_peak_memory_stats()
            subset=data_mode(scores,a.dataset,mode)
            if mode=="candidate":
                pred,cost=candidates(archive,model,tok,net,subset,depth,max(1,a.batch//4))
            else:
                pred,cost=generate(archive,model,tok,net,subset,mode,depth,a.batch,cap)
            save(raw,pred)
            compact=summarize(pred)
            minimal.dump(receipt,dict(plan=meta,rows_sha256=minimal.sha(raw),runtime=runtime,metrics=compact,cost=cost))
            print(stem,len(pred),compact,cost,flush=True)


if __name__=="__main__":
    main()
