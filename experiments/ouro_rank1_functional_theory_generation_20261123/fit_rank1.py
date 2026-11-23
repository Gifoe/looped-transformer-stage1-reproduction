"""Fitted generation weights with original Rank-1 retention on model-owned prefixes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import generation
from controller import load_controller, save_controller, controller_size
from minimal import dump, forward, label_prompt, runtime, sha
from train import BATCH, SEED, logprobs, partitions, ranking


def sequence(tokenizer, prompt, answer, q):
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    whole = tokenizer.encode(prompt + " " + answer, add_special_tokens=False)
    if whole[:len(prefix)] != prefix or len(whole) <= len(prefix):
        raise ValueError((q, prompt, answer))
    return whole, len(whole) - len(prefix), q, 0


def targets(raw, tok):
    letter, content, prefixes = [], [], []
    for q,row in enumerate(raw):
        prompt,_,gold = label_prompt(row)
        content_prompt = prompt.replace("Answer with the option letter.", "Answer with the full option text.")
        text = row["choices"]["text"][row["choices"]["label"].index(row["answerKey"])]
        letter.append(sequence(tok,prompt,gold,q))
        content.append(sequence(tok,content_prompt,text,q))
        prefixes.append(tok.encode(content_prompt,add_special_tokens=False))
    return letter, content, prefixes


@torch.no_grad()
def own_prefixes(tok, model, original, prompts, q, depth, steps):
    chosen = [prompts[i] for i in q]
    generated,_,_,_ = generation.generate_batch(tok,chosen,lambda packs,last:generation.model_next(
        model,None,generation.Arm("minimal",original,1.),depth,packs,last),steps,"cuda")
    # Include all past decoding positions in the feedback path. The dummy EOS
    # exposes the last predictor and does not contribute a target loss.
    return [(p+g+[tok.eos_token_id],len(g)+1,i,0) for p,g,i in zip(chosen,generated,q,strict=True)]


def fit(args):
    torch.manual_seed(SEED)
    old,model,tok,info=runtime(args.archive,args.previous,args.model)
    model.eval().requires_grad_(False)
    initial=args.control/"results/weights/rank1_t1.pt"
    original=load_controller(initial).requires_grad_(False)
    net=load_controller(initial).requires_grad_(True)
    raw=sorted(old.read_data(args.corpus,"train"),key=lambda r:hashlib.sha256(
        ("rank1-dev-20261123:"+r["id"]).encode()).digest())[256:]
    assert len(raw)==863
    items, meta=partitions(old,tok,raw)
    labels, text, prompts=targets(raw,tok)
    ids=np.random.default_rng(SEED).permutation(len(raw)).tolist()
    optimizer=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=.02)
    # The inherited runtime disables gradients globally for benchmark scoring.
    torch.set_grad_enabled(True)
    temperature=2.
    costs=[]
    began=time.perf_counter()
    for step,start in enumerate(range(0,len(ids),BATCH)):
        q=ids[start:start+BATCH]
        k=1+step%4
        optimizer.zero_grad(set_to_none=True)
        candidates=old.pack([item for i in q for item in items[i]])
        ranked,anchor=forward(model,candidates,k,net)
        logp,logits=logprobs(model,ranked,candidates[-1])
        rank,_=ranking(logp,candidates[1],meta,q)
        (.40*rank).backward()
        del ranked,anchor,logp,logits
        label_pack=old.pack([labels[i] for i in q])
        adjusted,anchor=forward(model,label_pack,k,net)
        lab,logits=logprobs(model,adjusted,label_pack[-1])
        (-.75*lab.mean()).backward()
        del adjusted,anchor,logits
        content_pack=old.pack([text[i] for i in q])
        adjusted,anchor=forward(model,content_pack,k,net)
        words,logits=logprobs(model,adjusted,content_pack[-1])
        (-.50*words.mean()).backward()
        del adjusted,anchor,logits
        preservation=rank.new_zeros(())
        calls=3
        if step%2==0:
            roll_steps=1+step%3
            replay=old.pack(own_prefixes(tok,model,original,prompts,q,k,roll_steps))
            group=replay[1]
            at_end=torch.cat((group[1:]!=group[:-1],group.new_ones(1,dtype=torch.bool)))
            with torch.no_grad():
                ref,_=forward(model,replay,k,original)
                distribution=(model.lm_head(ref[at_end]).float()/temperature).softmax(-1)
            prediction,_=forward(model,replay,k,net)
            posterior=(model.lm_head(prediction[at_end]).float()/temperature).log_softmax(-1)
            preservation=temperature**2*F.kl_div(posterior,distribution,reduction="batchmean")
            (args.kl*preservation).backward()
            calls+=2+roll_steps
        gradient=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        if not torch.isfinite(gradient):
            raise RuntimeError(f"Nonfinite gradient at step {step}")
        optimizer.step()
        costs.append(dict(step=step,depth=k,rank=float(rank.detach()),label=float(-lab.mean().detach()),
                          content=float(-words.mean().detach()),kl=float(preservation.detach()),max_forwards=calls))
        if step%24==0:
            print(costs[-1],flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    meta=dict(hyperparameters=dict(lr=args.lr,preservation_kl=args.kl,temperature=temperature,seed=SEED),
              initialized_from=sha(initial),train_ids=[r["id"] for r in raw],runtime=json.loads(json.dumps(info)),
              seconds=time.perf_counter()-began,steps=len(costs),size=controller_size(net))
    path=args.out/"rank1_generation.pt"
    save_controller(path,net,meta)
    dump(args.out/"TRAIN_RECEIPT.json",dict(meta,sha256=sha(path),costs=costs))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("archive","previous","model","corpus","control","out"):
        parser.add_argument("--"+arg,type=Path,required=True)
    parser.add_argument("--lr",type=float,default=.0003)
    parser.add_argument("--kl",type=float,default=.30)
    fit(parser.parse_args())
