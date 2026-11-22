"""Free generation and isolated feedback head profiling on fixed ARC Validation IDs."""
import argparse
from pathlib import Path
import re
import time

import numpy as np
import torch

from bridge import dump
import feedback as fb

SEED=261122997


@torch.no_grad()
def free(model,tok,records,arms):
    demo=[]
    for r in records:
        inline='\n'.join(f'{l}. {t}' for l,t in zip(r['choices']['label'],r['choices']['text'],strict=True))
        prefix=f"Question: {r['question']}\n{inline}\nAnswer with the option letter.\nAnswer:"
        demo.append(tok.encode(prefix,add_special_tokens=False))
    logs=[dict(id=r['id'],gold=r['answerKey'],method={}) for r in records]
    for name,(net,dose,read,rd) in arms.items():
        for k in fb.DEPTHS:
            for lo in range(0,len(records),8):
                batch=demo[lo:lo+8]; tokens=[[] for _ in batch]; done=[False]*len(batch)
                for _ in range(6):
                    lengths=[len(x)+len(y) for x,y in zip(batch,tokens,strict=True)]
                    edge=max(lengths)
                    seq=[x+y+[0]*(edge-n) for x,y,n in zip(batch,tokens,lengths,strict=True)]
                    positions=[list(range(len(q)-1,n)) for q,n in zip(batch,lengths,strict=True)]
                    group=[i for i,p in enumerate(positions) for _ in p]
                    ends=np.cumsum([len(p) for p in positions])-1
                    actual,wp,_=fb.forward(model,tuple(torch.tensor(x,device='cuda',dtype=torch.long) for x in
                                            (seq,group,[s for p in positions for s in p],[0]*len(group))),
                                            k,net,dose,read,rd)
                    use=wp if wp is not None else actual
                    selected=model.lm_head(use[ends]).argmax(-1).tolist()
                    for i,t in enumerate(selected):
                        if not done[i]:
                            tokens[i].append(t)
                            line=tok.decode(tokens[i],skip_special_tokens=True)
                            done[i]=t==tok.eos_token_id or bool(line.strip('\n ').count('\n'))
                    if all(done):
                        break
                for i,seq in enumerate(tokens):
                    text=tok.decode(seq,skip_special_tokens=True)
                    match=re.match(r'^\s*(?:\(|\[)?([A-E])(?:\b|[.)\]])',text)
                    option=match[1] if match else None
                    if option not in records[lo+i]['choices']['label']:
                        option=None
                    logs[lo+i]['method'][f'{name}_k{k}']=dict(text=text,tokens=seq,option=option)
    return logs


@torch.no_grad()
def profile(old,model,items,net,dose,read,rd):
    cost={}
    for k in fb.DEPTHS:
        for name in ('feedback_only','readout_plus_feedback'):
            def one(batch):
                inp=old.pack(batch)
                x,edited,_=fb.forward(model,inp,k,net,dose,read if name=='readout_plus_feedback' else None,rd)
                logits=model.lm_head(edited if edited is not None else x).float()
                return old.scores(logits,inp[1].cpu().numpy(),inp[-1],len(batch))
            one(items[:16]); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            durations=[]
            for off in range(0,len(items),16):
                start=time.perf_counter(); one(items[off:off+16]); durations.append(time.perf_counter()-start)
            torch.cuda.synchronize()
            cost[f'{name}_k{k}']=dict(wall_s=sum(durations),median_batch_ms=1000*float(np.median(durations)),
                                      peak_bytes=torch.cuda.max_memory_allocated(),batch_calls=len(durations),
                                      head_calls_per_batch=1,loops=k,candidates=len(items),batch_size=16)
    return cost


def run(a):
    old=fb.configuration(a)
    model,tok,_=old.load_runtime(a.model)
    model.requires_grad_(False)
    net,dose,read,rd=fb.frozen(a)
    raw=old.read_data(a.corpus,'validation')
    p=np.random.default_rng(SEED).permutation(len(raw))
    small=[raw[i] for i in p[:24]]
    pool=[raw[i] for i in p[:96]]
    rows=free(model,tok,small,dict(native=(None,0.,None,0.),dual=(None,0.,read,rd),
                               feedback=(net,dose,None,0.),combo=(net,dose,read,rd)))
    items,_=old.encode(tok,pool); items.sort(key=lambda x:len(x[0]))
    cost=profile(old,model,items,net,dose,read,rd)
    counts={key:sum(x['method'][key]['option']==x['gold'] for x in rows) for key in rows[0]['method']}
    dump(a.out/'GENERATION_DIAGNOSTIC.json',dict(seed=SEED,n=len(rows),scores=counts,raw=rows,
         scope='Validation prompts listing options; unrestricted greedy generation, six-token limit, no-cache'))
    dump(a.out/'ISOLATED_FEEDBACK_COST.json',dict(n=len(pool),ids=[r['id'] for r in pool],cost=cost))
    print('24-question free generation and 96-question solo cost complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('model','corpus','archive','prior','out'):
        p.add_argument(f'--{key}',type=Path,required=True)
    run(p.parse_args())
