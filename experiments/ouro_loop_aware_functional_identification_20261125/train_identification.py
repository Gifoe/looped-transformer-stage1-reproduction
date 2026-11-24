"""Matched full-coverage Rank-1 training with question-level weighting only."""

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata
import torch
import torch.nn.functional as F

from collect_identification import batches, pack, torch_save
from controller_io import controller
import minimal
from task_data import hash_order,load


ARMS=('uniform','ordinary','loop')


def score_records(root,sources):
    records={}
    for ds in sources:
        for path in sorted((root/ds/'candidate/k4').glob('fold_*.jsonl')):
            for r in load(path):
                key=(ds,r['id'])
                if key in records or 't1' not in r['slopes']:
                    raise ValueError(f'Changed T1 selector probe for {key}')
                records[key]=r
    return records


def selection(rows,records,forecast):
    available=defaultdict(list)
    higher=defaultdict(list)
    per_source=defaultdict(list)
    for key,row in rows.items():
        r=records[key]
        stratum=(key[0],int(r['native_direct']),int(r['native_loop']),int(np.log2(max(1,row['input_tokens']))))
        available[stratum].append(key)
        per_source[key[0]].append(key)
    for value in forecast['records']:
        if value['scope']=='leave_source_out':
            for variant,metric in (('ordinary','ordinary_functional'),('loop','loop_and_absolute')):
                higher[variant,value['source']].append(value['predicted'][metric])
    weights={'uniform':{key:1. for key in rows}}
    for arm in ('ordinary','loop'):
        quality={key:float(np.mean(higher[arm,key])) for key in per_source}
        source_rank=rankdata([quality[name] for name in per_source],method='average')
        source_weight={name:.5+rank/(len(per_source)+1) for name,rank in zip(per_source,source_rank,strict=True)}
        rank_weights={}
        for cell,items in available.items():
            raw=[]
            for key in items:
                p=records[key]['slopes']['t1']
                raw.append(p['loop'] if arm=='ordinary' else min(p['loop'],p['excess']))
            ranks=rankdata(raw,method='average')
            sample_weights=.5+ranks/(len(items)+1)
            rank_weights.update({key:float(sample)*source_weight[cell[0]] for key,sample in zip(items,sample_weights,strict=True)})
        den=np.mean(list(rank_weights.values()))
        weights[arm]={k:v/den for k,v in rank_weights.items()}
    return weights


def qloss(model,h,packet):
    tokens,seq,_,truth=packet.packed
    logits=model.lm_head(h).float()
    logp=-F.cross_entropy(logits,truth,reduction='none')
    sums=logp.new_zeros(len(tokens)).index_add(0,seq,logp)
    chars=sums.new_tensor([x for r in packet.metadata for x in r['characters']])
    raw,normal=sums/3,16*sums/chars
    sizes=[len(r['characters']) for r in packet.metadata]
    loss=[]
    for r,l,n in zip(packet.metadata,raw.split(sizes),normal.split(sizes),strict=True):
        g=r['gold']
        loss.append(.5*(l.logsumexp(0)-l[g]+n.logsumexp(0)-n[g]))
    return torch.stack(loss)


def frozen_schedule(sources,data,seed,limit):
    rows={}
    order=[]
    for name in sources:
        path=data/f'{name}.jsonl'
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta['output_sha256']!=minimal.sha(path):
            raise ValueError(f'Changed pool for {name}')
        for r in load(path):
            key=(name,r['sample_id'])
            if key in rows:
                raise ValueError(f'Duplicate exposure {key}')
            # Leave the previously sealed independent fourth fold out of all fits.
            if int(hash_order(r.get('fold_key',r['sample_id']),'loop-identification/fold'),16)%4==3:
                continue
            rows[key]=r
            order.append(key)
    order.sort(key=lambda t:hash_order(str(t),f'loop-identification/training/{seed}'))
    if limit:
        order=order[:limit]
        rows={key:rows[key] for key in order}
    for i in range(0,len(order),128):
        order[i:i+128]=sorted(order[i:i+128],key=lambda key:(rows[key]['input_tokens'],hash_order(str(key),'loop-bucket-tie')))
    return rows,[rows[k] for k in order]


def fit(a):
    forecast=a.forecast/'FORECAST.json'
    plan=json.loads(forecast.read_text())
    sources=a.include or plan['source_order']
    if any(plan['sources'][s]['mode']!='candidate' for s in sources):
        raise ValueError('First matched fit uses the historical candidate objective')
    archive,model,tok,hardware=minimal.runtime(a.archive,a.previous,a.model)
    rows, order=frozen_schedule(sources,a.data,a.seed,a.limit)
    recorded=score_records(a.moments,sources)
    if len(order)==0 or not rows.keys()<=recorded.keys():
        raise ValueError('Missing per-question functional statistics for matched control')
    weight=selection(rows,recorded,plan)[a.arm]
    device=next(model.parameters()).device
    torch.manual_seed(a.seed)
    net=(controller(a.initialize,device) if a.initialize else minimal.Minimal('linear',1,.45).to(device)).requires_grad_(True)
    assert sum(p.numel() for p in net.parameters())==4096 and not any(p.requires_grad for p in model.parameters())
    if not a.initialize and torch.count_nonzero(net.out.weight):
        raise ValueError('Cold start must be exactly zero write')
    lock=dict(arm=a.arm,seed=a.seed,lr=.001,n=len(order),sources=sources,
              maximum_padded_token_loops=a.max_work,matched_question_limit=a.limit,
              forecast_sha256=minimal.sha(forecast), initialization=minimal.sha(a.initialize) if a.initialize else 'zero-write',
              depth=[1,2,3,4],task_objective='historical 50/50 raw and character-normalized candidate cross-entropy',
              exposure_sha256=sha256(json.dumps([(r['dataset'],r['sample_id']) for r in order]).encode()).hexdigest(),
              selector_sha256=sha256(json.dumps(sorted((d,k,v) for (d,k),v in weight.items())).encode()).hexdigest(),
              code_sha256=minimal.sha(__file__), model_sha256=hardware['model_sha256'])
    a.out.mkdir(parents=True,exist_ok=True)
    receipt=a.out/'TRAIN_LOCK.json'
    if receipt.exists() and json.loads(receipt.read_text())!=lock:
        raise ValueError('Matched training identity changed')
    minimal.dump(receipt,lock)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.03)
    recovery=a.out/'resume.pt'
    done,step,work,stats=(0,0,0,[])
    if recovery.exists():
        state=torch.load(recovery,weights_only=True,map_location='cpu')
        if state['lock']!=minimal.sha(receipt):
            raise ValueError('Training resume belongs to another arm')
        done,step,work,stats=(state[k] for k in ('done','step','work','stats'))
        net.load_state_dict(state['net']); optimizer.load_state_dict(state['optimizer'])
    with torch.enable_grad():
        for segment in batches(order,done,6,2600,'candidate'):
            if a.max_work and work>=a.max_work:
                break
            k=1+step%4
            packet=pack(archive,tok,segment,'candidate')
            state,_=minimal.forward(model,packet.packed,k,net)
            losses=qloss(model,state,packet)
            scale=losses.new_tensor([weight[row['dataset'],row['sample_id']] for row in segment])
            value=(losses*scale).mean()
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            slope=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.))
            if not np.isfinite(slope):
                raise ValueError(f'Nonfinite gradient at update {step}')
            optimizer.step()
            q=float(value.detach())
            done+=len(segment);step+=1;work+=packet.packed[0].numel()*k
            stats.append(dict(step=step,done=done,depth=k,train_loss=q,gradient_norm=slope,weighted_mean=float(scale.mean())))
            if step%96==0 or done==len(order) or (a.max_work and work>=a.max_work):
                torch_save(recovery,dict(lock=minimal.sha(receipt),done=done,step=step,work=work,stats=stats,
                          net={k:v.cpu() for k,v in net.state_dict().items()},optimizer=optimizer.state_dict()))
                print('TRAIN',a.arm,step,done,work,q,flush=True)
    if done!=len(order) and not (a.max_work and work>=a.max_work):
        raise RuntimeError('Partial training exposure')
    final=a.out/'controller.pt'
    minimal.save_net(final,net,dict(arm=a.arm,training_lock_sha256=minimal.sha(receipt),steps=step))
    minimal.dump(a.out/'TRAIN.json',dict(**lock,updates=step,items=done,padded_token_loops=work,
                checkpoints_sha256=minimal.sha(final),exposures=dict(Counter(r['dataset'] for r in order[:done])),trace=stats))
    print('DONE',a.arm,step,done,work,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('archive','previous','model','data','moments','forecast','out'):
        p.add_argument(f'--{field}',type=Path,required=True)
    p.add_argument('--initialize',type=Path)
    p.add_argument('--include',nargs='+')
    p.add_argument('--limit',type=int)
    p.add_argument('--max-work',type=int)
    p.add_argument('--seed',type=int,default=261125137)
    p.add_argument('--arm',choices=ARMS,required=True)
    fit(p.parse_args())
