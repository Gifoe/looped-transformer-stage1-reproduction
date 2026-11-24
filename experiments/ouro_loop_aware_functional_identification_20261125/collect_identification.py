"""Collect pre-norm rank-one direct and loop adjoints with one native forward."""

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import time

import torch
import torch.nn.functional as F

from advantages import choice_batch, sequence_mean, token_margin
from controller_io import controller
import minimal
from task_data import encode_target, hash_order, load, multiple_choice, save
from train_generative import pack_entries


@dataclass
class Batch:
    packed: tuple
    ids: list
    example: torch.Tensor
    metadata: list | None


def pack(archive, tok, rows, mode):
    ids = [r['sample_id'] for r in rows]
    if mode == 'candidate':
        data = choice_batch(archive,tok,rows)
        counts = [len(r['characters']) for r in data.metadata]
        by_example = torch.repeat_interleave(torch.arange(len(rows),device=data.packed[0].device),
                                             torch.tensor(counts,device=data.packed[0].device))
        return Batch(data.packed,ids,by_example,data.metadata)
    encoded = [encode_target(tok,r) for r in rows]
    data = pack_entries(archive,encoded)
    return Batch(data,ids,torch.arange(len(rows),device=data[0].device),None)


def readout(model, state, batch, eos):
    tokens,sequences,positions,targets = batch.packed
    logits = model.lm_head(state[sequences,positions]).float()
    if batch.metadata is None:
        decisive = token_margin(logits,targets,eos)
        mean = sequence_mean(decisive,sequences,len(batch.ids))
        token_accuracy = sequence_mean((logits.argmax(-1)==targets).float(),sequences,len(batch.ids))
        return mean, token_accuracy
    pieces = -F.cross_entropy(logits,targets,reduction='none')
    sums = pieces.new_zeros(len(tokens)).index_add_(0,sequences,pieces)
    sizes = [len(r['characters']) for r in batch.metadata]
    lengths = sums.new_tensor([v for r in batch.metadata for v in r['characters']])
    scores = (16*sums/lengths).split(sizes)
    improvements, correct = [], []
    for row, vector in zip(batch.metadata,scores,strict=True):
        gold=row['gold']
        rivals=torch.cat((vector[:gold],vector[gold+1:]))
        improvements.append(vector[gold]-rivals.logsumexp(0)+rivals.new_tensor(rivals.numel()).log())
        correct.append(vector.argmax()==gold)
    return torch.stack(improvements),torch.stack(correct).float()


def native(model, batch, depth, intervention=None, differentiate=False, receiver=None):
    """Use a full pre-norm leaf or edit at the historical penultimate loop boundary."""
    tokens,seq,pos,_ = batch.packed
    core=model.model
    anchor=depth-1 if receiver is None else receiver
    counter=0
    leaf=None

    def hook(_, args):
        nonlocal leaf,counter
        counter+=1
        if counter!=anchor:
            return None
        original=args[0]
        leaf=original.detach().float()
        if differentiate:
            leaf.requires_grad_(True)
            return (leaf.to(original.dtype),*args[1:])
        if intervention is None:
            return None
        perturb=intervention(leaf,seq,pos)
        return ((leaf+perturb).to(original.dtype),*args[1:])

    old=core.total_ut_steps
    core.total_ut_steps=depth
    hand=core.norm.register_forward_pre_hook(hook)
    try:
        _,states,_=core(input_ids=tokens,use_cache=False)
    finally:
        core.total_ut_steps=old
        hand.remove()
    if leaf is None or counter!=depth:
        raise ValueError('Native Ouro did not reach the requested receiver')
    return states[anchor-1],states[-1],leaf


def derivatives(model,batch,depth,eos):
    with torch.enable_grad():
        direct,loop,leaf=native(model,batch,depth,differentiate=True)
        p,a=readout(model,direct,batch,eos)
        q,b=readout(model,loop,batch,eos)
        dp,=torch.autograd.grad(p.sum(),leaf,retain_graph=True)
        dq,=torch.autograd.grad(q.sum(),leaf)
    return leaf.detach(),dp.detach(),dq.detach(),p.detach(),q.detach(),a.detach(),b.detach()


def reduce_questions(batch,values):
    return values.new_zeros(len(batch.ids)).index_add_(0,batch.example[batch.packed[1]],values)


def accumulate(model,batch,depth,eos,metrics,operators):
    h,gd,gl,p,q,a,b=derivatives(model,batch,depth,eos)
    rows,pos=batch.packed[1:3]
    states=h[rows,pos]
    direct,loop=gd[rows,pos],gl[rows,pos]
    score=(loop-direct)
    metrics['direct'].addmm_(direct.T,states)
    metrics['loop'].addmm_(loop.T,states)
    if 'full_loop' in metrics:
        maximum=pos.new_full((h.shape[0],),-1).scatter_reduce_(0,rows,pos,reduce='amax')
        valid=torch.arange(h.shape[1],device=h.device)[None]<=maximum[:,None]
        metrics['full_loop'].addmm_(gl[valid].T,h[valid])
    # Direct readout depends only on its scored positions, while the loop can
    # propagate derivatives into other earlier positions in the same sequence.
    used=torch.zeros(h.shape[:2],dtype=torch.bool,device=h.device)
    used[rows,pos]=True
    off=~used
    diag=dict(direct_off=float(gd[off].square().sum()),loop_off=float(gl[off].square().sum()),
              loop_total=float(gl.square().sum()),predictors=len(rows),padded_tokens=h.shape[0]*h.shape[1])
    projections={}
    with torch.no_grad():
        for name,net in operators.items():
            proposed=net.delta(states)
            projections[name]=dict(direct=reduce_questions(batch,(direct*proposed).sum(-1)).tolist(),
                                   loop=reduce_questions(batch,(loop*proposed).sum(-1)).tolist(),
                                   excess=reduce_questions(batch,(score*proposed).sum(-1)).tolist())
    records=[]
    for i,key in enumerate(batch.ids):
        records.append(dict(id=key,q_direct=float(p[i]),q_loop=float(q[i]),native_direct=float(a[i]),native_loop=float(b[i]),
                            slopes={name:{part:v[i] for part,v in parts.items()} for name,parts in projections.items()}))
    return records,diag


def source_rows(path,dataset,mode):
    for raw in load(path):
        if 'choices' in raw and 'answerKey' in raw:
            row=multiple_choice(raw,dataset,'content')
        elif 'views' in raw:
            options=raw['views']
            row=options.get('process',options.get('short',options.get('content')))
        else:
            row=raw
        if row is None or (mode=='candidate' and 'candidate' not in row):
            raise ValueError(f'Invalid {mode} record for {dataset}')
        yield row


def torch_save(path,payload):
    tmp=path.with_suffix('.partial')
    torch.save(payload,tmp)
    tmp.replace(path)


def batches(rows,start,limit,budget,mode):
    while start<len(rows):
        end=start
        maximum=sequences=0
        while end<min(start+limit,len(rows)):
            row=rows[end]
            size=row.get('input_tokens',max(16,len(row.get('prompt',''))//3))
            count=len(row['candidate']['choices']['text']) if mode=='candidate' else 1
            length=max(maximum,size)
            if end>start and length*(sequences+count)>budget:
                break
            maximum=length
            sequences+=count
            end+=1
        yield rows[start:end]
        start=end


def collect(a):
    archive,model,tok,runtime=minimal.runtime(a.archive,a.previous,a.model)
    device=next(model.parameters()).device
    operators={name:controller(Path(path),device) for name,path in (arg.split('=',1) for arg in a.operator)}
    a.out.mkdir(parents=True,exist_ok=True)
    lock=dict(source_sha256=minimal.sha(a.input),code_sha256=minimal.sha(Path(__file__)),model_sha256=runtime['model_sha256'],
              depth=a.depth,mode=a.mode,dataset=a.dataset,folds=a.folds,batch=a.batch,limit=a.limit,
              token_budget=a.token_budget,full_context=a.full_context,
              operator_sha256={n:minimal.sha(Path(path)) for n,path in (arg.split('=',1) for arg in a.operator)})
    path=a.out/'LOCK.json'
    if path.exists() and json.loads(path.read_text())!=lock:
        raise ValueError(f'Changed frozen inputs in {a.out}')
    minimal.dump(path,lock)
    host=list(source_rows(a.input,a.dataset,a.mode))
    if a.limit:
        host=sorted(host,key=lambda r:hash_order(r['sample_id'],'loop-identification/diagnostic'))[:a.limit]
    if len(set(r['sample_id'] for r in host))!=len(host):
        raise ValueError('Duplicate IDs in processed training pool')
    for number in range(a.folds):
        rows=[r for r in host if int(hash_order(r.get('fold_key',r['sample_id']),'loop-identification/fold'),16)%a.folds==number]
        rows.sort(key=lambda r:(r.get('input_tokens',len(r.get('prompt',''))),r['sample_id']))
        size=model.config.hidden_size
        target=a.out/f'fold_{number}.pt'
        if target.exists():
            prior=torch.load(target,weights_only=True,map_location='cpu')
            if prior['count']<0 or prior['count']>len(rows):
                raise ValueError('Resume cursor out of the sealed cohort')
        else:
            keys=('direct','loop','full_loop') if a.full_context else ('direct','loop')
            prior=dict(moments={key:torch.zeros((size,size)) for key in keys},
                       count=0,records=[],diagnostics={},seconds=0.)
        if [r['id'] for r in prior['records']] != [r['sample_id'] for r in rows[:prior['count']]]:
            raise ValueError('Resume identity mismatch')
        if prior['count']==len(rows):
            save(a.out/f'fold_{number}.jsonl',prior['records'])
            print('COMPLETE',number,len(rows),'reused',flush=True)
            continue
        moments={key:mat.to(device) for key,mat in prior['moments'].items()}
        records=prior['records']
        diag=Counter(prior['diagnostics'])
        started=time.perf_counter()
        for i,segment in enumerate(batches(rows,prior['count'],a.batch,a.token_budget,a.mode),1):
            batch=pack(archive,tok,segment,a.mode)
            new,control=accumulate(model,batch,a.depth,tok.eos_token_id,moments,operators)
            records.extend(new)
            diag.update(control)
            if i%a.checkpoint_batches==0 or len(records)==len(rows):
                torch.cuda.synchronize()
                torch_save(target,dict(moments={key:mat.cpu() for key,mat in moments.items()},count=len(records),
                                       records=records,diagnostics=dict(diag),seconds=prior['seconds']+time.perf_counter()-started))
                print('SAVED',a.dataset,a.mode,a.depth,number,len(records),len(rows),flush=True)
        if rows:
            save(a.out/f'fold_{number}.jsonl',records)
    print('DONE',len(host),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('archive','previous','model','input','out'):
        p.add_argument(f'--{field}',required=True,type=Path)
    p.add_argument('--dataset',required=True)
    p.add_argument('--mode',choices=('candidate','generation'),required=True)
    p.add_argument('--depth',type=int,default=4)
    p.add_argument('--folds',type=int,default=4)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--limit',type=int)
    p.add_argument('--checkpoint-batches',type=int,default=48)
    p.add_argument('--token-budget',type=int,default=2304)
    p.add_argument('--full-context',action='store_true')
    p.add_argument('--operator',action='append',default=[])
    return p


if __name__=='__main__':
    collect(parser().parse_args())
