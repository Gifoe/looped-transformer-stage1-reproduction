"""Matched task, generation-compatible and distillation fits for minimal feedback."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from minimal import DEPTHS, Minimal, dump, forward, label_prompt, load_net, rms, runtime, save_net, sha

SEED=261123137
BATCH=4
PHASE1={**{f'linear{r}_t1':('linear',r,'t1') for r in (1,2,4,8,16)},
        'nonlinear8_t1':('nonlinear',8,'t1'),'history8_t1':('history',8,'t1'),
        'adapter8_t1':('plain',8,'t1'),'distill8_t1':('linear',8,'distill'),
        'constant_t1':('constant',1,'t1')}


def partitions(old,tok,raw):
    items,metadata=old.encode(tok,raw)
    groups=[[] for _ in raw]
    for x in items:
        groups[x[2]].append(x)
    return groups,metadata


def letter_targets(tok,raw):
    results=[]
    for q,r in enumerate(raw):
        prompt,_,gold=label_prompt(r)
        prefix=tok.encode(prompt,add_special_tokens=False)
        total=tok.encode(prompt+' '+gold,add_special_tokens=False)
        assert total[:len(prefix)]==prefix and len(total)>len(prefix)
        results.append((total,len(total)-len(prefix),q,0))
    return results


def ranking(logp,idx,metadata,choices):
    total=sum(len(metadata[q]['labels']) for q in choices)
    s=logp.new_zeros(total).index_add(0,idx,logp)
    options=[len(metadata[q]['labels']) for q in choices]
    chars=[c for q in choices for c in metadata[q]['characters']]
    gold=torch.tensor([metadata[q]['gold'] for q in choices],device=logp.device)
    v=s.new_full((len(options),max(options)),-1e6)
    norm=v.clone()
    mask=torch.arange(max(options),device=s.device)[None,:]<torch.tensor(options,device=s.device)[:,None]
    v[mask]=s/3
    norm[mask]=16*s/torch.tensor(chars,device=s.device)
    flags=torch.zeros(total,dtype=torch.bool,device=s.device)
    flags[mask.new_tensor(np.cumsum([0]+options[:-1]).tolist(),dtype=torch.long)+gold]=True
    return .5*F.cross_entropy(v,gold)+.5*F.cross_entropy(norm,gold),-logp[flags[idx]].mean()


def logprobs(model,h,target):
    logits=model.lm_head(h).float()
    return -F.cross_entropy(logits,target,reduction='none'),logits


@torch.enable_grad()
def fit(old,model,items,metadata,labels,kind,rank,objective,basis=None,fixed=False):
    torch.manual_seed(SEED)
    net=Minimal(kind,rank,None if kind=='plain' else .45,basis,fixed).cuda()
    trainable=[p for p in net.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=.001 if objective!='distill' else .0015,weight_decay=.03)
    trial=np.random.default_rng(SEED).permutation(len(items)).tolist()
    for i in range(0,len(trial),64):
        trial[i:i+64]=sorted(trial[i:i+64],key=lambda q:max(len(s[0]) for s in items[q]))
    losses=[]; grads=[]; network_calls=0; loop_calls=0; begin=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step,i in enumerate(range(0,len(trial),BATCH)):
        q=trial[i:i+BATCH]; k=1+step%4
        p=old.pack([a for j in q for a in items[j]])
        if objective=='distill':
            with torch.no_grad():
                end,anchor=forward(model,p,k)
            h=model.model.norm((anchor+net.delta(anchor)).to(end.dtype))
            loss=(h.float()-end.float()).square().mean()
        else:
            edited,anchor=forward(model,p,k,net)
            prob,logits=logprobs(model,edited,p[-1]); del logits
            task,content=ranking(prob,p[1],metadata,q)
            loss=task
            if objective in ('t2','t2kl'):
                label_pack=old.pack([labels[a] for a in q])
                prediction,_=forward(model,label_pack,k,net)
                nll,label_logits=logprobs(model,prediction,label_pack[-1]); network_calls+=1; loop_calls+=k
                loss=.35*content+.65*(-nll.mean())+.3*task
                if objective=='t2kl':
                    with torch.no_grad():
                        native,_=forward(model,label_pack,k)
                        distribution=model.lm_head(native).float().softmax(-1)
                    loss=loss+.1*F.kl_div(label_logits.log_softmax(-1),distribution,reduction='batchmean')
                    network_calls+=1; loop_calls+=k
        opt.zero_grad(set_to_none=True); loss.backward()
        g=nn.utils.clip_grad_norm_(trainable,1.)
        if step==0:
            assert torch.isfinite(g) and g>0
        opt.step(); losses.append(loss.detach()); grads.append(g.detach())
        network_calls+=1; loop_calls+=k
    torch.cuda.synchronize()
    return net.eval(),dict(objective=objective,n=len(trial),steps=len(losses),loss=float(torch.stack(losses).mean()),
        mean_gradient=float(torch.stack(grads).mean()),network_calls=network_calls,executed_loop_calls=loop_calls,
        trainable_parameters=sum(p.numel() for p in trainable),total_parameters=sum(p.numel() for p in net.parameters()),
        seconds=time.perf_counter()-begin,peak_bytes=torch.cuda.max_memory_allocated())


@torch.no_grad()
def evaluate(old,model,tok,raw,net,dose,depths=DEPTHS):
    items,meta=old.encode(tok,raw)
    items.sort(key=lambda x:len(x[0]))
    costs={}
    for k in depths:
        tag=f'k{k}'
        def run(batch):
            p=old.pack(batch); h,_=forward(model,p,k,net,dose)
            logits=model.lm_head(h).float()
            out=old.scores(logits,p[1].cpu().numpy(),p[-1],len(batch))
            return out,p[0].numel()
        run(items[:16]); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        elapsed=[]; padded=0
        for off in range(0,len(items),16):
            batch=items[off:off+16]; begin=time.perf_counter()
            values,n=run(batch); elapsed.append(time.perf_counter()-begin); padded+=n
            old.record(meta,batch,{tag:values})
        torch.cuda.synchronize()
        costs[tag]=dict(seconds=sum(elapsed),median_batch_ms=1000*float(np.median(elapsed)),
                        p95_batch_ms=1000*float(np.quantile(elapsed,.95)),peak_bytes=torch.cuda.max_memory_allocated(),
                        backbone_calls=len(elapsed),lm_head_calls=len(elapsed),loops=k,padded_token_loops=padded*k,
                        candidates=len(items),batch_size=16)
    return meta,costs


def count(rows):
    return {m:[sum(int(np.argmax(np.asarray(r['scores'][f'k{k}'])/(r['characters'] if m=='acc_norm' else 1)))==r['gold']
                   for r in rows) for k in DEPTHS] for m in ('acc','acc_norm')}


def validation_utility(summary):
    return sum(summary['acc'])+sum(summary['acc_norm']),sum(summary['acc_norm'])


def finish(out,records,basis_requested):
    def pick(names):
        return max(names,key=lambda a:(*validation_utility(records[a]['counts']),-records[a]['fit']['total_parameters'],a))
    requested=['linear8_t1','nonlinear8_t1','history8_t1','distill8_t1','adapter8_t1','constant_t1','linear8_t2','nonlinear8_t2']
    main=[a for a in records if (a.startswith(('linear','nonlinear','subspace')) and records[a]['fit']['total_parameters']<=32768)]
    performance=pick(main)
    generation=pick([a for a in records if a.endswith(('_t2','_t2kl'))])
    arms=list(dict.fromkeys(requested+[performance,generation]+[a for a in records if a.startswith('subspace')]))
    assert set(arms)<=set(records)
    frozen={a:dict(file=a+'.pt',sha256=sha(out/(a+'.pt')),dose=records[a]['dose'],params=records[a]['fit']['total_parameters'],
                   validation=records[a]['counts']) for a in arms}
    dump(out/'FREEZE.json',dict(arms=frozen,best=performance,generation=generation,seed=SEED,
          selection='max equal-weight validation correct counts across two candidate metrics and K1–K4, with <=32768 stored parameters for main selection',
          source={f:sha(Path(__file__).parent/f) for f in ('train.py','minimal.py')},subspace_initialization_attempted=basis_requested))


def run(args):
    old,model,tok,info=runtime(args.archive,args.previous,args.model)
    args.out.mkdir(parents=True,exist_ok=True)
    dump(args.out/'RUNTIME.json',info)
    train=old.read_data(args.corpus,'train'); dev=old.read_data(args.corpus,'validation')
    partition,metadata=partitions(old,tok,train)
    labels=letter_targets(tok,train) if args.stage=='refine' else None
    archive=args.out/'VALIDATION.json'
    import json
    results=json.loads(archive.read_text()) if archive.exists() else {}
    basis=None
    if args.stage=='screen':
        todo=PHASE1
    elif args.stage=='refine':
        todo={f'{model}8_t2':(model,8,'t2') for model in ('linear','nonlinear')}
    else:
        todo={}
    def trial(name,kind,rank,objective,basis=None,fixed=False):
        if name in results:
            return
        net,details=fit(old,model,partition,metadata,labels,kind,rank,objective,basis,fixed)
        choices=(.65,1.) if name in ('linear8_t1','nonlinear8_t1','history8_t1','linear8_t2','nonlinear8_t2') else (1.,)
        options={}; best=None
        for dose in choices:
            val,cost=evaluate(old,model,tok,dev,net,dose)
            score=count(val); options[str(dose)]=dict(counts=score,cost=cost)
            key=validation_utility(score)
            if best is None or key>best[0]:
                best=key,dose,score
                old.write_predictions(args.out/f'VAL_{name}.jsonl',val)
        save_net(args.out/f'{name}.pt',net,details)
        results[name]=dict(fit=details,dose=best[1],counts=best[2],options=options)
        dump(archive,results)
        print(name,details['seconds'],best[1],best[2],flush=True)
    for name,cfg in todo.items():
        trial(name,*cfg)
    if args.stage=='refine':
        winner=max(('linear8_t2','nonlinear8_t2'),key=lambda s:validation_utility(results[s]['counts']))
        trial(winner+'kl',winner.split('8')[0],8,'t2kl')
        if args.basis:
            data=torch.load(args.basis,map_location='cpu',weights_only=True)
            basis=data['basis']; assert basis.shape[0]==2048 and basis.shape[1]>=8
            trial('subspace_fixed_t1','linear',8,'t1',basis,True)
            trial('subspace_joint_t1','linear',8,'t1',basis,False)
        finish(args.out,results,bool(args.basis))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('screen','refine'))
    for a in ('model','archive','previous','corpus','out'):
        p.add_argument(a,type=Path)
    p.add_argument('--basis',type=Path)
    run(p.parse_args())
