"""Fit a shared dual-objective readout, matched auxiliaries and OOF utility gates."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bridge import configure, dump, rows, winner

DEPTHS=range(1,5)
SEED=261122919
KINDS=('teacher','shuffled','functional')


def scatter(token, group, n):
    return torch.zeros(n,device=token.device,dtype=token.dtype).index_add(0,group,token)


def loss_vectors(s, chars, options, gold):
    v=s.new_full((len(options),max(options)),-1e6)
    z=v.clone()
    eligible=torch.arange(max(options),device=s.device)[None,:]<torch.tensor(options,device=s.device)[:,None]
    v[eligible] = s/3
    z[eligible] = 16*s/torch.tensor(chars,device=s.device)
    target=torch.tensor(gold,device=s.device)
    return v,z,target


def rank_loss(s, original, chars, options, gold, weight, preserve, teacher_scores=None):
    a,b,g=loss_vectors(s,chars,options,gold)
    base_a,base_b,_=loss_vectors(original,chars,options,gold)
    raw,norm=F.cross_entropy(a,g),F.cross_entropy(b,g)
    if torch.is_grad_enabled():
        gr=torch.autograd.grad(raw,s,retain_graph=True)[0].detach().norm()
        gn=torch.autograd.grad(norm,s,retain_graph=True)[0].detach().norm()
        ratio=(gn/gr.clamp_min(1e-8)).sqrt().clamp(.35,2.85)
    else:
        ratio=1.
    loss=weight*ratio*raw+(1-weight)/ratio*norm
    for current,base in ((a,base_a),(b,base_b)):
        values,picks=base.topk(2,dim=-1)
        good=(picks[:,0]==g) & ((values[:,0]-values[:,1])>=.40)
        kl=F.kl_div(current.log_softmax(-1),base.softmax(-1),reduction='none').sum(-1)
        loss=loss+preserve*(kl*good).sum()/max(1,len(g))
    if teacher_scores is not None:
        ta,tb,_=loss_vectors(teacher_scores,chars,options,gold)
        for current,teacher in ((a,ta),(b,tb)):
            mask=teacher.argmax(-1)==g
            kl=F.kl_div(current.log_softmax(-1),teacher.softmax(-1),reduction='none').sum(-1)
            loss=loss+.20*(kl*mask).mean()
    return loss


def base_scores(amp,model,state,k,target,group,n):
    with torch.no_grad():
        tok=amp.nll_tokens(model,state['raw'][:,k-1],target)
        return scatter(tok,group,n)


def train(amp,model,data,indices,weight,kind,initial,epochs=3):
    torch.manual_seed(SEED)
    rng=torch.Generator().manual_seed(SEED)
    net=amp.operator(initial) if initial else amp.Amplifier('direction').cuda()
    net.train()
    opt=torch.optim.AdamW(net.parameters(),lr=.0007 if initial else .0015,weight_decay=.04)
    loss_values=[]
    for e in range(epochs):
        perm=[indices[j] for j in torch.randperm(len(indices),generator=rng).tolist()]
        ep=[]
        with torch.enable_grad():
            for batch,lo in enumerate(range(0,len(perm),12)):
                q=perm[lo:lo+12]
                k=1+(batch+e)%4
                future=k<4 and kind in KINDS
                state,target,group,chars,options,gold=data.batch(q,k,future)
                prior=base_scores(amp,model,state,k,target,group,len(chars))
                h,(g,shift,wn,w,_) = net(state,k)
                candidate=scatter(amp.nll_tokens(model,h,target),group,len(chars))
                teacher=base_scores(amp,model,state,k+1,target,group,len(chars)) if future and kind=='functional' else None
                objective=rank_loss(candidate,prior,chars,options,gold,weight,.12,teacher)
                if future and kind in ('teacher','shuffled'):
                    delta=amp.limit(amp.orthogonal(amp.writes(state,k+1)-w,w,wn),.5*wn)
                    if kind=='shuffled' and len(q)>1:
                        owner=torch.arange(len(q),device=group.device).repeat_interleave(torch.tensor(options,device=group.device))[group]
                        first=torch.cat((owner.new_zeros(1),torch.where(owner[1:]!=owner[:-1])[0]+1))
                        donor=first[(owner+1)%len(q)]
                        mixed=amp.orthogonal(delta[donor],w,wn)
                        delta=mixed/amp.rms(mixed)*amp.rms(delta)
                    objective=objective+.25*((shift-delta)/wn).square().mean()
                objective=objective+.015*((amp.rms(shift)/wn).square().mean()+g.square().mean())
                opt.zero_grad(set_to_none=True)
                objective.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step()
                ep.append(float(objective.detach()))
        loss_values.append(float(np.mean(ep)))
    return net.eval(),loss_values


def save_net(af,path,net,meta):
    af.atomic_save(dict(architecture='direction',settings=meta,state={k:x.detach().cpu() for k,x in net.state_dict().items()}),path)


def split_scores(v,options):
    end=0
    for n in options:
        yield v[end:end+n]
        end+=n


def features_for_gate(amp,state,meta,group,chars,options,base,k):
    gain,shift,wr,w,hr=meta
    token_stats=[gain[:,0].abs(),(amp.rms(shift)/wr)[:,0],(wr/hr)[:,0]]
    for channel in amp.CHANNELS:
        token_stats.append((amp.rms(state[channel][:,k-1].float())/hr)[:,0])
    if k>1:
        change=state['raw'][:,k-1].float()-state['raw'][:,k-2].float()
        token_stats.append((amp.rms(change)/hr)[:,0])
    else:
        token_stats.append(torch.zeros_like(token_stats[0]))
    counts=torch.bincount(group,minlength=len(chars)).float()
    f=torch.stack([scatter(x,group,len(chars))/counts for x in token_stats],dim=1).cpu().numpy()
    original=base.detach().cpu().numpy()
    features=[]
    for scores,lengths,g in zip(split_scores(original,options),split_scores(chars,options),split_scores(f,options),strict=True):
        sentences=np.asarray(lengths)
        normalized=scores/sentences
        ix=int(np.argmax(scores)); iy=int(np.argmax(normalized))
        r,n=np.sort(scores),np.sort(normalized)
        coarse=[(k-1)/3,r[-1]-r[-2],n[-1]-n[-2],r[-1],n[-1],float(ix==iy),
                np.mean(sentences),np.std(sentences),sentences[ix],sentences[iy]]
        features.append(coarse+g.mean(0).tolist()+g[ix].tolist()+g[iy].tolist())
    return np.asarray(features,dtype=np.float32)


class Gate(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n,12),nn.SiLU(),nn.Linear(12,1))

    def forward(self,x):
        return self.net(x).squeeze(-1).tanh()


def fit_gate(x,y,confidence=False):
    torch.manual_seed(SEED)
    if confidence:
        x=x[:,:6]
    center=x.mean(0)
    spread=x.std(0).clip(.02)
    xs=torch.as_tensor(np.clip((x-center)/spread,-6,6),dtype=torch.float32)
    ys=torch.as_tensor(y,dtype=torch.float32)
    mod=Gate(x.shape[-1]); opt=torch.optim.AdamW(mod.parameters(),lr=.007,weight_decay=.15)
    rng=torch.Generator().manual_seed(SEED)
    with torch.enable_grad():
        for _ in range(160):
            j=torch.randint(len(ys),(128,),generator=rng)
            loss=(mod(xs[j])-ys[j]).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    mod.eval()
    return dict(n=x.shape[-1],mean=center.tolist(),scale=spread.tolist(),state={k:v.detach().tolist() for k,v in mod.state_dict().items()})


def gate_engine(config):
    n=config['n']; mod=Gate(n)
    mod.load_state_dict({k:torch.tensor(v) for k,v in config['state'].items()})
    mean=np.asarray(config['mean'],dtype=np.float32); std=np.asarray(config['scale'],dtype=np.float32)
    @torch.no_grad()
    def predict(features):
        return mod(torch.from_numpy(np.clip((features[:,:n]-mean)/std,-6,6))).numpy()
    return predict


def weighted_dose(prob,threshold):
    return np.where(prob>=threshold,1.,0.).astype(np.float32)


def token_dose(weights, options, group):
    a=torch.as_tensor(weights,device=group.device,dtype=torch.float32)
    return a.repeat_interleave(torch.tensor(options,device=group.device))[group,None]


@torch.no_grad()
def iter_cached(amp,model,net,data,indices,dose=1.,policy=None):
    for k in DEPTHS:
        for lo in range(0,len(indices),32):
            q=indices[lo:lo+32]
            st,tok,group,chars,options,_=data.batch(q,k)
            orig=base_scores(amp,model,st,k,tok,group,len(chars))
            out,meta=net(st,k,dose)
            feat=features_for_gate(amp,st,meta,group,chars,options,orig,k)
            if policy:
                engine,threshold=policy
                active=weighted_dose(engine(feat),threshold)
                out,_=net(st,k,dose*token_dose(active,options,group))
            after=scatter(amp.nll_tokens(model,out,tok),group,len(chars)).cpu().tolist()
            yield k,q,list(split_scores(orig.cpu().tolist(),options)),list(split_scores(after,options)),feat


def evaluate_cached(amp,model,net,data,dose,policy=None):
    table=[dict(r,scores={}) for r in data.rows]
    for k,idx,_,new,_ in iter_cached(amp,model,net,data,list(range(len(data.rows))),dose,policy):
        for q,s in zip(idx,new,strict=True):
            table[q]['scores'][f'k{k}']=s
    return table


def matrix(records):
    return {m:[sum(winner(r['scores'][f'k{k}'],r,m=='acc_norm')==r['gold'] for r in records) for k in DEPTHS]
            for m in ('acc','acc_norm')}


def rank(metric,native):
    a,b=metric['acc'],metric['acc_norm']
    early=min(a[0]-native['acc'][0],a[1]-native['acc'][1])
    return sum(a)+sum(b)+.5*early,early,sum(b)


def run(model_path,previous,arc,out):
    old,af,amp=configure(previous,arc)
    m,_,_=old.load_runtime(model_path)
    m.requires_grad_(False)
    train_data=amp.Data(previous/'features/train'); dev=amp.Data(previous/'features/validation')
    native=rows(arc/'results/validation_predictions.jsonl')
    assert [r['id'] for r in native]==[r['id'] for r in dev.rows]
    base={t:[sum(winner(r['scores'][f'native_k{k}'],r,t=='acc_norm')==r['gold'] for r in native) for k in DEPTHS] for t in ('acc','acc_norm')}
    p=previous/'results/direction_round1.pt'
    logs={}; locked={}
    batch=list(range(len(train_data.rows)))
    best=None
    for weight in (.25,.50,.75):
        net,loss=train(amp,m,train_data,batch,weight,'dual',p)
        path=out/f'dual_{weight:.2f}.pt'
        save_net(af,path,net,dict(lambda_raw=weight,loss=loss))
        for dose in (.5,.75,1.):
            validation=evaluate_cached(amp,m,net,dev,dose)
            metric=matrix(validation)
            logs[f'{path.stem}/{dose}']=metric
            key=rank(metric,base)
            if best is None or key>best:
                best=key
                locked['dual']=dict(file=path.name,dose=dose,weight=weight,validation=metric)
                old.write_predictions(out/'VALIDATION_dual.jsonl',validation)
            print('dual',weight,dose,metric,flush=True)
            dump(out/'VALIDATION_SEARCH.json',logs)
    choice=locked['dual']
    for kind in KINDS:
        net,loss=train(amp,m,train_data,batch,choice['weight'],kind,p)
        path=out/f'{kind}.pt'
        save_net(af,path,net,dict(lambda_raw=choice['weight'],kind=kind,loss=loss))
        validation=evaluate_cached(amp,m,net,dev,choice['dose'])
        metric=matrix(validation)
        locked[kind]=dict(file=path.name,dose=choice['dose'],validation=metric)
        old.write_predictions(out/f'VALIDATION_{kind}.jsonl',validation)
        print('aux',kind,metric,flush=True)
    folds=[int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16)%2 for r in train_data.rows]
    X=[]; Y=[]; provenance=[]
    for fold in (0,1):
        fit=[i for i,f in enumerate(folds) if f!=fold]; hold=[i for i,f in enumerate(folds) if f==fold]
        net,loss=train(amp,m,train_data,fit,choice['weight'],'dual',None)
        save_net(af,out/f'oof_{fold}.pt',net,dict(training_n=len(fit),holdout_n=len(hold),loss=loss))
        for k,idx,orig,edited,feat in iter_cached(amp,m,net,train_data,hold,choice['dose']):
            for j,a,b,f in zip(idx,orig,edited,feat,strict=True):
                r=train_data.rows[j]; g=r['gold']
                dif=[int(winner(b,r,n)==g)-int(winner(a,r,n)==g) for n in (False,True)]
                X.append(f); Y.append(.5*sum(dif))
                provenance.append(dict(id=r['id'],fold=fold,depth=k,feature=f.tolist(),delta=dif,native=a,readout=b))
        print('oof',fold,len(fit),len(hold),flush=True)
    assert len({(r['id'],r['depth']) for r in provenance})==1119*4
    dump(out/'OOF_UTILITY.json',dict(rows=provenance))
    net=amp.operator(out/choice['file'])
    for name in ('gate','confidence'):
        g=fit_gate(np.asarray(X),np.asarray(Y),name=='confidence')
        engine=gate_engine(g)
        values=engine(np.asarray(X))
        best=None
        for frac in (.10,.25,.40,.60):
            threshold=float(np.quantile(values,frac))
            validation=evaluate_cached(amp,m,net,dev,choice['dose'],(engine,threshold))
            metric=matrix(validation)
            logs[f'{name}/{frac}']=metric
            key=rank(metric,base)
            if best is None or key>best:
                best=key
                locked[name]={**choice,'validation':metric,'gate':g,'threshold':threshold,'train_reject_fraction':frac}
                old.write_predictions(out/f'VALIDATION_{name}.jsonl',validation)
        print(name,locked[name]['validation'],flush=True)
    dump(out/'VALIDATION_SEARCH.json',logs)
    dump(out/'READOUT_FREEZE.json',dict(config=locked,validation_native=base,feature_hash=af.sha(previous/'results/FREEZE.json'),
                                       weights={f:af.sha(out/f) for f in {v['file'] for v in locked.values()}},
                                       qualified_data='ARC Train only; OOF initialized without pretrained ARC adapters'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','previous','archive','out'):
        p.add_argument(name,type=Path)
    a=p.parse_args(); run(a.model,a.previous,a.archive,a.out)
