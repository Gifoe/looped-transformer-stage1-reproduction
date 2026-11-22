"""Low-cost answer ranking from original scores and candidate lengths alone."""
import argparse
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bridge import dump, rows, winner

SEED=261122951


def inputs(row,original,k):
    s=np.asarray(original,dtype=np.float32)
    lengths=np.asarray(row['characters'],dtype=np.float32)
    tok=np.asarray(row['answer_tokens'],dtype=np.float32)
    normal=s/lengths
    return np.stack((s/12,(s-s.mean())/6,5*normal,5*(normal-normal.mean()),s/tok/4,
                     np.log(lengths)/4,np.log(tok)/3,np.log1p(lengths/tok),
                     np.full_like(s,(k-1)/3),np.full_like(s,s.std()/6),
                     np.full_like(s,lengths.std()/20)),axis=-1)


class Calibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(11,16),nn.SiLU(),nn.Linear(16,1))

    def forward(self,x,base):
        return base+5*self.head(x).squeeze(-1).tanh()


class LengthCalibration(nn.Module):
    def __init__(self, exponent, char, tok):
        super().__init__()
        self.exponent=exponent
        self.char=char
        self.tok=tok

    def forward(self,x,base):
        lengths=(4*x[...,5]).exp()
        tokens=(3*x[...,6]).exp()
        scale=(lengths/24.).pow(self.exponent)
        return base*scale+self.char*lengths+self.tok*tokens


def pack(examples,supervision=True):
    n=len(examples)
    max_options=max(len(s) for _,s,_ in examples)
    x=np.zeros((n,max_options,11),dtype=np.float32)
    score=np.full((n,max_options),-1e5,dtype=np.float32)
    lens=np.ones((n,max_options),dtype=np.float32)
    gold=[]
    for i,(r,s,k) in enumerate(examples):
        x[i,:len(s)]=inputs(r,s,k)
        score[i,:len(s)]=s
        lens[i,:len(s)]=r['characters']
        if supervision:
            gold.append(r['gold'])
    return tuple(torch.from_numpy(z) for z in (x,score,lens,np.asarray(gold)))


def load(config):
    if config.get('kind')=='analytic':
        return LengthCalibration(*config['coefficients']).eval()
    net=Calibrator()
    net.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in config['state'].items()})
    return net.eval()


@torch.no_grad()
def apply(net,table,label='native'):
    result=[dict(r,scores={}) for r in table]
    begin=time.perf_counter()
    for k in range(1,5):
        field=f'{label}_k{k}'
        xx,origin,_,_=pack([(r,r['scores'][field],k) for r in table],False)
        all_scores=net(xx,origin).tolist()
        for r,s in zip(result,all_scores,strict=True):
            r['scores'][f'scoreonly_k{k}']=s[:len(r['characters'])]
    return result,time.perf_counter()-begin


def metrics(table):
    return {m:[sum(winner(r['scores'][f'scoreonly_k{k}'],r,m=='acc_norm')==r['gold'] for r in table)
               for k in range(1,5)] for m in ('acc','acc_norm')}


def utility(score):
    return sum(score['acc'])+sum(score['acc_norm'])+.5*min(score['acc'][:2])


def run(out,prior,arc):
    torch.set_num_threads(4)
    oof=json.loads((out/'OOF_UTILITY.json').read_text())['rows']
    train={r['id']:r for r in json.loads((prior/'features/train/index.json').read_text())['rows']}
    xx,s,length,y=pack([(train[r['id']],r['native'],r['depth']) for r in oof])
    ref=rows(arc/'results/validation_predictions.jsonl')
    best=None; search={}
    for w in (.25,.50,.75):
        torch.manual_seed(SEED)
        net=Calibrator(); opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.03)
        rng=torch.Generator().manual_seed(SEED)
        for _ in range(280):
            j=torch.randint(len(y),(128,),generator=rng)
            candidate=net(xx[j],s[j]); raw=F.cross_entropy(candidate/3,y[j]); norm=F.cross_entropy(16*candidate/length[j],y[j])
            loss=w*raw+(1-w)*norm+.006*(candidate-s[j]).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        v,_=apply(net.eval(),ref)
        score=metrics(v); search[str(w)]=score
        simple=utility(score)
        if best is None or simple>best[0]:
            best=(simple,dict(kind='mlp',ratio=w,validation=score,parameter_count=sum(p.numel() for p in net.parameters()),
                              state={k:v.detach().tolist() for k,v in net.state_dict().items()}),v)
    # Include standard length/score corrections with only three shared scalars.
    for exponent,char,tok in product((-.20,-.10,0.,.10,.20),(-.015,0.,.015),(-.10,0.,.10)):
        setting=(exponent,char,tok)
        v,_=apply(LengthCalibration(*setting),ref)
        score=metrics(v); search['scale/'+str(setting)]=score
        value=utility(score)
        if value>best[0]:
            best=(value,dict(kind='analytic',coefficients=setting,validation=score,parameter_count=3),v)
    dump(out/'CALIBRATION_FREEZE.json',dict(chosen=best[1],validation_search=search))
    with (out/'VALIDATION_scoreonly.jsonl').open('w',encoding='utf-8') as f:
        for r in best[2]:
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print('score-only selected',best[1]['kind'],best[1].get('ratio',best[1].get('coefficients')),best[1]['validation'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','previous','archive'):
        p.add_argument(name,type=Path)
    a=p.parse_args();run(a.out,a.previous,a.archive)
