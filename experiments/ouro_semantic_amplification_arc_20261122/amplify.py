"""State-conditioned residual amplification and complete-candidate ARC training."""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from arc_features import archive, atomic_save, sha

WIDTH, RANK = 2048, 8
SEED = 261122713
CHANNELS = ('attention_first','attention_last','mlp_first','mlp_last')
INPUTS = ('raw', *CHANNELS, 'secant')
METHODS = ('magnitude','direction','completion','shuffled')
DEPTHS = (1,2,3,4)


def rms(x):
    return x.square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt()


def writes(state,k):
    return sum(state[name][:,k-1].float() for name in CHANNELS)


def orthogonal(x, w, wn):
    axis = w/wn
    return x - (x*axis).mean(-1,keepdim=True)*axis


def limit(x, ceiling):
    return x * (ceiling/rms(x)).clamp_max(1.)


class Amplifier(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.feature = nn.ModuleDict({name:nn.Linear(WIDTH,RANK,bias=False) for name in INPUTS})
        self.mixer = nn.Sequential(nn.Linear(RANK*len(INPUTS)+7,32),nn.SiLU(),nn.Linear(32,RANK),nn.SiLU())
        self.gain = nn.Linear(RANK,1)
        nn.init.zeros_(self.gain.weight)
        nn.init.zeros_(self.gain.bias)
        if method!='magnitude':
            self.align = nn.Linear(RANK,WIDTH,bias=False)
            nn.init.zeros_(self.align.weight)

    def forward(self,state,k,dose=1.,component='full'):
        h = state['raw'][:,k-1].float()
        v = {a:state[a][:,k-1].float() for a in CHANNELS}
        v['raw'] = h
        v['secant'] = h-state['raw'][:,k-2].float() if k>1 else torch.zeros_like(h)
        w = sum(v[a] for a in CHANNELS)
        wr = rms(w)
        hr = rms(h)
        measures = [(rms(v[a])/hr).clamp_min(1e-5).log().clamp(-8,3) for a in (*CHANNELS,'secant')]
        measures.extend((torch.full_like(hr,(k-1)/3), torch.full_like(hr,float(k>1))))
        projected = [self.feature[a](v[a]/rms(v[a])) for a in INPUTS]
        latent = self.mixer(torch.cat(projected+measures,-1))
        gain = .5*self.gain(latent).tanh()
        shift = orthogonal(self.align(latent)*wr,w,wr) if self.method!='magnitude' else torch.zeros_like(w)
        shift = limit(shift,.5*wr)
        mag = gain * w if component!='direction_only' else torch.zeros_like(w)
        delta = mag if component=='magnitude_only' else mag+shift
        return (h+dose*delta).to(torch.bfloat16), (gain,shift,wr,w,hr)


class Data:
    def __init__(self,path):
        self.info = json.loads((path/'index.json').read_text())
        self.rows = self.info['rows']
        fields = defaultdict(list)
        labels, offsets = [], []
        pos = 0
        for start in range(0,len(self.rows),self.info['chunk_questions']):
            shard = torch.load(path/f'{start:04}.pt',map_location='cpu',weights_only=True)
            for record in shard:
                q,a = record['q'],record['a']
                offsets.append((q,a,pos))
                t = record['target']
                labels.append(t)
                pos+=len(t)
                for key,val in record['state'].items():
                    if key!='hidden':
                        fields[key].append(val)
        expected = [(q,a) for q,row in enumerate(self.rows) for a in range(len(row['labels']))]
        assert [(q,a) for q,a,_ in offsets]==expected
        self.table = {k:torch.cat(v) for k,v in fields.items()}
        self.target = torch.cat(labels)
        self.candidates = len(expected)
        self.offsets = [[] for _ in self.rows]
        for (q,a,s),end in zip(offsets,[*([r[2] for r in offsets][1:]),pos],strict=True):
            self.offsets[q].append((s,end))

    def batch(self,questions,depth,teacher=False):
        tokens, group, choices, options, gold = [],[],[],[],[]
        for i,q in enumerate(questions):
            r = self.rows[q]
            gold.append(r['gold'])
            options.append(len(r['labels']))
            for choice,(s,e) in enumerate(self.offsets[q]):
                tokens.extend(range(s,e))
                group.extend([len(choices)]*(e-s))
                choices.append(r['characters'][choice])
        at = torch.tensor(tokens)
        last = depth+int(teacher)
        state = {k:v[:,:last].index_select(0,at).cuda() for k,v in self.table.items()}
        return state,self.target[at].cuda(),torch.tensor(group,device='cuda'),choices,options,gold


def normalized(model, augmented):
    return model.model.norm(augmented)


def nll_tokens(model,augmented,target):
    logits = model.lm_head(normalized(model,augmented)).float()
    return -F.cross_entropy(logits,target,reduction='none')


def group_choices(scores, options, gold):
    matrix = scores.new_full((len(options),max(options)),-1e6)
    valid = torch.arange(max(options),device=scores.device)[None,:]<torch.tensor(options,device=scores.device)[:,None]
    matrix[valid]=scores
    return F.cross_entropy(matrix, torch.tensor(gold,device=scores.device))


def training_loss(net,model,state,target,group,chars,options,gold,k,teacher,aux_weight,rng):
    h,(gain,shift,wr,w,hr) = net(state,k)
    token = nll_tokens(model,h,target)
    scores = torch.zeros(len(chars),device=token.device).index_add_(0,group,token)
    length = torch.tensor(chars,dtype=scores.dtype,device=scores.device)
    # All scores include the exact full-vocabulary partition. Character normalization
    # is used by the existing primary metric, with raw sums as a second objective.
    loss = .8*group_choices(16*scores/length,options,gold)+.2*group_choices(scores/3,options,gold)
    loss = loss + .01*(gain.square().mean() + (rms(shift)/wr).square().mean())
    if teacher:
        desired = limit(orthogonal(writes(state,k+1)-w,w,wr),.5*wr)
        if net.method=='shuffled':
            # Cyclically rotate target directions among tokens in the same depth.
            # Match target norm again after tangent projection in the recipient.
            owners = torch.arange(len(options),device=group.device).repeat_interleave(torch.tensor(options,device=group.device))[group]
            cross = 1+int(torch.randint(max(1,len(group)-1),(1,),generator=rng).item())
            donor = torch.arange(len(group),device=group.device).roll(cross)
            first = torch.cat((donor.new_zeros(1), torch.where(owners[1:]!=owners[:-1])[0]+1))
            donor = torch.where(owners==owners[donor],first[(owners+1)%len(options)],donor)
            random = orthogonal(desired[donor],w,wr)
            desired = random/rms(random)*rms(desired)
        loss = loss + aux_weight*((shift-desired)/wr).square().mean()
    return loss


@torch.no_grad()
def dev_scores(net,model,dataset,doses):
    full = {dose:[{} for _ in dataset.rows] for dose in doses}
    for k in DEPTHS:
        for lo in range(0,len(dataset.rows),32):
            hi = min(lo+32,len(dataset.rows))
            state,target,group,chars,options,_ = dataset.batch(list(range(lo,hi)),k)
            for dose in doses:
                x,_ = net(state,k,dose)
                v = nll_tokens(model,x,target).cpu().numpy()
                values = np.bincount(group.cpu().numpy(),weights=v,minlength=len(chars)).tolist()
                off = 0
                for i,n in enumerate(options):
                    full[dose][lo+i][f'k{k}'] = values[off:off+n]
                    off+=n
    return {dose:[dict(r,scores=s) for r,s in zip(dataset.rows,full[dose],strict=True)] for dose in doses}


def count(rows):
    return {metric:{f'k{k}':int(sum(r['gold']==np.argmax(np.array(r['scores'][f'k{k}'])/(
                        r['characters'] if metric=='acc_norm' else 1)) for r in rows)) for k in DEPTHS}
            for metric in ('acc','acc_norm')}


def rank(score,base):
    norm = score['acc_norm']
    return sum(norm.values()),min(norm[k]-base[k] for k in norm),sum(score['acc'].values())


def fit(model,train,dev,method,settings,checkpoint,history,base,old):
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    teacher_rng = torch.Generator().manual_seed(SEED+7)
    net = Amplifier(method).cuda()
    opt = torch.optim.AdamW(net.parameters(),lr=settings['lr'],weight_decay=.03)
    best, winner = None,None
    for e in range(settings['epochs']):
        net.train()
        order = torch.randperm(len(train.rows),generator=gen).tolist()
        losses = []
        with torch.enable_grad():
            for b,lo in enumerate(range(0,len(order),12)):
                k = 1+(b+e)%4
                teacher = method in ('completion','shuffled') and k<4
                args = train.batch(order[lo:lo+12],k,teacher)
                loss = training_loss(net,model,*args,k,teacher,settings['aux'],teacher_rng)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step()
                losses.append(loss.detach())
        net.eval()
        v = dev_scores(net,model,dev,(1.,))[1.]
        score = count(v)
        record = dict(epoch=e+1, loss=float(torch.stack(losses).mean()), metrics=score)
        history.append(record)
        print('validation',method,settings,record,flush=True)
        value = rank(score,base)
        if best is None or value>best:
            best,winner = value,record
            state = {k:x.detach().cpu() for k,x in net.state_dict().items()}
            atomic_save(dict(architecture=method,settings=settings,state=state,selected_epoch=e+1),checkpoint)
        old.dump(checkpoint.with_suffix('.progress.json'),dict(settings=settings,epochs=history))
    return best,winner


def operator(path):
    obj = torch.load(path,map_location='cpu',weights_only=True)
    net = Amplifier(obj['architecture'])
    net.load_state_dict(obj['state'])
    return net.cuda().eval()


def digest_bundle(out,chosen):
    paths = [out/x['file'] for x in chosen.values()]
    paths.extend(Path(__file__).parent/f for f in ('arc_features.py','amplify.py','evaluate.py'))
    return {p.name:sha(p) for p in paths}


def main(model_path,features,out,archived):
    old = archive(archived)
    model,_,_ = old.load_runtime(model_path)
    model.requires_grad_(False)
    baseline = json.loads((archived/'results/validation_metrics.json').read_text())
    base = {f'k{k}':baseline['acc_norm'][f'native_k{k}']['correct'] for k in DEPTHS}
    train, dev = [Data(features/split) for split in ('train','validation')]
    assert (len(train.rows),len(dev.rows))==(1119,299)
    assert {r['id'] for r in train.rows}.isdisjoint(r['id'] for r in dev.rows)
    out.mkdir(exist_ok=True,parents=True)
    old.dump(out/'TRAIN_INPUT.json',dict(train=train.info['dataset_sha256'],validation=dev.info['dataset_sha256'],
                                        validation_baseline=sha(archived/'results/validation_metrics.json'),seed=SEED))
    selections = {}
    choices = ((.0015,.25),(.0005,.75))
    logs = {}
    for round_number,(lr,aux) in enumerate(choices,1):
        for method in METHODS:
            name = f'{method}_round{round_number}.pt'
            ckpt = out/name
            settings = dict(lr=lr,aux=aux if method in ('completion','shuffled') else 0.,epochs=3)
            history = []
            fit(model,train,dev,method,settings,ckpt,history,base,old)
            logs[name] = history
            old.dump(out/'VALIDATION_TRAINING.json',logs)
    # Common dose is calibrated across all four depths. Calibration only scales
    # the mechanism's existing update and never creates a new depth-specific head.
    tuning = {}
    for method in METHODS:
        best = None
        rounds = (selections['completion']['round'],) if method=='shuffled' else (1,2)
        for r in rounds:
            name = f'{method}_round{r}.pt'
            ckpt = out/name
            doses = (selections['completion']['dose'],) if method=='shuffled' else (.5,.75,1.)
            v = dev_scores(operator(ckpt),model,dev,doses)
            for dose,rows in v.items():
                s = count(rows)
                tuning[f'{method}/round{r}/dose{dose}'] = s
                metric = rank(s,base)
                e = torch.load(ckpt,map_location='cpu',weights_only=True)['selected_epoch']
                if best is None or metric>best:
                    best = metric
                    selections[method] = dict(file=name,dose=dose,round=r,selected_epoch=e,metrics=s)
                    path = out/f'VALIDATION_{method}.jsonl'
                    old.write_predictions(path,rows)
        print('chosen',method,selections[method],flush=True)
    old.dump(out/'VALIDATION_CALIBRATION.json',tuning)
    loopcd = {f'k{k}':max((key for key in baseline['acc_norm'] if key.startswith('loopcd_') and key.endswith(f'k{k}')),
                          key=lambda key:(baseline['acc_norm'][key]['correct'],baseline['acc'][key]['correct'],key))
              for k in DEPTHS[1:]}
    old.dump(out/'FREEZE.json',dict(chosen=selections,hashes=digest_bundle(out,selections),loopcd_validation=loopcd,
                                   criterion='max total validation acc_norm over K1–K4, then min gain, then total acc',
                                   matched_parameters={a:sum(p.numel() for p in operator(out/b['file']).parameters())
                                                       for a,b in selections.items()}))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for opt in ('model','features','out','archive'):
        p.add_argument(opt,type=Path)
    args=p.parse_args()
    main(args.model,args.features,args.out,args.archive)
