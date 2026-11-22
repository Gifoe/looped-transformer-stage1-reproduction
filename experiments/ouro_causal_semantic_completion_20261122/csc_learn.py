"""Train current-pass-only semantic writeback and matched low-rank baselines."""

import argparse
from contextlib import contextmanager
from pathlib import Path
from statistics import median
import time

import torch
from torch import nn
import torch.nn.functional as F

from csc_bank import KINDS,LAYER,encode,fields,load_bank,metadata,whites
from csc_data import prompt,repair
from csc_oracle import RANKS,Write,last_prefix,stat,summary
from interchange import terminated
from native import Forward,load,output,pad,save,setup
from rule import MAX_NEW,text_results

ARCH=("linear","bilinear","gated")
ROLE=(slice(0,2),slice(2,4),slice(4,5),slice(5,7))
WIDTH=24


def fit_features(x):
    out=[]
    for s in ROLE:
        v=x[:,s].flatten(1).float()
        mean=v.mean(0)
        _,a,b=torch.linalg.svd(v-mean,full_matrices=False)
        out.append((mean,b[:WIDTH].contiguous(),(a[:WIDTH]/max(1,len(v)-1)**.5).clamp_min(.1)))
    return out


def compress(x,encoder):
    return torch.cat([((x[:,r].flatten(1).float()-p)@q.T)/scale for r,(p,q,scale) in zip(ROLE,encoder,strict=True)],-1)


class Operator(nn.Module):
    def __init__(self,name,rank,basis):
        super().__init__()
        self.name=name
        self.proj=nn.Linear(4*WIDTH if name!="bilinear" else 2*WIDTH,rank)
        if name!="linear":
            self.control=nn.Linear(4*WIDTH if name=="gated" else 2*WIDTH,rank)
            self.mix=nn.Linear(rank,rank)
        self.decoder=nn.Linear(rank,basis.shape[-1],bias=False)
        self.decoder.weight.data.copy_(basis[:rank].T)

    def latent(self,x):
        if self.name=="linear":
            return self.proj(x)
        if self.name=="bilinear":
            c,r=x.split(2*WIDTH,dim=-1)
            return self.mix(self.proj(c)*self.control(r))
        return self.mix(self.proj(x)*torch.sigmoid(self.control(x)))

    def forward(self,x):
        return self.decoder(self.latent(x))


def cap(x,limit):
    return x*(limit/x.norm(dim=-1,keepdim=True).clamp_min(limit))


def encoder_device(enc,device):
    return [tuple(v.to(device) for v in z) for z in enc]


def valid_sites(tok,case,p):
    w=whites(tok,{"tokens":p})
    return len(case["quad"][3])+len(w)-1


def proposed(x,sites,roles,enc,operator,gain,limit):
    data=torch.stack([encode(x[j],k,roles[j]) for j,k in enumerate(sites)])
    return cap(operator(compress(data,enc)),limit)*gain


@contextmanager
def earlier(f,sites,roles,enc,operator,gain,limit,branch,external=None):
    module=f.layers[LAYER]
    state={}

    def read(_,args):
        state["delta"]=external if external is not None else proposed(args[0],sites,roles,enc,operator,gain,limit)

    def write(_,args,x):
        row=torch.arange(len(sites),device=x.device)
        y=x.clone()
        y[row,sites]+=state["delta"]
        return y

    target=module.post_attention_layernorm_2 if branch=="mlp" else (
        module.input_layernorm_2 if branch=="attention" else module)
    a,b=module.register_forward_pre_hook(read),target.register_forward_hook(write)
    try:
        yield
    finally:
        a.remove()
        b.remove()


@torch.inference_mode()
def infer(f,tok,cases,operator=None,enc=None,gain=1.,limit=1.,depth=1,branch="residual"):
    active={i:[] for i in range(len(cases))}
    finished={}
    for _ in range(MAX_NEW):
        keys=list(active)
        if not keys:
            break
        for off in range(0,len(keys),24):
            j=keys[off:off+24]
            rows=[cases[i] for i in j]
            ids,at=pad([r["quad"][3]+active[i] for r,i in zip(rows,j,strict=True)])
            if operator is None:
                h=f.run(ids,depth)[-1]
            else:
                sites=[valid_sites(tok,r,active[i]) for r,i in zip(rows,j,strict=True)]
                roles=[r["roles"] for r in rows]
                start=f.run(ids,1)[0] if depth==2 else None
                delta=proposed(start,sites,roles,enc,operator,gain,limit) if start is not None else None
                with earlier(f,sites,roles,enc,operator,gain,limit,branch,delta):
                    h=f.run(ids,1,start=start)[0]
            values=output(f.model,h,at).argmax(-1).tolist()
            for i,t in zip(j,values,strict=True):
                active[i].append(t)
                if terminated(tok,active[i]):
                    finished[i]=active.pop(i)
    finished.update(active)
    return text_results(tok,[finished[i] for i in range(len(cases))])


def load_operator(path,device):
    x=torch.load(path,map_location="cpu",weights_only=True)
    shape=x["state"]["decoder.weight"].T
    op=Operator(x["architecture"],x["rank"],shape)
    op.load_state_dict(x["state"])
    return op.to(device).eval(),encoder_device(x["encoder"],device),x


def deploy(f,tok,prompts,state_path):
    op,enc,settings=load_operator(state_path,"cuda")
    examples=[]
    for text in prompts:
        ids,roles=fields(tok,text)
        examples.append(dict(quad=[None,None,None,ids],roles=roles))
    return infer(f,tok,examples,op,enc,settings["gain"],settings["limit"],settings["depth"],settings["branch"])


def samples(bank,rows,slot,depth):
    x=[]
    for row in rows:
        for p in row["prefixes"]:
            z=bank[row["id"],row["kind"],p]
            beneficial=row["repair"]
            causal=z["updates"][slot] if beneficial else torch.zeros_like(z["updates"][slot])
            x.append((row,p,z["features"][depth-1].float(),causal,z["base"][1]-z["base"][0]))
    return x


def fit_map(name,rank,basis,enc,rows,mode,device):
    torch.manual_seed(261134369+rank)
    op=Operator(name,rank,basis).to(device)
    x=compress(torch.stack([a[2] for a in rows]).to(device),enc)
    y=torch.stack([a[3 if mode=="causal" else 4] for a in rows]).to(device)
    latent=y@basis[:rank].to(device).T
    best=float(y.norm(dim=-1).median().clamp_min(1.0))
    optimizer=torch.optim.AdamW(op.parameters(),lr=.002,weight_decay=.02)
    rng=torch.Generator().manual_seed(261134379)
    peers={}
    first={}
    for i,(r,p,*_) in enumerate(rows):
        if r["repair"]:
            key=(r["form"],r["target"],len(p))
            if key not in first:
                first[key]=i
            elif rows[first[key]][0]["id"]!=r["id"]:
                peers[i]=first[key]
                peers[first[key]]=i
    for _ in range(144):
        sample=torch.randint(len(rows),(24,),generator=rng).tolist()
        q=torch.tensor(sample,device=device)
        z=op.latent(x[q])
        predicted=op.decoder(z)
        loss=(predicted-y[q]).square().sum(-1).mean()/best**2
        if mode=="causal":
            loss=loss+.4*(z-latent[q]).square().sum(-1).mean()/best**2
            events=[(j,peers[i]) for j,i in enumerate(sample) if i in peers]
            if events:
                a,b=zip(*events,strict=True)
                p=F.cosine_similarity(predicted[list(a)],op(x[list(b)]),dim=-1)
                q=F.cosine_similarity(y[[sample[j] for j in a]],y[list(b)],dim=-1)
                loss=loss+.08*(p-q).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return op,opt_count(op)


def opt_count(op):
    return sum(x.numel() for x in op.parameters() if x.requires_grad)


def tune_actual(f,tok,op,enc,rows,depth,limit,mode,steps=56):
    usable=[]
    for row,p,x,v,_ in rows:
        y=row["native"][1]
        if y["number"]==row["target"] and tuple(y["tokens"][:len(p)])==p and len(y["tokens"])>len(p):
            if tok.decode([y["tokens"][len(p)]],skip_special_tokens=True).strip():
                usable.append((row,p,x,v,y["tokens"][len(p)]))
    device=next(op.parameters()).device
    rng=torch.Generator().manual_seed(261134389)
    optimizer=torch.optim.AdamW(op.parameters(),lr=.001,weight_decay=.02)
    for _ in range(steps):
        b=[usable[i] for i in torch.randint(len(usable),(6,),generator=rng).tolist()]
        x=compress(torch.stack([r[2] for r in b]).to(device),enc)
        target=torch.tensor([r[4] for r in b],device=device)
        ids,at=pad([r[0]["quad"][3]+list(r[1]) for r in b])
        delta=cap(op(x),limit)
        start=f.run(ids,1)[0] if depth==2 else None
        with Write(f,LAYER,at,delta):
            h=f.run(ids,1,start=start)[0]
        loss=F.cross_entropy(output(f.model,h,at),target)
        if mode=="causal":
            y=torch.stack([r[3] for r in b]).to(device)
            loss=loss+(delta-y).square().sum(-1).mean()/limit**2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(op.parameters(),1.)
        optimizer.step()


def calculate(rows,outputs,path=1):
    result=[]
    for i,r in enumerate(rows):
        native=r["native"][path-1]
        result.append(dict({k:v for k,v in r.items() if k not in ("quad","roles","prefixes")},
                           native=[native,r["native"][1]],pred={k:out[i] for k,out in outputs.items()}))
    return result


@torch.inference_mode()
def predict_model(f,tok,dev,entry,depth,limit):
    op,enc=entry
    predictions={}
    for gain,branch in ((.5,"residual"),(1.,"residual"),(1.,"mlp"),(1.,"attention")):
        name=f"{gain}/{branch}"
        predictions[name]=infer(f,tok,dev,op,enc,gain,limit,depth,branch)
    best=max(predictions,key=lambda k:sum(p["number"]==r["target"] for p,r in zip(predictions[k],dev,strict=True)))
    return (float(best.split("/")[0]),best.split("/")[1]),predictions


def latency(f,tok,examples,op,enc,gain,limit,depth,branch):
    times=[]
    for case in examples:
        torch.cuda.synchronize()
        begin=time.perf_counter()
        infer(f,tok,[case],op,enc,gain,limit,depth,branch)
        torch.cuda.synchronize()
        times.append(1000*(time.perf_counter()-begin))
    return median(times)


def cpu_state(op,enc,arch,rank,gain,branch,limit,depth):
    return dict(state={k:v.detach().cpu() for k,v in op.state_dict().items()},
                encoder=[[x.cpu() for x in r] for r in enc],architecture=arch,rank=rank,
                gain=gain,branch=branch,limit=limit,depth=depth,parameter_count=opt_count(op))


def record_metric(outputs,cases):
    truth=[r["target"] for r in cases]
    choices=[int(a["number"]==t) for a,t in zip(outputs,truth,strict=True)]
    return sum(choices)


def accuracy_gate(raw,depth):
    sub=[r for r in raw if r["form"]=="borrow" and r["split"]=="test"]
    test=stat(sub,"causal","native")
    ce=stat(sub,"causal","answer_only")
    dist=stat(sub,"causal","state_distill")
    shift=stat([r for r in raw if r["form"]=="borrow" and r["split"]=="shift"],"causal","native")
    carry=stat([r for r in raw if r["form"]=="carry" and r["split"]=="test"],"causal","native")
    return dict(depth=depth,vs_native=test,vs_ce=ce,vs_distill=dist,shift=shift,carry=carry,
                passes=bool(test["interval"][0]>0 and ce["mean"]>0 and dist["mean"]>0 and shift["mean"]>=0 and carry["mean"]>=0))


def experiment(f,tok,cases,bank,decision,basis,depth,root):
    train=[r for r in cases if r["split"]=="train"]
    dev=[r for r in cases if r["split"]=="dev"]
    test=[r for r in cases if r["split"] in ("test","shift")]
    slot=decision["slot"]
    z=samples(bank,train,slot,depth)
    span=torch.stack([a[3] for a in z if a[0]["repair"]])
    limit=float(span.norm(dim=-1).quantile(.95).clamp_min(1.0))
    enc=encoder_device(fit_features(torch.stack([a[2] for a in z])),"cuda")
    scans=[]
    winner=None
    first_outputs={}
    for arch in ARCH:
        for rank in RANKS:
            op,count=fit_map(arch,rank,basis,enc,z,"causal","cuda")
            out=infer(f,tok,dev,op,enc,1.,limit,depth)
            valid=record_metric(out,dev)
            # New heads are always selected on development outputs.
            r=dict(arch=arch,rank=rank,parameters=count,correct=valid,n=len(dev))
            scans.append(r)
            first_outputs[f"{arch}/{rank}"]=out
            key=(valid,-count)
            if winner is None or key>winner[0]:
                winner=(key,op,arch,rank)
            print("CSC learned architecture",depth,arch,rank,valid,len(dev),flush=True)
    save(root/f"LEARNED_SCAN_K{depth}.json",dict(metrics=scans,outputs=first_outputs,dev_ids=[r["id"]+":"+r["kind"] for r in dev]))
    _,causal,arch,rank=winner
    control_state=torch.stack([a[4] for a in z])
    _,_,hidden_axis=torch.linalg.svd(control_state,full_matrices=False)
    distill,_=fit_map(arch,rank,hidden_axis,enc,z,"distillation","cuda")
    rng=torch.Generator().manual_seed(261134395)
    basis0=torch.linalg.qr(torch.randn((basis.shape[1],rank),generator=rng)).Q.T
    torch.manual_seed(261134397)
    ce=Operator(arch,rank,basis0).cuda()
    assert opt_count(ce)==opt_count(distill)==opt_count(causal)
    tune_actual(f,tok,causal,enc,z,depth,limit,"causal")
    tune_actual(f,tok,ce,enc,z,depth,limit,"answer_only")
    engines={"causal":causal,"answer_only":ce,"state_distill":distill}
    settings,dev_out={},{}
    for k,op in engines.items():
        settings[k],dev_out[k]=predict_model(f,tok,dev,(op,enc),depth,limit)
        print("CSC learned select",depth,k,settings[k],flush=True)
        scale,site=settings[k]
        torch.save(cpu_state(op,enc,arch,rank,scale,site,limit,depth),root/f"operator_{k}_K{depth}.pt")
    raw_dev=dict(settings=settings,variants=dev_out)
    save(root/f"LEARNED_DEV_K{depth}.json",raw_dev)
    outs={"native":infer(f,tok,test,depth=depth)}
    expected=[r["native"][depth-1] for r in test]
    assert outs["native"]==expected
    for k,op in engines.items():
        scale,site=settings[k]
        outs[k]=infer(f,tok,test,op,enc,scale,limit,depth,site)
    raw=calculate(test,outs,depth)
    path=root/f"LEARNED_RAW_K{depth}.json"
    save(path,dict(rows=raw,arch=arch,rank=rank,parameters=opt_count(causal),settings=settings))
    lat={}
    smoke=test[:12]
    infer(f,tok,smoke[:1],depth=depth)
    lat["native"]=latency(f,tok,smoke,None,None,1,limit,depth,"residual")
    for k,op in engines.items():
        scale,site=settings[k]
        lat[k]=latency(f,tok,smoke,op,enc,scale,limit,depth,site)
    s=summary(raw)
    g=accuracy_gate(raw,depth)
    save(root/f"LEARNED_SUMMARY_K{depth}.json",dict(summary=s,gate=g,parameters=opt_count(causal),arch=arch,rank=rank,
          settings=settings,no_cache_batch1_median_ms=lat,latency_examples=[r["id"]+":"+r["kind"] for r in smoke]))
    print("CSC learned gate",depth,g,flush=True)
    return g


def run(model,root):
    z=load(root/"ORACLE_SUMMARY.json")
    assert z["gate"]["full"]
    bank=load_bank(root)
    m,tok=setup(model)
    f=Forward(m)
    cases=[dict(metadata(tok,r,k),repair=k=="joint" and repair(r)) for r in load(root/"NATIVE.json")["rows"] for k in KINDS]
    basis=torch.load(root/"ORACLE_BASIS.pt",map_location="cpu",weights_only=True)["basis"]
    for depth in (1,2):
        path=root/f"LEARNED_SUMMARY_K{depth}.json"
        if not path.exists():
            experiment(f,tok,cases,bank,z["lock"],basis,depth,root)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("result",type=Path)
    a=p.parse_args()
    run(a.model,a.result)
