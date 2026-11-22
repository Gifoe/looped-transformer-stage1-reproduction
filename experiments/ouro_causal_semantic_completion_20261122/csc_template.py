"""Test whether a learned current-state selector can use a shared rank-one causal update."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from csc_bank import KINDS, load_bank, metadata
from csc_data import repair
from csc_learn import WIDTH, accuracy_gate, calculate, compress, encoder_device, fit_features, infer, latency, samples
from csc_oracle import decode, stat, summary
from native import Forward, load, save, setup


class Template(nn.Module):
    def __init__(self, direction, constant=None):
        super().__init__()
        self.left = nn.Linear(WIDTH, 16)
        self.right = nn.Linear(WIDTH, 16)
        self.read = nn.Linear(2*WIDTH+16,1)
        self.register_buffer("direction", direction)
        self.constant = constant

    def logit(self, x):
        a,b,rule,ctx = x.split(WIDTH,dim=-1)
        interaction = self.left(a).tanh()*self.right(b).tanh()
        return self.read(torch.cat((interaction,rule,ctx),dim=-1))

    def forward(self, x):
        strength = self.logit(x).sigmoid() if self.constant is None else self.constant
        return strength*self.direction.unsqueeze(0).expand(len(x),-1)


def fit_gate(train, encoder, direction):
    rng = torch.Generator().manual_seed(261134487)
    torch.manual_seed(261134473)
    op = Template(direction).cuda()
    x = compress(torch.stack([a[2] for a in train]).cuda(),encoder)
    y = torch.tensor([float(a[0]["repair"]) for a in train],device=x.device)[:,None]
    positive = (len(y)-y.sum())/y.sum()
    opt = torch.optim.AdamW(op.parameters(),lr=.002,weight_decay=.05)
    for _ in range(250):
        take = torch.randint(len(x),(64,),generator=rng).cuda()
        loss = F.binary_cross_entropy_with_logits(op.logit(x[take]),y[take],pos_weight=positive)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return op.eval()


def teacher_directions(train, bank, slot):
    stable, actual = [],[]
    for row,p,_,v,_ in train:
        if row["repair"] and row["form"] == "borrow" and not p:
            actual.append(v.float())
            stable.append(bank[row["id"],"safe",()]["updates"][slot].float())
    positives = torch.stack(actual)
    level = float(positives.norm(dim=-1).median())
    targets = {"mean":positives.mean(0),"paired":(positives-torch.stack(stable)).mean(0)}
    return {a:x/x.norm().clamp_min(1e-9)*level for a,x in targets.items()}


def need_oracle(f, tok, cases, bank, op, gain, limit, chosen, slot):
    transformed = {}
    delta = op.direction.cpu()*gain*min(1.,limit/float(op.direction.norm()))
    for row in cases:
        for p in row["prefixes"]:
            key = row["id"],row["kind"],p
            cached = bank[key]
            z = torch.zeros_like(cached["updates"])
            if row["repair"]:
                z[slot] = delta
            transformed[key] = dict(base=cached["base"],updates=z)
    return [r for i in range(0,len(cases),20)
            for r in decode(f,tok,cases[i:i+20],transformed,(chosen,),slot=slot)]


def load_template(path):
    state = torch.load(path,map_location="cpu",weights_only=True)
    op = Template(state["weights"]["direction"])
    op.load_state_dict(state["weights"])
    return op.cuda().eval(),encoder_device(state["encoder"],"cuda"),state["settings"]


def selector_audit(op, encoder, bank, cases):
    x = compress(torch.stack([bank[r["id"],r["kind"],()]["features"][0] for r in cases]).cuda(),encoder)
    scores = op.logit(x).sigmoid().squeeze(-1).cpu()
    result = {}
    for split in ("test","shift"):
        pick = [i for i,r in enumerate(cases) if r["split"]==split and r["form"]=="borrow"]
        positive = torch.tensor([cases[i]["repair"] for i in pick])
        p = scores[pick]
        diff = p[positive,None]-p[None,~positive]
        a = (diff.gt(0).float()+.5*diff.eq(0).float()).mean()
        result[split] = dict(auc=float(a),positive=int(positive.sum()),negative=int((~positive).sum()),
                             median_positive=float(p[positive].median()),median_negative=float(p[~positive].median()))
    return result


@torch.inference_mode()
def deployment(f,tok,cases,op,encoder,s):
    return infer(f,tok,cases,op,encoder,s["gain"],s["limit"],1,"residual")


def run(model, root):
    z = load(root / "ORACLE_SUMMARY.json")
    assert z["gate"]["full"]
    chosen,slot = (z["lock"][k] for k in ("update","slot"))
    bank = load_bank(root)
    m,tok = setup(model)
    f = Forward(m)
    cases = [dict(metadata(tok,r,k),repair=k=="joint" and repair(r))
             for r in load(root / "NATIVE.json")["rows"] for k in KINDS]
    train = samples(bank,[r for r in cases if r["split"]=="train"],slot,1)
    encoder = encoder_device(fit_features(torch.stack([r[2] for r in train])),"cuda")
    dirs = teacher_directions(train,bank,slot)
    limit = float(torch.stack([r[3] for r in train if r[0]["repair"]]).norm(dim=-1).quantile(.95))
    op = fit_gate(train,encoder,dirs["mean"])
    dev = [r for r in cases if r["split"] == "dev"]
    values = {}
    for name,d in dirs.items():
        op.direction.copy_(d.cuda())
        for gain in (.5,1.,2.):
            s = dict(direction=name,gain=gain,depth=1,limit=limit,branch="residual")
            pred = deployment(f,tok,dev,op,encoder,s)
            n = sum(p["number"]==r["target"] for p,r in zip(pred,dev,strict=True))
            values[f"{name}/{gain}"] = dict(correct=n,settings=s,outputs=pred)
            print("CSC rank-one dev",name,gain,n,len(dev),flush=True)
    key = max(values,key=lambda a:values[a]["correct"])
    settings = values[key]["settings"]
    save(root / "TEMPLATE_DEV.json",dict(dev_ids=[r["id"]+":"+r["kind"] for r in dev],
                                        selection=key,choices=values))
    op.direction.copy_(dirs[settings["direction"]].cuda())
    counts = sum(x.numel() for x in op.parameters())
    torch.save(dict(weights={k:v.detach().cpu() for k,v in op.state_dict().items()},
                    encoder=[[x.cpu() for x in group] for group in encoder],settings=settings,parameters=counts),
               root / "operator_template_K1.pt")
    test = [r for r in cases if r["split"] in ("test","shift")]
    ref = load(root / "LEARNED_RAW_K1.json")["rows"]
    assert [(r["id"],r["kind"]) for r in ref] == [(r["id"],r["kind"]) for r in test]
    pred = {a:[r["pred"][a] for r in ref] for a in ref[0]["pred"]}
    pred["previous_causal"] = pred.pop("causal")
    pred["causal"] = deployment(f,tok,test,op,encoder,settings)
    with torch.inference_mode():
        inputs = compress(torch.stack([r[2] for r in train]).cuda(),encoder)
        const = float(op.logit(inputs).sigmoid().mean())
        op.constant = const
        pred["constant_gate"] = deployment(f,tok,test,op,encoder,settings)
        op.constant = None
        forced = need_oracle(f,tok,test,bank,op,settings["gain"],limit,chosen,slot)
        pred["oracle_gate"] = [r[chosen] for r in forced]
        selectors = selector_audit(op,encoder,bank,test)
    results = calculate(test,pred,1)
    save(root / "TEMPLATE_RAW.json",dict(rows=results,configuration=settings))
    k = accuracy_gate(results,1)
    ablation = [r for r in results if (r["form"],r["split"]) == ("borrow","test")]
    report = dict(summary=summary(results),parameters=counts,settings=settings,gate=k,constant_gate=const,selector=selectors,
                  versus_constant=stat(ablation,"causal","constant_gate"),
                  no_cache_batch1_median_ms=latency(f,tok,test[:12],op,encoder,settings["gain"],limit,1,"residual"))
    save(root / "TEMPLATE_SUMMARY.json",report)
    print("CSC rank-one gate",k,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model",type=Path)
    parser.add_argument("root",type=Path)
    args = parser.parse_args()
    run(args.model,args.root)
