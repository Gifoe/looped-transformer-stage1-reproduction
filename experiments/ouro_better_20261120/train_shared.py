"""One matched epoch of shared Q/V rank-8 Ouro adaptation."""
from contextlib import contextmanager
import argparse
import json
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from prepare import dump, key, load, sha

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"ouro_arc_pareto_20261119"))
from run_arc import load_runtime, states, variants

DEPTHS=(1,2,3,4,6)
ARMS=("binding","ce","distill","dropout")
SEED=101
RANK=8
ALPHA=16
LAYERS=24


def load_cohort(path, source, manifest, split, label=True):
    selected={a["id"]:a for a in manifest["cohorts"][split]}
    rows=load(path,source,labels=label,ids=set(selected))
    assert len(rows)==len(selected)
    current={r["id"]:r for r in rows}
    for rid,r in current.items():
        assert r["full_key"]==selected[rid]["content_sha256"]
    return [current[rid] for rid in selected]


def order_for(row, purpose):
    n=len(row["choices"]["text"])
    indices=list(range(n))
    if purpose=="original":
        return indices
    if purpose=="train":
        shift=1+(int(key("training_rotation_seed101",row["id"]),16)%(n-1))
        return indices[shift:]+indices[:shift]
    if purpose=="eval":
        # Nonidentity assignment across the full permutation space, not only rotations.
        gen=random.Random(int(key("evaluation_permutation_seed101",row["id"]),16))
        while True:
            trial=gen.sample(indices,n)
            if trial!=indices:
                return trial
    raise ValueError(purpose)


def make_view(tok, row, purpose, assigned_order=None):
    order=order_for(row,purpose) if assigned_order is None else assigned_order
    choices=row["choices"]["text"]
    prompt=f'Question: {row["question"]}'+"".join(
        f"\n{chr(65+i)}. {choices[c]}" for i,c in enumerate(order))+"\nAnswer:"
    ids=tok.encode(prompt,add_special_tokens=False)
    display=[]
    for j in range(len(choices)):
        both=tok.encode(prompt+" "+chr(65+j),add_special_tokens=False)
        assert len(both)==len(ids)+1 and both[:-1]==ids, (row["id"],purpose)
        display.append(both[-1])
    assert len(set(display))==len(display)
    by_content=[display[order.index(i)] for i in range(len(order))]
    v={"id":row["id"], "ids":ids,"order":order,"prompt":prompt, "display":display, "content":by_content}
    if "gold" in row:
        v["gold_content"]=row["gold"]
        v["gold_token"]=by_content[row["gold"]]
    return v


def batch(views):
    width=max(len(v["ids"]) for v in views)
    option_count=max(len(v["content"]) for v in views)
    inp=torch.tensor([v["ids"]+[0]*(width-len(v["ids"])) for v in views],device="cuda")
    end=torch.tensor([len(v["ids"])-1 for v in views],device="cuda")
    take=torch.tensor([v["content"]+[0]*(option_count-len(v["content"])) for v in views],device="cuda")
    valid=torch.tensor([[True]*len(v["content"])+[False]*(option_count-len(v["content"])) for v in views],device="cuda")
    rows=torch.arange(len(views),device="cuda")
    return inp,rows,end,take,valid


def readouts(model, views, max_budget=6):
    inp,rows,end,take,valid=batch(views)
    hidden,_,_=states(model,inp,max_budget,capture_raw=False)
    logits=torch.stack([model.lm_head(hidden[k-1][rows,end]).float() for k in DEPTHS])
    cand=logits.gather(-1,take.unsqueeze(0).expand(logits.shape[0],-1,-1))
    return logits,cand.masked_fill(~valid.unsqueeze(0),-1e9),int(inp.numel())


class SharedRank8(nn.Module):
    def __init__(self, dense):
        super().__init__()
        self.dense=dense
        self.a=nn.Parameter(torch.empty(RANK,dense.in_features,device=dense.weight.device,dtype=torch.float32))
        self.b=nn.Parameter(torch.zeros(dense.out_features,RANK,device=dense.weight.device,dtype=torch.float32))
        self.gate=None
        nn.init.kaiming_uniform_(self.a,a=5**.5)

    def forward(self,x):
        delta=F.linear(F.linear(x,self.a.to(x.dtype)),self.b.to(x.dtype))
        if self.gate is not None:
            delta=delta*self.gate
        return self.dense(x)+delta*(ALPHA/RANK)


class Adapter:
    def __init__(self,model):
        self.model=model
        self.targets={}
        self.attentions=[]
        self.gates=None
        for index,layer in enumerate(model.model.layers):
            attn=layer.self_attn
            original_forward=attn.forward
            self.attentions.append((attn,original_forward))
            for attr in ("q_proj","v_proj"):
                wrapper=SharedRank8(getattr(attn,attr))
                self.targets[(index,attr)]=(attn,wrapper)
                setattr(attn,attr,wrapper)
            def call(*args, _attn=attn, _forward=original_forward, **kwargs):
                gate=None if self.gates is None else self.gates[kwargs["current_ut"],:,None,None]
                _attn.q_proj.gate=gate
                _attn.v_proj.gate=gate
                return _forward(*args,**kwargs)
            attn.forward=call

    def parameters(self):
        for attn,w in self.targets.values():
            yield w.a
            yield w.b

    def checkpoint(self,path):
        d={}
        for (i,an),(attn,w) in self.targets.items():
            d[f"{i}.{an}.a"]=w.a.detach().cpu().contiguous()
            d[f"{i}.{an}.b"]=w.b.detach().cpu().contiguous()
        save_file(d,str(path))

    def cleanup(self):
        for (_,attr),(attn,w) in self.targets.items():
            setattr(attn,attr,w.dense)
        for attn,original_forward in self.attentions:
            attn.forward=original_forward


@contextmanager
def merged(model,checkpoint):
    state=load_file(str(checkpoint),device="cpu")
    saved=[]
    try:
        with torch.no_grad():
            for i,layer in enumerate(model.model.layers):
                for attr in ("q_proj","v_proj"):
                    w=getattr(layer.self_attn,attr).weight
                    saved.append((w,w.clone()))
                    a,b=(state[f"{i}.{attr}.{factor}"].to(w.device) for factor in "ab")
                    w.copy_((w.float()+(b@a)*(ALPHA/RANK)).to(torch.bfloat16))
        yield
    finally:
        with torch.no_grad():
            for w,orig in saved:
                w.copy_(orig)


def joint_loss(full, candidates, views, arm, teacher):
    gold=torch.tensor([v["gold_token"] for v in views],device="cuda")
    a,b,vocab=full.shape
    ce=F.cross_entropy(full.reshape(a*b,vocab),gold.repeat(a))
    other=ce.new_zeros(())
    if arm=="binding":
        logs=candidates.log_softmax(-1).reshape(a,b//2,2,-1)
        mixture=torch.logsumexp(logs,dim=2,keepdim=True)-0.6931471805599453
        other=(logs.exp()*(logs-mixture)).sum(dim=-1).mean()
    elif arm=="distill":
        temperature=2.
        t=teacher.unsqueeze(0).expand(a-1,-1,-1) / temperature
        student=candidates[:-1]/temperature
        teacher_lp=t.log_softmax(-1)
        other=(teacher_lp.exp()*(teacher_lp-student.log_softmax(-1))).sum(-1).mean() * temperature**2
    return ce+other,ce,other


def teacher_cache(model, views, output):
    by_id={}
    for start in range(0,len(views),16):
        selected=views[start:start+16]
        _,scores,_=readouts(model,selected)
        for v,score in zip(selected,scores[-1].cpu()):
            by_id.setdefault(v["id"],[]).append(score[:len(v["content"])].tolist())
    assert all(len(v)==2 for v in by_id.values())
    dump(output,by_id)
    return by_id


def teacher_batch(teacher, views):
    rows=[]
    width=max(len(v["content"]) for v in views)
    for i,v in enumerate(views):
        x=teacher[v["id"]][i%2]
        assert len(v["content"])==len(x)
        rows.append(x+[-1e9]*(width-len(x)))
    return torch.tensor(rows,device="cuda")


def fit_arm(model, views, order, arm, teacher, output, microbatch):
    torch.manual_seed(SEED)
    mask_rng=torch.Generator(device="cuda").manual_seed(SEED)
    adapter=Adapter(model)
    params=list(adapter.parameters())
    assert len(params)==4*LAYERS and sum(p.numel() for p in params)==1_572_864
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=0)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.train()
    ledger=[]
    total_tokens=0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for step in range(len(order)//4):
        opt.zero_grad(set_to_none=True)
        questions=order[4*step:4*step+4]
        tally=[0.,0.,0.]
        gates=[]
        for chunk in range(0,len(questions),microbatch):
            subset=questions[chunk:chunk+microbatch]
            b=[views[2*q+j] for q in subset for j in (0,1)]
            active=None
            if arm=="dropout":
                active=(torch.rand((6,len(b)),device="cuda",generator=mask_rng)>.5).to(torch.bfloat16)*2.
                gates.append(active.cpu().int().tolist())
            adapter.gates=active
            with torch.enable_grad():
                z,cand,tokens=readouts(model,b)
                labels=teacher_batch(teacher,b) if arm=="distill" else None
                loss,ce,other=joint_loss(z,cand,b,arm,labels)
                assert torch.isfinite(loss)
                (loss*(len(subset)/len(questions))).backward()
            adapter.gates=None
            total_tokens+=tokens
            for i,v in enumerate((loss,ce,other)):
                tally[i]+=float(v.detach())*(len(subset)/len(questions))
        grad=nn.utils.clip_grad_norm_(params,1.)
        assert torch.isfinite(grad)
        if step==0:
            assert float(grad)>0
        opt.step()
        rec={"update":step+1,"loss":tally[0],"ce":tally[1],"regularizer":tally[2],
             "grad_norm":float(grad),"gates":gates}
        ledger.append(rec)
        if (step+1)%16==0 or step==0:
            print(arm, step+1, f"loss {tally[0]:.5f}",f"{time.perf_counter()-start:.1f} s",flush=True)
    torch.cuda.synchronize()
    path=output/f"adapter.{arm}.safetensors"
    adapter.checkpoint(path)
    info={"arm":arm,"train_items":len(order),"updates":len(ledger),"prompt_views":2*len(order),
          "full_physical_layer_calls_per_prompt":6*24, "padded_input_tokens":total_tokens,
          "trainable_parameters":sum(p.numel() for p in params),"seconds":time.perf_counter()-start,
          "peak_vram_bytes":torch.cuda.max_memory_allocated(),"checkpoint_sha256":sha(path),
          "optimizer":{"name":"AdamW","lr":1e-4,"weight_decay":0.,"clip":1.,"seed":SEED},
          "shape":{"depths":DEPTHS,"rank":RANK,"alpha":ALPHA,"microbatch":microbatch,"effective_pairs":4},
          "training_history":ledger}
    dump(output/f"train.{arm}.json",info)
    adapter.cleanup()
    model.gradient_checkpointing_disable()
    model.eval()
    torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser()
    for k in ("model","arc","manifest","out","freeze"):
        p.add_argument(k,type=Path)
    p.add_argument("--microbatch",type=int,choices=(1,2,4),default=4)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    freeze=json.loads(args.freeze.read_bytes())
    for name,digest in freeze["code_hashes"].items():
        assert sha(Path(__file__).with_name(name))==digest
    assert freeze["split_sha256"]==sha(args.manifest)
    assert freeze["microbatch"]==args.microbatch
    manifest=json.loads(args.manifest.read_bytes())
    assert manifest["source"]["arc"]["sha256"]==sha(args.arc)
    rows=load_cohort(args.arc,"arc",manifest,"fit")
    assert len(rows)==512
    model, tok, runtime=load_runtime(args.model)
    model.requires_grad_(False)
    views=[make_view(tok,r,k) for r in rows for k in ("original","train")]
    order=list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    dump(args.out/"TRAIN_EXECUTION.json",{"runtime":runtime,"item_order":[rows[i]["id"] for i in order],
                                         "freeze_hash":sha(args.freeze),"maximum_prompt_tokens":max(len(v["ids"]) for v in views)})
    teacher_path=args.out/"frozen_teacher_k6_candidates.json"
    if teacher_path.exists():
        teacher=json.loads(teacher_path.read_bytes())
    else:
        teacher=teacher_cache(model,views,teacher_path)
    assert set(teacher)=={q["id"] for q in rows}
    for arm in ARMS:
        path=args.out/f"train.{arm}.json"
        if path.exists():
            assert json.loads(path.read_bytes())["checkpoint_sha256"]==sha(args.out/f"adapter.{arm}.safetensors")
            print("reusing",arm,flush=True)
            continue
        fit_arm(model,views,order,arm,teacher,args.out,args.microbatch)
    print("TRAIN_COMPLETE",flush=True)


if __name__=="__main__":
    main()

