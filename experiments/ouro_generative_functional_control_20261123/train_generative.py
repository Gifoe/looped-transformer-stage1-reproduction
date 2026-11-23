"""Fit a single 4096-parameter controller with full-answer teacher forcing."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import generation
import minimal
from task_data import SEED, encode_target, load
from train import ranking


def pack_entries(archive, encoded):
    return archive.pack([(seq,size,idx,0) for idx,(_,seq,size) in enumerate(encoded)])


def next_logits(model,packed,depth,net=None):
    h,_=minimal.forward(model,packed,depth,net)
    return model.lm_head(h).float()


def normalized_nll(logits,packed,n):
    _,group,_,target=packed
    losses=F.cross_entropy(logits,target,reduction="none")
    totals=losses.new_zeros(n).index_add(0,group,losses)
    sizes=torch.bincount(group,minlength=n)
    return (totals/sizes).mean()


def check_protocol(archive,model,tok,net,rows,depths,out):
    device=next(model.parameters()).device
    receipts=[]
    with torch.no_grad():
        for record in rows:
            encoded=encode_target(tok,record)
            prefix,whole,n=encoded
            suffix=whole[-n:]
            for k in depths:
                packed=pack_entries(archive,[encoded])
                all_logits=next_logits(model,packed,k,net)
                for j in range(n):
                    forced_prefix=prefix+suffix[:j]
                    deployed,last,_=generation.pack_generation([prefix],[suffix[:j]],[0],0,device)
                    # Score the very same sequence and visible answer prefix, with the matching teacher mask.
                    teacher=pack_entries(archive,[(prefix,forced_prefix+[suffix[j]],j+1)])
                    if any(not torch.equal(a,b) for a,b in zip(deployed[:3],teacher[:3],strict=True)):
                        raise RuntimeError("Visible training/decode evidence differs")
                    earlier=next_logits(model,teacher,k,net)[-1]
                    actual=next_logits(model,deployed,k,net)[last[0]]
                    if not torch.equal(earlier,actual):
                        raise RuntimeError("On identical complete prefix, teacher and deployment logits differ")
                    score_diff=float((actual.log_softmax(-1)[suffix[j]]-all_logits[j].log_softmax(-1)[suffix[j]]).abs())
                    receipts.append(dict(id=record["sample_id"],depth=k,position=j,target=suffix[j],
                                         same_prefix_bitwise=True,full_teacher_target_logp_error=score_diff,
                                         full_teacher_top1_equal=bool(actual.argmax()==all_logits[j].argmax())))
    if max(r["full_teacher_target_logp_error"] for r in receipts)>.25:
        raise RuntimeError("Teacher mask causality check exceeded preflight BF16 tolerance")
    minimal.dump(out,dict(protocol="all visible causal response predictor positions, native no cache",
                          model_tokenizer_eos=tok.eos_token_id,receipts=receipts))
    print("PARITY",len(receipts),max(r["full_teacher_target_logp_error"] for r in receipts),flush=True)


def plan(rows,encoded,steps,batch,seed,preserve_order):
    order=np.arange(len(rows))
    rng=np.random.default_rng(seed)
    full=[]
    while len(full)<steps*batch:
        epoch=rng.permutation(order)
        if not preserve_order:
            for i in range(0,len(epoch),192):
                epoch[i:i+192]=sorted(epoch[i:i+192],key=lambda j:len(encoded[j][1]))
        full.extend(epoch.tolist())
    return [full[s:s+batch] for s in range(0,steps*batch,batch)]


def rank_loss(archive,model,tok,net,rows,k):
    items,info=archive.encode(tok,rows)
    packed=archive.pack(items)
    logits=next_logits(model,packed,k,net)
    lp=-F.cross_entropy(logits,packed[-1],reduction="none")
    score,_=ranking(lp,packed[1],info,list(range(len(info))))
    return score,packed[0].numel()*k


def save_partial(out,net,opt,step,history,lock_sha):
    p=out/"resume.pt"
    tmp=p.with_suffix(".partial")
    torch.save(dict(net=net.state_dict(),optimizer=opt.state_dict(),next_step=step+1,history=history,lock=lock_sha),tmp)
    tmp.replace(p)


def fit(a,archive,model,tok):
    rows=load(a.data)
    packed_rows=[encode_target(tok,r) for r in rows]
    if max(len(x[1]) for x in packed_rows)>a.max_length:
        raise ValueError("Prepare the sample pool with the agreed length cap before fitting")
    torch.manual_seed(a.seed)
    net=minimal.Minimal("linear",1,.45).cuda()
    if a.initial is not None:
        raw=torch.load(a.initial,map_location="cpu",weights_only=True)
        net.load_state_dict(raw["state"])
    params=list(net.parameters())
    assert all(p.dtype==torch.float32 for p in params) and sum(p.numel() for p in params)==4096
    optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=.03)
    chunks=plan(rows,packed_rows,a.steps,a.batch,a.seed,a.preserve_order)
    manifest=dict(data_sha256=minimal.sha(a.data),steps=a.steps,max_work=a.max_work,seed=a.seed,batch=a.batch,
        script_sha256=minimal.sha(Path(__file__)),architecture="2 x 2048, FP32 rank-one, BF16 base",
        learning_rate=a.lr,rank_weight=a.rank_weight,initial_sha256=minimal.sha(a.initial) if a.initial else None,
        shared_depths=a.depths,train_order="historical" if a.preserve_order else "shuffled length buckets",
        loss="per-sequence full token average with EOS; candidate auxiliary uses unchanged ranking CE")
    info=a.out/"TRAIN_LOCK.json"
    if info.exists() and json.loads(info.read_text())!=manifest:
        raise ValueError(f"Existing model lock differs {info}")
    minimal.dump(info,manifest)
    lock=minimal.sha(info)
    trace=[]
    start=0
    path=a.out/"resume.pt"
    if path.exists():
        state=torch.load(path,map_location="cpu",weights_only=True)
        if state["lock"]!=lock:
            raise ValueError("Training restart configuration changed")
        net.load_state_dict(state["net"])
        optimizer.load_state_dict(state["optimizer"])
        trace=state["history"]
        start=state["next_step"]
    if (a.out/"TRAIN.json").exists():
        return
    work=sum(r["backbone_padded_token_loops"] for r in trace)
    observed=0
    clock=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.enable_grad():
        for step,idx in enumerate(chunks[start:],start=start):
            k=a.depths[step%len(a.depths)]
            packed=pack_entries(archive,[packed_rows[j] for j in idx])
            logits=next_logits(model,packed,k,net)
            gen=normalized_nll(logits,packed,len(idx))
            g_value=float(gen.detach())
            optimizer.zero_grad(set_to_none=True)
            gen.backward()
            del logits
            objective2=None
            extra=0
            aux=[rows[j]["candidate"] for j in idx if "candidate" in rows[j]]
            if a.rank_weight and aux:
                objective2,extra=rank_loss(archive,model,tok,net,aux,k)
                (a.rank_weight*objective2).backward()
            grad=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(grad) or (step==0 and float(grad)<=0):
                raise RuntimeError(f"Unusable controller gradient at step {step}")
            optimizer.step()
            model_work=packed[0].numel()*k+extra
            work+=model_work
            observed+=1
            trace.append(dict(step=step+1,depth=k,generation_nll=g_value,ranking=float(objective2.detach()) if objective2 is not None else None,
                              target_tokens=packed[-1].numel(),backbone_padded_token_loops=model_work,
                              ids=[rows[j]["sample_id"] for j in idx],gradient_norm=float(grad),
                              gpu_seconds=time.perf_counter()-clock))
            if ((step+1)%50==0 or step==0):
                print(step+1,k,g_value,trace[-1]["ranking"],work,wtime(clock),flush=True)
            if (step+1)%100==0:
                save_partial(a.out,net,optimizer,step,trace,lock)
            if a.max_work and work>=a.max_work:
                break
    torch.cuda.synchronize()
    final=a.out/"controller.pt"
    minimal.save_net(final,net.eval(),dict(seed=a.seed,parameters=4096,source=a.data.name))
    minimal.dump(a.out/"TRAIN.json",dict(lock_sha256=lock,checkpoint_sha256=minimal.sha(final),
        steps=len(trace),steps_this_run=observed,padded_token_loops=work,supervised_tokens=sum(r["target_tokens"] for r in trace),
        unique_samples=len({i for r in trace for i in r["ids"]}),seconds=time.perf_counter()-clock,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),logs=trace))


def wtime(clock):
    return round(time.perf_counter()-clock,2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("archive","previous","model","data","out"):
        p.add_argument(f"--{key}",required=True,type=Path)
    p.add_argument("--phase",choices=("parity","train"),default="train")
    p.add_argument("--initial",type=Path)
    p.add_argument("--depths",type=int,nargs="+",default=[2,3,4])
    p.add_argument("--steps",type=int,default=282)
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--max-length",type=int,default=640)
    p.add_argument("--max-work",type=int)
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--rank-weight",type=float,default=.25)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--preserve-order",action="store_true")
    a=p.parse_args()
    archive,model,tok,env=minimal.runtime(a.archive,a.previous,a.model)
    model.eval().requires_grad_(False)
    a.out.mkdir(parents=True,exist_ok=True)
    minimal.dump(a.out/"RUNTIME.json",env)
    if a.phase=="parity":
        net=minimal.load_net(a.initial).requires_grad_(False)
        selected=load(a.data)[:3]
        check_protocol(archive,model,tok,net,selected,a.depths,a.out/"PARITY.json")
    else:
        fit(a,archive,model,tok)


if __name__=="__main__":
    main()
