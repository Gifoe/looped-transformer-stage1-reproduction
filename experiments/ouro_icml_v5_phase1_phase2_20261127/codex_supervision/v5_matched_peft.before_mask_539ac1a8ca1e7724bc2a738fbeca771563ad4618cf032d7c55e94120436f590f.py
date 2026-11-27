"""Strict V5 1C matched PEFT matrix with pinned sources and held-out S1 selection."""
import argparse
import bisect
import contextlib
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import minimal
import p1_gpu0_theory_and_peft as parts
import v4_t5_locked as t5
from v4_common import (
    BASE_SHA, FastUnifiedDevEvaluator, cloze_option_logits_differentiable,
    load_base_runtime, load_jsonl, make_candidate_row, save_json, score_cloze_probs, sha256_file,
)

SEEDS=(26112711,26112712)
STEPS=840
SOURCES=("S1","S2","S3")
METHODS=("M0_InLoop_Rank1","M2_Learned_Constant","M3_SoftPrompt","M4_PostLoop_LMHead_Rank1",
         "M5a_IA3_Last4","M5b_IA3_All","M6a_LoRA_LastO","M6b_LoRA_AllQV")
STATIC=("M1a_CAA","M1b_RepE")
ALPHAS=(.05,.10,.20,.30,.45)
RATES={name:1e-3 for name in METHODS}
RATES.update(M3_SoftPrompt=3e-3,M5a_IA3_Last4=1e-3,M5b_IA3_All=1e-3,M6b_LoRA_AllQV=3e-4)


class IA3ProjectionOutput(parts.HookAdapterBase):
    """Projection-output IA3, including down-projection output per the V5 specification."""
    def __init__(self,model,indices):
        super().__init__()
        self.scales=nn.ParameterDict()
        for i in indices:
            layer=model.model.layers[i]
            for tag, projection in (("k",layer.self_attn.k_proj),("v",layer.self_attn.v_proj),
                                    ("down",layer.mlp.down_proj)):
                self.scales[f"{i}_{tag}"]=nn.Parameter(torch.ones(projection.out_features))

    @contextlib.contextmanager
    def attach(self,model):
        def for_scale(p):
            return lambda _, __, out: (out.float()*p).to(out.dtype)
        handles=[]
        try:
            for key,p in self.scales.items():
                idx,tag=key.split("_")
                layer=model.model.layers[int(idx)]
                target={"k":layer.self_attn.k_proj,"v":layer.self_attn.v_proj,"down":layer.mlp.down_proj}[tag]
                handles.append(target.register_forward_hook(for_scale(p)))
            yield
        finally:
            for handle in handles:
                handle.remove()


def build(method,model,tokenizer,seed):
    torch.manual_seed(seed)
    width=model.config.hidden_size
    layers=list(range(len(model.model.layers)))
    if method==METHODS[0]:
        return minimal.Minimal("linear",1,.45)
    if method==METHODS[1]:
        return minimal.Minimal("constant",1,.45)
    if method==METHODS[2]:
        tok=tokenizer.encode("Question:",add_special_tokens=False)
        if len(tok)<2:
            raise ValueError("Two embedding initializers are required")
        with torch.no_grad():
            initial=model.model.embed_tokens(torch.tensor(tok[:2],device=model.device)).cpu()
        return parts.SoftPromptAdapter(initial)
    if method==METHODS[3]:
        return parts.PostLoopLMHeadRank1(width,.45)
    if method in METHODS[4:6]:
        return IA3ProjectionOutput(model,layers[-4:] if method==METHODS[4] else layers)
    all_mod=dict(model.named_modules())
    paths=[f"model.layers.{layers[-1]}.self_attn.o_proj"] if method==METHODS[6] else [
        f"model.layers.{layer}.self_attn.{name}_proj" for layer in layers for name in ("q","v")]
    dims=[(path, all_mod[path].in_features, all_mod[path].out_features) for path in paths]
    return parts.LoRAAdapter(dims,rank=1,lora_alpha=4.)


def read(path):
    return json.loads(path.read_text())


def digest(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def source_plan(cfg,root):
    original=t5.original.resolve_pool_path
    t5root=Path(t5.__file__).resolve().parents[1]
    arc=Path(cfg["pools"]["arc_challenge"]["path"])
    candidate=read(root/"track_b_and_theory/ARC_BANK_S1.json")
    expected=[t5.raw_item(row,"arc_challenge") for row in load_jsonl(arc)]
    expected=[row for row in expected if 2<=len(row["options"])<=5]
    if len(candidate)!=1013 or len(expected)!=len(candidate):
        raise ValueError("Expected the declared original 860/153 ARC teacher split")
    s1=[]
    for row,exp in zip(candidate,expected,strict=True):
        if any(row[key]!=exp[key] for key in ("id","question","options")):
            raise ValueError("The original native ARC teacher bank ordering differs")
        if row["pseudo_label"]!=max(range(len(exp["options"])),key=row["q_mc"].__getitem__):
            raise ValueError("Stored soft probabilities and teacher answer differ")
        s1.append({key:row[key] for key in ("id","question","options","pseudo_label","q_mc","conf")})
    s2,_=t5.setup_arm(t5root,read(t5root/"handoff_codex/T5_TRAINING_LOCK.json"),"T5_CFIT_Top6_4096")
    med=original("medmcqa")
    med_obs=t5root/"cfit_14_features/regime3_samples/medmcqa.jsonl"
    med_data={r["sample_id"]:r for r in load_jsonl(med)}
    med_records=load_jsonl(med_obs)
    random.Random("t3b_rand/medmcqa").shuffle(med_records)
    if len(med_records)<1024:
        raise ValueError("The complete V4 MedMCQA sampled cohort is unavailable")
    s3=[t5.raw_item(med_data[r["sample_id"]],"medmcqa") for r in med_records[:512]]
    if len({r["id"] for r in s3})!=512:
        raise ValueError("Expected exactly the archived 512 MedMCQA rows")
    source_sha={str(path):sha256_file(path) for path in (arc,med,med_obs,t5root/"handoff_codex/T5_TRAINING_LOCK.json")}
    plan=dict(S1=s1[:860],S2=s2,S3=s3)
    return plan,s1[860:],source_sha


def stream(source,bank,seed):
    if source=="S2":
        source_rng=random.Random(f"t5/T5_CFIT_Top6_4096/{seed}/source")
        item_rngs=[random.Random(f"t5/T5_CFIT_Top6_4096/{seed}/{d}") for d,_,_ in bank]
        arrays,ends=[],[]
        for _,weight,rows in bank:
            items,weights=zip(*rows,strict=True)
            cumulative=[]
            for w in weights:
                cumulative.append((cumulative[-1] if cumulative else 0.)+w)
            arrays.append((items,cumulative))
            ends.append((ends[-1] if ends else 0.)+weight)

        def draw():
            group=bisect.bisect_right(ends,source_rng.random()*ends[-1])
            items,cs=arrays[group]
            return items[bisect.bisect_right(cs,item_rngs[group].random()*cs[-1])]
    else:
        rng=random.Random(f"v5/phase1c/{source}/{seed}")
        indices=list(range(len(bank)))
        rng.shuffle(indices)
        at=0

        def draw():
            nonlocal at
            if at==len(indices):
                rng.shuffle(indices)
                at=0
            it=bank[indices[at]]
            at+=1
            return it
    return [draw() for _ in range(STEPS*4)]


def lock(root,cfg):
    directory=root/"track_b_baselines"
    directory.mkdir(parents=True,exist_ok=True)
    bank,heldout,source_sha=source_plan(cfg,root)
    streams={(source,seed):stream(source,bank[source],seed) for source in SOURCES for seed in SEEDS}
    target=directory/"PROTOCOL.json"
    definition=dict(
        phase="1C strict",model_sha256=BASE_SHA,runner_sha256=sha256_file(Path(__file__)),
        common_sha256=sha256_file(Path(parts.__file__)), data_source_sha256=source_sha,
        methods=list(METHODS+STATIC),curve_alpha=list(ALPHAS),rates=RATES,
        train_steps=STEPS,effective_batch=4,depths=[2,3,4],seeds=list(SEEDS),
        ordered_draws_sha256={f"{s}.{v}":digest([x["id"] for x in seq]) for (s,v),seq in streams.items()},
        s1_holdout_sha256=digest(heldout),s1_fit=len(bank["S1"]),s1_val=len(heldout),
        s2_full_source_sizes={source:len(rows) for source,_,rows in bank["S2"]},
        s3_random_512=digest([r["id"] for r in bank["S3"]]),
        official_test_used=False, external_development_used_for_model_selection=False,
        dose_and_final_selection="Same held-out ARC pseudolabel cohort; mean agreement across two frozen seeds; break ties by lower calibration NLL, then identifier.",
    )
    if target.exists():
        if read(target)!=definition:
            raise ValueError("The 1C frozen method contract changed")
    else:
        save_json(target,definition)
    return directory,streams,heldout


def batch_logits(archive,model,tokenizer,rows,depth,net):
    candidates=[make_candidate_row(r["id"],r["question"],r["options"],0) for r in rows]
    if isinstance(net,parts.HookAdapterBase):
        with net.attach(model):
            return cloze_option_logits_differentiable(archive,model,tokenizer,candidates,depth,None)[0]
    return cloze_option_logits_differentiable(archive,model,tokenizer,candidates,depth,net)[0]


def train(path,archive,model,tokenizer,samples,seed,method,source,contract):
    net=build(method,model,tokenizer,seed).to(model.device)
    opt=torch.optim.AdamW(net.parameters(),lr=RATES[method],weight_decay=.03)
    progress=path.with_suffix(".resume.pt")
    elapsed,start,loss=0.,0,[]
    if progress.exists():
        old=torch.load(progress,map_location="cpu",weights_only=True)
        if old["lock"]!=contract or old["method"]!=method or old["source"]!=source or old["seed"]!=seed:
            raise ValueError("Cannot use a checkpoint from a different arm")
        net.load_state_dict(old["state"])
        opt.load_state_dict(old["optimizer"])
        start,elapsed,loss=old["step"],old["seconds"],old["loss"]
    torch.cuda.reset_peak_memory_stats()
    begin=time.monotonic()
    key="pseudo_label" if source=="S1" else "gold"
    for n in range(start,STEPS):
        mb=samples[4*n:4*n+4]
        longest=max(len(r["question"])+max(map(len,r["options"])) for r in mb)
        maxopts=max(len(r["options"]) for r in mb)
        micro=1 if longest>1400 or longest*maxopts>5200 else (2 if longest>650 or longest*maxopts>2600 else 4)
        weights=[max(.05,r["conf"]) for r in mb] if source=="S1" else [1.]*4
        denom=sum(weights)
        opt.zero_grad(set_to_none=True)
        loss_n=0.
        with torch.enable_grad():
            for k in range(0,4,micro):
                sub=mb[k:k+micro]
                raw=batch_logits(archive,model,tokenizer,sub,(2,3,4)[n%3],net)
                obj=sum(-F.log_softmax(x,dim=-1)[r[key]] * w/denom
                        for x,r,w in zip(raw,sub,weights[k:k+micro],strict=True))
                obj.backward()
                loss_n+=float(obj.detach())
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        if not grad.isfinite().item():
            raise ValueError("Nonfinite adapter gradients")
        opt.step()
        loss=(loss+[loss_n])[-70:]
        if (n+1)%140==0:
            hold=progress.with_suffix(".tmp")
            torch.save(dict(lock=contract,method=method,source=source,seed=seed,step=n+1,loss=loss,
                            seconds=elapsed+time.monotonic()-begin,state=net.state_dict(),
                            optimizer=opt.state_dict()),hold)
            hold.replace(progress)
            print("V5_1C_TRAIN",source,method,seed,n+1,round(elapsed+time.monotonic()-begin,1),flush=True)
    seconds=elapsed+time.monotonic()-begin
    return net,dict(n_params=sum(p.numel() for p in net.parameters()),steps=STEPS,train_seconds=seconds,
                    mean_step_seconds=seconds/STEPS,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    stream_sha256=digest([r["id"] for r in samples]),loss_tail=sum(loss)/len(loss))


@torch.inference_mode()
def validate(archive,model,tokenizer,heldout,net):
    rows=[make_candidate_row(r["id"],r["question"],r["options"],0) for r in heldout]
    if isinstance(net,parts.HookAdapterBase):
        with net.attach(model):
            _,prob=score_cloze_probs(archive,model,tokenizer,rows,batch_size=8)
    else:
        _,prob=score_cloze_probs(archive,model,tokenizer,rows,net=net,batch_size=8)
    correct, logloss=0,0.
    agreements=[]
    for p,row in zip(prob,heldout,strict=True):
        ref=row["pseudo_label"]
        flag=int(max(range(len(p)),key=p.__getitem__)==ref)
        logloss-=__import__("math").log(max(p[ref],1e-9))
        correct+=flag
        agreements.append(flag)
    return dict(accuracy=correct/len(heldout),cross_entropy=logloss/len(heldout), flags=agreements)


def output_weight(path,state,meta):
    temporary=path.with_suffix(".pt.partial")
    torch.save(dict(state={k:v.detach().cpu() for k,v in state.items()},meta=meta),temporary)
    temporary.replace(path)
    return sha256_file(path)


def run(root,source=None,seed=None,methods=None):
    from v4_common import E0_CONFIG_PATH
    directory,streams,heldout=lock(root,read(E0_CONFIG_PATH))
    if source is None:
        print("V5_1C_PROTOCOL_LOCK",sha256_file(directory/"PROTOCOL.json"),flush=True)
        return
    cfg,archive,model,tokenizer=load_base_runtime()
    scorer=FastUnifiedDevEvaluator(cfg,archive,model,tokenizer)
    contract=sha256_file(directory/"PROTOCOL.json")
    cohort=streams[source,seed]
    view="pseudo_label" if source=="S1" else "gold"
    lat_rows=[make_candidate_row(r["id"],r["question"],r["options"],0) for r in heldout[:48]]
    for method in methods or METHODS+STATIC:
        name=f"{source}.{method}.s{seed}"
        path=directory/f"{name}.pt"
        receipt=directory/f"{name}.json"
        if receipt.exists():
            keep=read(receipt)
            if keep["protocol_sha256"] != contract or sha256_file(path)!=keep["weights_sha256"]:
                raise ValueError("The completed adapter or its receipt changed")
            continue
        if method in STATIC:
            uniq=list({row["id"]:row for row in reversed(cohort)}.values())[::-1]
            first=time.monotonic()
            vectors=parts.fit_caa_repe_vectors(model,tokenizer,uniq,target_key=view,max_items=256)
            direction=vectors[STATIC.index(method)]
            sweep={}
            for alpha in ALPHAS:
                mod=parts.StaticSteeringController(direction,alpha).to(model.device)
                sweep[alpha]=validate(archive,model,tokenizer,heldout,mod)
            alpha=max(ALPHAS,key=lambda a:(sweep[a]["accuracy"],-sweep[a]["cross_entropy"],-a))
            net=parts.StaticSteeringController(direction,alpha).to(model.device)
            training=dict(n_params=direction.numel(),trainable_n=0,steps=0,fit_seconds=time.monotonic()-first,
                          alpha=alpha,alpha_sweep=sweep,match_fit_limit=256)
            selection=sweep[alpha]
        else:
            net,training=train(path,archive,model,tokenizer,cohort,seed,method,source,contract)
            selection=validate(archive,model,tokenizer,heldout,net)
        score,unlab,evaluation_seconds=parts.evaluate_any(scorer,model,tokenizer,net,include_mmlu_dev158=True)
        latency=parts.measure_cloze_latency_ms(archive,model,tokenizer,lat_rows,net)
        detail=dict(method=method,source=source,seed=seed,protocol_sha256=contract)
        sha=output_weight(path,net.state_dict(),{**detail,"training":training})
        save_json(receipt,dict(**detail,weights_sha256=sha,training=training,selection=selection,
                               development=score,unlabeled=unlab,latency_ms_per_question=latency,
                               evaluation_seconds=evaluation_seconds))
        print("V5_1C_ARM_DONE",name,selection["accuracy"],score["macro5"],flush=True)
        del net
        torch.cuda.empty_cache()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--source",choices=SOURCES)
    parser.add_argument("--seed",choices=SEEDS,type=int)
    parser.add_argument("--methods",choices=METHODS+STATIC,nargs="+")
    args=parser.parse_args()
    if (args.source is None)!=(args.seed is None):
        parser.error("Provide both --source and --seed to execute a worker")
    torch.set_num_threads(4)
    run(args.root,args.source,args.seed,args.methods)

