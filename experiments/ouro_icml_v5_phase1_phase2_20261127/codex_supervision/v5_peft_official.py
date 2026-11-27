"""Freeze V5 Track B choices from S1 alone, then run the historical complete K4 scorer."""
import argparse
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import minimal
import official_v3 as historical_gate
from aggregate_v3 import summarize
from aggregate_full import closed, legacy_rows
from run_full import LEGACY, freeze
from v4_common import E0_CONFIG_PATH, save_json, sha256_file
from v5_matched_peft import METHODS, SEEDS, STATIC, build, parts, read


def pin_source(folder,source,method,seed):
    name=f"{source}.{method}.s{seed}"
    protocol=sha256_file(folder/"PROTOCOL.json")
    weights, receipt=folder/f"{name}.pt",folder/f"{name}.json"
    data=read(receipt)
    if (data["protocol_sha256"]!=protocol or sha256_file(weights)!=data["weights_sha256"]
            or (data["source"],data["method"],data["seed"])!=(source,method,seed)):
        raise ValueError(f"Invalid registered adapter {name}")
    return weights,receipt,data


def family(method):
    return method.split("_",1)[0][:2]


def seal(root):
    frame=root/"track_b_baselines"
    candidates={}
    provenance={}
    means={}
    options={}
    for method in METHODS+STATIC:
        pins=[pin_source(frame,"S1",method,seed) for seed in SEEDS]
        for _, p, payload in pins:
            provenance[p.name]=sha256_file(p)
        if family(method)=="M0":
            continue
        rates=-sum(p[2]["selection"]["cross_entropy"] for p in pins)/len(pins)
        accuracy=sum(p[2]["selection"]["accuracy"] for p in pins)/len(pins)
        means[method]={"agreement":accuracy,"cross_entropy":-rates}
        within=max(pins,key=lambda p:(p[2]["selection"]["accuracy"],-p[2]["selection"]["cross_entropy"],-p[2]["seed"]))
        options[method]=(accuracy,rates,within)
    best=[]
    used=set()
    for method in sorted(options,key=lambda m:(-options[m][0],-options[m][1],m)):
        key=family(method)
        if key not in used:
            used.add(key)
            best.append(method)
        if len(best)==2:
            break
    for method in best:
        w,p,data=options[method][2]
        node=dict(controller=str(w),controller_sha256=data["weights_sha256"],method=method,seed=data["seed"],
                  development=str(p),development_sha256=sha256_file(p),
                  training=str(p),training_sha256=sha256_file(p),historical=str(p),historical_sha256=sha256_file(p))
        candidates[method]=node
    spec=dict(status="FROZEN_BEFORE_OFFICIAL_EVALUATION",development_only=True,
              rule="S1 pseudo-label holdout only, two-seed mean agreement, lower cross entropy, each chosen from a different method family.",
              official_test_used=False,means=means,provenance=provenance,candidates=candidates,
              env_sha256=read(E0_CONFIG_PATH)["approved_env_sha256"],
              protocol_sha256=sha256_file(frame/"PROTOCOL.json"), runner_sha256=sha256_file(Path(__file__)))
    target=root/"SELECTION_PHASE1_TRACK_B.json"
    if target.exists():
        if read(target)!=spec:
            raise ValueError("Selection cannot change after the frozen official lock")
    else:
        save_json(target,spec)
    signature=target.with_suffix(".sha256")
    value=f"{sha256_file(target)}  {target.name}\n"
    if signature.exists() and signature.read_text()!=value:
        raise ValueError("Phase 1C selection signature changed")
    if not signature.exists():
        staged=signature.with_suffix(".tmp")
        staged.write_text(value)
        staged.replace(signature)
    print("V5_1C_FROZEN",sha256_file(target),best,flush=True)


def runtime_for(root,alias,task=None):
    target=root/"SELECTION_PHASE1_TRACK_B.json"
    sig=target.with_suffix(".sha256").read_text().split()
    if sig!=[sha256_file(target),target.name]:
        raise ValueError("Missing exact Track B selection signature")
    auth=read(target)
    if auth["runner_sha256"]!=sha256_file(Path(__file__)) or not auth["development_only"]:
        raise ValueError("Mismatched B evaluation wrapper")
    if len(auth["candidates"])!=2 or alias not in auth["candidates"]:
        raise ValueError("Run only the two locked external finalists")
    data=auth["candidates"][alias]
    for slot in ("controller","training","development","historical"):
        if sha256_file(Path(data[slot]))!=data[slot+"_sha256"]:
            raise ValueError(f"Official selection asset changed: {slot}")
    args=SimpleNamespace(selection=target,config=E0_CONFIG_PATH,out=root/"official",alias=alias,task=task)
    rt=historical_gate.arguments(args,auth)
    rt.minimal=rt.minimal_code
    rt.release=rt.panel.parents[1]
    return auth,data,rt


def run(root,alias,task):
    auth,data,args=runtime_for(root,alias,task)
    import evaluate_benchmarks as archived
    import seven_unified
    import stage2_score
    args.freeze,_=freeze(args)
    if task is None:
        print("V5_1C_OFFICIAL_CONFIG_READY",alias,sha256_file(args.freeze),flush=True)
        return
    stage2_score.APPLE_BENCHMARKS=dict.fromkeys(stage2_score.APPLE_BENCHMARKS,0)
    seven_unified.TASKS={name:(0,n,metrics) for name,(_,n,metrics) in seven_unified.TASKS.items()}
    args.arm,args.task,args.out="new",task,args.root/"legacy"
    original=minimal.runtime
    path=Path(data["controller"])
    with ExitStack() as context:
        active={"net":None}
        def initialize(*args,**kwargs):
            output=original(*args,**kwargs)
            _,model,tokenizer,_=output
            saved=torch.load(path,map_location="cpu",weights_only=True)
            meta=saved["meta"]
            if (meta["method"],meta["seed"],meta["protocol_sha256"])!=(alias,data["seed"],auth["protocol_sha256"]):
                raise ValueError("The stored PEFT variant differs from the locked declaration")
            if alias in STATIC:
                net=parts.StaticSteeringController(saved["state"]["u_unit"],meta["training"]["alpha"])
            else:
                net=build(alias,model,tokenizer,data["seed"])
            net.load_state_dict(saved["state"],strict=True)
            net.to(model.device).eval().requires_grad_(False)
            if isinstance(net,parts.HookAdapterBase):
                context.enter_context(net.attach(model))
            else:
                active["net"]=net
            return output
        def loader(address,device):
            if address!=path:
                raise ValueError("The archived inference runner requested a different controller")
            return active["net"]
        context.enter_context(patch.object(minimal,"runtime",initialize))
        context.enter_context(patch.object(archived,"controller",loader))
        archived.run(args)


def summarize_all(root):
    v4=Path(__import__("v4_common").__file__).resolve().parent
    history=v4/"official_run/official/Best_ZeroLabel_4096"
    level=root/"SELECTION_PHASE1_TRACK_B.json"
    auth=read(level)
    scores={}
    for alias in auth["candidates"]:
        _,_,args=runtime_for(root,alias)
        results={}
        for task,n in LEGACY.items():
            rows,clusters=legacy_rows(args,task,n)
            own, _, own_receipt=rows["new"]
            mode="generated" if task=="mmlu_pro" else "candidate"
            anchor=history/"FREEZE.json"
            history_file=history/"legacy"/f"{task}__new__k4__{mode}.jsonl"
            rows["best_zerolabel_v4"]=closed(history_file,[r["id"] for r in own],anchor,
                                             read(anchor)["arms"]["new"]["sha256"],own_receipt["plan"])
            results[task]=summarize(task,rows,clusters)
        scores[alias]=results
    save_json(root/"track_b_baselines/PEFT_FULL8_PAIRED.json",
              dict(n=sum(LEGACY.values()),selection_sha256=sha256_file(level),methods=scores))
    print("V5_1C_FULL8_CLOSED",sum(LEGACY.values()),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--phase",choices=("seal","freeze","evaluate","aggregate"),required=True)
    p.add_argument("--alias",choices=METHODS+STATIC)
    p.add_argument("--task",choices=tuple(LEGACY))
    args=p.parse_args()
    torch.set_num_threads(4)
    if args.phase=="seal":
        seal(args.root)
    elif args.phase=="aggregate":
        summarize_all(args.root)
    else:
        run(args.root,args.alias,args.task)

