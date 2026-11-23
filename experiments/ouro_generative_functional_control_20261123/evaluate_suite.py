"""Run a sealed test suite against one weight while loading Ouro only once."""

import argparse
import json
from pathlib import Path

import torch

import minimal
from evaluate_generation import candidates,data_mode,generate,summarize
from evaluation_lock import code_hashes
from task_data import hash_order,load,save


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("archive","previous","model","protocol","out"):
        p.add_argument(f"--{key}",required=True,type=Path)
    p.add_argument("--arm",required=True)
    p.add_argument("--stage", choices=("k4", "qa_transfer", "instruction_transfer"))
    a=p.parse_args()
    config=json.loads(a.protocol.read_text(encoding="utf-8"))
    if config["evaluation_code_sha256"]!=code_hashes():
        raise RuntimeError("Evaluation code differs from the sealed protocol")
    record=config["arms"][a.arm]
    weight=Path(record["file"]) if record["file"] else None
    if weight is not None and minimal.sha(weight)!=record["sha256"]:
        raise ValueError("The requested controller no longer matches the frozen protocol")
    archive,model,tok,runtime=minimal.runtime(a.archive,a.previous,a.model)
    torch_version = runtime.get("torch")
    if not torch_version or runtime["dtype"]!="bfloat16":
        raise RuntimeError("Unexpected Ouro runtime")
    net=None
    if weight is not None:
        net=minimal.Minimal("linear",1,.45).cuda().eval()
        net.load_state_dict(torch.load(weight,map_location="cpu",weights_only=True)["state"])
        net.requires_grad_(False)
        if sum(p.numel() for p in net.parameters())!=4096:
            raise RuntimeError("Controller shape differs")
    a.out.mkdir(parents=True,exist_ok=True)
    protocol=minimal.sha(a.protocol)
    for case in config["cases"]:
        if a.stage=="qa_transfer" and case["mode"]=="other":
            continue
        if a.stage=="instruction_transfer" and case["mode"]!="other":
            continue
        f=Path(case["file"])
        if minimal.sha(f)!=case["data_sha256"]:
            raise ValueError(f"The sealed input has changed: {f}")
        base=load(f)
        mode=case["mode"]
        if mode=="gsm_process":
            for row in base:
                alt="Solve the problem step by step. End with the final answer after ####."
                row["instruction"]=alt
                row["prompt"]=f"Question: {row['context']}\n{alt}\nAnswer:"
        for depth in case["depths"]:
            if a.stage is not None and (depth==4)!=(a.stage=="k4"):
                continue
            n=min(len(base),case["full_limit"] if depth==4 else case["transfer_limit"])
            preds=sorted(base,key=lambda row:hash_order(row["sample_id"],f"test/{case['dataset']}"))[:n]
            pred=data_mode(preds,case["dataset"],mode)
            raw=a.out/f"{case['dataset']}__{a.arm}__k{depth}__{mode}.jsonl"
            receipt=raw.with_suffix(".json")
            plan=dict(arm=a.arm,dataset=case["dataset"],mode=mode,depth=depth,n=n,
                      protocol_sha256=protocol,data_sha256=case["data_sha256"],checkpoint_sha256=record["sha256"],
                      max_new=case["max_new"],batch=case["batch"])
            if receipt.exists():
                lock=json.loads(receipt.read_text())
                if lock["plan"]!=plan or minimal.sha(raw)!=lock["rows_sha256"]:
                    raise ValueError(f"Untrusted resume shard {raw}")
                continue
            torch.cuda.reset_peak_memory_stats()
            if mode=="candidate":
                vector,cost=candidates(archive,model,tok,net,pred,depth,case["batch"])
            else:
                vector,cost=generate(archive,model,tok,net,pred,mode,depth,case["batch"],case["max_new"])
            save(raw,vector)
            metric=summarize(vector)
            minimal.dump(receipt,dict(plan=plan,rows_sha256=minimal.sha(raw),metrics=metric,cost=cost,runtime=runtime))
            print(raw.stem,n,{k:round(v["value"],4) for k,v in metric.items()},round(cost["seconds"],2),flush=True)


if __name__=="__main__":
    main()
