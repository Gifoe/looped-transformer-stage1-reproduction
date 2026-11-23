"""Check the frozen study's raw outputs, paired metrics and GPU lease."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from accept import evaluate
from content_eval import compare
from minimal import dump, load_jsonl, sha
from screen import paired


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_ledger(path):
    charged=0
    completed=[]
    failed=[]
    started=None
    last_finish=0
    for row in path.read_text().splitlines():
        tag,phase,start,*tail=row.split("\t")
        start=int(start)
        if tag=="START":
            assert started is None and start>=last_finish and int(tail[0])==28800-charged
            started=(phase,start)
        else:
            assert tag=="END" and started==(phase,start)
            end,seconds,exit_code=map(int,tail)
            assert seconds==end-start and seconds>=0
            charged+=seconds
            rec=dict(phase=phase,seconds=seconds,exit_code=exit_code)
            (failed if exit_code else completed).append(rec)
            started=None
            last_finish=end
    assert started is None and charged<=28800
    assert set(r["phase"] for r in completed)=={
        "schedule_dev","fit_rank1","rescreen","native_a","content_b","generation_confirm","content_gsm8k","content_mmlu_pro"
    }
    return dict(charged_seconds=charged,completed=completed,failed=failed)


def raw_outputs(root, control, history):
    expected={"arc_validation":299,"openbookqa_test":500}
    main=root/"results/BENCHMARK_RESULTS"
    folder=main/"OPTION_GENERATION"
    base=control/"results/GENERATION_RESULTS"
    option=read(folder/"CONFIRM.json")
    for split,n in expected.items():
        for depth in range(1,5):
            old=load_jsonl(base/f"{split}__rank1_t1__k{depth}.jsonl")
            new=load_jsonl(folder/f"{split}__locked__k{depth}.jsonl")
            assert len(old)==len(new)==n
            for metric in ("correct","valid","strict_format"):
                observed=json.loads(json.dumps(paired(old,new,metric)))
                assert observed==option[split][f"k{depth}"]["comparisons"]["t1"][metric]
    for name,n in (("gsm8k",1319),("mmlu_pro",12032)):
        folder=main/name
        original=load_jsonl(folder/"ORIGINAL.jsonl")
        summary=read(folder/"SUMMARY.json")
        other=original if summary["exact_reuse"] else load_jsonl(folder/"LOCKED.jsonl")
        canonical={r["id"]:r["gold"] for r in load_jsonl(history/name/"native.jsonl")}
        assert len(original)==len(other)==len(canonical)==n
        assert {r["id"]:r["gold"] for r in original}=={r["id"]:r["gold"] for r in other}==canonical
        assert compare(name,original,other)==summary["vs_rank1_t1"]
    return dict(option=expected,content={"gsm8k":1319,"mmlu_pro":12032})


def main(args):
    root=args.study
    results=root/"results"
    plan=read(results/"frozen/PROPOSAL.json")
    screen=results/"frozen/DEV_RESULTS.json"
    assert sha(screen)==plan["validation_results_sha256"]
    trained=results/"fit/rank1_generation.pt"
    assert sha(trained)==read(results/"fit/TRAIN_RECEIPT.json")["sha256"]==plan["weights_sha256"]["train"]
    observed=raw_outputs(root,args.control,args.history)
    receipt=evaluate(results)
    assert receipt==read(results/"frozen/ACCEPTANCE.json")
    run=gpu_ledger(root/"GPU_LEDGER.tsv")
    vector_geometry={}
    old=args.control/"results/weights/rank1_t1.pt"
    ref=torch.load(old,map_location="cpu",weights_only=True)["state"]
    reference_norm=float(ref["out.weight"].norm()*ref["encode.weight"].norm())
    for name,path in (("t2",args.control/"results/weights/rank1_t2.pt"),("train",trained)):
        assert sha(path)==plan["weights_sha256"][name]
        current=torch.load(path,map_location="cpu",weights_only=True)["state"]
        out_cos=float(F.cosine_similarity(ref["out.weight"].reshape(-1),current["out.weight"].reshape(-1),dim=0))
        enc_cos=float(F.cosine_similarity(ref["encode.weight"].reshape(-1),current["encode.weight"].reshape(-1),dim=0))
        current_norm=float(current["out.weight"].norm()*current["encode.weight"].norm())
        vector_geometry[name]=dict(output_cosine=out_cos,encoder_cosine=enc_cos,operator_cosine=out_cos*enc_cos,
                                   relative_unbounded_frobenius_norm=current_norm/reference_norm)
    dump(root/"FINAL_AUDIT.json",dict(output_rows=observed,ledger=run,
                    gate=receipt["generation_deployment_accepted"],failed_cells=receipt["failed_cells"],
                    proposal_sha256=receipt["proposal_sha256"],weight_geometry_relative_to_t1=vector_geometry))
    print(f"Verified frozen method, full paired outputs and GPU lease ({run['charged_seconds']} s)",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("study","control","history"):
        p.add_argument(name,type=Path)
    main(p.parse_args())
