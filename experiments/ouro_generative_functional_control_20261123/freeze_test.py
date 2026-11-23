"""Seal one controller using development generation only, then define the test matrix."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import minimal
from evaluation_lock import code_hashes


MULTIPLE_CHOICE=("arc_challenge","arc_easy","csqa","obqa")
GENERAL=("no_robots","smol-magpie-ultra","smol-rewrite","smol-summarize","smol-constraints")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def score_dev(base,arm):
    def get(ds,mode,key):
        f=base/f"{ds}__{arm}__k4__{mode}.json"
        if not f.is_file():
            raise FileNotFoundError(f)
        doc=read(f)
        return doc["metrics"][key]["value"]
    semantic={
        **{f"{ds}/{mode}":get(ds,mode,"loose_correct" if mode=="letter" else "correct")
           for ds in ("csqa","obqa") for mode in ("letter","content")},
        "squad/EM":get("squad","span","correct"),
        "gsm8k/final":get("gsm8k","gsm_final","correct"),
    }
    format_=float(np.mean([get(d,"letter","strict") for d in ("csqa","obqa")]))
    valid=float(np.mean([get(d,"letter","valid") for d in ("csqa","obqa")]))
    choice=float(np.mean([get(d,"candidate","acc_norm") for d in ("csqa","obqa")]))
    return dict(semantic_macro=float(np.mean(list(semantic.values()))),semantic=semantic,
                strict_letter=format_,valid_letter=valid,paired_candidate_macro=choice)


def make_cases(data):
    specs=[]
    for ds in MULTIPLE_CHOICE:
        for mode,limit,batch in (("content",24,48),("letter",8,48),("candidate",1,12)):
            specs.append((ds,mode,[2,3,4,5,6],99999,192,limit,batch))
    specs.extend([("squad","span",[2,3,4,5,6],360,96,24,32),
                  ("gsm8k","gsm_final",[2,3,4,5,6],320,96,28,32),
                  ("gsm8k","gsm_process",[4],96,96,192,24)])
    specs.extend((ds,"other",[2,3,4,5,6],64,20,160,16) for ds in GENERAL)
    return [dict(dataset=ds,mode=mode,depths=depths,full_limit=n,transfer_limit=nt,
                 max_new=max_new,batch=batch,file=str(data/f"test_{ds}.jsonl"),
                 data_sha256=minimal.sha(data/f"test_{ds}.jsonl"))
            for ds,mode,depths,n,nt,max_new,batch in specs]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("result","data","historical","out"):
        p.add_argument(f"--{name}",required=True,type=Path)
    p.add_argument("--multi",default="multitask_full")
    p.add_argument("--extra",action="append",default=[])
    p.add_argument("--filtered",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("Test configuration was already frozen")
    root=a.result
    choices=["gen_only","gen_rank",a.multi,*a.extra]
    all_arm=["native","rank_only",*choices]
    reference={"native":None,"rank_only":a.historical}
    reference.update({label:root/label/"controller.pt" for label in choices})
    dev=root/"development_rescored"
    evidence={arm:score_dev(dev,arm) for arm in all_arm}
    def criteria(arm):
        x=evidence[arm]
        return (x["semantic_macro"],x["strict_letter"],x["valid_letter"],x["paired_candidate_macro"],arm)
    winner=max(choices,key=criteria)
    candidate=root/"FINAL_CONTROLLER.pt"
    shutil.copyfile(reference[winner],candidate)
    arms={arm:dict(file=str(f) if f is not None else None,sha256=minimal.sha(f) if f else "original_backbone")
          for arm,f in reference.items()}
    # Test carries five scientific arms. If an extra development candidate wins, include it as a sixth.
    if winner in a.extra:
        run_names=["native","rank_only","gen_only","gen_rank",a.multi,winner]
    else:
        run_names=["native","rank_only","gen_only","gen_rank",a.multi]
    final=dict(selected_arm=winner,checkpoint_sha256=minimal.sha(candidate),controller_file=str(candidate),
        trainable_parameters=4096,development_evidence=evidence,data_lock_sha256=minimal.sha(a.data/"DATA_LOCK.json"),
        expanded_pool_sha256=minimal.sha(a.data/"SCALE_LOCK.json"),
        contamination_sha256=minimal.sha(a.filtered/"CONTAMINATION.json"),
        evaluation_code_sha256=code_hashes(),
        allowed_run_order=[winner,*[x for x in run_names if x!=winner]],arms={k:arms[k] for k in run_names},
        run_stages=["k4","qa_transfer","instruction_transfer"],
        cases=make_cases(a.filtered),decode="full vocabulary greedy, EOS; short tasks preserve intervening historical newline stop",
        selector="Development semantic correctness first. Exact tie-breaks strict format, parsed validity, acc_norm.",
        no_test_model_selection=True)
    minimal.dump(a.out,final)
    print(winner,final["checkpoint_sha256"],evidence,flush=True)


if __name__=="__main__":
    main()
