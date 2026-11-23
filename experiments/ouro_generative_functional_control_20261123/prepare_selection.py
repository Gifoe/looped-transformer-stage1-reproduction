"""Make the final development-only decision for the ranking-retention ablation."""

import argparse
from pathlib import Path

import minimal
from freeze_test import score_dev
from task_data import hash_order,load,save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("."))
    a=p.parse_args()
    data=a.root/"data"
    results=a.root/"results"
    dev=results/"development_rescored"
    candidates=("gen_only","gen_rank","multitask_full")
    base={arm:score_dev(dev,arm) for arm in (*candidates,"native","rank_only")}
    improved=lambda x:(x["semantic_macro"]>base["native"]["semantic_macro"]+.02 and
                       x["paired_candidate_macro"]>=base["rank_only"]["paired_candidate_macro"]-.02)
    accepted=[arm for arm in candidates if improved(base[arm])]
    chosen=max(candidates,key=lambda arm:base[arm]["semantic_macro"])
    run=not accepted
    if run:
        pool=load(data/"full_mixed_train.jsonl")
        train=[]
        for ds in ("csqa","obqa"):
            records=[r for r in pool if r["dataset"]==ds]
            train.extend(sorted(records,key=lambda r:hash_order(r["sample_id"],"retention"))[:564])
        train=sorted(train,key=lambda r:hash_order(r["sample_id"],"retention-combined"))
        assert len(train)==1128
        save(data/"retention_train.jsonl",train)
    decision=dict(run_paired_retention=run,shared_initial=chosen,passed_dev=accepted,scores=base,
                  predeclared_gate="semantic over Native by at least 2 pp and candidate within 2 pp of same-source ranking arm",
                  paired_ablation="same 1128 balanced MC examples, same start and optimizer steps, rank .25 vs .90")
    minimal.dump(results/"ROUND4_DECISION.json",decision)
    print(decision,flush=True)


if __name__=="__main__":
    main()
