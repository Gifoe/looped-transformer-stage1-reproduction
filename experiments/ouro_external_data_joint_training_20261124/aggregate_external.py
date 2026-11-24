"""Verify complete raw results and report one shared deployment checkpoint."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from evaluate_core import TASKS
from evaluate_generation import summarize
import minimal
import paired_analysis
from task_data import load


paired_analysis.METRICS.update({"strict_format", "process_correct", "premature_eos", "token_limit", "repetitive", "parseable", "process"})


def label(root, dataset, arm, depth, mode):
    return root / f"{dataset}__{arm}__k{depth}__{mode}.jsonl"


def receipts(paths, freeze_sha):
    rows, costs = {}, {}
    for arm,path in paths.items():
        doc=json.loads(path.with_suffix('.json').read_text())
        if doc["plan"]["freeze_sha256"]!=freeze_sha or doc["rows_sha256"]!=minimal.sha(path):
            raise ValueError(f"Non-frozen or changed per-question artifact: {path}")
        raw=load(path)
        if len({r['id'] for r in raw})!=len(raw) or summarize(raw)!=doc['metrics']:
            raise ValueError(f"Per-question metrics or IDs differ at {path}")
        rows[arm],costs[arm]=raw,doc['cost']
    return rows,costs


def entries(freeze, root):
    seven=('arc_challenge','arc_easy','sciq','mmlu','hellaswag','winogrande','piqa')
    for ds in seven:
        yield 'primary',ds,4,'candidate',root/'test/benchmark',freeze['primary'][ds]
    yield 'primary','mmlu_pro',4,'generated',root/'test/benchmark','correct'
    for ds,modes in TASKS.items():
        for mode in modes:
            primary=freeze['primary'][ds] if ds not in seven and mode in ('candidate','span','gsm_process') else None
            yield 'primary' if primary else 'supplement',ds,4,mode,root/'test/core',primary
            for depth in (2,3,5,6):
                yield 'transfer',ds,depth,mode,root/'transfer/core',None


def run(a):
    freeze=json.loads(a.freeze.read_text())
    checksum=minimal.sha(a.freeze)
    coverage=[]
    output=[]
    primary=[]
    groups=defaultdict(list)
    for scope,dataset,k,mode,path,chosen in entries(freeze,a.results):
        expected={arm:label(path,dataset,arm,k,mode) for arm in freeze['arms']}
        rows,costs=receipts(expected,checksum)
        own=rows['joint']
        for arm, pred in rows.items():
            if [row['id'] for row in pred]!=[row['id'] for row in own]:
                raise ValueError(f"Differing paired cohort {scope}/{dataset}/K{k}/{mode}/{arm}")
        statistics={arm:paired_analysis.compare(pred,own) for arm,pred in rows.items() if arm!='joint'}
        raw_summary={arm:summarize(pred) for arm,pred in rows.items()}
        coverage.append(dict(scope=scope,dataset=dataset,depth=k,mode=mode,n=len(own),arms=len(rows),metric=chosen,
                             costs=costs,summary=raw_summary,paired=statistics))
        for name,score in raw_summary['joint'].items():
            for arm,by_metric in statistics.items():
                if name not in by_metric:
                    continue
                p=by_metric[name]
                output.append(dict(scope=scope,dataset=dataset,depth=k,mode=mode,metric=name,baseline=arm,
                                   n=p['n'], accuracy=p['controller'], baseline_accuracy=p['reference'], delta=p['delta'],
                                   rescued=p['rescued'],harmed=p['harmed'],ci95_low=p['paired_ci95'][0],ci95_high=p['paired_ci95'][1],
                                   exact_mcnemar=p['exact_mcnemar'],seconds=costs['joint'].get('seconds',costs['joint'].get('elapsed_seconds'))))
        if chosen:
            native=statistics['native'][chosen]
            refs=('t1','t2','retention','mixed')
            best=max(refs,key=lambda arm:statistics[arm][chosen]['reference'])
            strongest=statistics[best][chosen]
            family='reasoning' if dataset=='gsm8k' else ('short' if dataset in ('squad','mmlu_pro') else 'candidate')
            groups[family].append(native)
            primary.append(dict(dataset=dataset,phase=scope,metric=chosen,n=len(own),native=native,best_reference=best,
                                paired_vs_best=strongest))
        elif scope=='supplement' and mode=='content':
            groups['short'].append(statistics['native']['correct'])
    if len(primary)!=12:
        raise ValueError(f"Expected twelve complete primary benchmarks, got {len(primary)}")
    observed=all(row['native']['delta']>=0 for row in primary)
    confirmed={family:any(cell['paired_ci95'][0]>0 for cell in cells) for family,cells in groups.items()}
    result=dict(freeze_sha256=checksum, selected=freeze['selection'], final_sha256=freeze['arms']['joint']['sha256'],
                primary=primary, observed_no_primary_degradation=observed, all_primary_ci_nonnegative=all(
                    row['native']['paired_ci95'][0]>=0 for row in primary), confirmed_improvement_by_family=confirmed,
                achieved_primary_observed_and_confirmed=observed and all(confirmed.values()), coverage=coverage,
                seed_results=json.loads(a.selection.read_text())['seed_results'])
    minimal.dump(a.results/'RESULTS.json',result)
    with (a.results/'RESULTS.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=output[0])
        writer.writeheader()
        writer.writerows(output)
    print("DONE",len(primary),observed,confirmed,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results'))
    parser.add_argument('--freeze',type=Path,default=Path('results/FREEZE.json'))
    parser.add_argument('--selection',type=Path,default=Path('results/DEVELOPMENT_SELECTION.json'))
    run(parser.parse_args())
