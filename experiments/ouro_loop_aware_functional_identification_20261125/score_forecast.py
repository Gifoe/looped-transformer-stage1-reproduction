"""Audit pretraining forecast ranks, magnitude calibration, and held-out regret."""

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import kendalltau, spearmanr

import minimal
from task_data import load


def scores(results,lock_hash):
    evaluations={}
    for receipt in sorted(results.glob('*.json')):
        result=json.loads(receipt.read_text())
        plan=result['plan']
        path=receipt.with_suffix('.jsonl')
        if plan['forecast_sha256']!=lock_hash or result['predictions_sha256']!=minimal.sha(path):
            raise ValueError(f'Changed sealed intervention: {receipt}')
        groups=defaultdict(list)
        for row in load(path):
            groups[row['direction'],row['dose']].append(row)
        for (direction,dose), group in groups.items():
            if [r['id'] for r in group]!=plan['ids']:
                raise ValueError(f'Incorrect evaluation cohort for {direction}/{dose}')
            dataset=plan['target']
            fields={name:statistics.mean(r[name] for r in group) for name in ('direct_gain','loop_gain','interaction')}
            fields['accuracy_delta']=statistics.mean(r['edited_loop']-r['native_loop'] for r in group)
            fields['positive_but_loop_damaged']=statistics.mean(r['interaction']>0 and r['loop_gain']<0 for r in group)
            fields['joint_positive']=statistics.mean(r['loop_gain']>0 and r['interaction']>0 for r in group)
            evaluations[dataset,plan['depth'],direction,dose]=fields
    return evaluations


def correlation(fn,a,b):
    if len(a)<3 or max(a)==min(a) or max(b)==min(b):
        return None
    return float(fn(a,b).statistic)


def evaluate(rows,key):
    by_target=defaultdict(list)
    for r in rows:
        by_target[r['target']].append(r)
    report=[]
    for target, held in by_target.items():
        other=[r for r in rows if r['target']!=target]
        estimate=[r['predicted'][key] for r in held]
        actual=[r['actual']['scaled_loop'] for r in held]
        chosen=max(held,key=lambda r:(r['predicted'][key],r['source']))
        if other:
            x=np.asarray([r['predicted'][key] for r in other])
            y=np.asarray([r['actual']['scaled_loop'] for r in other])
            scale=np.std(x)
            # Calibration leaves the entire reported target out, including all its sources.
            if scale<1.e-12:
                estimated=np.full(len(held),np.mean(y))
            else:
                z=(x-np.mean(x))/scale
                slope=float(z@ (y-np.mean(y))/(z@z+1.e-8))
                estimated=np.mean(y)+slope*(np.asarray(estimate)-np.mean(x))/scale
            error=float(np.abs(estimated-actual).mean())
        else:
            error=None
        report.append(dict(target=target,n=len(held),spearman=correlation(spearmanr,estimate,actual),
                    kendall=correlation(kendalltau,estimate,actual),loo_calibrated_mae=error,
                    selected=chosen['source'],actual=chosen['actual']['scaled_loop'],
                    accuracy_delta=chosen['actual']['accuracy_delta'],interaction=chosen['actual']['interaction'],
                    regret=max(actual)-chosen['actual']['scaled_loop']))
    stats={}
    for metric in ('spearman','kendall','loo_calibrated_mae','regret','accuracy_delta','actual'):
        values=[r[metric] for r in report if r[metric] is not None]
        stats[metric+'_mean']=statistics.mean(values) if values else None
    stats['weakest_selected_loop']=min(r['actual'] for r in report)
    stats['weakest_selected_accuracy']=min(r['accuracy_delta'] for r in report)
    return dict(targets=report,aggregate=stats)


def run(a):
    plan=json.loads(a.forecast.read_text())
    lock=minimal.sha(a.forecast)
    outcomes=scores(a.results,lock)
    if not outcomes:
        raise ValueError('No closed held-out interventions')
    entries=[]
    kinds=sorted({(key[1],key[3]) for key in outcomes})
    scopes=sorted(set(r['scope'] for r in plan['records']))
    for (depth,dose),scope in itertools.product(kinds,scopes):
        rows=[]
        for r in plan['records']:
            if r['scope']!=scope:
                continue
            target=r['target']
            observed=outcomes.get((target,depth,r['source']+'__excess',dose))
            simpler=outcomes.get((target,depth,r['source']+'__loop',dose))
            if observed is None or simpler is None:
                continue
            scale=plan['sources'][target]['fit_q_robust_scale']
            rows.append(dict(**r,actual=dict(observed,scaled_loop=observed['loop_gain']/scale),ordinary=simpler))
        if not rows:
            continue
        names=[k for k,v in rows[0]['predicted'].items() if v is not None and k!='ordinary_policy']
        summary={name:evaluate(rows,name) for name in names}
        shifts=[r['actual']['loop_gain']-r['ordinary']['loop_gain'] for r in rows]
        entries.append(dict(depth=depth,dose=dose,scope=scope,rows=rows,methods=summary,
                            excess_direction_minus_ordinary_mean_loop=statistics.mean(shifts),
                            false_positive_interaction_mean=statistics.mean(r['actual']['positive_but_loop_damaged'] for r in rows)))
    minimal.dump(a.out,dict(forecast_sha256=lock,confirmation=entries))
    for k in entries:
        if k['depth']==4 and k['dose']==1.:
            print(k['scope'],{n:round(a['aggregate']['spearman_mean'],3) if a['aggregate']['spearman_mean'] is not None else None
                              for n,a in k['methods'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for item in ('forecast','results','out'):
        p.add_argument(f'--{item}',type=Path,required=True)
    run(p.parse_args())
