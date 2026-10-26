"""Measured extrapolation; never change method identities to satisfy runtime."""
import argparse
import json
import numpy as np
from common import ROOT,BASE,CFG,read,save

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generation',action='store_true');parser.add_argument('--selected',action='store_true');args=parser.parse_args()
    ledger=ROOT/'logs/gpu_budget.json';used=json.loads((ledger if ledger.exists() else BASE/'logs/gpu_budget.json').read_text())['used_seconds']
    if args.generation:
        estimates=[]
        for model in ['Ouro-1.4B','Ouro-2.6B']:
            rows=read('trajectories/math_'+model+'_validation.jsonl')
            by_method={method:float(np.mean([r['elapsed_s'] for r in rows if r['method']==method])) for method in sorted({r['method'] for r in rows})}
            validation=0 if args.selected else sum(value*(CFG['math']['validation_count']-5) for value in by_method.values())
            extra=read('trajectories/math_'+model+'_pilot_extra.jsonl')
            costs={m:float(np.mean([r['elapsed_s'] for r in extra if r['method']==m])) for m in ['depth_2','depth_3','logits_paper_03','logits_adaptive_paper']}
            costs.update(baseline=by_method['baseline'],hidden_fixed_025=by_method['hidden_0.25'])
            if args.selected:
                selected=json.loads((ROOT/'configs/math_selected.json').read_text())[model]
                identities=set()
                for space in ['logits','hidden']:
                    omega=selected[space]['omega'];key='baseline' if omega==0 else f'{space}_{omega:g}'
                    duplicate=key in identities or key=='baseline' or key=='hidden_0.25'
                    costs[space+'_validation_selected']=0 if duplicate else by_method[key];identities.add(key)
            else:
                # Provisional prediction uses the grid median, not the slowest
                # often-degenerate high-strength run for every test baseline.
                for space in ['logits','hidden']:costs[space+'_validation_selected']=float(np.median([v for k,v in by_method.items() if k.startswith(space+'_')]))
            estimates.append(dict(model=model,mean_seconds_by_validation_method=by_method,test_seconds_by_method=costs,remaining_validation_seconds=validation,test_seconds_per_question=sum(costs.values())))
        reserve=3600;per_question=sum(r['test_seconds_per_question'] for r in estimates)*1.5
        validation=sum(r['remaining_validation_seconds'] for r in estimates)*1.5
        available=50*3600-used-validation-reserve
        count=min(CFG['math']['test_count'],max(0,int(available/max(per_question,1))))
        assert count>=50,'Even 50 aligned test questions exceed the measured budget; inspect before further GPU work'
        remaining=validation+per_question*count+reserve
        result=dict(used_seconds=used,estimated_remaining_seconds=remaining,models=estimates,test_count=count,uncertainty_multiplier=1.5,reserve_seconds=reserve,passes_hard_budget=True,selection_source='validation only' if args.selected else 'pre-validation method-specific pilot; provisional selected costs are grid medians')
        save('results/generation_budget_'+('selected' if args.selected else 'pilot')+'.json',result)
        if args.selected:
            # Frozen before ANY test generation; every method/model gets the
            # same deterministic prefix of the already-frozen fresh test IDs.
            assert not any((ROOT/f'trajectories/math_{m}_test.jsonl').exists() for m in ['Ouro-1.4B','Ouro-2.6B'])
            save('configs/math_execution.json',dict(test_count=count,planned_count=CFG['math']['test_count'],reason='method-specific validation/pilot runtime under cumulative 50h cap; no test outcome used',budget=result))
    else:
        estimates=[]
        for model in CFG['models']:
            methods=CFG['qwen_methods'] if model.startswith('Qwen') else [None]
            for method in methods:
                name=f'{model}_{method}_pilot' if method else model+'_pilot'
                rows=read('trajectories/'+name+'.jsonl')
                category=lambda r:'mmlu' if r['task'].startswith('mmlu_') else r['task']
                seconds=0
                for task in ['arc_challenge','openbookqa','mmlu']:
                    mean=float(np.mean([r['elapsed_s'] for r in rows if category(r)==task]))
                    count=CFG['new_mc_validation'][task]+(CFG['new_qwen_test'] if method else CFG['new_ouro_test'])[task]
                    seconds+=mean*count
                estimates.append(dict(model=model,method=method,estimated_mc_seconds=seconds))
        remaining=sum(r['estimated_mc_seconds'] for r in estimates)*1.5
        assert used+remaining<50*3600
        save('results/mc_budget_estimate.json',dict(used_seconds=used,estimated_remaining_mc_seconds=remaining,models=estimates,passes_hard_budget=True))
    print(json.dumps(dict(used_gpu_hours=used/3600,estimated_remaining_hours=remaining/3600)),flush=True)

if __name__=='__main__':main()
