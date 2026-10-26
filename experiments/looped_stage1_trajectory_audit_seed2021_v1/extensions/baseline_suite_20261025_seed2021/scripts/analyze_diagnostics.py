"""Trajectory attribution, scoring sensitivity, cases and offline oracle bound."""
import json
import numpy as np
import pandas as pd
from common import ROOT,CFG,read,save
from analyze_suite import quality,category,groups,wilson

def main():
    selected=json.loads((ROOT/'configs/mc_selected.json').read_text());attribution=[];sensitivity=[];oracles=[];cases=[];depth_summary=[];strata=[];pairwise=[]
    manifests={r['id']:r for r in read('data/validation.jsonl')+read('data/ouro_test.jsonl')}
    for method in CFG['qwen_methods']:
        rows=read(f'trajectories/Qwen3-4B-Base_{method}_test.jsonl');assert len(rows)==600
        for task,part in groups(rows).items():
            sensitivity.append(dict(model='Qwen3-4B-Base',method=method,task=task,n=len(part),raw_accuracy=float(np.mean([r['result']['raw_correct'] for r in part])),normalized_accuracy=float(np.mean([r['result']['normalized_correct'] for r in part])),primary_accuracy=float(np.mean([r['result']['correct'] for r in part])),prediction_disagreements=sum(r['result']['raw_prediction']!=r['result']['normalized_prediction'] for r in part)))
    for model in ['Ouro-1.4B','Ouro-2.6B']:
        rows=read(f'trajectories/{model}_test.jsonl');assert len(rows)==300
        validation=read(f'trajectories/{model}_validation.jsonl');assert len(validation)==250
        for row in rows:
            task=category(row)
            for space in ['logits','hidden']:
                strength=selected['ouro'][model][space][task]
                row['predictions'][space+'_validation_selected']=row['predictions'][f'{space}_{strength:g}']
            temperature=selected['ouro'][model]['temperature'][task]
            row['predictions']['temperature_validation_selected']=row['predictions'][f'temperature_{temperature:g}']
        def mask(row,pattern):
            p=[s['correct'] for s in row['trajectory']]
            if pattern=='CWC':return p[-1] and any(p[a] and not p[b] for a in range(2) for b in range(a+1,3))
            if pattern=='WCW':return not p[-1] and any(not p[a] and p[b] for a in range(2) for b in range(a+1,3))
            return ('C' if p[0] else 'W')+('C' if p[-1] else 'W')==pattern
        for task,part in groups(rows).items():
            for depth in range(1,5):
                states=[r['trajectory'][depth-1] for r in part]
                summary=dict(model=model,task=task,depth=depth,n=len(part),accuracy=float(np.mean([s['correct'] for s in states])),mean_gold_nll_per_token=float(np.mean([r['predictions'][f'r{depth}']['gold_nll_per_token'] for r in part])))
                for metric in ['confidence','entropy','probability_margin','logit_margin','choice_confidence','choice_entropy','choice_probability_margin','choice_logit_margin','full_vocab_kl','choice_kl','hidden_relative_l2','hidden_cosine']:
                    values=[s[metric] for s in states if s.get(metric) is not None]
                    summary['mean_'+metric]=float(np.mean(values)) if values else None
                depth_summary.append(summary)
            comparisons=[(method,'temperature_validation_selected') for method in ['logits_0.5','adaptive_paper','hidden_0.5','logits_validation_selected','hidden_validation_selected']]
            comparisons += [('hidden_0.5','orthogonal_05'),('logits_0.5','parallel_05')]
            for method,reference in comparisons:
                pairwise.append(dict(model=model,task=task,method=method,reference=reference,**quality([r['predictions'][reference]['correct'] for r in part],[r['predictions'][method]['correct'] for r in part]),scope='Direct paired question comparison; fixed paper or validation-selected settings; pointwise unadjusted intervals'))
            for method in rows[0]['predictions']:
                sensitivity.append(dict(model=model,method=method,task=task,n=len(part),raw_accuracy=float(np.mean([r['predictions'][method]['raw_correct'] for r in part])),normalized_accuracy=float(np.mean([r['predictions'][method]['normalized_correct'] for r in part])),primary_accuracy=float(np.mean([r['predictions'][method]['correct'] for r in part])),prediction_disagreements=sum(r['predictions'][method]['raw_prediction']!=r['predictions'][method]['normalized_prediction'] for r in part)))
            oracle_depths=[next((i+1 for i,s in enumerate(r['trajectory']) if s['correct']),4) for r in part]
            correct=[any(s['correct'] for s in r['trajectory']) for r in part]
            oracles.append(dict(model=model,task=task,average_depth=float(np.mean(oracle_depths)),**quality([r['trajectory'][-1]['correct'] for r in part],correct),scope='Offline earliest-correct upper bound with gold/future access; not a deployable method'))
            for pattern in ['CC','CW','WC','WW','CWC','WCW']:
                subset=[r for r in part if mask(r,pattern)]
                for method in ['logits_0.5','adaptive_paper','hidden_0.5','logits_validation_selected','hidden_validation_selected','temperature_validation_selected']:
                    metrics=quality([r['trajectory'][-1]['correct'] for r in subset],[r['predictions'][method]['correct'] for r in subset]) if subset else dict(n=0,accuracy=None,delta=None,ci_low=None,ci_high=None,wrong_to_correct=0,correct_to_wrong=0)
                    attribution.append(dict(model=model,task=task,pattern=pattern,method=method,**metrics,scope='Post-hoc trajectory attribution; CC/CW/WC/WW partition, CWC/WCW overlap'))
        for task in ['arc_challenge','openbookqa','mmlu']:
            vv=[r for r in validation if category(r)==task];tt=[r for r in rows if category(r)==task]
            for variable in ['question_chars','initial_vocab_confidence']:
                value=lambda row:len(manifests[row['id']]['question']) if variable=='question_chars' else row['trajectory'][0]['confidence']
                cuts=np.quantile([value(r) for r in vv],[.25,.5,.75])
                # Ties go to the higher quartile; bins can be empty. Cuts use
                # validation only and never define a deployed stopping policy.
                indices=np.searchsorted(cuts,[value(r) for r in tt],side='right')
                for quartile in range(4):
                    part=[r for r,k in zip(tt,indices) if k==quartile]
                    for pattern in ['CC','CW','WC','WW','CWC','WCW']:
                        count=sum(mask(r,pattern) for r in part);lo,hi=wilson(count,len(part))
                        strata.append(dict(model=model,task=task,variable=variable,quartile=quartile+1,validation_cuts=json.dumps(cuts.tolist()),n=len(part),pattern=pattern,count=count,rate=count/len(part) if part else None,ci_low=lo,ci_high=hi,scope='Post-hoc descriptive strata with validation quartiles; CW/WC are first/final classes; not a learned routing policy'))
        for task in ['arc_challenge','openbookqa','mmlu']:
            for pattern in ['CW','WC','CWC','WCW']:
                first=next((r for r in rows if category(r)==task and mask(r,pattern)),None)
                if first is None:continue
                doc=manifests[first['id']]
                cases.append(dict(model=model,task=task,pattern=pattern,id=first['id'],question=doc['question'],choices=doc['choices'],gold=doc['gold'],trajectory=first['trajectory'],guided_predictions={m:first['predictions'][m] for m in ['logits_0.5','adaptive_paper','hidden_0.5']},selection_rule='First frozen-manifest ID matching this benchmark/pattern; qualitative illustration, not a population estimate'))
    for name,rows in [('loopcd_trajectory_attribution.csv',attribution),('score_rule_sensitivity.csv',sensitivity),('offline_oracle.csv',oracles),('depth_diagnostics.csv',depth_summary),('trajectory_strata.csv',strata),('readout_pairwise_controls.csv',pairwise)]:pd.DataFrame(rows).to_csv(ROOT/'results'/name,index=False)
    save('results/representative_cases.json',cases)
    print(json.dumps(dict(completed=True,cases=len(cases))),flush=True)

if __name__=='__main__':main()
