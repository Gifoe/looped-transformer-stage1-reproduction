import json
import numpy as np
import pandas as pd
from decimal import Decimal,InvalidOperation
from analyze import ROOT,read,paired_ci,trajectory_pattern_masks,wilson

rows=read('ouro_test.jsonl');selected=json.loads((ROOT/'configs/validation_selected.json').read_text())
groups={name:[] for name in ['early_correct_final_correct','early_correct_final_wrong','early_wrong_final_correct','early_wrong_final_wrong']}
for row in rows:
    early=row['trajectory'][0]['correct'];final=row['trajectory'][-1]['correct']
    key='early_'+('correct' if early else 'wrong')+'_final_'+('correct' if final else 'wrong')
    groups[key].append(row)
table=[]
for group,data in groups.items():
    for method in rows[0]['predictions']:
        if not method.startswith(('logits_','hidden_','temp_')):continue
        baseline=np.array([x['predictions']['r4']['correct'] for x in data]);pred=np.array([x['predictions'][method]['correct'] for x in data])
        table.append(dict(group=group,method=method,n=len(data),accuracy=float(pred.mean()),wrong_to_correct=int((~baseline&pred).sum()),correct_to_wrong=int((baseline&~pred).sum()),
                          mean_gold_choice_logp_change=float(np.mean([x['predictions'][method]['gold_choice_logp']-x['predictions']['r4']['gold_choice_logp'] for x in data])),
                          gold_choice_logp_decreased=sum(x['predictions'][method]['gold_choice_logp']<x['predictions']['r4']['gold_choice_logp']-1e-7 for x in data)))
pd.DataFrame(table).to_csv(ROOT/'results/loopcd_by_trajectory.csv',index=False)
comparisons=[]
temp=selected['temperature']
for method in [selected['logits'],selected['hidden'],'logits_0.25','hidden_0.25']:
    for task in ['all','arc_challenge','openbookqa','mmlu']:
        subset=[r for r in rows if task=='all' or r['task']==task or task=='mmlu' and r['task'].startswith('mmlu_')]
        a=[r['predictions'][temp]['correct'] for r in subset];b=[r['predictions'][method]['correct'] for r in subset]
        comparisons.append(dict(task=task,method=method,control=temp,n=len(subset),**paired_ci(a,b)))
pd.DataFrame(comparisons).to_csv(ROOT/'results/loopcd_vs_temperature.csv',index=False)
math=ROOT/'trajectories/math_generation.jsonl'
if math.exists():
    data=[json.loads(l) for l in math.read_text(encoding='utf-8').splitlines()]
    extra=ROOT/'trajectories/math_depth_and_hidden.jsonl'
    if extra.exists():data += [json.loads(l) for l in extra.read_text(encoding='utf-8').splitlines()]
    base={r['id']:r for r in data if r['method']=='baseline'};out=[]
    for method in sorted({r['method'] for r in data}):
        subset=[r for r in data if r['method']==method];assert len(subset)==100
        a=np.array([base[r['id']]['correct'] for r in subset]);b=np.array([r['correct'] for r in subset])
        def numeric(row):
            if row['prediction'] is None:return False
            try:return Decimal(row['prediction'].replace(',',''))==Decimal(row['gold'].replace(',',''))
            except InvalidOperation:return False
        out.append(dict(method=method,n=len(subset),accuracy=float(b.mean()),numeric_accuracy=float(np.mean([numeric(r) for r in subset])),wrong_to_correct=int((~a&b).sum()),correct_to_wrong=int((a&~b).sum()),**paired_ci(a,b),
                        generated_tokens=sum(r['new_tokens'] for r in subset),truncated=sum(r['reached_token_limit'] for r in subset),identity_reused=sum(r.get('identity_reused',False) for r in subset),
                        total_elapsed_s=sum(r['elapsed_s'] for r in subset),mean_elapsed_s=float(np.mean([r['elapsed_s'] for r in subset])),process_peak_memory_bytes_upper_bound=max(r['peak_memory_bytes'] for r in subset)))
    pd.DataFrame(out).to_csv(ROOT/'results/math_paired_metrics.csv',index=False)
    aligned={(r['id'],r['method']):r for r in data}
    methods=['depth_1','depth_2','depth_3','baseline']
    if all((uid,method) in aligned for uid in base for method in methods):
        paths=[]
        for uid in base:
            entry=dict(id=uid,gold=base[uid]['gold'],scope='different autoregressive prefixes across depths')
            for i,method in enumerate(methods,1):
                entry['prediction_r'+str(i)]=aligned[uid,method]['prediction'];entry['correct_r'+str(i)]=aligned[uid,method]['correct']
            paths.append(entry)
        pd.DataFrame(paths).to_csv(ROOT/'results/math_fixed_depth_paths.csv',index=False)
        masks=trajectory_pattern_masks([[r['correct_r'+str(i)] for i in range(1,5)] for r in paths])
        patterns=[]
        for name,mask in masks.items():
            lo,hi=wilson(int(mask.sum()),len(paths))
            patterns.append(dict(pattern=name,n=len(paths),count=int(mask.sum()),rate=float(mask.mean()),ci_low=lo,ci_high=hi,scope='different autoregressive prefixes; no same-prefix mechanistic claim'))
        pd.DataFrame(patterns).to_csv(ROOT/'results/math_generation_patterns.csv',index=False)
    teacherforced=ROOT/'trajectories/ouro_math_teacherforced.jsonl'
    if teacherforced.exists():
        tf=[json.loads(l) for l in teacherforced.read_text(encoding='utf-8').splitlines()]
        assert len(tf)==100 and {r['id'] for r in tf}==set(base)
        means=[]
        for depth in range(4):
            states=[r['trajectory'][depth] for r in tf]
            entry=dict(depth=depth+1,n=len(states),scope='same prompt endpoint; gold reference solution likelihood; separate autoregressive final-answer linkage')
            for key in ['reference_solution_nll_per_token','confidence','entropy','margin','full_vocab_kl','hidden_relative_l2','hidden_cosine']:
                values=[s[key] for s in states if s[key] is not None]
                entry['mean_'+key]=float(np.mean(values)) if values else None
            means.append(entry)
        pd.DataFrame(means).to_csv(ROOT/'results/math_teacherforced_diagnostics.csv',index=False)
        absolute=[]
        for depth in [2,3]:
            for metric,cuts in [('full_vocab_kl',[.01,.05,.1]),('hidden_relative_l2',[.05,.1,.25])]:
                for cut in cuts:
                    part=[r for r in tf if r['trajectory'][depth-1][metric]<=cut]
                    fixed=sum(not r['trajectory'][depth-1]['generated_answer_correct'] and r['trajectory'][-1]['generated_answer_correct'] for r in part)
                    lost=sum(r['trajectory'][depth-1]['generated_answer_correct'] and not r['trajectory'][-1]['generated_answer_correct'] for r in part)
                    lo,hi=wilson(fixed,len(part))
                    absolute.append(dict(depth=depth,metric=metric,absolute_cut=cut,n=len(part),generated_final_corrections=fixed,generated_final_degradations=lost,ci_low=lo,ci_high=hi,scope='post-hoc prompt endpoint metric versus final answers from different autoregressive prefixes; no intervention'))
        pd.DataFrame(absolute).to_csv(ROOT/'results/math_prompt_stability_audit.csv',index=False)
