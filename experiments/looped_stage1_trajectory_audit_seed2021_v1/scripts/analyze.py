"""Rebuild statistics and figures exclusively from aligned raw JSONL records."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/protocol.json').read_text(encoding='utf-8-sig'))

def read(name):
    rows=[json.loads(s) for s in (ROOT/'trajectories'/name).read_text(encoding='utf-8').splitlines()]
    gate_path=ROOT/'trajectories/ouro_native_gate.jsonl'
    gates={r['id']:r['exit_cdf'] for r in (json.loads(s) for s in gate_path.read_text(encoding='utf-8').splitlines())} if gate_path.exists() else {}
    for row in rows:
        for state in row.get('trajectory',[]):
            scores=sorted(row['predictions']['r'+str(state['depth'])]['scores'])
            state['choice_logit_margin']=scores[-1]-scores[-2]
            if row['id'] in gates:state['native_gate_cdf']=gates[row['id']][state['depth']-1]
    return rows

def paired_ci(a,b,seed=2021):
    delta=np.asarray(b,dtype=float)-np.asarray(a,dtype=float)
    rng=np.random.default_rng(seed);means=[]
    for start in range(0,10000,200):
        idx=rng.integers(0,len(delta),size=(min(200,10000-start),len(delta)))
        means.extend(delta[idx].mean(1))
    return dict(delta=float(delta.mean()),ci_low=float(np.quantile(means,.025)),ci_high=float(np.quantile(means,.975)))

def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(center-half),float(center+half)]

def trajectory_pattern_masks(paths):
    paths=np.asarray(paths,dtype=bool)
    assert paths.ndim==2 and paths.shape[1]==4
    return {'early_correct_final_correct':paths[:,0]&paths[:,-1],
            'early_correct_final_wrong':paths[:,0]&~paths[:,-1],
            'early_wrong_final_correct':~paths[:,0]&paths[:,-1],
            'early_wrong_final_wrong':~paths[:,0]&~paths[:,-1],
            'correct_wrong_correct':np.array([p[3] and any(p[a] and not p[b] for a in range(2) for b in range(a+1,3)) for p in paths]),
            'wrong_correct_wrong':np.array([not p[3] and any(not p[a] and p[b] for a in range(2) for b in range(a+1,3)) for p in paths])}

def policy_index(trajectory,metric,threshold):
    for i,state in enumerate(trajectory[:-1]):
        value=state.get(metric)
        if value is None:continue
        if (value<=threshold if metric in ['full_vocab_kl','choice_kl','hidden_relative_l2'] else value>=threshold):return i
    return len(trajectory)-1

def policy(rows,metric,threshold):
    indices=[policy_index(x['trajectory'],metric,threshold) for x in rows]
    correct=np.array([x['trajectory'][i]['correct'] for x,i in zip(rows,indices)],dtype=bool)
    baseline=np.array([x['trajectory'][-1]['correct'] for x in rows],dtype=bool)
    return dict(accuracy=float(correct.mean()),average_depth=float(np.mean(indices)+1),
                theoretical_recurrent_compute_reduction=float(1-(np.mean(indices)+1)/4),
                missed_final_corrections=int((~correct&baseline).sum()),avoided_final_degradations=int((correct&~baseline).sum()),
                wrong_to_correct=int((correct&~baseline).sum()),correct_to_wrong=int((~correct&baseline).sum()),
                **paired_ci(baseline,correct)),indices

def selections(rows):
    # No test record enters this function.
    selected={}
    for prefix in ['logits_','hidden_']:
        candidates=[prefix+str(x) for x in CFG['loopcd_strength_grid']]
        selected[prefix.rstrip('_')]=min(candidates,key=lambda k:(-np.mean([r['predictions'][k]['correct'] for r in rows]),float(k[len(prefix):])))
    candidates=['temp_'+str(x) for x in CFG['temperature_grid']]
    selected['temperature']=min(candidates,key=lambda k:(-np.mean([r['predictions'][k]['correct'] for r in rows]),np.mean([r['predictions'][k]['gold_nll_per_token'] for r in rows]),abs(float(k[5:])-1)))
    return selected

def summary(rows):
    result=[];base=np.array([x['predictions']['r4']['correct'] for x in rows])
    for method in rows[0]['predictions']:
        correct=np.array([x['predictions'][method]['correct'] for x in rows])
        item=dict(method=method,n=len(rows),accuracy=float(correct.mean()),
                  correct_to_wrong=int((base&~correct).sum()),wrong_to_correct=int((~base&correct).sum()),
                  correct_to_correct=int((base&correct).sum()),wrong_to_wrong=int((~base&~correct).sum()),
                  mean_gold_nll_per_token=float(np.mean([x['predictions'][method]['gold_nll_per_token'] for x in rows])),
                  mean_choice_entropy=float(np.mean([-sum(p*np.log(p) for p in x['predictions'][method]['choice_probabilities']) for x in rows])),
                  mean_gold_choice_logp=float(np.mean([x['predictions'][method]['gold_choice_logp'] for x in rows])),**paired_ci(base,correct))
        result.append(item)
    return result

def main():
    validation=read('ouro_validation.jsonl');test=read('ouro_test.jsonl')
    assert not {r['id'] for r in validation}&{r['id'] for r in test}
    assert len({r['id'] for r in test})==len(test)
    selected=selections(validation)
    (ROOT/'configs/validation_selected.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    groups={'all':test}
    for row in test:
        groups.setdefault(row['task'],[]).append(row)
        if row['task'].startswith('mmlu_'):groups.setdefault('mmlu',[]).append(row)
    metrics=[]
    for group,rows in groups.items():
        metrics.extend(dict(task=group,**r) for r in summary(rows))
    pd.DataFrame(metrics).to_csv(ROOT/'results/ouro_metrics.csv',index=False)
    pd.DataFrame(summary(validation)).to_csv(ROOT/'results/ouro_validation_metrics.csv',index=False)
    transitions=[];failure_cases=[]
    for group,rows in groups.items():
        paths=np.array([[t['correct'] for t in r['trajectory']] for r in rows])
        patterns=trajectory_pattern_masks(paths)
        for name,mask in patterns.items():
            count=int(mask.sum());lo,hi=wilson(count,len(rows))
            transitions.append(dict(task=group,pattern=name,n=len(rows),count=count,rate=count/len(rows),ci_low=lo,ci_high=hi))
            if group=='all':
                for i in np.flatnonzero(mask)[:5]:failure_cases.append(dict(pattern=name,**rows[i]))
    pd.DataFrame(transitions).to_csv(ROOT/'results/trajectory_patterns.csv',index=False)
    (ROOT/'results/failure_cases.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in failure_cases),encoding='utf-8')
    curves=[];threshold_grid={}
    halting_metrics=['full_vocab_kl','choice_kl','choice_confidence','choice_margin','choice_logit_margin','hidden_relative_l2']
    if all('native_gate_cdf' in r['trajectory'][0] for r in validation+test):halting_metrics.append('native_gate_cdf')
    for metric in halting_metrics:
        values=[s[metric] for row in validation for s in row['trajectory'][:-1] if s[metric] is not None]
        thresholds=np.linspace(.05,1,20).tolist() if metric=='native_gate_cdf' else np.unique(np.quantile(values,np.linspace(0,1,31))).tolist()
        thresholds=[float('-inf')]+thresholds+[float('inf')]
        threshold_grid[metric]=thresholds
        for threshold in thresholds:
            val,_=policy(validation,metric,threshold);te,indices=policy(test,metric,threshold)
            curves.append(dict(method=metric,threshold=str(threshold),validation_accuracy=val['accuracy'],validation_average_depth=val['average_depth'],**te))
    # Select deployable operating points using only validation and target budget.
    operating=[]
    for metric in threshold_grid:
        for target in [1.5,2,2.5,3,3.5,4]:
            candidates=[c for c in curves if c['method']==metric and c['validation_average_depth']<=target+1e-9]
            if not candidates:continue
            winner=min(candidates,key=lambda c:(-c['validation_accuracy'],c['validation_average_depth']))
            operating.append(dict(target_validation_depth=target,**winner))
    pd.DataFrame(curves).to_csv(ROOT/'results/halting_curves.csv',index=False)
    pd.DataFrame(operating).to_csv(ROOT/'results/halting_selected.csv',index=False)
    per_task=[]
    for point in operating:
        for task in ['arc_challenge','openbookqa','mmlu']:
            subset=[r for r in test if r['task']==task or task=='mmlu' and r['task'].startswith('mmlu_')]
            stats,_=policy(subset,point['method'],float(point['threshold']))
            per_task.append(dict(task=task,n=len(subset),method=point['method'],threshold=point['threshold'],target_validation_depth=point['target_validation_depth'],**stats))
    pd.DataFrame(per_task).to_csv(ROOT/'results/halting_selected_by_task.csv',index=False)
    baseline=np.array([r['trajectory'][-1]['correct'] for r in test]);fixed=[]
    for r in range(4):
        c=np.array([x['trajectory'][r]['correct'] for x in test])
        fixed.append(dict(method='fixed',depth=r+1,accuracy=float(c.mean()),**paired_ci(baseline,c)))
    pd.DataFrame(fixed).to_csv(ROOT/'results/fixed_depth.csv',index=False)
    oracle_indices=[next((i for i,s in enumerate(x['trajectory']) if s['correct']),3) for x in test]
    oracle=dict(type='offline oracle: uses gold and future; not deployable',accuracy=float(np.mean([x['trajectory'][i]['correct'] for x,i in zip(test,oracle_indices)])),average_depth=float(np.mean(oracle_indices)+1))
    (ROOT/'results/oracle.json').write_text(json.dumps(oracle,indent=2),encoding='utf-8')
    diagnostic=[]
    for row in test:
        for depth,state in enumerate(row['trajectory'][:-1]):
            future=row['trajectory'][depth+1:]
            diagnostic.append(dict(id=row['id'],task=row['task'],prompt_tokens=row['prompt_tokens'],**state,
                                  future_any_correction=(not state['correct'] and any(s['correct'] for s in future)),
                                  future_any_degradation=(state['correct'] and any(not s['correct'] for s in future)),
                                  final_correction=(not state['correct'] and future[-1]['correct']),
                                  final_degradation=(state['correct'] and not future[-1]['correct'])))
    df=pd.DataFrame(diagnostic);df.to_csv(ROOT/'results/convergence_diagnostics.csv',index=False)
    draw(pd.DataFrame(metrics),pd.DataFrame(transitions),pd.DataFrame(curves),pd.DataFrame(fixed),df,selected)
    print(json.dumps(dict(n_test=len(test),selected=selected,oracle=oracle)),flush=True)

def draw(metrics,transitions,curves,fixed,diagnostics,selected):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def savefig(name,rect=None):
        plt.tight_layout(rect=rect);plt.savefig(ROOT/'figures'/(name+'.png'));plt.savefig(ROOT/'figures'/(name+'.pdf'));plt.close()
    fig,ax=plt.subplots(figsize=(6,4));ax.plot(fixed.depth,fixed.accuracy,'o-');ax.set(xlabel='Recurrent depth',ylabel='Accuracy',xticks=[1,2,3,4]);savefig('figure2_depth_accuracy')
    all_patterns=transitions[transitions.task=='all']
    fig,ax=plt.subplots(figsize=(9,4));ax.barh(all_patterns.pattern,all_patterns.rate);ax.set(xlabel='Fraction of held-out questions');savefig('figure2_prediction_transitions')
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,metric in zip(axes,['full_vocab_kl','hidden_relative_l2','choice_margin']):
        values=diagnostics[diagnostics.depth==2].dropna(subset=[metric]).copy();values['bin']=pd.qcut(values[metric],5,duplicates='drop')
        agg=values.groupby('bin',observed=True)[['final_correction','final_degradation']].mean()
        agg.plot.bar(ax=ax);ax.set(title=metric+' (depth 2)',xlabel='Test quantiles (descriptive only)',ylabel='Fraction changing correctness by depth 4');ax.set_xticklabels(range(1,len(agg)+1),rotation=0)
        ax.get_legend().remove()
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2)
    savefig('figure3_convergence_prediction_changes',rect=(0,0,1,.91))
    fig,ax=plt.subplots(figsize=(7,4.5))
    for method,rows in curves.groupby('method'):
        rows=rows.sort_values('average_depth');ax.scatter(rows.average_depth,rows.accuracy,s=9,alpha=.25)
        frontier=[];best=-1
        for _,point in rows.sort_values(['average_depth','accuracy'],ascending=[True,False]).iterrows():
            if point.accuracy>best+1e-12:frontier.append(point);best=point.accuracy
        labels={'choice_confidence':'Choice confidence','choice_kl':'Choice KL','choice_margin':'Choice probability margin','choice_logit_margin':'Choice score/logit margin','full_vocab_kl':'Vocabulary KL','hidden_relative_l2':'Relative hidden L2','native_gate_cdf':'Released ponder gate'}
        frontier=pd.DataFrame(frontier);ax.plot(frontier.average_depth,frontier.accuracy,'.-',label=labels.get(method,method),alpha=.8)
    oracle=json.loads((ROOT/'results/oracle.json').read_text());ax.scatter([oracle['average_depth']],[oracle['accuracy']],marker='*',s=110,c='black',label='Offline oracle (uses gold)')
    ax.plot(fixed.depth,fixed.accuracy,'ko-',label='Fixed depth');ax.set(xlabel='Average recurrent depth (offline FLOP proxy)',ylabel='Accuracy',title='Validation-frozen thresholds; descriptive test Pareto envelopes');ax.legend(fontsize=8);savefig('figure4_halting_pareto')
    rows=metrics[(metrics.task=='all') & (metrics.method.str.startswith('logits_') | metrics.method.str.startswith('hidden_') | (metrics.method==selected['temperature']))]
    fig,ax=plt.subplots(figsize=(11,4));x=np.arange(len(rows));ax.bar(x-.18,rows.wrong_to_correct/rows.n,.36,label='Wrong → correct');ax.bar(x+.18,rows.correct_to_wrong/rows.n,.36,label='Correct → wrong');ax.set_xticks(x,rows.method,rotation=40,ha='right');ax.set(ylabel='Fraction vs. depth-4 baseline',title='All predeclared strengths; validation selected omega=0');ax.legend();savefig('figure5_loopcd_transitions')

if __name__=='__main__':main()
