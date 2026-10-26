"""Deterministic CPU reconstruction; validation selection precedes test use."""
import argparse
import hashlib
import json
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from common import ROOT,BASE,CFG,read,save
from policies import METRICS,LOWER,exit_index,bf16_scalar

CATEGORIES=['arc_challenge','openbookqa','mmlu']
category=lambda row:'mmlu' if row['task'].startswith('mmlu_') else row['task']

def paired_ci(a,b):
    delta=np.asarray(b,dtype=float)-np.asarray(a,dtype=float);rng=np.random.default_rng(2021);values=[]
    for start in range(0,10000,200):
        samples=rng.integers(0,len(delta),size=(200,len(delta)));values.extend(delta[samples].mean(1))
    return dict(delta=float(delta.mean()),ci_low=float(np.quantile(values,.025)),ci_high=float(np.quantile(values,.975)))

def wilson(k,n):
    if not n:return None,None
    z=1.959963984540054;p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(center-half),float(center+half)

def groups(rows):
    result={'all':rows}
    for task in CATEGORIES:result[task]=[row for row in rows if category(row)==task]
    for task in sorted({row['task'] for row in rows if row['task'].startswith('mmlu_')}):result[task]=[row for row in rows if row['task']==task]
    return result

def quality(base,correct):
    base=np.asarray(base,dtype=bool);correct=np.asarray(correct,dtype=bool)
    result=dict(n=len(base),accuracy=float(correct.mean()),wrong_to_correct=int((~base&correct).sum()),correct_to_wrong=int((base&~correct).sum()),correct_to_correct=int((base&correct).sum()),wrong_to_wrong=int((~base&~correct).sum()),**paired_ci(base,correct))
    for transition in ['wrong_to_correct','correct_to_wrong','correct_to_correct','wrong_to_wrong']:
        lo,hi=wilson(result[transition],len(base))
        result[transition+'_rate']=result[transition]/len(base)
        result[transition+'_rate_ci_low']=lo;result[transition+'_rate_ci_high']=hi
    return result

def csv(name,rows):pd.DataFrame(rows).to_csv(ROOT/'results'/name,index=False)

def qwen_analysis():
    data={split:{} for split in ['validation','test']}
    for split in data:
        for method in CFG['qwen_methods']:
            rows=read(f'trajectories/Qwen3-4B-Base_{method}_{split}.jsonl')
            assert len(rows)==(250 if split=='validation' else 600)
            data[split][method]={r['id']:r for r in rows};assert len(data[split][method])==len(rows)
        assert all(set(rows)==set(data[split]['A0']) for rows in data[split].values())
    selected={}
    for task in CATEGORIES:
        selected[task]=min(['dola_low','dola_high'],key=lambda method:(-np.mean([r['result']['correct'] for r in data['validation'][method].values() if category(r)==task]),['dola_low','dola_high'].index(method)))
    aligned=data['test'];ids=list(aligned['A0'])
    aligned['dola_validation_selected']={uid:aligned[selected[category(aligned['A0'][uid])]][uid] for uid in ids}
    table=[]
    for group,base_rows in groups(list(aligned['A0'].values())).items():
        subset=[r['id'] for r in base_rows];base=[r['result']['correct'] for r in base_rows]
        for method,lookup in aligned.items():
            rows=[lookup[uid] for uid in subset];correct=[r['result']['correct'] for r in rows]
            nll=[r['result']['gold_nll_per_token'] for r in rows if r['result']['gold_nll_per_token'] is not None]
            table.append(dict(model='Qwen3-4B-Base',task=group,method=method,**quality(base,correct),
                mean_gold_nll_per_token=float(np.mean(nll)) if nll else None,
                mean_vocab_entropy=float(np.mean([r['full_vocab_stats']['entropy'] for r in rows])),total_elapsed_s=sum(r['elapsed_s'] for r in rows),
                peak_memory_bytes=max(r['peak_memory_bytes'] for r in rows),theoretical_layer_ratio=1 if method in ['A0','dola_low','dola_high','dola_validation_selected'] else 1+(1 if method=='A1' else 5)*4/36,
                time_note='Separate complete scoring runs; selected DoLa composes validation-selected per-task pools'))
    csv('qwen_metrics.csv',table)
    return selected

def threshold_grid(validation,metric):
    if metric=='native_gate_cdf':return np.linspace(.01,1,100).tolist()
    values=[state[metric] for row in validation for state in row['trajectory'][:-1] if state.get(metric) is not None]
    return np.unique(np.quantile(values,np.linspace(.005,.995,51))).tolist()

def policy(rows,metric,threshold,include_ci=True):
    indices=[exit_index(row['trajectory'],metric,threshold) for row in rows]
    correct=np.array([row['trajectory'][index]['correct'] for row,index in zip(rows,indices)],dtype=bool)
    base=np.array([row['trajectory'][-1]['correct'] for row in rows],dtype=bool)
    result=dict(n=len(rows),accuracy=float(correct.mean()),average_depth=float(np.mean(indices)+1),
                recurrent_work_saved=float(1-(np.mean(indices)+1)/4),missed_final_corrections=int((~correct&base).sum()),avoided_final_degradations=int((correct&~base).sum()))
    if include_ci:result.update(quality(base,correct))
    return result,indices

def halting(model,validation,test):
    curves=[];operating=[];selection={};matched=[]
    for metric in METRICS:
        print(json.dumps(dict(stage='halting_analysis',model=model,metric=metric)),flush=True)
        selection[metric]={}
        for task in CATEGORIES:
            vv=[r for r in validation if category(r)==task];tt=[r for r in test if category(r)==task]
            points=[]
            for threshold in threshold_grid(vv,metric):
                stats,_=policy(vv,metric,threshold,False);result,_=policy(tt,metric,threshold,False)
                entry=dict(model=model,task=task,method=metric,threshold=float(threshold),effective_threshold=bf16_scalar(threshold) if metric=='native_gate_cdf' else float(threshold),validation_accuracy=stats['accuracy'],validation_average_depth=stats['average_depth'],**result)
                points.append(entry);curves.append(entry)
            selection[metric][task]={}
            for target in CFG['halting']['validation_budget_targets']:
                feasible=[point for point in points if point['validation_average_depth']<=target+1e-9]
                if not feasible:continue
                best=min(feasible,key=lambda point:(-point['validation_accuracy'],point['validation_average_depth'],point['threshold']))
                selection[metric][task][str(target)]=best['threshold']
                result,_=policy(tt,metric,best['threshold']);operating.append(dict(model=model,task=task,method=metric,validation_budget=target,threshold=best['threshold'],validation_accuracy=best['validation_accuracy'],validation_average_depth=best['validation_average_depth'],**result))
        for target in CFG['halting']['validation_budget_targets']:
            if not all(str(target) in selection[metric][task] for task in CATEGORIES):continue
            thresholds={task:selection[metric][task][str(target)] for task in CATEGORIES}
            indices=[exit_index(row['trajectory'],metric,thresholds[category(row)]) for row in test]
            correct=[row['trajectory'][index]['correct'] for row,index in zip(test,indices)];base=[row['trajectory'][-1]['correct'] for row in test]
            result=dict(model=model,task='all',method=metric,validation_budget=target,thresholds=json.dumps(thresholds),average_depth=float(np.mean(indices)+1),recurrent_work_saved=float(1-(np.mean(indices)+1)/4),**quality(base,correct))
            result.update(missed_final_corrections=result['correct_to_wrong'],avoided_final_degradations=result['wrong_to_correct']);operating.append(result)
            # Random assignment stays within benchmark, preserving each exact
            # benchmark-specific exit-depth histogram as well as total compute.
            seed=2021+int(hashlib.sha256((model+metric+str(target)).encode()).hexdigest()[:8],16)
            rng=np.random.default_rng(seed);outcomes=np.array([[s['correct'] for s in row['trajectory']] for row in test]);depths=np.array(indices);values=[]
            positions=[np.array([i for i,r in enumerate(test) if category(r)==task]) for task in CATEGORIES]
            for _ in range(10000):
                shuffled=depths.copy()
                for pos in positions:shuffled[pos]=rng.permutation(depths[pos])
                values.append(outcomes[np.arange(len(test)),shuffled].mean())
            matched.append(dict(model=model,method=metric,validation_budget=target,average_depth=result['average_depth'],accuracy=result['accuracy'],matched_random_mean=float(np.mean(values)),matched_random_central95_low=float(np.quantile(values,.025)),matched_random_central95_high=float(np.quantile(values,.975))))
    return curves,operating,selection,matched

def trajectory_diagnostics(model,rows):
    patterns=[];stability=[];predictability=[]
    for task,subset in groups(rows).items():
        paths=np.asarray([[s['correct'] for s in row['trajectory']] for row in subset],dtype=bool)
        masks={'early_correct_final_correct':paths[:,0]&paths[:,-1],'early_correct_final_wrong':paths[:,0]&~paths[:,-1],
            'early_wrong_final_correct':~paths[:,0]&paths[:,-1],'early_wrong_final_wrong':~paths[:,0]&~paths[:,-1],
            'correct_wrong_correct':np.array([p[3] and any(p[a] and not p[b] for a in range(2) for b in range(a+1,3)) for p in paths]),
            'wrong_correct_wrong':np.array([not p[3] and any(not p[a] and p[b] for a in range(2) for b in range(a+1,3)) for p in paths])}
        for name,mask in masks.items():
            lo,hi=wilson(int(mask.sum()),len(subset));patterns.append(dict(model=model,task=task,pattern=name,n=len(subset),count=int(mask.sum()),rate=float(mask.mean()),ci_low=lo,ci_high=hi))
        for depth in [2,3]:
            for metric,cuts in [('full_vocab_kl',[.001,.01,.05,.1]),('choice_kl',[.0001,.001,.01]),('hidden_relative_l2',[.01,.05,.1,.25])]:
                for threshold in cuts:
                    part=[row for row in subset if row['trajectory'][depth-1][metric]<=threshold]
                    fixed=sum(not row['trajectory'][depth-1]['correct'] and row['trajectory'][-1]['correct'] for row in part)
                    lost=sum(row['trajectory'][depth-1]['correct'] and not row['trajectory'][-1]['correct'] for row in part)
                    lo,hi=wilson(fixed,len(part));stability.append(dict(model=model,task=task,depth=depth,metric=metric,threshold=threshold,n=len(part),final_corrections=fixed,final_degradations=lost,ci_low=lo,ci_high=hi,scope='post-hoc descriptive absolute cut; not a selected policy'))
            for metric in METRICS:
                part=[row for row in subset if not row['trajectory'][depth-1]['correct'] and row['trajectory'][depth-1].get(metric) is not None]
                y=np.array([row['trajectory'][-1]['correct'] for row in part],dtype=bool);score=np.array([row['trajectory'][depth-1][metric] for row in part])
                if metric in ['confidence','choice_confidence','logit_margin','choice_logit_margin','native_gate_cdf']:score=-score
                npos=int(y.sum());nneg=len(y)-npos
                auc=None if not npos or not nneg else float((rankdata(score)[y].sum()-npos*(npos+1)/2)/(npos*nneg))
                predictability.append(dict(model=model,task=task,depth=depth,metric=metric,n_current_wrong=len(part),final_corrections=npos,auc=auc,auc_reversed_sign=None if auc is None else 1-auc,orientation='negative raw value' if metric in ['confidence','choice_confidence','logit_margin','choice_logit_margin','native_gate_cdf'] else 'positive raw value',scope='descriptive single-variable ranking among currently wrong; both signs shown, no selected orientation, fitted predictor or cross-task generalization claim'))
    return patterns,stability,predictability

def ouro_analysis():
    metrics=[];all_curves=[];all_operating=[];randoms=[];selected={};halt_selected={};patterns=[];stable=[];auc=[];geometry=[]
    for model in ['Ouro-1.4B','Ouro-2.6B']:
        print(json.dumps(dict(stage='ouro_analysis',model=model)),flush=True)
        validation=read(f'trajectories/{model}_validation.jsonl');test=read(f'trajectories/{model}_test.jsonl')
        assert len(validation)==250 and len(test)==300
        selected[model]={space:{} for space in ['logits','hidden']};selected[model]['temperature']={}
        for task in CATEGORIES:
            vv=[r for r in validation if category(r)==task]
            for space in ['logits','hidden']:
                strength=min(CFG['loopcd_strength_grid'],key=lambda w:(-np.mean([r['predictions'][f'{space}_{w:g}']['correct'] for r in vv]),w))
                selected[model][space][task]=strength
            selected[model]['temperature'][task]=min(CFG['temperature_grid'],key=lambda value:(-np.mean([r['predictions'][f'temperature_{value:g}']['correct'] for r in vv]),abs(value-1)))
        for row in test:
            task=category(row)
            for space in ['logits','hidden']:row['predictions'][space+'_validation_selected']=row['predictions'][f'{space}_{selected[model][space][task]:g}']
            row['predictions']['temperature_validation_selected']=row['predictions'][f'temperature_{selected[model]["temperature"][task]:g}']
        for task,subset in groups(test).items():
            base=[r['predictions']['r4']['correct'] for r in subset]
            for method in subset[0]['predictions']:
                values=[r['predictions'][method] for r in subset]
                metrics.append(dict(model=model,task=task,method=method,**quality(base,[v['correct'] for v in values]),mean_gold_nll_per_token=float(np.mean([v['gold_nll_per_token'] for v in values])),mean_gold_choice_log_weight=float(np.mean([v['gold_choice_log_weight'] for v in values]))))
            geometry.append(dict(model=model,task=task,n=len(subset),hidden_vs_orthogonal_cosine=float(np.mean([r['geometry']['hidden_vs_orthogonal_cosine'] for r in subset])),hidden_vs_full_cosine=float(np.mean([r['geometry']['hidden_vs_full_cosine'] for r in subset]))))
        print(json.dumps(dict(stage='ouro_metrics_complete',model=model)),flush=True)
        curves,operating,selection,matched=halting(model,validation,test)
        all_curves+=curves;all_operating+=operating;randoms+=matched;halt_selected[model]=selection
        p,s,a=trajectory_diagnostics(model,test);patterns+=p;stable+=s;auc+=a
    for name,rows in [('ouro_metrics.csv',metrics),('halting_curves.csv',all_curves),('halting_selected.csv',all_operating),('halting_matched_random.csv',randoms),('trajectory_patterns.csv',patterns),('absolute_stability.csv',stable),('continuation_auc.csv',auc),('projection_geometry.csv',geometry)]:csv(name,rows)
    save('configs/halting_selected.json',halt_selected)
    return selected

def math_selection():
    selected={};grid_results=[]
    for model in ['Ouro-1.4B','Ouro-2.6B']:
        rows=read(f'trajectories/math_{model}_validation.jsonl');assert len(rows)==900
        assert len({(r['id'],r['method']) for r in rows})==900
        for method in CFG['math']['validation_methods']:
            part=[r for r in rows if r['method']==method];assert len(part)==100
            grid_results.append(dict(model=model,method=method,n=len(part),accuracy=float(np.mean([r['correct'] for r in part])),numeric_accuracy=float(np.mean([r['numeric_correct'] for r in part])),truncated=sum(r['reached_token_limit'] for r in part),mean_new_tokens=float(np.mean([r['new_tokens'] for r in part])),total_elapsed_s=sum(r['elapsed_s'] for r in part),scope='validation only; baseline tracing adds heads, times are budget costs not clean speedups'))
        selected[model]={}
        for space in ['logits','hidden']:
            candidates=[(0,'baseline')]+[(w,f'{space}_{w:g}') for w in CFG['loopcd_strength_grid'] if w]
            strength,method=min(candidates,key=lambda pair:(-np.mean([r['correct'] for r in rows if r['method']==pair[1]]),pair[0]))
            selected[model][space]=dict(omega=strength,validation_method=method,validation_accuracy=float(np.mean([r['correct'] for r in rows if r['method']==method])))
    save('configs/math_selected.json',selected)
    csv('math_validation_grid.csv',grid_results)
    print(json.dumps(selected),flush=True)

def math_analysis():
    table=[];token_rows=[]
    execution=ROOT/'configs/math_execution.json';count=json.loads(execution.read_text())['test_count'] if execution.exists() else CFG['math']['test_count']
    for model in ['Ouro-1.4B','Ouro-2.6B']:
        rows=read(f'trajectories/math_{model}_test.jsonl');assert len(rows)==count*len(CFG['math']['test_methods'])
        baseline={r['id']:r for r in rows if r['method']=='baseline'}
        for method in CFG['math']['test_methods']:
            subset=[r for r in rows if r['method']==method];assert len(subset)==count and {r['id'] for r in subset}==set(baseline)
            table.append(dict(model=model,method=method,**quality([baseline[r['id']]['correct'] for r in subset],[r['correct'] for r in subset]),numeric_accuracy=float(np.mean([r['numeric_correct'] for r in subset])),truncated=sum(r['reached_token_limit'] for r in subset),identity_reused=sum(bool(r.get('identity_reused_from')) for r in subset),total_elapsed_s=sum(r['elapsed_s'] for r in subset),generated_tokens=sum(r['new_tokens'] for r in subset),process_peak_memory_bytes_upper_bound=max(r['peak_memory_bytes'] for r in subset)))
        for row in baseline.values():
            for token in row.get('token_trajectory',[]):
                states=token['by_depth']
                for depth in [2,3]:
                    token_rows.append(dict(model=model,id=row['id'],token_position=token['position'],depth=depth,final_answer_correct=row['correct'],full_vocab_kl=states[depth-1]['full_vocab_kl'],hidden_relative_l2=states[depth-1]['hidden_relative_l2'],next_token_agrees_final=states[depth-1]['prediction']==states[-1]['prediction'],scope='next-token agreement, not gold-token correctness'))
    csv('math_metrics.csv',table);csv('math_token_diagnostics.csv',token_rows)

def draw():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':160,'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def finish(name):
        plt.tight_layout()
        for extension in ['png','pdf']:plt.savefig(ROOT/'figures'/(name+'.'+extension))
        plt.close()
    q=pd.read_csv(ROOT/'results/qwen_metrics.csv');fig,ax=plt.subplots(figsize=(10,4))
    for task in CATEGORIES:
        p=q[(q.task==task)&(q.method!='dola_validation_selected')];ax.plot(p.method,p.accuracy,'o-',label=task)
    ax.set(ylabel='Primary benchmark accuracy',xlabel='Frozen Qwen method');ax.legend();finish('figure6_qwen_training_free_baselines')
    o=pd.read_csv(ROOT/'results/ouro_metrics.csv');fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    selected_methods=['r4','logits_0.5','adaptive_paper','hidden_0.5','parallel_05','orthogonal_05','temperature_validation_selected']
    method_labels=['Fixed R4','Logits .5','Adaptive L','Hidden .5','Parallel .5','Orthogonal .5','Temp (val)']
    for ax,model in zip(axes,['Ouro-1.4B','Ouro-2.6B']):
        p=o[(o.model==model)&(o.task=='all')].set_index('method').loc[selected_methods]
        ax.bar(range(len(p)),100*p.delta);ax.errorbar(range(len(p)),100*p.delta,yerr=np.stack([100*(p.delta-p.ci_low),100*(p.ci_high-p.delta)]),fmt='none',c='black',capsize=3)
        ax.axhline(0,c='gray',lw=.7);ax.set_xticks(range(len(p)),method_labels,rotation=35,ha='right');ax.set(title=model,ylabel='Accuracy change (pp)')
    finish('figure7_loopcd_paper_and_geometry')
    h=pd.read_csv(ROOT/'results/halting_selected.csv');fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,model in zip(axes,['Ouro-1.4B','Ouro-2.6B']):
        for metric in ['full_vocab_kl','choice_kl','choice_entropy','hidden_relative_l2','native_gate_cdf']:
            p=h[(h.model==model)&(h.method==metric)&(h.task=='all')].sort_values('average_depth')
            ax.plot(p.average_depth,p.accuracy,'o-',markersize=3,label=metric)
        fixed=o[(o.model==model)&(o.task=='all')&o.method.isin(['r1','r2','r3','r4'])].sort_values('method')
        ax.plot([1,2,3,4],fixed.accuracy,'ko-',label='fixed, pooled')
        ax.set(title=model+'; validation-selected budgets',xlabel='Pooled average recurrent depth',ylabel='Primary benchmark accuracy');ax.legend(fontsize=7)
    finish('figure8_validation_frozen_halting')
    if (ROOT/'results/math_metrics.csv').exists():
        m=pd.read_csv(ROOT/'results/math_metrics.csv');fig,ax=plt.subplots(figsize=(11,4))
        for model,p in m.groupby('model'):ax.plot(p.method,p.accuracy,'o-',label=model)
        ax.tick_params(axis='x',labelrotation=35);ax.set(ylabel='GSM8K strict exact match');ax.legend();finish('figure9_actual_autoregressive_generation')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['mc','select_math','all'],required=True);args=parser.parse_args()
    if args.stage=='select_math':math_selection();return
    if args.stage in ['mc','all']:
        print(json.dumps(dict(stage='qwen_analysis')),flush=True)
        q=qwen_analysis();o=ouro_analysis();save('configs/mc_selected.json',dict(qwen_dola=q,ouro=o))
    if args.stage=='all':math_selection();math_analysis()
    draw();print(json.dumps(dict(stage=args.stage,completed=True)),flush=True)

if __name__=='__main__':main()
