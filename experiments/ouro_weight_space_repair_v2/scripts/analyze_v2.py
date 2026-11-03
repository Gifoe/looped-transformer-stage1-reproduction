"""CPU-only final paired analysis. Refuses incomplete or unfrozen evaluations."""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from runtime import ROOT, configure, save
configure();os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')

METHODS=['native_k3','native_k4','native_k5','native_k6','loopcd_k4_fixed','loopcd_k4_adaptive']
LABELS={'native_k3':'Native K3','native_k4':'Native K4','native_k5':'Native K5','native_k6':'Native K6',
        'loopcd_k4_fixed':'Fixed LoopCD K4','loopcd_k4_adaptive':'Adaptive LoopCD K4'}


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def rows(p):return [json.loads(l) for l in Path(p).read_text(encoding='utf8').splitlines() if l]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def writecsv(path,items):
    path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in items for k in r))
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(items)


def validate(data):
    assert len({r['id'] for r in data})==len(data)
    for r in data:
        assert set(METHODS)<=set(r['methods'])
        for name in METHODS:
            v=r['methods'][name];s=np.asarray(v['sum_scores'],dtype=float)
            z=np.asarray(v['normalized_scores'],dtype=float)
            assert np.isfinite(s).all() and np.isfinite(z).all()
            lengths=np.asarray(r['choice_char_lengths'],dtype=float)
            assert len(s)==len(z)==len(lengths) and len(s)>=2
            assert np.array_equal(z,s/np.maximum(lengths,1))
            assert int(z.argmax())==v['prediction'] and bool(v['correct'])==(v['prediction']==r['gold'])
            assert abs(v['margin']-float(z[r['gold']]-np.delete(z,r['gold']).max()))<1e-12


def paired(before,after,method,scope):
    from scipy.stats import binomtest
    b=np.array([r['methods'][method]['correct'] for r in before],dtype=int)
    a=np.array([r['methods'][method]['correct'] for r in after],dtype=int)
    n=len(a);assert n==len(b)>0
    rescue=int(((b==0)&(a==1)).sum());damage=int(((b==1)&(a==0)).sum())
    seed=2021+int(hashlib.sha256((scope+'|'+method).encode()).hexdigest()[:8],16)
    # Sampling paired (rescue,damage,unchanged) categories is exactly equivalent
    # to iid question bootstrap for mean(correct_after-correct_before).
    boot=np.random.default_rng(seed).multinomial(n,[rescue/n,damage/n,1-(rescue+damage)/n],size=50000)
    delta=100*(boot[:,0]-boot[:,1])/n;lo,hi=np.quantile(delta,[.025,.975])
    bp=np.array([r['methods'][method]['prediction'] for r in before]);ap=np.array([r['methods'][method]['prediction'] for r in after])
    return dict(scope=scope,method=method,n=n,original_correct=int(b.sum()),edited_correct=int(a.sum()),
        corrections=rescue,regressions=damage,net_correct=rescue-damage,delta_accuracy_pp=100*(rescue-damage)/n,
        original_accuracy_percent=100*float(b.mean()),edited_accuracy_percent=100*float(a.mean()),
        both_correct=int(((a==1)&(b==1)).sum()),both_wrong=int(((a==0)&(b==0)).sum()),
        answer_changes=int((bp!=ap).sum()),wrong_to_different_wrong=int(((a==0)&(b==0)&(bp!=ap)).sum()),
        rescue_rate_on_original_errors=rescue/int((b==0).sum()) if (b==0).any() else None,
        damage_rate_on_original_correct=damage/int((b==1).sum()) if (b==1).any() else None,
        paired_bootstrap_95_low_pp=float(lo),paired_bootstrap_95_high_pp=float(hi),
        exact_mcnemar_p=1. if rescue+damage==0 else float(binomtest(rescue,rescue+damage,.5).pvalue),
        mean_delta_gold_margin=float(np.mean([a['methods'][method]['margin']-b['methods'][method]['margin'] for b,a in zip(before,after)])),
        bootstrap_replicates=50000,bootstrap_seed=seed,bootstrap_unit='question',
        interval='pointwise paired percentile 95%; not simultaneous',
        interpretation='locked regression benchmark; previously used; no blind-test claim' if scope=='test' else 'search/selection evidence, not confirmatory')


def adjust_holm(items,key):
    ordered=sorted(items,key=lambda r:r['exact_mcnemar_p']);running=0.
    for i,r in enumerate(ordered):
        running=max(running,min(1.,r['exact_mcnemar_p']*(len(ordered)-i)));r[key]=running


def align(before,after):
    b={r['id']:r for r in before};a={r['id']:r for r in after}
    assert len(b)==len(a)==len(before)==len(after) and set(b)==set(a)
    original=[b[r['id']] for r in after]
    for x,y in zip(original,after):
        assert x['gold']==y['gold'] and x['prompt_sha256']==y['prompt_sha256']
        assert x['choice_char_lengths']==y['choice_char_lengths']
        assert x['choice_token_counts']==y['choice_token_counts']
    validate(original);validate(after)
    return original,after


def plot(summary,contrasts):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['DejaVu Sans','sans-serif'],'font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.3,
        'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    red,blue='#B64342','#0F4D92'
    fig,axs=plt.subplots(1,2,figsize=(10,3.9))
    for model,color,marker in [('Original Ouro',red,'s'),('Edited Ouro',blue,'o')]:
        v=[r for r in summary if r['model']==model and r['readout']=='native']
        axs[0].plot([r['K'] for r in v],[r['accuracy_percent'] for r in v],color=color,marker=marker,label=model,lw=2)
    for model,color,offset in [('Original Ouro',red,-.045),('Edited Ouro',blue,.045)]:
        fixed=next(r for r in summary if r['model']==model and r['readout']=='fixed_loopcd')
        axs[0].scatter([4+offset],[fixed['accuracy_percent']],marker='*',s=120,color=color,edgecolors='black',linewidths=.4,zorder=4)
    axs[0].set(xlabel='Recurrent depth K',ylabel='ARC-Challenge acc_norm (%)',xticks=[3,4,5,6],
               title='(a) 1,172-question locked regression')
    axs[0].legend(loc='lower left',fontsize=9)
    axs[0].text(.03,.88,'Stars: Fixed LoopCD (K4 only)',transform=axs[0].transAxes,va='top',fontsize=8)
    show=[next(c for c in contrasts if c['method']==name) for name in METHODS]
    labels=['K3','K4','K5','K6','Fixed\nK4','Adaptive\nK4'];x=np.arange(len(show))
    vals=np.array([c['delta_accuracy_pp'] for c in show]);low=np.array([c['paired_bootstrap_95_low_pp'] for c in show]);high=np.array([c['paired_bootstrap_95_high_pp'] for c in show])
    # Percentile bounds can exclude the point estimate in a degenerate discrete
    # sample; draw bounds directly rather than assuming symmetric yerr lengths.
    axs[1].vlines(x,low,high,color=blue,lw=1.5)
    axs[1].scatter(x,vals,color=blue,s=30,zorder=3)
    axs[1].hlines(low,x-.07,x+.07,color=blue);axs[1].hlines(high,x-.07,x+.07,color=blue)
    axs[1].axhline(0,color='#767676',lw=1,ls='--')
    axs[1].set(xticks=x,xticklabels=labels,ylabel='Edited - corresponding original (pp)',
               title='(b) Paired 95% bootstrap intervals')
    fig.tight_layout(pad=1.6)
    for ext in ['pdf','svg','png']:
        p=ROOT/('accuracy_by_loop.pdf' if ext=='pdf' else 'figures/accuracy_by_loop.'+ext)
        p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,dpi=300,facecolor='white')
    plt.close(fig)


def main():
    completed=read(ROOT/'results/PIPELINE_COMPLETE.json');assert completed['status']=='complete'
    final_lock=read(ROOT/'configs/FINAL_LOCK.json');selected=read(ROOT/'selected_edit_manifest.json')
    cid=selected['selected_candidate_id'];assert cid==completed['selected_candidate_id']==final_lock['selected_candidate_id']
    assert sha(ROOT/'selected_edit_manifest.json')==final_lock['selected_manifest_sha256']
    assert sha(ROOT/'edited_weights/selected/manifest.json')==final_lock['bundle_manifest_sha256']
    frozen=read(ROOT/'configs/CANDIDATES_FROZEN.json');specs={s['id']:s for s in frozen['candidates']}
    search_lock=read(ROOT/'configs/SEARCH_LOCK.json')
    stage_candidates={'screen':set(specs),
        'selection':set(read(ROOT/'configs/SELECTION_LOCK.json')['candidates']),
        'validation':set(read(ROOT/'configs/VALIDATION_LOCK.json')['candidates']),'final_test':{cid}}
    data_names={'screen':'discovery_screen','selection':'selection','validation':'validation','final_test':'test'}
    stage_sizes={'screen':64,'selection':192,'validation':299,'final_test':1172}
    stage_data={stage:rows(ROOT/'data'/(name+'.jsonl')) for stage,name in data_names.items()}
    data_hashes={stage:hashlib.sha256(json.dumps(rr,sort_keys=True,separators=(',',':')).encode()).hexdigest() for stage,rr in stage_data.items()}
    for stage,candidates in stage_candidates.items():
        assert len(stage_data[stage])==stage_sizes[stage]
        assert {p.parent.name for p in (ROOT/'results'/stage).glob('*/COMPLETE.json')}==candidates
    original={'screen':rows(ROOT/'results/training_statistics/scores.jsonl')}
    for split in ['selection','validation','test']:
        original[split]=rows(ROOT/'results/original_reference'/split/'scores.jsonl')
        manifest_sha=sha(ROOT/'data'/(split+'.jsonl'))
        assert all(r['data_sha256']==manifest_sha and r['cached'] is False and r['fewshot']==25
                   and r['demonstrations']=='fixed_global_25' for r in original[split])
    assert all(len(rr)==len({r['id'] for r in rr}) for rr in original.values())
    search=[];selected_splits={};edit_audit=[]
    for candidate,spec in specs.items():
        manifest=read(ROOT/'edited_weights/candidates'/candidate/'manifest.json')
        for item in manifest['edits']:
            edit_audit.append(dict(candidate_id=candidate,family=spec['family'],tensor_name=item['tensor_name'],
                shape=json.dumps(item['shape']),actual_changed_coordinates=item['actual_changed_coordinates'],
                intended_support_coordinates=item['intended_support_coordinates'],
                relative_delta_frobenius=item['relative_delta_frobenius'],actual_delta_frobenius=item['actual_delta_frobenius'],
                max_abs_delta=item['max_abs_delta'],base_sha256=item['base_sha256'],edited_sha256=item['edited_sha256'],
                specification=json.dumps(item['specification'],sort_keys=True)))
        for stage in ['screen','selection','validation','final_test']:
            d=ROOT/'results'/stage/candidate
            if not (d/'COMPLETE.json').exists():continue
            receipt=read(d/'COMPLETE.json');rr=rows(d/'scores.jsonl')
            assert candidate in stage_candidates[stage]
            assert receipt['n']==len(rr)==stage_sizes[stage] and sha(d/'scores.jsonl')==receipt['scores_sha256']
            assert receipt['candidate_id']==candidate and receipt['stage']==stage
            assert receipt['implementation_sha256']==search_lock['implementation_sha256']
            assert receipt['stage_data_sha256']==data_hashes[stage]
            assert receipt['bundle_manifest_sha256']==sha(ROOT/'edited_weights/candidates'/candidate/'manifest.json')
            assert {r['id'] for r in rr}=={r['id'] for r in stage_data[stage]}
            for r in rr:
                assert r['scope']==stage and r['candidate_id']==candidate
                assert r['implementation_sha256']==receipt['implementation_sha256']
                assert r['bundle_manifest_sha256']==receipt['bundle_manifest_sha256']
                assert r['stage_data_sha256']==data_hashes[stage]
                assert set(r['methods'])=={f'native_k{k}' for k in range(1,7)}|{'loopcd_k4_fixed','loopcd_k4_adaptive'}
            if stage=='final_test':assert candidate==cid
            split='test' if stage=='final_test' else stage
            ref={r['id']:r for r in original[split]};before=[ref[r['id']] for r in rr]
            before,rr=align(before,rr)
            pair=paired(before,rr,'native_k4',split)
            item=dict(candidate_id=candidate,family=spec['family'],description=spec['description'],stage=stage,n=len(rr),
                selected_final=candidate==cid,actual_changed_parameters=manifest['actual_changed_parameters'],
                changed_fraction_of_model=manifest['actual_changed_fraction_of_model'],delta_frobenius=manifest['actual_delta_frobenius'])
            for method in METHODS:
                item[method+'_correct']=sum(r['methods'][method]['correct'] for r in rr)
                item[method+'_accuracy_percent']=100*item[method+'_correct']/len(rr)
            item.update({k:pair[k] for k in ['original_correct','corrections','regressions','net_correct','delta_accuracy_pp','mean_delta_gold_margin']})
            search.append(item)
            if candidate==cid:selected_splits[split]=(before,rr)
    assert len([r for r in search if r['stage']=='screen'])==14
    assert len([r for r in search if r['stage']=='selection'])==4
    assert len([r for r in search if r['stage']=='validation'])==3
    assert len([r for r in search if r['stage']=='final_test'])==1
    writecsv('edit_search_results.csv',search);writecsv('candidate_edit_audit.csv',edit_audit)
    before,after=selected_splits['test'];assert len(before)==len(after)==1172
    contrasts=[paired(before,after,m,'test') for m in METHODS]
    adjust_holm(contrasts,'holm_p_all6_comparisons')
    adjust_holm([r for r in contrasts if r['method'] in ['native_k4','loopcd_k4_fixed']],'holm_p_two_primary_comparisons')
    writecsv('paired_statistics.csv',contrasts)
    summary=[];wide=[];transitions=[];failures=[]
    test_docs={r['id']:r for r in rows(ROOT/'data/test.jsonl')}
    for model,rr in [('Original Ouro',before),('Edited Ouro',after)]:
        for readout,names in [('native',METHODS[:4]),('fixed_loopcd',['loopcd_k4_fixed']),('adaptive_loopcd',['loopcd_k4_adaptive'])]:
            role='K4_primary_K3_K5_K6_secondary' if readout=='native' else 'primary' if readout=='fixed_loopcd' else 'secondary'
            w=dict(method=model+('' if readout=='native' else ' + '+readout),n=1172,role=role)
            for k in [3,4,5,6]:w['K'+str(k)+'_correct']='NA';w['K'+str(k)+'_accuracy_percent']='NA'
            for method in names:
                k=int(method[-1]) if method.startswith('native') else 4
                c=sum(r['methods'][method]['correct'] for r in rr)
                summary.append(dict(model=model,readout=readout,method=method,K=k,n=1172,correct=c,accuracy_percent=100*c/1172,
                    role='primary_contrast' if method in ['native_k4','loopcd_k4_fixed'] else 'secondary'))
                w['K'+str(k)+'_correct']=c;w['K'+str(k)+'_accuracy_percent']=100*c/1172
            wide.append(w)
    for method in METHODS:
        for b,a in zip(before,after):
            x,y=b['methods'][method],a['methods'][method]
            transition=('C' if x['correct'] else 'W')+'_to_'+('C' if y['correct'] else 'W')
            r=dict(id=a['id'],method=method,transition=transition,gold=a['gold'],
                original_prediction=x['prediction'],edited_prediction=y['prediction'],
                original_correct=x['correct'],edited_correct=y['correct'],
                original_margin=x['margin'],edited_margin=y['margin'],delta_margin=y['margin']-x['margin'],
                original_normalized_scores=json.dumps(x['normalized_scores']),edited_normalized_scores=json.dumps(y['normalized_scores']),
                prompt_sha256=a['prompt_sha256'],selected_candidate_id=cid)
            transitions.append(r)
            if x['prediction']!=y['prediction']:
                doc=test_docs[a['id']]
                failures.append(dict(**r,correctness_changed=x['correct']!=y['correct'],question=doc['question'],choices=json.dumps(doc['choices'],ensure_ascii=False)))
    writecsv('final_comparison.csv',wide);writecsv('final_comparison_long.csv',summary)
    writecsv('paired_transitions.csv',transitions);writecsv('changed_answer_cases.csv',failures)
    generalization=[]
    for split,(b,a) in selected_splits.items():
        generalization.extend(paired(b,a,m,split) for m in ['native_k4','loopcd_k4_fixed','loopcd_k4_adaptive'])
    writecsv('generalization_by_split.csv',generalization)
    save('results/FINAL_ANALYSIS.json',dict(status='complete',selected_candidate_id=cid,n_test=1172,
        search_rows=len(search),paired_rows=len(transitions),contrasts=contrasts,summary=summary,
        generalization=generalization,selected_edit=selected,
        bootstrap='50000 paired question resamples, implemented by equivalent multinomial transition counts',
        coefficients_frozen=True,no_test_selection=True,original_baseline='V1 exact fixed25/native-call BF16; not rerun in V2',
        warning='Official test is a previously used locked regression benchmark, not blind.'))
    plot(summary,contrasts)
    print(json.dumps(dict(status='complete',candidate=cid,contrasts=contrasts),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
