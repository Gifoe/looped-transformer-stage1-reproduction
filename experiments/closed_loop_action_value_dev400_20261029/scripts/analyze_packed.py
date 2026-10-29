"""Locked development analysis; no sign changes, threshold search or fitting."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.special import log_softmax
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'frozen_loop_update_value_seed2021_v1/scripts'))
from analysis_core import paired,holm
ALPHAS=['0.0','-1.0','-0.5','0.5']
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summarize(rows):
    n=len(rows);gold=np.array([r['gold'] for r in rows]);pred=np.array([[r['actions'][a]['prediction'] for a in ALPHAS] for r in rows]);right=pred==gold[:,None]
    selected=np.array([ALPHAS.index(str(r['decision']['alpha'])) for r in rows]);conf=np.array([ALPHAS.index(str(r['decision']['confidence_alpha'])) for r in rows])
    chosen=right[np.arange(n),selected];baseline=right[:,0]
    terminal=np.array([r['terminal'][str(r['decision']['alpha'])]['prediction']==r['gold'] for r in rows])
    policies={'adaptive':baseline,**{'fixed_'+a:right[:,i] for i,a in enumerate(ALPHAS[1:],1)},'confidence':right[np.arange(n),conf],'terminal_only':terminal,'probe':chosen,'oracle':right.any(1)}
    summary={name:paired(y,baseline) for name,y in policies.items()}
    comparisons={name:paired(chosen,y) for name,y in policies.items() if name not in ['probe','oracle']}
    adjusted=holm({name:s['mcnemar_p'] for name,s in comparisons.items()})
    available=np.array(['losses' in r['decision'] for r in rows]);idx=np.flatnonzero(available);rng=np.random.default_rng(2021)
    random_correct=[]
    for _ in range(10000):
        actions=selected.copy();actions[idx]=rng.permutation(actions[idx]);random_correct.append(int(right[np.arange(n),actions].sum()))
    random_correct=np.array(random_correct);observed=int(chosen.sum());p=float((1+(random_correct>=observed).sum())/10001)
    signals={};events={}
    for j,a in enumerate(ALPHAS[1:],1):
        delta=right[:,j].astype(int)-baseline.astype(int)
        score=np.array([min(r['decision']['losses']['0.0']['A']-r['decision']['losses'][a]['A'],r['decision']['losses']['0.0']['B']-r['decision']['losses'][a]['B']) if 'losses' in r['decision'] else np.nan for r in rows])
        mask=available & (delta!=0);pos=np.flatnonzero(mask&(delta>0));neg=np.flatnonzero(mask&(delta<0))
        auc=None;ci=None
        if len(pos) and len(neg):
            auc=float(roc_auc_score(delta[mask]>0,score[mask]));boot=[];rr=np.random.default_rng(2021)
            for _ in range(2000):
                ix=np.concatenate([rr.choice(pos,len(pos)),rr.choice(neg,len(neg))]);boot.append(roc_auc_score(delta[ix]>0,score[ix]))
            ci=np.quantile(boot,[.025,.975]).tolist()
        loggain=np.array([float(log_softmax(np.array(r['actions'][a]['sum_scores'])/r['choice_char_lengths'])[r['gold']]-log_softmax(np.array(r['actions']['0.0']['sum_scores'])/r['choice_char_lengths'])[r['gold']]) for r in rows])
        rho=spearmanr(score[available],loggain[available]).statistic if available.sum()>2 else np.nan
        signals[a]=dict(beneficial=len(pos),harmful=len(neg),auc=auc,auc_ci95=ci,spearman_normalized_gold_logweight=float(rho) if np.isfinite(rho) else None)
        events[a]=dict(beneficial=int((delta>0).sum()),harmful=int((delta<0).sum()),neutral=int((delta==0).sum()),beneficial_action_selected=int(((delta>0)&(selected==j)).sum()),beneficial_action_not_selected=int(((delta>0)&(selected!=j)).sum()),harmful_action_selected=int(((delta<0)&(selected==j)).sum()),harmful_action_not_selected=int(((delta<0)&(selected!=j)).sum()))
    timing={}
    baseline_s=sum(r['actions']['0.0']['total_seconds']-r['actions']['0.0']['terminal_diagnostic_seconds'] for r in rows)
    chosen_s=sum(r['decision_seconds']+r['actions'][str(r['decision']['alpha'])]['total_seconds']-r['actions'][str(r['decision']['alpha'])]['terminal_diagnostic_seconds'] for r in rows)
    timing=dict(baseline_branch_seconds=baseline_s,controller_replay_seconds=chosen_s,controller_to_baseline_ratio=chosen_s/baseline_s,interpretation='Serial replay estimate including confidence diagnostics in decision overhead; not optimized serving performance')
    passed=all(s['ci95_pp'][0]>0 and adjusted[name]<.05 for name,s in comparisons.items()) and p<.05
    return dict(n=n,summary_vs_adaptive=summary,probe_vs_comparators=comparisons,holm_six_p=adjusted,selected_action_counts={a:int((selected==i).sum()) for i,a in enumerate(ALPHAS)},probe_available=int(available.sum()),paired_action_events=events,signals=signals,random=dict(draws=10000,mean_correct=float(random_correct.mean()),correct_interval95=np.quantile(random_correct,[.025,.975]).tolist(),observed_correct=observed,upper_tail_p=p),timing=timing,passes_development_controller_screen=passed)

def main():
    lock=load(ROOT/'configs/LOCK.json')
    for rel,h in lock['files'].items():assert sha(ROOT/rel)==h,rel
    complete=load(ROOT/'results/COMPLETE_PACKED.json');assert complete['n']==400 and complete['sha256']==sha(ROOT/'results/development_packed.jsonl')
    with (ROOT/'results/development_packed.jsonl').open(encoding='utf8') as f:rows=[json.loads(l) for l in f]
    assert [r['id'] for r in rows]==lock['ids']
    result=dict(scope='Exposed development intervention evidence, not independent confirmation',full400=summarize(rows),exclude_implementation_pilot392=summarize(rows[8:]),settings_changed=False,final_tasks_evaluated=False)
    (ROOT/'results/ANALYSIS.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))
if __name__=='__main__':main()
