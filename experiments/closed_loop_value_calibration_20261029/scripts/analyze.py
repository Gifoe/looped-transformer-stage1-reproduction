"""Precommitted held-out-development analysis of tiny state-value heads."""
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'frozen_loop_update_value_seed2021_v1/scripts'))
import runtime_bootstrap
import numpy as np
from analysis_core import paired,holm
from value_core import ALPHAS,KINDS,observation,choose
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf8')
def ranking(scores,target,available):
    result={}
    for j,a in enumerate(ALPHAS[1:]):
        pos=np.flatnonzero(available&(target[:,j]>0));neg=np.flatnonzero(available&(target[:,j]<0));report=dict(beneficial=len(pos),harmful=len(neg),heads={})
        if len(pos) and len(neg):
            names=list(scores);credit=[]
            for name in names:
                v=scores[name][:,j];x=v[pos,None];y=v[None,neg]
                credit.append((x>y).astype(float)+.5*(x==y))
            credit=np.stack(credit);auc=credit.mean(axis=(1,2));rng=np.random.default_rng(2021);boot=[]
            for _ in range(2000):
                p=rng.integers(len(pos),size=len(pos));n=rng.integers(len(neg),size=len(neg));boot.append(credit[:,p[:,None],n[None,:]].mean(axis=(1,2)))
            boot=np.stack(boot)
            for i,name in enumerate(names):report['heads'][name]=dict(auc=float(auc[i]),ci95=np.quantile(boot[:,i],[.025,.975]).tolist())
            report['paired_auc_difference_vs_confidence']={name:dict(difference=float(auc[names.index(name)]-auc[names.index('confidence')]),ci95=np.quantile(boot[:,names.index(name)]-boot[:,names.index('confidence')],[.025,.975]).tolist()) for name in ['input','combined']}
        else:
            for name in scores:report['heads'][name]=dict(auc=None,ci95=None)
        result[a]=report
    return result
def main():
    start=time.monotonic();lock=load(ROOT/'configs/LOCK.json');freeze=load(ROOT/'configs/TRAIN_LOCK.json');models=load(ROOT/'configs/heads.json')
    for rel,h in lock['files'].items():assert sha(ROOT/rel)==h,rel
    assert freeze['source_lock_sha256']==sha(ROOT/'configs/LOCK.json') and freeze['heads_sha256']==sha(ROOT/'configs/heads.json')
    path=ROOT/'results/validation.jsonl';complete=load(ROOT/'results/validation_COMPLETE.json')
    assert complete['n']==800 and complete['sha256']==sha(path)
    with path.open(encoding='utf8') as f:rows=[json.loads(l) for l in f]
    assert [r['id'] for r in rows]==lock['manifests']['validation']['ids']
    n=len(rows);available=np.array(['losses' in r['decision'] for r in rows]);gold=np.array([r['gold'] for r in rows]);pred=np.array([[r['actions'][a]['prediction'] for a in ALPHAS] for r in rows]);right=pred==gold[:,None];target=right[:,1:].astype(int)-right[:,0,None].astype(int)
    actions={};values={};head_cpu={}
    for kind,m in models.items():
        t=time.perf_counter();choices=[choose(observation(r),m) for r in rows];head_cpu[kind]=time.perf_counter()-t
        actions[kind]=np.array([ALPHAS.index(a) for a,v in choices]);values[kind]=np.array([v[1:] for a,v in choices])
    policies={'adaptive':right[:,0],**{'fixed_'+a:right[:,i] for i,a in enumerate(ALPHAS[1:],1)},'raw_probe':np.array([r['actions'][str(r['decision']['alpha'])]['prediction']==r['gold'] for r in rows]),'raw_confidence':np.array([r['actions'][str(r['decision']['confidence_alpha'])]['prediction']==r['gold'] for r in rows]),'best_fixed_train':right[:,ALPHAS.index(freeze['best_fixed_train'])],'training_mean':right[np.arange(n),np.where(available,ALPHAS.index(freeze['constant_action_on_available']),0)],'oracle':right.any(1)}
    for kind,act in actions.items():
        policies[kind]=right[np.arange(n),act]
        policies['terminal_'+kind]=np.array([r['terminal'][ALPHAS[act[i]]]['prediction']==r['gold'] for i,r in enumerate(rows)])
    summary={name:paired(y,policies['adaptive']) for name,y in policies.items()}
    comparisons={};raw_p={};permutation={};idx=np.flatnonzero(available)
    for kind in ['input','combined']:
        comparisons[kind]={name:paired(policies[kind],policies[name]) for name in ['adaptive','confidence','best_fixed_train']}
        for name,s in comparisons[kind].items():raw_p[f'{kind}_vs_{name}']=s['mcnemar_p']
        rng=np.random.default_rng(2021);counts=[]
        for _ in range(10000):
            act=actions[kind].copy();act[idx]=rng.permutation(act[idx]);counts.append(int(right[np.arange(n),act].sum()))
        counts=np.array(counts);p=float((1+(counts>=policies[kind].sum()).sum())/10001)
        permutation[kind]=dict(draws=10000,mean_correct=float(counts.mean()),correct_interval95=np.quantile(counts,[.025,.975]).tolist(),observed_correct=int(policies[kind].sum()),p=p)
        raw_p[f'{kind}_vs_shuffle']=p
    adjusted=holm(raw_p);eligible=[]
    for kind in ['input','combined']:
        if all(s['net']>=5 and s['ci95_pp'][0]>0 and adjusted[f'{kind}_vs_{name}']<.05 for name,s in comparisons[kind].items()) and adjusted[f'{kind}_vs_shuffle']<.05:eligible.append(kind)
    selected=eligible[0] if eligible else 'adaptive'
    raw_response=np.array([[min(r['decision']['losses']['0.0'][g]-r['decision']['losses'][a][g] for g in ['A','B']) if 'losses' in r['decision'] else np.nan for a in ALPHAS[1:]] for r in rows])
    signals=ranking({**values,'raw_input_response':raw_response},target,available)
    calibration={};mse={}
    for kind in KINDS:
        mse[kind]=float(np.mean((values[kind]-target)**2));bins={}
        for j,a in enumerate(ALPHAS[1:]):
            boundaries=models[kind]['actions'][a]['training_quintile_boundaries'];b=np.searchsorted(boundaries,values[kind][:,j],side='right');entries=[]
            for k in range(5):
                mask=available&(b==k);entries.append(dict(bin=k,n=int(mask.sum()),predicted_mean=float(values[kind][mask,j].mean()) if mask.any() else None,observed_mean=float(target[mask,j].mean()) if mask.any() else None))
            bins[a]=dict(training_boundaries=boundaries,validation_bins=entries)
        calibration[kind]=bins
    mse['zero']=float(np.mean(target**2));means=np.array(freeze['training_mean_gains']);mse['training_mean']=float(np.mean((np.where(available[:,None],means,0)-target)**2))
    baseline_seconds=sum(r['actions']['0.0']['total_seconds']-r['actions']['0.0']['terminal_diagnostic_seconds'] for r in rows);timing={}
    for kind,act in actions.items():
        seconds=sum(r['decision_seconds']+r['actions'][ALPHAS[act[i]]]['total_seconds']-r['actions'][ALPHAS[act[i]]]['terminal_diagnostic_seconds'] for i,r in enumerate(rows))+head_cpu[kind]
        timing[kind]=dict(head_cpu_seconds=head_cpu[kind],controller_replay_seconds=seconds,baseline_seconds=baseline_seconds,ratio=seconds/baseline_seconds)
    report=dict(scope='Head-held-out but historically exposed development validation; not independent confirmation',n=n,n_available=int(available.sum()),summary_vs_adaptive=summary,primary_comparisons=comparisons,primary_holm_eight_p=adjusted,shuffled=permutation,eligible=eligible,selected_policy=selected,selected_action_counts={k:{a:int((v==i).sum()) for i,a in enumerate(ALPHAS)} for k,v in actions.items()},signals=signals,all_row_mse=mse,calibration=calibration,values_outside_possible_range={k:int((np.abs(v)>1).sum()) for k,v in values.items()},terminal_paired={k:paired(policies[k],policies['terminal_'+k]) for k in KINDS},timing=timing,analysis_cpu_seconds=time.monotonic()-start,final_tasks_evaluated=False)
    save(ROOT/'results/VALIDATION_ANALYSIS.json',report)
    with (ROOT/'results/validation_head_predictions.jsonl').open('x',encoding='utf8') as out:
        for i,r in enumerate(rows):out.write(json.dumps(dict(id=r['id'],heads={k:dict(action=ALPHAS[actions[k][i]],values=np.r_[0.,values[k][i]].tolist(),prediction=int(pred[i,actions[k][i]])) for k in KINDS}))+'\n')
    selection=dict(selected_policy=selected,eligible=eligible,reason='Precommitted eight-test validation gate',heads_sha256=sha(ROOT/'configs/heads.json'),train_lock_sha256=sha(ROOT/'configs/TRAIN_LOCK.json'),validation_sha256=sha(path),analysis_sha256=sha(ROOT/'results/VALIDATION_ANALYSIS.json'),frozen_unix=time.time())
    save(ROOT/'configs/SELECTION_LOCK.json',selection);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
