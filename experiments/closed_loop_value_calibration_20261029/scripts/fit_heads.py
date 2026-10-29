"""Fit only tiny calibration heads on training intervention outcomes."""
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'frozen_loop_update_value_seed2021_v1/scripts'))
import runtime_bootstrap
import numpy as np
from sklearn.linear_model import Ridge
from value_core import ALPHAS,KINDS,features,observation,choose
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf8')
def main():
    start=time.monotonic();lock=load(ROOT/'configs/LOCK.json')
    for rel,h in lock['files'].items():assert sha(ROOT/rel)==h,rel
    assert not (ROOT/'configs/heads.json').exists(),'No refit after model freeze'
    assert not (ROOT/'results/validation.jsonl').exists(),'Validation must not precede model freeze'
    path=ROOT/'results/head_train.jsonl';complete=load(ROOT/'results/head_train_COMPLETE.json')
    assert complete['n']==3200 and sha(path)==complete['sha256']
    with path.open(encoding='utf8') as f:rows=[json.loads(l) for l in f]
    assert [r['id'] for r in rows]==lock['manifests']['head_train']['ids']
    available=np.array(['losses' in r['decision'] for r in rows])
    gold=np.array([r['gold'] for r in rows]);pred=np.array([[r['actions'][a]['prediction'] for a in ALPHAS] for r in rows]);right=pred==gold[:,None]
    target=right[:,1:].astype(int)-right[:,0,None].astype(int)
    counts={a:dict(beneficial=int((target[available,j]>0).sum()),harmful=int((target[available,j]<0).sum()),neutral=int((target[available,j]==0).sum())) for j,a in enumerate(ALPHAS[1:])}
    if any(min(x['beneficial'],x['harmful'])<20 for x in counts.values()):
        save(ROOT/'results/INSUFFICIENT_SUPERVISION.json',dict(events=counts,n_available=int(available.sum()),minimum_per_class_per_action=20));raise RuntimeError('Insufficient genuine intervention supervision; no head fitted')
    models={};fitrows=[r for r,v in zip(rows,available) if v]
    for kind in KINDS:
        actions={}
        for j,a in enumerate(ALPHAS[1:]):
            X=np.stack([features(observation(r),kind,a) for r in fitrows]);mean=X.mean(0);scale=X.std(0);scale[scale<1e-8]=1
            Z=(X-mean)/scale;fit=Ridge(alpha=100).fit(Z,target[available,j]);prediction=Z@fit.coef_+fit.intercept_
            names=['input_A_gain','input_B_gain'] if kind=='input' else ['margin_gain','entropy_reduction'] if kind=='confidence' else ['input_A_gain','input_B_gain','margin_gain','entropy_reduction']
            actions[a]=dict(features=names,mean=mean.tolist(),scale=scale.tolist(),coef=fit.coef_.tolist(),intercept=float(fit.intercept_),training_quintile_boundaries=np.quantile(prediction,[.2,.4,.6,.8]).tolist())
        parameters=sum(len(m['coef'])+1 for m in actions.values());assert parameters=={'input':9,'confidence':9,'combined':15}[kind]
        models[kind]=dict(kind=kind,actions=actions,linear_coefficients_including_intercepts=parameters,standardization_statistics_stored_separately=True,alpha=100,threshold=0,missing_probes='adaptive')
    means=target[available].mean(0);constant=ALPHAS[int(np.argmax(np.r_[0.,means]))];best_fixed=ALPHAS[int(np.argmax(right.sum(0)))]
    path_heads=ROOT/'configs/heads.json';save(path_heads,models)
    with (ROOT/'results/training_head_predictions.jsonl').open('x',encoding='utf8') as out:
        for r in rows:
            predictions={k:dict(zip(['action','values'],choose(observation(r),m))) for k,m in models.items()}
            out.write(json.dumps(dict(id=r['id'],heads=predictions))+'\n')
    report=dict(n=len(rows),n_available=int(available.sum()),training_action_events=counts,best_fixed_train=best_fixed,constant_action_on_available=constant,training_mean_gains=means.tolist(),model_sha256=sha(path_heads),cpu_seconds=time.monotonic()-start)
    save(ROOT/'results/FIT.json',report)
    freeze=dict(heads_sha256=sha(path_heads),training_outcomes_sha256=sha(path),source_lock_sha256=sha(ROOT/'configs/LOCK.json'),best_fixed_train=best_fixed,constant_action_on_available=constant,training_mean_gains=means.tolist(),training_head_predictions_sha256=sha(ROOT/'results/training_head_predictions.jsonl'),frozen_unix=time.time())
    save(ROOT/'configs/TRAIN_LOCK.json',freeze);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
