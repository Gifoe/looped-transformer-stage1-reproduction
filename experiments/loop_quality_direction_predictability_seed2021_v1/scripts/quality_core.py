"""Gold-free feature contract and small, serializable development-only gates."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.special import softmax
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'looped_stage1_trajectory_audit_seed2021_v1'
PLAN = json.loads((ROOT/'configs/analysis_plan.json').read_text(encoding='utf8'))


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return [json.loads(line) for line in f if line.strip()]


def put(path, value):
    p=ROOT/path; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group(task):
    return 'mmlu' if task.startswith('mmlu_') else task


def visible(record):
    # Deliberate positive allowlist. Gold and correctness never reach features.
    r={k:record[k] for k in ['id','scores','choice_probabilities','full_vocab_stats',
            'window','prompt_tokens','choices']}
    if 'full_vocab_kl_vs_original' in record:r['full_vocab_kl_vs_original']=record['full_vocab_kl_vs_original']
    return r


def features(original, euler):
    """Takes only the allowlisted inference-visible records, not evaluation labels."""
    a,b=original,euler
    f={}
    pa,pb=map(lambda r:np.asarray(r['choice_probabilities'],float),[a,b])
    assert len(pa)==len(pb)==len(a['choices'])
    for label,r,p in [('original',a,pa),('euler',b,pb)]:
        s=np.asarray(r['scores'],float); order=np.argsort(-s,kind='stable')
        assert np.isclose(p.sum(),1) and np.allclose(softmax(s),p,atol=1e-7)
        f['choice_'+label+'_confidence']=float(p.max())
        f['choice_'+label+'_entropy']=float(-np.sum(p*np.log(np.clip(p,1e-300,None))))
        f['choice_'+label+'_margin']=float(np.sort(p)[-1]-np.sort(p)[-2])
        f['choice_'+label+'_top_score']=float(s[order[0]])
        f['choice_'+label+'_second_score']=float(s[order[1]])
        f['choice_'+label+'_score_gap']=float(s[order[0]]-s[order[1]])
        for k in ['confidence','entropy','margin']:
            f['vocab_'+label+'_'+k]=float(r['full_vocab_stats'][k])
    for stem,metrics in [('choice',['confidence','entropy','margin','top_score','second_score','score_gap']),
                         ('vocab',['confidence','entropy','margin'])]:
        for k in metrics:f[stem+'_'+k+'_delta']=f[stem+'_euler_'+k]-f[stem+'_original_'+k]
    def kl(p,q):return float(np.sum(p*(np.log(p)-np.log(q))))
    mix=(pa+pb)/2
    f.update(choice_kl_euler_original=kl(pb,pa),choice_kl_original_euler=kl(pa,pb),
             choice_js=(kl(pa,mix)+kl(pb,mix))/2,choice_tv=float(np.abs(pa-pb).sum()/2))
    # The historical runner kept original vocabulary vectors only in RAM. After
    # a resume some KL records were absent; missing is not a measured zero.
    f['vocab_kl_euler_original']=float(b['full_vocab_kl_vs_original']) if 'full_vocab_kl_vs_original' in b else None
    oa=np.argsort(-np.asarray(a['scores']),kind='stable');ob=np.argsort(-np.asarray(b['scores']),kind='stable')
    f['prediction_changed']=float(oa[0]!=ob[0])
    f['original_top_rank_in_euler']=float(np.where(ob==oa[0])[0][0]+1)
    f['euler_top_rank_in_original']=float(np.where(oa==ob[0])[0][0]+1)
    delta=np.asarray(b['scores'])-np.asarray(a['scores'])
    f.update(score_delta_mean=float(delta.mean()),score_delta_abs_mean=float(np.abs(delta).mean()),
             score_delta_std=float(delta.std()),score_delta_max=float(delta.max()),score_delta_min=float(delta.min()))
    window=b['window'];assert len(window['steps'])==6
    for dest,source in [('output','input_to_output'),('natural','input_to_natural')]:
        f['hidden_'+dest+'_l2']=float(window[source]['relative_l2'])
        f['hidden_'+dest+'_cosine']=float(window[source]['cosine'])
    step=np.array([x['relative_l2'] for x in window['steps']]);diff=np.diff(step)
    for k,v in [('mean',step.mean()),('max',step.max()),('last',step[-1]),('first',step[0]),
                ('std',step.std()),('slope',np.polyfit(np.arange(6),step,1)[0]),
                ('adjacent_delta_mean',diff.mean()),('adjacent_delta_max',diff.max()),
                ('monotone_increasing',np.all(diff>=0)),('monotone_decreasing',np.all(diff<=0))]:
        f['step_l2_'+k]=float(v)
    f['prompt_tokens']=float(a['prompt_tokens']);assert a['prompt_tokens']==b['prompt_tokens']
    lengths=np.array([len(x.lstrip(' ')) for x in a['choices']])
    f.update(candidate_count=float(len(lengths)),candidate_char_mean=float(lengths.mean()),
             candidate_char_max=float(lengths.max()),candidate_char_std=float(lengths.std()))
    f['random_feature']=int(hashlib.sha256(('2021/random/'+a['id']).encode()).hexdigest()[:13],16)/16**13
    assert all(v is None or np.isfinite(v) for v in f.values())
    return f


def cv_splits(indices, gain, n, seed):
    # g has adequate counts in the 600-question discovery set. Never consult test g.
    counts=np.unique(gain[indices],return_counts=True)[1]
    folds=min(n,int(counts.min()))
    if folds<2:raise ValueError('Insufficient training classes for stratified CV')
    return [(indices[tr],indices[va]) for tr,va in StratifiedKFold(folds,shuffle=True,random_state=seed).split(indices,gain[indices])]


def fit_ridge(X, gain, indices, alpha):
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    scaler=StandardScaler()
    Z=scaler.fit_transform(imputer.fit_transform(X[indices]))
    model=Ridge(alpha=alpha).fit(Z,gain[indices])
    return dict(kind='linear',alpha=alpha,median=imputer.statistics_.tolist(),mean=scaler.mean_.tolist(),
                scale=scaler.scale_.tolist(),coefficient=model.coef_.tolist(),intercept=float(model.intercept_))


def predict_ridge(model,X):
    Z=np.where(np.isfinite(X),X,np.asarray(model['median']))
    return ((Z-np.asarray(model['mean']))/np.asarray(model['scale']))@np.asarray(model['coefficient'])+model['intercept']


def fit_linear(X, gain, indices, seed=2021):
    # Nested validation chooses both penalty and threshold from inner OOF decisions.
    records=[];best=None
    inner=cv_splits(indices,gain,PLAN['inner_folds'],seed)
    for alpha in PLAN['ridge_alpha_grid']:
        score=np.full(len(gain),np.nan)
        for tr,va in inner:
            model=fit_ridge(X,gain,tr,alpha);score[va]=predict_ridge(model,X[va])
        for threshold in PLAN['linear_threshold_grid']:
            objective=int(np.sum(gain[indices]*(score[indices]>threshold)))
            records.append(dict(alpha=alpha,threshold=threshold,net_corrections=objective))
            key=(objective,-threshold,alpha)
            if best is None or key>best[0]:best=(key,alpha,threshold)
    model=fit_ridge(X,gain,indices,best[1]);model['threshold']=best[2];model['selection_records']=records
    model['inner_folds']=[dict(train=tr.tolist(),validation=va.tolist()) for tr,va in inner]
    return model


def fit_gate(F, gain, indices, candidates):
    best=((int(gain[indices].sum()),len(indices),1),'always',None,None,None)
    # Fixed deterministic order and minimum rejection tie-breaking.
    trials=[dict(feature='always_euler',net_corrections=int(gain[indices].sum()))]
    for name in candidates:
        x=np.array([r[name] for r in F])
        for sign in [1,-1]:
            v=sign*x[indices]
            for q in np.linspace(0,1,11):
                threshold=float(np.quantile(v,q));accept=v>threshold
                objective=int(np.sum(gain[indices]*accept))
                trials.append(dict(feature=name,sign=sign,quantile=float(q),threshold=threshold,net_corrections=objective))
                key=(objective,int(accept.sum()),0)
                if key>best[0]:best=(key,'threshold',name,sign,threshold)
    # Never Euler is present as an explicit endpoint.
    if (0,0,0)>best[0]:best=((0,0,0),'never',None,None,None)
    return dict(kind=best[1],feature=best[2],sign=best[3],threshold=best[4],selection_records=trials)


def gate_decisions(model,F):
    if model['kind']=='always':return np.ones(len(F),bool)
    if model['kind']=='never':return np.zeros(len(F),bool)
    return model['sign']*np.array([r[model['feature']] for r in F])>model['threshold']
