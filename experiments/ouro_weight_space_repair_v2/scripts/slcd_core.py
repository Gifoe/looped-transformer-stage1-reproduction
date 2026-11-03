"""Train-only development math and auditable, immutable method definitions."""
from pathlib import Path
import csv,hashlib,json,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REVISION='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'


def read(path):return json.loads((ROOT/path).read_text(encoding='utf8'))
def rows(path):return [json.loads(l) for l in (ROOT/path).read_text(encoding='utf8').splitlines() if l]
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(path,value):
    p=(ROOT/path).resolve();assert p.is_relative_to(ROOT.resolve()) and p.parent.is_dir()
    assert p.name.startswith('slcd_') or p.name in ['SELF_ITERATION_LOG.md','FINAL_METHOD_LOCK.json']
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8');tmp.replace(p)


def writecsv(path,values,fields=None):
    p=ROOT/path;assert p.parent.is_dir() and p.name.startswith('slcd_')
    fields=fields or list(dict.fromkeys(k for r in values for k in r))
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)


def load_development(split):
    assert split in ['discovery','selection'],'Development cannot open validation/test'
    data=rows('data/slcd_'+split+'.jsonl');scores=rows('results/slcd_original_'+split+'.jsonl')
    n={'discovery':384,'selection':192}[split]
    assert len(data)==len(scores)==len({r['id'] for r in data})==len({r['id'] for r in scores})==n
    source={r['id']:r for r in scores};out=[]
    for r in data:
        s=source[r['id']];assert r['gold']==s['gold'] and r['prompt_sha256']==s['prompt_sha256']
        assert r['choice_char_lengths']==s['choice_char_lengths']
        m={}
        for key,value in s['methods'].items():
            if key not in [f'native_k{k}' for k in range(1,7)]+['loopcd_k4_fixed','loopcd_k4_adaptive']:continue
            x=np.asarray(value['sum_scores'],dtype=float)/np.maximum(np.asarray(r['choice_char_lengths']),1)
            assert np.isfinite(x).all() and np.array_equal(x,np.asarray(value['normalized_scores']))
            assert int(x.argmax())==value['prediction'] and (value['prediction']==r['gold'])==value['correct']
            m[key]=x
        out.append(dict(id=r['id'],gold=r['gold'],methods=m,source=r,saved=s))
    return out


def top_margin(x):
    x=np.sort(x);return float(x[-1]-x[-2])


def scores_for(row,spec,registry,token=None):
    kind=spec['kind'];m=row['methods']
    if kind=='baseline':return m[spec['method']].copy()
    if kind=='score_late':
        fixed=m['loopcd_k4_fixed']
        return fixed.copy() if spec['gamma']==0 else fixed+spec['gamma']*(m['native_k4']-m['native_k'+str(spec.get('late',5))])
    if kind=='token':
        assert token is not None
        return np.asarray(token[row['id']]['methods'][spec['recipe_id']]['normalized_scores'],dtype=float)
    if kind=='mix':
        a=scores_for(row,registry[spec['a']],registry,token);b=scores_for(row,registry[spec['b']],registry,token)
        return (1-spec['weight'])*a+spec['weight']*b
    if kind=='gate':
        a=scores_for(row,registry[spec['a']],registry,token);fallback=scores_for(row,registry[spec['fallback']],registry,token)
        k4=m['native_k4'];k5=m['native_k5'];cd=m['loopcd_k4_fixed']
        signal=spec['signal'];threshold=spec.get('threshold',0.)
        if signal=='k4_k5_disagree':use=int(k4.argmax())!=int(k5.argmax())
        elif signal=='k4_cd_disagree':use=int(k4.argmax())!=int(cd.argmax())
        elif signal=='cd_agrees_k5':use=int(cd.argmax())==int(k5.argmax())
        elif signal=='k4_margin_below':use=top_margin(k4)<=threshold
        elif signal=='cd_margin_below':use=top_margin(cd)<=threshold
        elif signal=='late_opposes_cd':
            j=int(cd.argmax());i=int(k4.argmax());use=(k4[j]-k5[j])-(k4[i]-k5[i])>=threshold
        else:raise ValueError('Unknown deterministic inference signal '+signal)
        return a if use else fallback
    if kind=='trajectory_reference':
        return m['native_k4']+spec['omega']*(m['native_k4']-m['native_k'+str(spec['reference'])])
    raise ValueError(kind)


def summarize(data,spec,registry,token=None):
    from scipy.stats import binomtest
    outcomes=[];rescues=regressions=0;margins=[];before_margins=[];answer_changes=0
    for r in data:
        x=scores_for(r,spec,registry,token);assert len(x)==len(r['methods']['native_k4']) and np.isfinite(x).all()
        y=r['gold'];prediction=int(x.argmax());base=r['methods']['loopcd_k4_fixed'];bp=int(base.argmax())
        good=prediction==y;old=bp==y;rescues+=int(good and not old);regressions+=int(old and not good)
        margin=float(x[y]-np.delete(x,y).max());bm=float(base[y]-np.delete(base,y).max())
        margins.append(margin);before_margins.append(bm);answer_changes+=prediction!=bp
        outcomes.append(dict(id=r['id'],prediction=prediction,correct=good,base_correct=old,margin=margin,base_margin=bm))
    n=len(data);correct=sum(v['correct'] for v in outcomes);original=sum(v['base_correct'] for v in outcomes)
    return dict(n=n,correct=correct,baseline_correct=original,accuracy=correct/n,corrections=rescues,regressions=regressions,
        net_correct=rescues-regressions,delta_accuracy_pp=100*(correct-original)/n,
        mean_gold_margin=float(np.mean(margins)),mean_delta_margin=float(np.mean(np.asarray(margins)-before_margins)),
        answer_changes=int(answer_changes),exact_mcnemar_p=1. if rescues+regressions==0 else float(binomtest(rescues,rescues+regressions,.5).pvalue),
        required_depth=spec['required_depth'],outcomes=outcomes)


def register(registry,spec):
    assert spec['id'] not in registry or registry[spec['id']]==spec
    assert spec['required_depth'] in [4,5,6]
    registry[spec['id']]=spec


def add_iteration(number,hypothesis,evidence,changes,results,decision,next_action):
    p=ROOT/'SELF_ITERATION_LOG.md';assert p.parent.is_dir()
    if not p.exists():p.write_text('# Training-free LoopCD self-iteration\n\nDevelopment uses only train discovery384 / selection192. Official validation is reserved for one final selection stage. Historical test726 is not selection evidence.\n',encoding='utf8')
    text=f'\n## Iteration {number}\n\nHypothesis: {hypothesis}\n\nMotivating evidence: {evidence}\n\nModification/configuration: {changes}\n\nObserved development results: {json.dumps(results,ensure_ascii=False)}\n\nDecision: {decision}\n\nNext action: {next_action}\n'
    with p.open('a',encoding='utf8') as f:f.write(text)
    path=ROOT/'results/slcd_hypotheses_and_decisions.csv'
    record=dict(iteration=number,hypothesis=hypothesis,evidence=evidence,changes=changes,results=json.dumps(results),decision=decision,next_action=next_action)
    exists=path.exists()
    with path.open('a',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(record));
        if not exists:w.writeheader()
        w.writerow(record)


def evaluate_registry(ids,registry,token_by_split=None):
    results=[]
    for split in ['discovery','selection']:
        data=load_development(split);token=(token_by_split or {}).get(split)
        for name in ids:
            r=summarize(data,registry[name],registry,token);r.pop('outcomes')
            results.append(dict(split=split,method=name,spec_sha256=digest(registry[name]),**r))
    return results


def development_order(name,results,registry):
    r={v['split']:v for v in results if v['method']==name}
    gains=[r[s]['net_correct']/r[s]['n'] for s in ['discovery','selection']]
    return (-min(gains),-r['selection']['net_correct'],-r['discovery']['net_correct'],registry[name]['required_depth'],name)
