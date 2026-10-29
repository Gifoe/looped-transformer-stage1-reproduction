"""Read-only failure analysis of sealed heads; no fitting or policy selection.

Training/validation labels are used only for target decomposition. Final-task
rows have gold removed immediately and contribute observable diagnostics only.
No alternative policy accuracy, threshold sweep or final-label calibration.
"""
import json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'frozen_loop_update_value_seed2021_v1'
sys.path.insert(0,str(BASE/'scripts'))
from analysis_core import features,ACTIONS
def load(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(rows,models,allow_labels):
 pred=np.array([[int(np.argmax(np.asarray(r['views']['official']['sum_scores'][a])/r['choice_char_lengths'])) for a in ACTIONS] for r in rows])
 different=pred[:,1:]!=pred[:,0,None]
 X={k:np.stack([features(r,k)[0] for r in rows]) for k in ['controls','new','combined']}
 result={'n':len(rows),'answer_disagreements':dict(zip(ACTIONS[1:],map(int,different.sum(0)))),'heads':{},'feature_shift':{}}
 if allow_labels:
  gold=np.array([r['gold'] for r in rows]);correct=pred==gold[:,None];delta=correct[:,1:].astype(int)-correct[:,0,None].astype(int)
  assert np.all(delta[~different]==0)
  result['target_decomposition']={a:dict(same_prediction=int((~different[:,i]).sum()),different_prediction=int(different[:,i].sum()),beneficial=int((delta[:,i]>0).sum()),harmful=int((delta[:,i]<0).sum()),both_wrong_disagreements=int((different[:,i]&(delta[:,i]==0)).sum())) for i,a in enumerate(ACTIONS[1:])}
 for name,m in models.items():
  xx=(X[m['kind']]-m['mean'])/m['scale'];v=xx@np.array(m['coef']).T+m['intercept'];best=v.argmax(1);maxv=v[np.arange(len(rows)),best];override=maxv>m['threshold'];chosen_diff=different[np.arange(len(rows)),best]
  eligible=(v>m['threshold'])&different
  result['heads'][name]=dict(override_questions=int(override.sum()),answer_changed_questions=int((override&chosen_diff).sum()),same_answer_overrides=int((override&~chosen_diff).sum()),blocked_different_answer_above_threshold=int((override&~chosen_diff&eligible.any(1)).sum()),any_different_answer_above_threshold=int(eligible.any(1).sum()),positive_value_for_same_answer_pairs=int(((v>m['threshold'])&~different).sum()),predicted_gain_quantiles=np.quantile(maxv,[0,.25,.5,.75,.95,.99,1]).tolist(),target=m['target'],interpretation='Exact zero-accuracy-gain identity applies only to correctness-delta heads; log-weight targets can change while answers agree')
  if m['target']=='accuracy':
   z=np.abs(xx);names=m['features'];frac=(z>5).mean(0);order=np.argsort(-frac)[:8]
   result['feature_shift'][m['kind']]=dict(abs_standardized_quantiles=np.quantile(z,[.5,.95,.99,.999]).tolist(),question_fraction_with_any_abs_z_over_5=float((z>5).any(1).mean()),top_features_by_abs_z_over_5=[dict(feature=names[i],fraction=float(frac[i])) for i in order])
 return result
def main():
 lock=load(BASE/'configs/FINAL_LOCK.json')
 for rel,h in lock['frozen_files'].items():assert sha(BASE/rel)==h
 models=load(BASE/'configs/heads.json');reports={};inputs={}
 for split in ['head_train','validation','final_sciq','final_piqa']:
  path=BASE/'results'/(split+'.jsonl');allow_labels=split in ['head_train','validation'];rows=[]
  with path.open(encoding='utf8') as f:
   for line in f:
    row=json.loads(line)
    if not allow_labels:row.pop('gold',None)
    rows.append(row)
  if not allow_labels:assert all('gold' not in r for r in rows)
  reports[split]=audit(rows,models,allow_labels);inputs[split]=sha(path)
 result=dict(scope='Post-completion failure audit, not confirmatory evidence; no new policy evaluated, no fitting, final labels excluded',source_commit='3fe2bac590c515056966cd7add12e62fd8951cee',heads_sha256=sha(BASE/'configs/heads.json'),input_sha256=inputs,datasets=reports,additional_gpu_seconds=0)
 (ROOT/'DECISION_DIAGNOSTICS.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 for split,r in reports.items():
  print(split,json.dumps({k:v for k,v in r['heads'].items() if k.endswith('_accuracy')}))
if __name__=='__main__':main()
