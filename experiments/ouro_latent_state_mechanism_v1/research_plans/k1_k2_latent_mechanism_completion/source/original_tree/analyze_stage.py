"""Predeclared AB1 discovery analysis: runs only after full 64-case audits."""
import pathlib,json,hashlib,statistics,math,sys
import numpy as np
from fractions import Fraction
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--stage',required=True,choices=['train64','train1030','validation299']);args=ap.parse_args()
BASE=pathlib.Path(args.root);STAGE=args.stage;SPLIT='validation' if STAGE=='validation299' else 'train';D=BASE/'data'/STAGE;OUT=BASE/'analysis_results'/STAGE;LABEL=BASE/'analysis_inputs/analysis_labels.jsonl';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def gap(v,g,w):return {'allwrong':v[g]-sum(x for i,x in enumerate(v) if i!=g)/(len(v)-1),'fixedwrong':v[g]-v[w],'dynamic':v[g]-max(x for i,x in enumerate(v) if i!=g)}
def main():
 p=json.loads((D/'PENDING.json').read_text());a=json.loads((D/'INDEPENDENT_COVERAGE_AUDIT.json').read_text());assert p['status']=='COMPLETE' and a['status']=='PASS' and a['pending_sha256']==sha(D/'PENDING.json');assert len(p['cases'])==len(a['records'])==p['n'] and [x['id'] for x in p['cases']]==[x['id'] for x in a['records']];freeze_path=BASE/'FULL_SOURCE_FREEZE.json';f=json.loads(freeze_path.read_text());assert sha(freeze_path)==p['source_freeze_sha256'];freeze={'source_sha256':{str(BASE/k):v for k,v in f['sources'].items()}};assert all(sha(k)==v for k,v in freeze['source_sha256'].items());label_sha=sha(LABEL)
 if 'labels_sha256' in p:assert label_sha==p['labels_sha256']
 ys={(z['split'],z['id']):z['gold'] for line in LABEL.read_text().splitlines() if (z:=json.loads(line))};rows=[]
 for c in p['cases']:
  key=hashlib.sha256(c['id'].encode()).hexdigest();mp=D/c['metadata_file'];assert sha(mp)==c['metadata_sha256'];m=json.loads(mp.read_text());g=ys[(SPLIT,c['id'])];k1=m['identity']['K1_scores'];n=m['identity']['N_scores'];g1=k1['prediction'];g2=n['prediction'];group='rescue' if g1!=g and g2==g else 'damage' if g1==g and g2!=g else 'stable_correct' if g1==g else 'stable_wrong';fw=max((i for i in range(len(k1['scores'])) if i!=g),key=lambda i:k1['scores'][i]);r={'id':c['id'],'split':SPLIT,'gold':g,'natural_group':group,'natural_no_tie':len(k1['ties'])==len(n['ties'])==1,'fixedwrong':fw,'conditions':{},'contrasts':{},'frozen_e2':m['frozen_e2']}
  for name,item in m['conditions'].items():
   if item['status']!='OK':r['conditions'][name]={'status':item['status']};continue
   z=item['scores'];raw=[Fraction(v) for v in z['candidate_logits']];gg=gap(z['scores'],g,fw);exact=gap(raw,g,fw);r['conditions'][name]={'status':'OK','scores':z['scores'],'candidate_logits':z['candidate_logits'],'prediction':z['prediction'],'ties':z['ties'],'correct':z['prediction']==g,'production_gaps':gg,'fraction_gaps':{k:str(v) for k,v in exact.items()}}
  C=r['conditions'];edge={'Q_R_E':('H10','H00'),'Q_R_B':('H11','H01'),'R_Q_E':('H01','H00'),'R_Q_B':('H11','H10')}
  for nm,(x,y) in edge.items():
   r['contrasts'][nm]={}
   for metric in ('allwrong','fixedwrong','dynamic'):
    d=C[x]['production_gaps'][metric]-C[y]['production_gaps'][metric];fr=Fraction(C[x]['fraction_gaps'][metric])-Fraction(C[y]['fraction_gaps'][metric]);cell={'effect':d,'fraction_effect':str(fr)}
    for mode in ('norm','random'):
     z=C[nm+'_'+mode]
     if z['status']!='OK':cell[mode]=None;cell[mode+'_fraction']=None;continue
     bg='H00' if nm.endswith('_E') else 'H11';sign=1 if bg=='H00' else -1;cell[mode]=sign*(z['production_gaps'][metric]-C[bg]['production_gaps'][metric]);cell[mode+'_fraction']=str(sign*(Fraction(z['fraction_gaps'][metric])-Fraction(C[bg]['fraction_gaps'][metric])))
    cell['specificity']=None if cell['random'] is None else d-cell['random'];cell['specificity_fraction']=None if cell['random_fraction'] is None else str(fr-Fraction(cell['random_fraction']));r['contrasts'][nm][metric]=cell
  for metric in ('allwrong','fixedwrong','dynamic'):
   r['contrasts'].setdefault('Q9',{})[metric]={'restoration':C['U_Q9_N']['production_gaps'][metric]-C['H01']['production_gaps'][metric],'transfer_loss':C['H11']['production_gaps'][metric]-C['N_Q9_U']['production_gaps'][metric]}
   r['contrasts'].setdefault('interaction',{})[metric]=C['H11']['production_gaps'][metric]-C['H10']['production_gaps'][metric]-C['H01']['production_gaps'][metric]+C['H00']['production_gaps'][metric]
   r['contrasts'].setdefault('Q9_fraction',{})[metric]={'restoration':str(Fraction(C['U_Q9_N']['fraction_gaps'][metric])-Fraction(C['H01']['fraction_gaps'][metric])),'transfer_loss':str(Fraction(C['H11']['fraction_gaps'][metric])-Fraction(C['N_Q9_U']['fraction_gaps'][metric]))}
   r['contrasts'].setdefault('interaction_fraction',{})[metric]=str(Fraction(C['H11']['fraction_gaps'][metric])-Fraction(C['H10']['fraction_gaps'][metric])-Fraction(C['H01']['fraction_gaps'][metric])+Fraction(C['H00']['fraction_gaps'][metric]));r['contrasts'].setdefault('Q9_background_interaction',{})[metric]=r['contrasts']['Q9'][metric]['transfer_loss']-r['contrasts']['Q9'][metric]['restoration'];r['contrasts'].setdefault('Q9_background_interaction_fraction',{})[metric]=str(Fraction(r['contrasts']['Q9_fraction'][metric]['transfer_loss'])-Fraction(r['contrasts']['Q9_fraction'][metric]['restoration']))
  rows.append(r)
 result={}
 for group in ('all','rescue','damage','stable_correct','stable_wrong'):
  for nt in (False,True):
   subset=[r for r in rows if (group=='all' or r['natural_group']==group) and (not nt or r['natural_no_tie'])];gkey=group+('_no_tie' if nt else '');result[gkey]={'n':len(subset),'edges':{}}
   for edge in ('Q_R_E','Q_R_B','R_Q_E','R_Q_B'):
    result[gkey]['edges'][edge]={}
    for x in ('effect','norm','random','specificity'):
     vals=[r['contrasts'][edge]['allwrong'][x] for r in subset if r['contrasts'][edge]['allwrong'][x] is not None];result[gkey]['edges'][edge][x]={'n':len(vals),'mean':statistics.mean(vals) if vals else None,'median':statistics.median(vals) if vals else None,'positive':sum(v>0 for v in vals),'negative':sum(v<0 for v in vals),'zero':sum(v==0 for v in vals)}
 for groupkey,summary in result.items():
  nt=groupkey.endswith('_no_tie');group=groupkey[:-7] if nt else groupkey;subset=[r for r in rows if (group=='all' or r['natural_group']==group) and (not nt or r['natural_no_tie'])];summary['q9']={}
  for k in ('restoration','transfer_loss'):
   vals=[r['contrasts']['Q9']['allwrong'][k] for r in subset];summary['q9'][k]={'n':len(vals),'mean':statistics.mean(vals) if vals else None,'median':statistics.median(vals) if vals else None,'positive_ids':sorted(r['id'] for r in subset if r['contrasts']['Q9']['allwrong'][k]>0),'negative_ids':sorted(r['id'] for r in subset if r['contrasts']['Q9']['allwrong'][k]<0),'zero_ids':sorted(r['id'] for r in subset if r['contrasts']['Q9']['allwrong'][k]==0)}
  summary['condition_transitions']={}
  for name in sorted({n for r in subset for n in r['conditions']}):
   if name.startswith('Q_R_B_') or name.startswith('R_Q_B_') or name=='N_Q9_U':base='H11'
   elif name.startswith('Q_R_E_') or name.startswith('R_Q_E_'):base='H00'
   elif name=='U_Q9_N':base='H01'
   else:continue
   valid=[r for r in subset if r['conditions'][name]['status']=='OK'];summary['condition_transitions'][name]={'defined':len(valid),'rescued_ids':sorted(r['id'] for r in valid if not r['conditions'][base]['correct'] and r['conditions'][name]['correct']),'damaged_ids':sorted(r['id'] for r in valid if r['conditions'][base]['correct'] and not r['conditions'][name]['correct']),'wrong_to_other_wrong_ids':sorted(r['id'] for r in valid if not r['conditions'][base]['correct'] and not r['conditions'][name]['correct'] and r['conditions'][base]['prediction']!=r['conditions'][name]['prediction']),'post_tie_ids':sorted(r['id'] for r in valid if len(r['conditions'][name]['ties'])>1)}
  for edge,stats in summary['edges'].items():
   for x,cell in stats.items():
    vr=[(r['id'],r['contrasts'][edge]['allwrong'][x]) for r in subset if r['contrasts'][edge]['allwrong'][x] is not None];vals=sorted(v for _,v in vr);cell['quantiles']={str(q):(vals[round((len(vals)-1)*q)] if vals else None) for q in (.1,.25,.5,.75,.9)};cell['positive_ids']=sorted(i for i,v in vr if v>0);cell['negative_ids']=sorted(i for i,v in vr if v<0);cell['zero_ids']=sorted(i for i,v in vr if v==0)
 # Original preregistered route criteria: rescue group, positive signs, natural-no-tie no reversal.
 rescue=[r for r in rows if r['natural_group']=='rescue']; rescue_nt=[r for r in rescue if r['natural_no_tie']]; rng=np.random.default_rng(20261114)
 def boot(values,alpha):
  if not values:return {'n':0,'mean':None,'lower_one_sided_bonferroni':None,'ci95':None}
  arr=np.asarray(values,dtype=np.float64); draws=arr[rng.integers(0,len(arr),size=(5000,len(arr)))].mean(axis=1)
  return {'n':len(arr),'mean':float(arr.mean()),'lower_one_sided_bonferroni':float(np.quantile(draws,alpha,method='linear')),'ci95':[float(x) for x in np.quantile(draws,[.025,.975],method='linear')]}
 primary={'group':'same-site fresh natural rescue','bootstrap_draws':5000,'seed':20261114,'family_alpha':0.05,'edges':{},'q9':{}}
 for edge in ('Q_R_E','Q_R_B','R_Q_E','R_Q_B'):
  primary['edges'][edge]={}
  for kind in ('effect','norm','specificity'):
   vals=[r['contrasts'][edge]['allwrong'][kind] for r in rescue if r['contrasts'][edge]['allwrong'][kind] is not None]; nv=[r['contrasts'][edge]['allwrong'][kind] for r in rescue_nt if r['contrasts'][edge]['allwrong'][kind] is not None]; cell=boot(vals,.05/12);cell['natural_no_tie_n']=len(nv);cell['natural_no_tie_mean']=float(statistics.mean(nv)) if nv else None;cell['pass']=bool(vals and nv and cell['mean']>0 and cell['lower_one_sided_bonferroni']>0 and cell['natural_no_tie_mean']>=0); rawkey={'effect':'fraction_effect','norm':'norm_fraction','specificity':'specificity_fraction'}[kind]; raw=[(r['id'],Fraction(r['contrasts'][edge]['allwrong'][rawkey])) for r in rescue if r['contrasts'][edge]['allwrong'][rawkey] is not None]; cell['fraction_positive_ids']=sorted(i for i,v in raw if v>0);cell['fraction_negative_ids']=sorted(i for i,v in raw if v<0);cell['fraction_zero_ids']=sorted(i for i,v in raw if v==0);primary['edges'][edge][kind]=cell
 for kind in ('restoration','transfer_loss'):
  vals=[r['contrasts']['Q9']['allwrong'][kind] for r in rescue];nv=[r['contrasts']['Q9']['allwrong'][kind] for r in rescue_nt];cell=boot(vals,.05/2);cell['natural_no_tie_n']=len(nv);cell['natural_no_tie_mean']=float(statistics.mean(nv)) if nv else None;cell['pass']=bool(vals and nv and cell['mean']>0 and cell['lower_one_sided_bonferroni']>0 and cell['natural_no_tie_mean']>=0);primary['q9'][kind]=cell
 primary['Q_route_pass']=all(primary['edges'][e][k]['pass'] for e in ('Q_R_E','Q_R_B') for k in ('effect','norm','specificity'));primary['R_route_pass']=all(primary['edges'][e][k]['pass'] for e in ('R_Q_E','R_Q_B') for k in ('effect','norm','specificity'));primary['Q9_path_pass']=primary['Q_route_pass'] and all(primary['q9'][k]['pass'] for k in ('restoration','transfer_loss'));result['PRIMARY_PREDECLARED']=primary
 assert all(sha(k)==v for k,v in freeze['source_sha256'].items()) and sha(LABEL)==label_sha and not OUT.exists();OUT.mkdir(parents=True);(OUT/'rows.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows));(OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(OUT/'SOURCE.json').write_text(json.dumps({'status':'AB1_HOLDOUT_'+SPLIT.upper()+'_ANALYSIS_COMPLETE','pending_sha256':sha(D/'PENDING.json'),'audit_sha256':sha(D/'INDEPENDENT_COVERAGE_AUDIT.json'),'analysis_sha256':sha(pathlib.Path(__file__)),'freeze_sha256':sha(freeze_path),'rows_sha256':sha(OUT/'rows.jsonl'),'summary_sha256':sha(OUT/'summary.json'),'scope':'Predeclared within-project '+SPLIT+' confirmation; no claim of blind holdout; no post-treatment tie filtering.'},indent=2)+'\n');print('AB1_HOLDOUT_'+SPLIT.upper()+'_ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':
 main()
 source=json.loads((OUT/'SOURCE.json').read_text());source.update(status='COMPLETE',stage=STAGE,labels_sha256=sha(LABEL),scope='Stage analysis only; no global goal completion; train64 exploratory; train64 and train1030 IDs are disjoint, with historical exposure. Positive rescue primary retained.')
 (OUT/'COMPLETE.json').write_text(json.dumps(source,indent=2)+'\n')
 summary=json.loads((OUT/'summary.json').read_text())
 (OUT/'AB_INTERIM.md').write_text('# A/B interim: '+STAGE+'\n\nStage analysis complete. Full goal is not complete. Fixed E2 analysis requires separate descriptive output. Primary uses positive rescue effects, 5000 paired bootstrap draws, seed 20261114, 12-comparison family; Q9 uses separate 2-comparison family. Negative and exact Fraction rows remain in summary/rows.\n\n```json\n'+json.dumps(summary['PRIMARY_PREDECLARED'],indent=2)+'\n```\n')
