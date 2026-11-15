"""Frozen B2 positive-effect families. Saved-score analysis only; no model forward."""
import argparse,pathlib,json,hashlib,sys,statistics,numpy as np
from fractions import Fraction
FAMILIES={}
def contrasts():
 a={'loss':{'N':1,'T':-1},'restore':{'U_plus':1,'U':-1},'norm_loss':{'N':1,'upstream_norm':-1},'norm_restore':{'U_plus_norm_preserve':1,'U':-1},'specific_loss':{'upstream_unrelated':1,'T':-1},'specific_restore':{'U_plus':1,'U_plus_unrelated':-1}}
 m={'loss':{'N':1,'M':-1},'restore':{'U_M_plus':1,'U':-1},'norm_loss':{'N':1,'M_norm_preserve':-1},'norm_restore':{'U_M_plus_norm_preserve':1,'U':-1},'specific_loss':{'M_unrelated':1,'M':-1},'specific_restore':{'U_M_plus':1,'U_M_plus_unrelated':-1}}
 b={}
 for branch,restore,remove in [('X','TX','TA'),('A','TA','TX')]:
  b[branch+'_restore']={restore:1,'T':-1};b[branch+'_loss']={'N':1,remove:-1}
  for kind in ('norm','unrelated'):
   b[branch+'_restore_vs_'+kind]={restore:1,f'T_{branch}_{kind}':-1};b[branch+'_loss_vs_'+kind]={f'N_{branch}_{kind}':1,remove:-1}
 return {'attention':a,'L18_branch':b,'MLP_secondary':m}
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def gap(v,g,w):return dict(meanwrong=v[g]-sum(x for j,x in enumerate(v) if j!=g)/(len(v)-1),fixedwrong=v[g]-v[w],dynamic=v[g]-max(x for j,x in enumerate(v) if j!=g))
def bootstrap(v,alpha,rng):
 if not v:return dict(n_defined=0,mean=None,lower=None)
 arr=np.asarray(v);means=[]
 for _ in range(50):means.extend(arr[rng.integers(0,len(arr),size=(100,len(arr)))].mean(axis=1).tolist())
 return dict(n_defined=len(v),mean=float(arr.mean()),lower=float(np.quantile(means,alpha)),ci95=np.quantile(means,[.025,.975]).tolist())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--stage',required=True,choices=['train64','train1030','validation299']);a=ap.parse_args();root=pathlib.Path(a.root);sys.path[:0]=[str(root/'code'),str(root/'storage')];d=root/'b2_data/treatments'/a.stage;pending=d/'PENDING.json';audit=d/'INDEPENDENT_COVERAGE_AUDIT.json';p=json.loads(pending.read_text());au=json.loads(audit.read_text());assert p['status']=='COMPLETE' and au['status']=='PASS' and au['pending_sha256']==sha(pending);split=p['split'];labels=root/'analysis_inputs/analysis_labels.jsonl';ys={(z['split'],z['id']):z['gold'] for line in labels.read_text().splitlines() if (z:=json.loads(line))};rows=[];freeze=root/'B2_FULL_SOURCE_FREEZE.json';fz=json.loads(freeze.read_text());assert sha(freeze)==p['source_freeze_sha256'];before={str(x.relative_to(root)):sha(x) for x in (pending,audit,labels,freeze,pathlib.Path(__file__))};assert all(sha(root/k)==v for k,v in fz['sources'].items())
 for c in p['cases']:
  mp=d/c['metadata_file'];assert sha(mp)==c['metadata_sha256'];m=json.loads(mp.read_text());ab=json.loads((root/m['AB_metadata_path']).read_text());ref=json.loads((root/m['reference_metadata_path']).read_text());g=ys[(split,m['id'])];k1=ab['identity']['K1_scores'];n=ab['identity']['N_scores'];w=max((j for j in range(len(k1['scores'])) if j!=g),key=lambda j:k1['scores'][j]);group='rescue' if k1['prediction']!=g and n['prediction']==g else 'damage' if k1['prediction']==g and n['prediction']!=g else 'stable_correct' if k1['prediction']==g else 'stable_wrong';conditions={'N':{'status':'OK','scores':n},'U':ab['conditions']['H01'],'K1':{'status':'OK','scores':k1},**ref['conditions'],**m['conditions']};r=dict(id=m['id'],split=split,gold=g,natural_group=group,natural_no_tie=len(k1['ties'])==len(n['ties'])==1,conditions={},families={})
  for name,item in conditions.items():
   if item['status']!='OK':r['conditions'][name]={'status':item['status']};continue
   z=item['scores'];r['conditions'][name]=dict(status='OK',prediction=z['prediction'],ties=z['ties'],correct=z['prediction']==g,gaps=gap(z['scores'],g,w),fraction_gaps={k:str(v) for k,v in gap([Fraction(x) for x in z['candidate_logits']],g,w).items()})
  for family,cs in contrasts().items():
   r['families'][family]={}
   for name,coef in cs.items():
    valid=all(r['conditions'][k]['status']=='OK' for k in coef);r['families'][family][name]=None if not valid else dict(production={metric:sum(v*r['conditions'][k]['gaps'][metric] for k,v in coef.items()) for metric in ('meanwrong','fixedwrong','dynamic')},fraction={metric:str(sum((v*Fraction(r['conditions'][k]['fraction_gaps'][metric]) for k,v in coef.items()),Fraction(0))) for metric in ('meanwrong','fixedwrong','dynamic')})
  rows.append(r)
 summary={};rng=np.random.default_rng(20261114)
 for group in ('all','rescue','damage','stable_correct','stable_wrong'):
  for nt in (False,True):
   rr=[r for r in rows if (group=='all' or r['natural_group']==group) and (not nt or r['natural_no_tie'])];key=group+('_no_tie' if nt else '');summary[key]=dict(n=len(rr),families={},conditions={})
   for family,cs in contrasts().items():
    cells={}
    for name in cs:
     valid=[r for r in rr if r['families'][family][name] is not None];v=[r['families'][family][name]['production']['meanwrong'] for r in valid];cell=bootstrap(v,.05/len(cs),rng);cell.update(n_total=len(rr),full_defined_coverage=len(valid)==len(rr),positive_ids=[r['id'] for r in valid if Fraction(r['families'][family][name]['fraction']['meanwrong'])>0],strict_counterexample_ids=[r['id'] for r in valid if Fraction(r['families'][family][name]['fraction']['meanwrong'])<0],zero_ids=[r['id'] for r in valid if Fraction(r['families'][family][name]['fraction']['meanwrong'])==0]);nv=[r['families'][family][name]['production']['meanwrong'] for r in valid if r['natural_no_tie']];cell['natural_no_tie_mean']=statistics.mean(nv) if nv else None;cell['positive_pass']=bool(v and len(valid)==len(rr) and cell['mean']>0 and cell['lower']>0 and nv and cell['natural_no_tie_mean']>=0);cells[name]=cell
    summary[key]['families'][family]=dict(contrasts=cells,all_positive_pass=all(c['positive_pass'] for c in cells.values()))
   for name in rows[0]['conditions'] if rows else []:
    valid=[r for r in rr if r['conditions'][name]['status']=='OK'];summary[key]['conditions'][name]=dict(n_defined=len(valid),correct_ids=[r['id'] for r in valid if r['conditions'][name]['correct']],tie_ids=[r['id'] for r in valid if len(r['conditions'][name]['ties'])>1],prediction_changed_from_N_ids=[r['id'] for r in valid if r['conditions'][name]['prediction']!=r['conditions']['N']['prediction']],rescued_from_N_ids=[r['id'] for r in valid if not r['conditions']['N']['correct'] and r['conditions'][name]['correct']],damaged_from_N_ids=[r['id'] for r in valid if r['conditions']['N']['correct'] and not r['conditions'][name]['correct']])
 assert all(sha(root/k)==v for k,v in before.items()) and all(sha(root/k)==v for k,v in fz['sources'].items());out=root/'b2_analysis'/a.stage;assert not out.exists();out.mkdir(parents=True);(out/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',stage=a.stage,pending_sha256=sha(pending),audit_sha256=sha(audit),labels_sha256=sha(labels),script_sha256=sha(__file__),source_sha256=before,primary='fresh AB native rescue; three positive Bonferroni families independently; no natural no-tie sign reversal; full defined coverage required',bootstrap_draws=5000,seed=20261114,scope='Saved-score analysis only. TXA identity excluded from explanatory fractions. No whole-goal completion claim.'),indent=2)+'\n')
if __name__=='__main__':main()
