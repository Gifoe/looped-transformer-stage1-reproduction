"""Complete-ledger-only descriptive 2x2 gap summary; not mechanistic proof."""
import argparse,json,hashlib,math,collections
from pathlib import Path
if not __debug__:raise RuntimeError('optimized Python forbidden')
import pyarrow.parquet as pq
from cohort_runner import sha,split_data_path
from round2_intervention_core import CORE,NORM
from round2_intervention_runner import canonical,write_once
P=Path(__file__).resolve();ROOT=P.parents[2].parent
def factorial_metrics(m22,m12,m21,m11):
 return {'context_replacement_K2_to_K1_at_R2':m12-m22,'readout_replacement_K2_to_K1_at_C2':m21-m22,'joint_replacement_K2_to_K1':m11-m22,'context_contribution_at_R2':m22-m12,'context_contribution_at_R1':m21-m11,'readout_contribution_at_C2':m22-m21,'readout_contribution_at_C1':m12-m11,'interaction':m22-m12-m21+m11}
def stats(values):
 v=sorted(values);n=len(v)
 if not n:return {'n':0}
 return {'n':n,'mean':sum(v)/n,'min':v[0],'q25':v[(n-1)//4],'median':(v[(n-1)//2]+v[n//2])/2,'q75':v[3*(n-1)//4],'max':v[-1],'positive':sum(x>0 for x in v),'negative':sum(x<0 for x in v),'zero':sum(x==0 for x in v)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();run=Path(a.run).resolve();out=Path(a.out).resolve();run.relative_to(ROOT.resolve());out.relative_to(ROOT.resolve());assert not out.exists();allout={}
 for split in ('train','validation','test'):
  d=run/split;cp=run/f'coverage_{split}.json';cov=json.loads(cp.read_text());rows=pq.read_table(split_data_path(split)).to_pylist();ids={x['id'] for x in rows};ledger=cov['questions'];assert cov['status']=='COMPLETE_CUDA_REPLAY_ZERO_TOLERANCE' and cov['expected_questions']==cov['complete_questions']==len(ids) and cov['replayed_condition_logits']==11*len(ids) and len(ledger)==len(ids) and {x['id'] for x in ledger}==ids and cov['coverage_script_sha256']==sha(P.with_name('round2_intervention_coverage.py')) and cov['receipt_sha256']==sha(d/'_receipt.json')
  for path,h in cov['source']['source_sha256'].items():assert sha(Path(path))==h
  buckets=collections.defaultdict(lambda:collections.defaultdict(list));changes=collections.Counter();cases=[]
  for q in ledger:
   assert q['status']=='complete';key=hashlib.sha256(q['id'].encode()).hexdigest();p=d/(key+'.json');t=d/(key+'.safetensors');assert sha(p)==q['metadata_sha256'] and sha(t)==q['logits_sha256'];j=json.loads(p.read_text());v=j['payload'];assert j['payload_sha256']==hashlib.sha256(canonical(v)).hexdigest() and v['identity']['id']==q['id'];r=v['result'];g=r['gold'];a1=r['baseline_K1']['prediction']==g;a2=r['baseline_K2']['prediction']==g;group=('stable_correct' if a1 else 'rescue') if a2 else ('damage' if a1 else 'stable_wrong')
   c=r['conditions'];n=r['norm_conditions'];f00=c['K2C_K2R']['gold_vs_fixed_wrong_gap'];f10=c['K1C_K2R']['gold_vs_fixed_wrong_gap'];f01=c['K2C_K1R']['gold_vs_fixed_wrong_gap'];f11=c['K1C_K1R']['gold_vs_fixed_wrong_gap'];metrics={**factorial_metrics(f00,f10,f01,f11),'C_K1DIR_K2NORM_effect':n['C_K1DIR_K2NORM']['gold_vs_fixed_wrong_gap']-f00,'R_K1DIR_K2NORM_effect':n['R_K1DIR_K2NORM']['gold_vs_fixed_wrong_gap']-f00,'C_K2DIR_K1NORM_effect':n['C_K2DIR_K1NORM']['gold_vs_fixed_wrong_gap']-f00,'R_K2DIR_K1NORM_effect':n['R_K2DIR_K1NORM']['gold_vs_fixed_wrong_gap']-f00};assert all(math.isfinite(x) for x in metrics.values())
   groups=['all',group,'demonstration' if v['identity']['is_demonstration'] else 'non_demonstration']
   for groupname in groups:
    for metric,value in metrics.items():buckets[groupname][metric].append(value)
   changes['questions']+=1
   for name in CORE:
    if c[name]['prediction']!=c['K2C_K2R']['prediction']:changes['prediction_changed_'+name]+=1
   cases.append({'id':q['id'],'outcome':group,'is_demonstration':v['identity']['is_demonstration'],'metrics':metrics,'predictions':{k:c[k]['prediction'] for k in CORE}})
  assert changes['questions']==len(ids);allout[split]={'n':len(ids),'coverage_path':str(cp),'coverage_sha256':sha(cp),'groups':{g:{m:stats(v) for m,v in d.items()} for g,d in buckets.items()},'prediction_changes':dict(changes),'questions':cases}
 report={'status':'COMPLETE_COVERAGE_BOUND_DESCRIPTIVE_INTERVENTION','scope':'pre-registered interventions; outcomes do not by themselves establish a complete mechanism','splits':allout,'formulas':{'notation':'mXY = gap with C from KX and R from KY; positive contribution means K2 source increases the fixed gold-vs-K1-wrong gap','context_contribution_at_R2':'m22-m12','context_contribution_at_R1':'m21-m11','readout_contribution_at_C2':'m22-m21','readout_contribution_at_C1':'m12-m11','interaction':'m22-m12-m21+m11','replacement_and_norm_effects':'replacement gap minus m22'},'analysis_script_sha256':sha(P),'limitations':['All ARC splits exposed before this analysis; no independent benchmark validation.','Hybrid states may be out of distribution.','Historical endpoint scores differ and the reason remains unresolved.']};write_once(out,report);print('ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':main()
