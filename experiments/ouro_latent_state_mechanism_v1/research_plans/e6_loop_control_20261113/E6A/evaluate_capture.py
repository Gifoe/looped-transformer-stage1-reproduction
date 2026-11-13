"""Same-device evaluation with development freeze preceding Test capture/evaluation."""
import argparse,json,hashlib,collections,datetime
from pathlib import Path
import numpy as np
from E6A_METHODS import infer,fixed_specs,calibration_grid
from evaluate import evaluate as fixed_evaluate
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf8')
def extra_metrics(rows,name):
 records=[]
 for q in rows:
  x=q['capture']['replica_baselines'][name];v=np.array(x['scores']);c=np.array(q['historical_scores']);pred=int(v.argmax());b=int(c[3].argmax());g=q['gold'];records.append({'split':q['split'],'id':q['id'],'method':name,'track':'ZL existing repository baseline','prediction':pred,'scores':v.tolist(),'correct':pred==g,'native_K4_correct':b==g,'native_K4_prediction':b,'selected_depth':4,'executed_loops':4,'top_tie':bool(sum(v==v.max())>1),'all_native_and_method_no_tie':all(sum(z==z.max())==1 for z in c) and sum(v==v.max())==1})
 d=np.array([int(r['correct'])-int(r['native_K4_correct']) for r in records]);rng=np.random.default_rng(20261113);ci=np.quantile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1)*100,[.025,.975]);mask=np.array([r['all_native_and_method_no_tie'] for r in records]);s={'method':name,'track':'ZL existing repository baseline','n':len(d),'correct':sum(r['correct'] for r in records),'accuracy':float(np.mean([r['correct'] for r in records])),'pp_vs_K4':float(d.mean()*100),'rescue':int((d==1).sum()),'damage':int((d==-1).sum()),'net_gain':int(d.sum()),'paired_bootstrap95_pp':ci.tolist(),'no_tie_n':int(mask.sum()),'no_tie_pp_vs_K4':float(d[mask].mean()*100) if mask.any() else None,'mean_loops':4.,'relative_forward_loop_count':1.,'latency':'measured collector resource in separate capture receipt, includes output and latent observer overhead'}
 return s,records
def main():
 p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--labels',required=True);p.add_argument('--out',required=True);p.add_argument('--phase',choices=['dev','test'],required=True);p.add_argument('--transfer-spec');p.add_argument('--aggregation',choices=['scores','raw_sum','mean_token'],default='scores');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 ll=Path(a.labels);labels=json.loads(ll.read_text()) if ll.suffix=='.json' else {r['split']+':'+r['id']:r for r in map(json.loads,ll.read_text().splitlines())}
 rows=[];sources=[];identities=set();seen=set()
 for f in sorted(Path(a.capture).rglob('*.json')):
  if 'technical' in str(f).lower():continue
  q=json.loads(f.read_text())
  if not isinstance(q,dict) or not {'id','split','native','identity'}<=set(q):continue
  if a.phase=='dev' and q['split']=='test':continue
  if a.phase=='test' and q['split']!='test':continue
  key=q['split']+':'+q['id'];assert key not in seen;seen.add(key);l=labels[key]
  assert q['identity']['model_sha256']=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af' and '5090' in q['identity']['gpu'];assert len(q['native'])==4
  identities.add((q['identity']['gpu'],q['identity']['torch'],q['identity']['source_sha256'],q['identity']['collector_sha256']))
  if l.get('is_demonstration',l.get('demo',False)):continue
  if a.aggregation!='scores':
   for n in q['native']:n['scores']=n[a.aggregation]
   for n in q['replica_baselines'].values():n['scores']=n[a.aggregation]
  rows.append({'id':q['id'],'split':q['split'],'gold':l['gold'],'demo':False,'historical_scores':[n['scores'] for n in q['native']],'capture':q});sources.append({'file':str(f),'sha256':sha(f)})
 assert len(identities)==1,identities
 specs={**fixed_specs(),**calibration_grid()};freeze=out/'E6A_FROZEN_METHOD_SPEC.json'
 if a.phase=='dev':
  assert collections.Counter(r['split'] for r in rows)=={'train':1094,'validation':299};assert not freeze.exists(),'Never overwrite frozen development choices'
  summaries={};allrecords=[]
  for split in ['train','validation']:
   rr=[r for r in rows if r['split']==split];s,records=fixed_evaluate(rr,specs)
   for name in ['loopcd_adaptive_cap1','loopcd_fixed_0.5']:
    ss,ar=extra_metrics(rr,name);s.append(ss);records+=ar
   summaries[split]=s;allrecords+=records
  if a.transfer_spec:
   fs=json.loads(Path(a.transfer_spec).read_text());primary=fs['primary_Cal'];selection='Unchanged Joint-development calibrated method; zero-shot full-answer transfer, not selected using Full-Answer labels'
  else:
   primary=sorted(summaries['validation'],key=lambda x:(-x['correct'],x['mean_loops'],x['method']))[0]['method'];selection='Validation best among frozen32 core rules and2 fixed full-vocabulary LoopCD controls, lower loops/name ties; any label-based method selection is Cal'
  f={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'primary_ZL':'weighted4','primary_Cal':primary,'Cal_selection':selection,'methods':specs,'preset_existing_controls':['loopcd_adaptive_cap1','loopcd_fixed_0.5'],'source_sha256':sha(__file__),'methods_sha256':sha(Path(__file__).with_name('E6A_METHODS.py')),'metrics_source_sha256':sha(Path(__file__).with_name('evaluate.py')),'development_sources':sources,'native_identity':list(identities)[0],'protocol':rows[0]['capture']['identity'].get('score_protocol','joint_option_letter'),'primary_metric':'Test accuracy paired versus same-protocol5090 Native K4','test_policy':'Once frozen evaluation; no parameter changes from Test; previous historical exposure disclosed'}
  f['aggregation']=a.aggregation
  write(freeze,f);write(out/'DEVELOPMENT_SUMMARY.json',summaries);(out/'DEVELOPMENT_RESULTS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in allrecords));print({'phase':'dev','primary_Cal':primary,'freeze_sha256':sha(freeze)})
 else:
  f=json.loads(freeze.read_text());assert f['source_sha256']==sha(__file__) and f['methods_sha256']==sha(Path(__file__).with_name('E6A_METHODS.py'));assert f['aggregation']==a.aggregation;assert len(rows)==1172 and not (out/'TEST_COMPLETE.json').exists();assert list(identities)[0]==tuple(f['native_identity'])
  s,records=fixed_evaluate(rows,f['methods'])
  for name in f['preset_existing_controls']:
   ss,ar=extra_metrics(rows,name);s.append(ss);records+=ar
  write(out/'TEST_SUMMARY.json',s);(out/'E6A_ALL_RESULTS.jsonl').write_text((out/'DEVELOPMENT_RESULTS.jsonl').read_text()+''.join(json.dumps(r)+'\n' for r in records));write(out/'TEST_COMPLETE.json',{'status':'COMPLETE','n':1172,'frozen_spec_sha256':sha(freeze),'source':sources,'native_identity':list(identities)[0]});print({'phase':'test','n':1172,'correct':{r['method']:r['correct'] for r in s if r['method'] in ['K4',f['primary_ZL'],f['primary_Cal']]}})
if __name__=='__main__':main()
