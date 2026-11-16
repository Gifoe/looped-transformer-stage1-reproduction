"""CPU analysis of exactly the three authorized critical experiments."""
import pathlib,json,os,numpy as np,datetime,time,hashlib
from E6A_METHODS import infer,fixed_specs
R=pathlib.Path(os.environ.get('CATC_CRITICAL_ROOT','/root/ouro-catc-critical-sol-20261116'));O=pathlib.Path('/root/ouro-catc-sol-20261116');load=lambda p:json.loads(p.read_text());spec=load(R/'FREEZE.json');selected=load(R/'SELECTED_METHOD.json');start=time.monotonic()
def ce(v,g):
 x=np.array(v,dtype=np.float64);a=x.max();return float(a+np.log(np.exp(x-a).sum())-x[g])
def margin(v,g):return float(v[g]-max(x for k,x in enumerate(v) if k!=g))
def paired(d):
 d=np.asarray(d,dtype=np.float64);rng=np.random.default_rng(20261116);boot=[]
 for k in range(0,5000,250):boot.extend(d[rng.integers(0,len(d),(250,len(d)))].mean(1).tolist())
 return {'mean':float(d.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'lower_bonferroni3':float(np.quantile(boot,.05/3)),'n':len(d)}
def frame(q):
 n=q['native'];r={'native_K'+d:z for d,z in n['scores'].items()};r.update({'selected_K'+d:z for d,z in q['selected']['scores'].items()})
 for name,v in q.get('controls',{}).items():r[name+'_K4']=v['scores']['4']
 for name,v in q['output_only'].items():r['E6_'+name]={'prediction':v['prediction'],'candidate_logits':v['scores']}
 x1=np.array(n['scores']['1']['candidate_logits'],dtype=np.float32);x4=np.array(n['scores']['4']['candidate_logits'],dtype=np.float32)
 for w in [.25,.5,1.]:
  v=x4+w*(x4-x1);r['LoopCD_'+str(w)]={'prediction':int(v.argmax()),'candidate_logits':v.tolist()}
 return r
if not (R/'QUESTION_LEVEL_RESULTS.jsonl').exists():
 assert load(R/'EXPERIMENT2_COMPLETE.json')['status']=='COMPLETE' and load(R/'EXPERIMENT3_COMPLETE.json')['status']=='COMPLETE'
 inputs={q['id']:q for q in map(json.loads,(O/'INPUTS.jsonl').read_text().splitlines())};old={q['id']:q for q in map(json.loads,(O/'TRAINING_FREE_ALL_ROWS.jsonl').read_text().splitlines())};dev={q['id']:q for q in map(json.loads,(R/'DEVELOPMENT_ROWS.jsonl').read_text().splitlines())};rows=[]
 for cohort,ids in spec['cohorts'].items():
  for id in ids:
   q=old[id];records={z['name']:z for z in q['records']};native=records['T0'];outputs={n:infer([native['scores'][str(k)]['candidate_logits'] for k in range(1,5)],v) for n,v in fixed_specs().items()};control=load(R/'results'/f'{id}.json')['methods'] if cohort=='confirmation' else {};row={'id':id,'dataset':'train','cohort':cohort,'gold':q['gold'],'input_ids_sha256':inputs[id]['input_ids_sha256'],'native':native,'selected':records[selected['original_record']],'controls':control,'output_only':outputs,'selected_setting':selected['name'],'reused_original_capture':True}
   if id in dev:row['fixed_direction_methods']=dev[id]['methods'];row['offline_diagnostics']=dev[id]['offline_diagnostics']
   rows.append(row)
 for p in sorted((R/'validation').glob('*.json')):
  q=load(p);rows.append({'id':q['id'],'dataset':'validation','cohort':'exposed_validation','gold':q['gold'],'input_ids_sha256':q['input_ids_sha256'],'native':q['native'],'selected':q['selected'],'controls':{},'output_only':q['output_only'],'selected_setting':selected['name'],'selected_reused':q['selected_reused']})
 assert len(rows)==1393 and len({q['id'] for q in rows})==1393
 (R/'QUESTION_LEVEL_RESULTS.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in rows))
else:rows=[json.loads(l) for l in (R/'QUESTION_LEVEL_RESULTS.jsonl').read_text().splitlines()]
datasets={c:[q for q in rows if q['cohort']==c] for c in spec['cohorts']};datasets.update({'all_train':[q for q in rows if q['dataset']=='train'],'combined_confirmation':[q for q in rows if q['dataset']=='train' and q['cohort']!='development'],'exposed_validation':[q for q in rows if q['dataset']=='validation']});summary={};repair=load(R/'COHORT_DUPLICATE_REPAIR.json');excluded=set(repair['excluded_confirmation_ids']);datasets['confirmation_original512']=datasets['confirmation'];datasets['confirmation']=[q for q in datasets['confirmation'] if q['id'] not in excluded];datasets['combined_confirmation']=[q for q in datasets['combined_confirmation'] if q['id'] not in excluded]
for cohort,rs in datasets.items():
 frames=[frame(q) for q in rs];common=[n for n in frames[0] if all(n in f for f in frames)];gold=np.array([q['gold'] for q in rs]);acc={n:np.array([r[n]['prediction']==g for r,g in zip(frames,gold)],dtype=int) for n in common};losses={n:np.array([ce(r[n]['candidate_logits'],g) for r,g in zip(frames,gold)]) for n in common};margins={n:np.array([margin(r[n]['candidate_logits'],g) for r,g in zip(frames,gold)]) for n in common};b=acc['native_K4'];allwrong=np.logical_and.reduce([acc['native_K'+str(d)]==0 for d in range(1,5)]);methods={}
 for n,v in acc.items():
  methods[n]={'correct':int(v.sum()),'n':len(v),'accuracy':float(v.mean()),'pp_vs_nativeK4':float((v-b).mean()*100),'rescue':int(((v==1)&(b==0)).sum()),'damage':int(((v==0)&(b==1)).sum()),'paired_accuracy_vs_nativeK4':paired(v-b),'mean_candidate_CE':float(losses[n].mean()),'mean_gold_margin':float(margins[n].mean()),'paired_CE_benefit_vs_nativeK4':paired(losses['native_K4']-losses[n]),'new_correct_all_native_loops_wrong':int(((v==1)&allwrong).sum())}
 summary[cohort]={'methods':methods,'K2_same_depth_net_correct':int(acc['selected_K2'].sum()-acc['native_K2'].sum()),'K4_net_correct':int(acc['selected_K4'].sum()-b.sum()),'K2_rescue_count':int(((acc['selected_K2']==1)&(acc['native_K2']==0)).sum()),'K2_rescues_with_correct_selectedK4':int(((acc['selected_K2']==1)&(acc['native_K2']==0)&(acc['selected_K4']==1)).sum())}
 if cohort=='confirmation':
  names=['native_K4','matched_random_K4','matched_scaling_K4'];a={n:paired(acc['selected_K4']-acc[n]) for n in names};c={n:paired(losses[n]-losses['selected_K4']) for n in names};m={n:paired(margins['selected_K4']-margins[n]) for n in names};summary[cohort]['control_contrasts']={'accuracy':a,'CE_benefit':c,'margin_benefit':m};summary[cohort]['primary_accuracy_gate']=all(v['lower_bonferroni3']>0 for v in a.values());summary[cohort]['secondary_score_specificity_gate']=all(v['lower_bonferroni3']>0 for v in c.values()) and int(acc['selected_K4'].sum())>=int(b.sum())
(R/'CRITICAL_SUMMARY.json').write_text(json.dumps(summary,indent=2));doses=[]
for q in rows:
 for n,v in q['controls'].items():
  for z in v['dose_matches']:doses.append({'id':q['id'],'kind':n,**z})
assert len(doses)==512*2*7 and all(z['actual_norm']<=z['cap'] for z in doses)
dose_summary={'sites':len(doses),'matched_sites':sum(z['matched_within_1percent'] for z in doses),'unmatched_sites':[z for z in doses if not z['matched_within_1percent']],'max_relative_error':max(z['relative_error'] for z in doses),'mean_relative_error':float(np.mean([z['relative_error'] for z in doses])),'questions_with_unmatched_sites':len({z['id'] for z in doses if not z['matched_within_1percent']}),'all_retained':True};(R/'DOSE_MATCHING_AUDIT.json').write_text(json.dumps(dose_summary,indent=2));(R/'ANALYSIS_COMPLETE.json').write_text(json.dumps({'status':'PASS','question_rows':len(rows),'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'wall_seconds':time.monotonic()-start,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2));print('CRITICAL_ANALYSIS_COMPLETE',summary['confirmation']['primary_accuracy_gate'],dose_summary['matched_sites'],dose_summary['sites'])
