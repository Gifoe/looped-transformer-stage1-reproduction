"""Frozen E7 CPU analysis: complete natural grids, Train decisions, then exposed-Val review. No GPU/Test."""
import argparse,json,hashlib,datetime,gzip,time,importlib.util,collections,os
from pathlib import Path
import numpy as np
SEED=20261113;BOOT=10000
EXPECTED={'train':{3:1,4:1092,5:1},'validation':{3:3,4:295,5:1}}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def default(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,set):return sorted(x)
 raise TypeError(type(x).__name__)
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial');tmp.write_text(json.dumps(x,indent=2,default=default,allow_nan=False)+'\n');os.replace(tmp,p)
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 opener=gzip.open if str(p).endswith('.gz') else open
 with opener(p,'rt',encoding='utf8') as f:
  for l in f:
   if l.strip():yield json.loads(l)
def save_rows(p,rs):
 with gzip.open(str(p),'wt',encoding='utf8',compresslevel=9) as f:
  for r in rs:f.write(json.dumps(r,default=default,allow_nan=False)+'\n')
def import_methods(p):
 s=importlib.util.spec_from_file_location('frozen_E6A_methods',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def contrasts(x):
 # [P offset,S offset], nonzero offsets averaged within question, never selected.
 return np.array([x[1:,:].mean()-x[0,:].mean(),x[:,1:].mean()-x[:,0].mean(),x[1:,1:].mean()-x[1:,0].mean()-x[0,1:].mean()+x[0,0]],dtype=float)
def intervals(values,ids):
 x=np.asarray(values,dtype=np.float64)
 if x.ndim==1:x=x[:,None]
 assert len(ids)==len(x) and list(ids)==sorted(ids) and len(x)>0 and np.isfinite(x).all();rng=np.random.default_rng(SEED);draws=[]
 for _ in range(100):draws.append(x[rng.integers(0,len(x),(100,len(x)))].mean(1))
 z=np.concatenate(draws);return [{'n_questions':len(ids),'estimate':float(x[:,j].mean()),'percentile95':np.quantile(z[:,j],[.025,.975]).tolist(),'percentile98_333333':np.quantile(z[:,j],[.05/6,1-.05/6]).tolist(),'bootstrap_questions':BOOT,'seed':SEED,'descriptive_small_n':len(ids)<30} for j in range(x.shape[1])]
def paired_accuracy(gold,pred,reference,cost,ids):
 y=np.asarray(gold);p=np.asarray(pred);r=np.asarray(reference);d=(p==y).astype(int)-(r==y);ci=intervals(d,ids)[0]
 return {'n':len(y),'correct':int((p==y).sum()),'reference_correct':int((r==y).sum()),'accuracy':float((p==y).mean()),'rescue':int(((p==y)&(r!=y)).sum()),'damage':int(((p!=y)&(r==y)).sum()),'net_correct':int(d.sum()),'gain_pp':float(d.mean()*100),'paired_question95_pp':(np.asarray(ci['percentile95'])*100).tolist(),'mean_nominal_loop_budget':float(np.mean(cost)),'actual_online_latency':'NOT_RUN'}
def file_hashes(a):
 r=Path(a.root);return {name:sha(path) for name,path in {'analysis_source':__file__,'CPU_queue_source':r/'E7_CPU_QUEUE.py','synthetic_QA_source':r/'E7_CPU_SYNTHETIC_QA.py','synthetic_QA_receipt':r/'E7_CPU_SYNTHETIC_QA.json','Astra_formal_order':a.order,'E7_protocol':r/'E7_PROTOCOL_FREEZE.json','E7_collector':r/'E7_REENCODING_COLLECT.py','E7_generator':r/'E7_GENERATE_VARIANTS.py','variant_inputs':a.inputs,'variant_manifest':r/'E7_VARIANT_MANIFEST.json','technical_complete':r/'technical/COMPLETE.json','independent_semantic_audit':r/'E7_ALL_VARIANT_SEMANTIC_AUDIT.json','within_question_length_audit':r/'E7_WITHIN_QUESTION_LENGTH_AUDIT.json','old_Train_only_labels':a.train_labels,'old_exposed_Val_only_labels':a.val_labels,'E6A_methods':a.methods,'E6A_frozen_methods':a.e6a_spec}.items()}
def load(a,split,frozen):
 labels=list(lines(a.train_labels if split=='train' else a.val_labels));assert len(labels)==(1094 if split=='train' else 299) and all(q['split']==split and not q['is_demonstration'] for q in labels);lab={q['id']:q['gold'] for q in labels};cap=Path(a.capture);core=read(cap/'IDENTITY.json');assert core['collector_sha256']==frozen['hashes']['E7_collector'];assert core['model_sha256']=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af';assert core['dtype']=='bfloat16' and core['attention']=='sdpa' and core['use_cache'] is False
 questions={};manifest=[];expected_count=0
 for q in lines(a.inputs):
  assert q['split'] in ['train','validation'],'No Test inputs accepted'
  if q['split']!=split:continue
  assert q['id'] in lab;file=cap/(split+'_'+hashlib.sha256(q['variant_id'].encode()).hexdigest()+'.json');assert file.exists(),'Missing required completed variant: '+q['variant_id'];u=read(file);checksum=u['unit_payload_sha256'];assert canonical({k:v for k,v in u.items() if k!='unit_payload_sha256'})==checksum;assert u['status']=='COMPLETE' and u['identity']==core and u['variant_row_sha256']==canonical(q);assert u['input_ids_sha256']==q['input_ids_sha256'] and u['generator_sha256']==q['generator_sha256'];assert u['display_to_original']==q['display_to_original'] and u['targets']==q['targets'] and u['labels']==q['labels'];assert u['production_actual_counts']=={'model_body_forwards':1,'head_calls':4,'first_layer_calls':4};assert u['weight_versions_unchanged'] and len(u['native'])==4
  m=len(q['targets']);assert m in [3,4,5];mapping=q['display_to_original'];assert mapping==[(j+q['p'])%m for j in range(m)] and u['original_to_display']==[mapping.index(j) for j in range(m)];idx=np.argsort(mapping);logits=[];scores=[];pred=[];spred=[];tops=[];ties=[]
  for k,n in enumerate(u['native']):
   v=np.asarray(n['candidate_logits'],float);lp=np.asarray(n['scores'],float);assert len(v)==len(lp)==m and np.isfinite(v).all() and np.isfinite(lp).all();top=np.flatnonzero(v==v.max()).tolist();semantic_top=sorted(mapping[j] for j in top);assert n['prediction']==int(v.argmax()) and n['prediction_semantic']==mapping[int(v.argmax())] and n['top_semantic_indices']==semantic_top and n['top_tie']==(len(top)>1) and n['semantic_index_tiebreak_prediction']==min(semantic_top);assert n['depth']==k+1
   logits.append(v[idx]);scores.append(lp[idx]);pred.append(n['prediction_semantic']);spred.append(n['semantic_index_tiebreak_prediction']);tops.append(semantic_top);ties.append(n['top_tie'])
  v={'p':q['p'],'s':q['s'],'m':m,'variant_id':q['variant_id'],'logits':np.asarray(logits),'scores':np.asarray(scores),'pred':np.asarray(pred),'semantic_first_pred':np.asarray(spred),'tops':tops,'ties':np.asarray(ties),'token_count':u['token_count'],'unit_file':str(file),'unit_sha256':sha(file),'capture_seconds':u['production_forward_scoring_seconds'],'gate':u['gate_R'],'replica':u['replica_baselines'],'display_to_original':mapping,'targets':q['targets'],'labels':q['labels']}
  question=questions.setdefault(q['id'],{'id':q['id'],'split':split,'gold':lab[q['id']],'m':m,'views':{}});assert question['m']==m and (q['p'],q['s']) not in question['views'];question['views'][q['p'],q['s']]=v
  if q['p']==q['s']==0:
   bridge=u['identity_E6_bridge'];assert bridge['status']=='PASS' and bridge['input_tokens_sha256_exact'] and bridge['four_native_full_vocabulary_R_hashes_exact'];oldpath=Path(bridge['E6_source_file']);assert oldpath.parent.name==split and sha(oldpath)==bridge['E6_source_sha256'];old=read(oldpath);assert [n['full_logits_sha256'] for n in old['native']]==[n['full_logits_sha256'] for n in u['native']];assert u['original_input_ids_sha256']==u['input_ids_sha256'];assert 'loopcd_adaptive_cap1' in v['replica'];assert all(len(u['gate_R'][name])==4 for name in ['logits','hazards','exit_pdf','exit_cdf'])
  manifest.append({'id':q['id'],'variant_id':q['variant_id'],'p':q['p'],'s':q['s'],'path':str(file),'sha256':v['unit_sha256'],'input_ids_sha256':q['input_ids_sha256'],'unit_payload_sha256':checksum});expected_count+=1
 assert set(questions)==set(lab);counts=collections.Counter(q['m'] for q in questions.values());assert dict(counts)==EXPECTED[split]
 for q in questions.values():assert set(q['views'])=={(p,s) for p in range(q['m']) for s in range(4)}
 assert expected_count==(17504 if split=='train' else 4776);return [questions[k] for k in sorted(questions)],manifest,core
def native_grid(q,prediction='pred'):
 m=q['m'];return np.array([[[q['views'][p,s][prediction][d]==q['gold'] for d in range(4)] for s in range(4)] for p in range(m)],int)
def analyse_causal(qs,stratum):
 ids=[q['id'] for q in qs];endpoints=[];values=[]
 for prediction in ['pred','semantic_first_pred']:
  descriptors=[('accuracy_K'+str(k+1),'accuracy',k,None) for k in range(4)]+[('gain_K'+str(k+1)+'_K'+str(k+2),'gain',k,k+1) for k in range(3)]
  for name,kind,d,e in descriptors:
   for effect in ['P','S','interaction']:endpoints.append({'endpoint':name,'kind':kind,'depth_from':d+1,'depth_to':None if e is None else e+1,'tie_rule':prediction,'effect':effect,'primary':prediction=='pred' and name=='gain_K1_K2','stratum':stratum})
  for qi,q in enumerate(qs):
   if len(values)<=qi:values.append([])
   grid=native_grid(q,prediction)
   for name,kind,d,e in descriptors:values[qi].extend(contrasts(grid[:,:,d] if kind=='accuracy' else grid[:,:,e]-grid[:,:,d]).tolist())
 ci=intervals(values,ids);result=[{**desc,**r,'estimate_pp':r['estimate']*100,'percentile95_pp':(np.asarray(r['percentile95'])*100).tolist(),'percentile98_333333_pp':(np.asarray(r['percentile98_333333'])*100).tolist(),'inferential_scope':'Three primary adjusted intervals use Bonferroni family alpha.05; secondary intervals descriptive.'} for desc,r in zip(endpoints,ci)];perq=[{'id':key,'values':v} for key,v in zip(ids,values)]
 no_tie=[q for q in qs if not any(v['ties'][:2].any() for v in q['views'].values())];sensitivity=[]
 if no_tie:
  vals=[contrasts(native_grid(q)[:,:,1]-native_grid(q)[:,:,0]) for q in no_tie];cs=intervals(vals,[q['id'] for q in no_tie]);sensitivity=[{'effect':effect,**r,'stratum':stratum,'subset':'No exact BF16 candidate top tie at K1 or K2 in any required view; question list fixed before contrast calculation'} for effect,r in zip(['P','S','interaction'],cs)]
 return result,{'endpoints':endpoints,'per_question':perq},sensitivity
def view_records(qs):
 for q in qs:
  y=q['gold'];m=q['m'];wrong=[j for j in range(m) if j!=y];ident=q['views'][0,0];identity_wrong=min(wrong,key=lambda j:(-ident['logits'][0,j],j))
  for (p,s),v in sorted(q['views'].items()):
   receiver_wrong=min(wrong,key=lambda j:(-v['logits'][0,j],j));depths=[]
   for d in range(4):
    z=v['logits'][d];rank=np.sort(z);depths.append({'depth':d+1,'semantic_logits':z,'semantic_LP_scores':v['scores'][d],'prediction_semantic_first_displayed':v['pred'][d],'prediction_semantic_index_tie':v['semantic_first_pred'][d],'semantic_top_set':v['tops'][d],'exact_BF16_top_tie':v['ties'][d],'correct':v['pred'][d]==y,'semantic_tie_correct':v['semantic_first_pred'][d]==y,'meanwrong_gap':z[y]-z[wrong].mean(),'dynamic_wrong_margin':z[y]-z[wrong].max(),'top1_top2_gap':rank[-1]-rank[-2],'identity_K1_fixedwrong_semantic_index':identity_wrong,'identity_fixedwrong_gap':z[y]-z[identity_wrong],'receiver_K1_fixedwrong_semantic_index':receiver_wrong,'receiver_fixedwrong_gap':z[y]-z[receiver_wrong]})
   yield {'id':q['id'],'split':q['split'],'gold_offline_only':y,'stratum':m,'p':p,'s':s,'variant_id':v['variant_id'],'display_to_original':v['display_to_original'],'targets':v['targets'],'labels':v['labels'],'token_count':v['token_count'],'unit_file':v['unit_file'],'unit_sha256':v['unit_sha256'],'depths':depths,'scope':'All gold-conditioned gaps are descriptive offline metrics; none enters inference.'}
def tables(qs):
 cells=[];consistency=[];diagonal=[]
 for m in sorted({q['m'] for q in qs}):
  group=[q for q in qs if q['m']==m]
  for p in range(m):
   for s in range(4):
    c=np.array([[q['views'][p,s]['pred'][d]==q['gold'] for d in range(4)] for q in group]);tiers=np.array([q['views'][p,s]['ties'] for q in group]);trans=[]
    for d in range(3):
     before,after=c[:,d],c[:,d+1];rescue=int((~before&after).sum());damage=int((before&~after).sum());rw=int((~before).sum());rc=int(before.sum());trans.append({'from':d+1,'to':d+2,'net_gain':rescue-damage,'rescue':rescue,'damage':damage,'rescue_risk_n':rw,'damage_risk_n':rc,'rescue_rate':rescue/rw if rw else None,'damage_rate':damage/rc if rc else None,'stable_wrong':int((~before&~after).sum()),'stable_correct':int((before&after).sum())})
    cells.append({'m':m,'n_questions':len(group),'p':p,'s':s,'correct_K1_K4':c.sum(0),'accuracy_K1_K4':c.mean(0),'exact_top_tie_counts_K1_K4':tiers.sum(0),'transitions':trans})
  for q in group:
   for d in range(4):
    vs=list(q['views'].values());pred=[int(v['pred'][d]) for v in vs];top=[set(v['tops'][d]) for v in vs];it=set(q['views'][0,0]['tops'][d]);correct=[v==q['gold'] for v in pred];consistency.append({'id':q['id'],'m':m,'depth':d+1,'view_count':len(vs),'correct_view_count':sum(correct),'all_correct':all(correct),'all_wrong':not any(correct),'mixed_correctness':any(correct) and not all(correct),'all_predictions_same':len(set(pred))==1,'all_predictions_same_wrong':len(set(pred))==1 and not any(correct),'all_topsets_same':all(t==top[0] for t in top),'all_topsets_contain_gold':all(q['gold'] in t for t in top),'mean_topset_exact_agreement_to_identity':np.mean([t==it for t in top]),'mean_topset_Jaccard_to_identity':np.mean([len(t&it)/len(t|it) for t in top]),'offline_full_view_correctness_oracle':any(correct),'oracle_is_not_a_method':True})
  if m==4:
   for q in group:
    diagonal.append({'id':q['id'],'accuracy_K1_K4_by_p_equals_s':[[q['views'][p,p]['pred'][d]==q['gold'] for d in range(4)] for p in range(4)],'scope':'Query semantic candidate original answer symbol retained, but demos change with s; not a pure whole-prompt position-only intervention.'})
 return cells,consistency,diagonal
def methods_for(q,amod):
 identity=q['views'][0,0];pair=q['views'][q['m']//2,0];x1=identity['scores'][1].astype(np.float64);x2=pair['scores'][1].astype(np.float64);aggregate=((x1-x1.mean())+(x2-x2.mean()))/2;pred=int(aggregate.argmax());result={'fixed_pair_K2_K2':{'prediction':pred,'top_semantic_set':np.flatnonzero(aggregate==aggregate.max()).tolist(),'scores_semantic':aggregate,'nominal_loops':4,'model_body_forwards':2,'head_calls':2,'token_loop_units':2*identity['token_count']+2*pair['token_count'],'permission':'Fixed input-multiview output-only; no gold, no search'}}
 specs={'identity_K4':{'kind':'native','depth':4},'E6_primary_ZL_weighted4':amod.fixed_specs()['weighted4'],'strong_output_cal_early_0.6':amod.calibration_grid()['cal_early_0.6']}
 for name,s in specs.items():
  v=amod.infer(identity['scores'],s);result[name]={'prediction':v['prediction'],'nominal_loops':v['executed_loops'],'selected_depth':v['selected_depth'],'top_tie':v['top_tie'],'model_body_forwards':1,'head_calls':v['executed_loops'] if s['kind']=='early' else (4 if s['kind']=='score' else 1),'token_loop_units':v['executed_loops']*identity['token_count']}
 r=identity['replica']['loopcd_adaptive_cap1'];result['E6_primary_Cal_loopcd_adaptive_cap1']={'prediction':r['prediction_semantic'],'nominal_loops':4,'model_body_forwards':1,'head_calls':2,'token_loop_units':4*identity['token_count'],'uses_real_full_vocabulary_adjustment':True}
 return result
def practical(qs,amod,train=False):
 summaries=[];decisions=[];counter=[];promotion=False
 for m in [3,4,5]:
  group=[q for q in qs if q['m']==m];ids=[q['id'] for q in group];y=[q['gold'] for q in group];rr=[methods_for(q,amod) for q in group];pair=[r['fixed_pair_K2_K2']['prediction'] for r in rr]
  for ref in ['identity_K4','E6_primary_ZL_weighted4','E6_primary_Cal_loopcd_adaptive_cap1','strong_output_cal_early_0.6']:
   r=paired_accuracy(y,pair,[v[ref]['prediction'] for v in rr],[4]*len(group),ids);r.update(m=m,reference=ref,method='fixed_pair_K2_K2',scope='Main four-choice, extensions separately; no offset/pair/weight search',nominal_budget_warning='Two K2 forwards are not established latency-equivalent to one K4; actual online profile NOT_RUN');summaries.append(r)
  for q,r in zip(group,rr):
   decisions.append({'id':q['id'],'split':q['split'],'m':m,'gold_evaluation_only':q['gold'],'methods':r})
   for ref in ['identity_K4','strong_output_cal_early_0.6']:
    delta=int(r['fixed_pair_K2_K2']['prediction']==q['gold'])-int(r[ref]['prediction']==q['gold'])
    if delta:counter.append({'id':q['id'],'m':m,'reference':ref,'delta_correct':delta,'gold':q['gold'],'pair_prediction':r['fixed_pair_K2_K2']['prediction'],'reference_prediction':r[ref]['prediction']})
 main=[r for r in summaries if r['m']==4 and r['reference'] in ['identity_K4','strong_output_cal_early_0.6']];promotion=all(r['paired_question95_pp'][0]>0 for r in main) if train else None;assert len(main)==2
 return summaries,decisions,counter,promotion
def gate_analysis(qs,amod,selected=None,train=False):
 ids=[q['id'] for q in qs];gold=np.array([q['gold'] for q in qs]);controls={**{k:v for k,v in amod.fixed_specs().items() if k in ['K1','K2','K3','K4','early_fixed']},**{k:v for k,v in amod.calibration_grid().items() if k.startswith('cal_early_')}};points=[];rows=[];candidate=[]
 for name,spec in controls.items():
  v=[amod.infer(q['views'][0,0]['scores'],spec) for q in qs];points.append({'name':name,'correct':sum(x['prediction']==q['gold'] for x,q in zip(v,qs)),'n':len(qs),'mean_estimated_prefix_depth':np.mean([x['executed_loops'] for x in v]),'source':'Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid'})
 frontier=[p for p in points if not any(c['correct']>=p['correct'] and c['mean_estimated_prefix_depth']<=p['mean_estimated_prefix_depth'] and (c['correct']>p['correct'] or c['mean_estimated_prefix_depth']<p['mean_estimated_prefix_depth']) for c in points)]
 for threshold in [.25,.5,.75,1.]:
  rr=[q['views'][0,0]['gate']['threshold_readouts'][str(threshold)] for q in qs];prediction=np.array([r['prediction_semantic'] for r in rr]);depth=np.array([r['selected_depth'] for r in rr]);correct=int((prediction==gold).sum());point={'threshold':threshold,'correct':correct,'n':len(qs),'accuracy':correct/len(qs),'mean_estimated_prefix_depth':float(depth.mean()),'offline_actual_full_body_depth':4,'source':'Actual BF16 R gate PDF/CDF after full four-loop capture; technical stock-default full-sequence parity validated separately'};dominated=any(c['correct']>=correct and c['mean_estimated_prefix_depth']<=point['mean_estimated_prefix_depth'] and (c['correct']>correct or c['mean_estimated_prefix_depth']<point['mean_estimated_prefix_depth']) for c in frontier);equal=any(c['correct']==correct and c['mean_estimated_prefix_depth']==point['mean_estimated_prefix_depth'] for c in frontier);point['frontier_eligible']=not dominated and not equal;point['dominated']=dominated;point['exact_existing_point']=equal;candidate.append(point)
  for q,r in zip(qs,rr):rows.append({'id':q['id'],'m':q['m'],'threshold':threshold,'prediction_semantic':r['prediction_semantic'],'selected_depth':r['selected_depth'],'correct':r['prediction_semantic']==q['gold'],'offline_executed_body_depth':4,'possible_prefix_depth_not_measured_latency':r['selected_depth']})
 if not train:
  for point in candidate:point['descriptive_nondominated_Val_only']=point.pop('frontier_eligible');point['online_eligibility_decided_on_Train_only']=True
 choice=None
 if train:
  eligible=[r for r in candidate if r['frontier_eligible']];choice=min(eligible,key=lambda r:(-r['correct'],r['mean_estimated_prefix_depth'],r['threshold']))['threshold'] if eligible else None
 else:choice=selected
 stock={**next(r for r in candidate if r['threshold']==1.),'name':'stock_default_threshold1_full_body','scope':'R readout corresponds to per-token public default at R; all4 body passes executed; not sample-level runtime halting'}
 return {'selection_population':'All original identity Train1094 for Train selection; all exposedVal299 only frozen review','all_fixed_thresholds':candidate,'stock_default':stock,'existing_output_controls':points,'existing_output_frontier':frontier,'selected_threshold':choice,'selection':'Train only: nondominated and not an exact existing point; correct desc, prefixdepth asc, numericthreshold asc. Validation never reselects.','actual_online_latency':'NOT_RUN'},rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['freeze','train','validation'],required=True);p.add_argument('--root',default='/root/autodl-tmp/ouro-e7/reencoding');p.add_argument('--out');p.add_argument('--inputs');p.add_argument('--capture');p.add_argument('--order',default='/root/autodl-tmp/ouro-e7/ASTRA_E7_NATURAL_REENCODING_ORDER.md');p.add_argument('--train-labels',default='/root/autodl-tmp/ouro-e6a/next_step_train_diagnosis/TRAIN_ONLY_INPUTS.jsonl');p.add_argument('--val-labels',default='/root/autodl-tmp/ouro-e7/H2_VALIDATION_ONLY_INPUTS.jsonl');p.add_argument('--methods',default='/root/autodl-tmp/ouro-e6a/E6A/E6A_METHODS.py');p.add_argument('--e6a-spec',default='/root/autodl-tmp/ouro-e6a/E6A/joint5090/E6A_FROZEN_METHOD_SPEC.json');a=p.parse_args();r=Path(a.root);a.out=a.out or str(r/'analysis');a.inputs=a.inputs or str(r/'E7_VARIANTS.jsonl');a.capture=a.capture or str(r/'full_capture');out=Path(a.out);out.mkdir(parents=True,exist_ok=True);sourcefreeze=out/'E7_ANALYSIS_SOURCE_FREEZE.json';hashes=file_hashes(a)
 if a.phase=='freeze':
  assert not sourcefreeze.exists();write(sourcefreeze,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hashes':hashes,'permissions':'CPU only; no model/forward/GPU/fitting/Test; gold only isolated oldTrain1094 or oldexposedVal299','chronology':'Formal causal definitions and unique practical candidate frozen by Astra order/E7 protocol before new variant outcomes. This CPU implementation hash is frozen before any new variant scientific outcome/label comparison was read; technical outputs already exist and were used only for native parity.','coverage_expected':EXPECTED,'units_expected':{'train':17504,'validation':4776,'total':22280},'primary':'fourchoice production-first-displayed semantic correctness gain K1->K2; P/S/interaction whole-question contrast','intervals':{'bootstrap':BOOT,'seed':SEED,'primary_family_n':3,'ordinary_percentile':[.025,.975],'Bonferroni_percentile':[.05/6,1-.05/6],'primary_practical_equivalence_band_pp':[-1,1],'equivalence_rule':'All3 Trainadjusted and samecontrast Val95 intervals entirely within band; no nondetection-as-invariance'},'gap_definitions':{'meanwrong':'gold semantic BF16 logit minus mean all wrong semantic logits','dynamic_wrong_margin':'gold semantic BF16 logit minus maximum wrong semantic logit at current view/depth','identity_fixedwrong':'identity K1 strongest wrong original semantic ID, semantic index breaks ties, tracked across allviews/depths','receiver_fixedwrong':'eachview K1 strongest wrong original semantic ID, held across its laterdepths','top1_top2_gap':'unconditional top1 minus top2 semantic BF16 candidate logit, retained separately','permission':'Gold-conditioned gaps only descriptive; never used by method'},'pair':'Only identityK2 plus p=floor(m/2),s0,K2; FP32LP->FP64 each centered/equalmean; exact ties originalsemanticindex; no pair/grid/weight search','pair_promotion':'fourchoice Train paired accuracy95 lower bounds vsidentityK4 AND fixedstrong cal_early_0.6 entropy.5 both>0; root alone may run realprofile','gate_selection_population':'Original identity allTrain1094, matching oldfrontier; extensions never used to alter primary fourchoice causal estimates','gate_selection':'All4 fixedCDFthresholds .25/.5/.75/1 reported; selectTrain only nondominated strictnewpoint vs oldnative1..4+6Calearly+earlyfixed.9entropy.3 frontier; tie correct desc/depth asc/threshold asc','Val':'Only after Trainfreeze; source definitions unchanged and no newValoutcome read inTrain stage','secondary':'Otheradjacentgains, all-depthP/S accuracy, eachoffset, exacttie semanticsensitivity and noK1/K2-related-tie subset, semantic topset consistency/correct-vs-wrong stability, diagonal boundary, raw gaps'});print('E7_CPU_SOURCE_FROZEN',sha(sourcefreeze));return
 frozen=read(sourcefreeze);assert frozen['hashes']==hashes;split='train' if a.phase=='train' else 'validation';completion=out/(split.upper()+'_COMPLETE.json');assert not completion.exists(),'Never repeat/overwrite a completed scientific evaluation';trainfreeze=out/'E7_TRAIN_FROZEN_DECISIONS.json'
 if split=='validation':assert trainfreeze.exists() and read(trainfreeze)['source_freeze_sha256']==sha(sourcefreeze)
 else:assert not trainfreeze.exists()
 t=time.perf_counter();qs,manifest,core=load(a,split,frozen);amod=import_methods(a.methods);summary=[];perq={};sensitivity=[]
 for m in [3,4,5]:
  group=[q for q in qs if q['m']==m];rr,pr,ss=analyse_causal(group,str(m)+'_choice');summary.extend(rr);perq[str(m)]=pr;sensitivity.extend(ss)
 cells,consistency,diagonal=tables(qs);practical_summary,method_rows,counter,promote=practical(qs,amod,split=='train');gate,gate_rows=gate_analysis(qs,amod,None if split=='train' else read(trainfreeze)['selected_gate_threshold'],split=='train');prefix=split.upper();write(out/(prefix+'_CAUSAL_CONTRASTS.json'),summary);write(out/(prefix+'_NO_RELATED_TIE_SENSITIVITY.json'),sensitivity);save_rows(out/(prefix+'_QUESTION_CONTRASTS.jsonl.gz'),({'m':m,**row} for m,v in perq.items() for row in v['per_question']));write(out/(prefix+'_QUESTION_CONTRAST_DEFINITIONS.json'),{m:v['endpoints'] for m,v in perq.items()});save_rows(out/(prefix+'_VIEW_METRICS.jsonl.gz'),view_records(qs));write(out/(prefix+'_ALL_OFFSET_ACCURACY_TRANSITIONS.json'),cells);save_rows(out/(prefix+'_SEMANTIC_CONSISTENCY.jsonl.gz'),consistency);save_rows(out/(prefix+'_DIAGONAL.jsonl.gz'),diagonal);write(out/(prefix+'_FIXED_PAIR_SUMMARY.json'),practical_summary);save_rows(out/(prefix+'_METHOD_DECISIONS.jsonl.gz'),method_rows);save_rows(out/(prefix+'_PAIR_COUNTEREXAMPLES.jsonl.gz'),counter);write(out/(prefix+'_GATE_BASELINES.json'),gate);save_rows(out/(prefix+'_GATE_DECISIONS.jsonl.gz'),gate_rows);write(out/(prefix+'_INPUT_AND_CAPTURE_QA.json'),{'status':'PASS','n_questions':len(qs),'n_complete_views':len(manifest),'expected_total_study_views':22280,'identity_native_bridges':len(qs),'four_depths_per_view':True,'per_view_prod_first_layer_calls':4,'same_source':core,'all_checks':'Input canonical row SHA, atomicunitchecksum,source/model identities,mappings,predictions,ties,gatefieldcoverage,physicalcounter,alloffsetcoverage,oldidentityhash bridges','records':manifest});write(out/(prefix+'_RUNTIME.json'),{'CPU_seconds':time.perf_counter()-t,'CPU_peak_memory':'NOT_MEASURED','GPU_forwards':0,'model_weights_changed':False,'views':len(manifest),'captured_production_forward_scoring_seconds':sum(v['capture_seconds'] for q in qs for v in q['views'].values()),'actual_online_methods':'NOT_RUN','score_dataset_capture_scope':'Eachview1bodyforward/4physicalpasses/4fullheads; technical extras outside these production counts'})
 consistency_summary=[]
 for m in [3,4,5]:
  group=[q for q in qs if q['m']==m];sc=[c for c in cells if c['m']==m];mc=[np.mean([c['accuracy_K1_K4'][d] for c in sc]) for d in range(4)]
  for d in range(1,5):
   cc=[r for r in consistency if r['m']==m and r['depth']==d];keys=['correct_view_count','all_correct','all_wrong','mixed_correctness','all_predictions_same','all_predictions_same_wrong','all_topsets_same','all_topsets_contain_gold','mean_topset_exact_agreement_to_identity','mean_topset_Jaccard_to_identity','offline_full_view_correctness_oracle'];consistency_summary.append({'m':m,'depth':d,'n_questions':len(group),'mean_equal_question_offset_accuracy':mc[d-1],**{key:np.mean([r[key] for r in cc]) for key in keys},'oracle_is_not_a_method':True})
 write(out/(prefix+'_ACCURACY_AND_CONSISTENCY_SUMMARY.json'),consistency_summary)
 if split=='train':write(trainfreeze,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_freeze_sha256':sha(sourcefreeze),'source_hashes':hashes,'identity_core':core,'selected_gate_threshold':gate['selected_threshold'],'pair_profile_eligible':promote,'pair_promotion_population':'fourchoiceTrain1092','selected_gate_population':'identityTrain1094','strong_output_fixed':'E6 cal_early_0.6 entropy.5, fixed from oldTrain before newvariant outcomes','E6_primary_ZL':'weighted4','E6_primary_Cal':'loopcd_adaptive_cap1','no_fit_or_new_grid':True,'no_new_Val_or_Test_outcome_read':True,'Train_capture_manifest_sha256':sha(out/(prefix+'_INPUT_AND_CAPTURE_QA.json'))})
 else:
  assert read(trainfreeze)['source_hashes']==hashes and core==read(trainfreeze)['identity_core'];tr=[x for x in read(out/'TRAIN_CAUSAL_CONTRASTS.json') if x['primary'] and x['stratum']=='4_choice'];va=[x for x in summary if x['primary'] and x['stratum']=='4_choice'];verdict=[]
  for effect in ['P','S','interaction']:
   trow=next(x for x in tr if x['effect']==effect);vrow=next(x for x in va if x['effect']==effect);tc=trow['percentile98_333333_pp'];vc=vrow['percentile95_pp'];positive=tc[0]>0 and vc[0]>0;negative=tc[1]<0 and vc[1]<0;verdict.append({'effect':effect,'Train_adjusted_interval_pp':tc,'Val95_interval_pp':vc,'replicated_same_direction_protocol_dependency':positive or negative,'direction':'positive' if positive else 'negative' if negative else None,'practical_equivalence_both_intervals_within_1pp':min(tc+vc)>=-1 and max(tc+vc)<=1})
  write(out/'E7_CAUSAL_REVIEW_VERDICT.json',{'contrasts':verdict,'replicated_protocol_dependency':any(r['replicated_same_direction_protocol_dependency'] for r in verdict),'all_primary_average_contrasts_practically_equivalent_within1pp':all(r['practical_equivalence_both_intervals_within_1pp'] for r in verdict),'neither_means':'Inconclusive rather than proof of invariance','interpretation':'For these tested mean loop-gain contrasts only; no writer/label-only attribution or independent benchmark confirmation.'})
 write(completion,{'status':'COMPLETE','split':split,'n':len(qs),'views':len(manifest),'source_sha256':sha(__file__),'source_freeze_sha256':sha(sourcefreeze),'train_decisions_sha256':sha(trainfreeze),'pair_profile_eligible_Train_only':promote if split=='train' else read(trainfreeze)['pair_profile_eligible'],'new_fits':0,'new_GPU_forwards':0,'CPU_seconds':time.perf_counter()-t});report(out,split,summary,practical_summary,gate,sensitivity);print('E7_CPU_PHASE_COMPLETE',split,gate['selected_threshold'],promote if split=='train' else 'TRAIN_DECISION_UNCHANGED',flush=True)
def report(out,split,causal,pair,gate,sensitivity):
 text='# E7 '+split+' frozen natural-reencoding analysis\n\nScientific direction: Astra. All CPU implementation, analysis and review: Sol. Natural inputs only; no latent intervention, new model fit or Test data. Historically exposed ARC/Val is not independent benchmark confirmation.\n\n## Main four-choice K1→K2 gain contrasts\n\nP/S nonzerooffsets average within each question before question averaging; all16 cells retained. Estimates and intervals are percentage points. Primary support uses three-family Bonferroni98.333333% interval; ordinary95% retained.\n\n| Effect | Estimate pp | 95% | 98.333333% |\n|---|---:|---|---|\n'
 for r in causal:
  if r['primary'] and r['stratum']=='4_choice':text+=f"| {r['effect']} | {r['estimate_pp']:+.4f} | {r['percentile95_pp']} | {r['percentile98_333333_pp']} |\n"
 text+='\nAll secondary endpoints, both tie rules, alloffset heterogeneity and separate3/5-choice extensions are retained in machine-readable tables. No-related-tie subset excludes an entire question if any K1/K2 top tie occurs anywhere in its grid; its coverage is explicit. Nonsignificance is not invariance: practical equivalence requires every relevant primary interval within±1pp. A consistent semantic prediction can be consistently wrong; perdepth allcorrect/allsamewrong/mixedcorrectness and exact top-set/Jaccard summaries are separate. Full-view correctness oracle is offline gold only. The p=s query-symbol-preserving diagonal still changes demo symbols, so it is not a pure whole-prompt position-only contrast.\n\n## Unique fixed pair versus frozen controls\n\n| Choices | Comparator | Pair correct/N | Net | Gain95% pp |\n|---|---|---:|---:|---|\n'
 for r in pair:text+=f"| {r['m']} | {r['reference']} | {r['correct']}/{r['n']} | {r['net_correct']:+d} | {r['paired_question95_pp']} |\n"
 text+='\nTwo K2 views nominally total4 loop passes but execute two prompts/forwards/heads. Token-loop units and full method records are retained; no equality of real latency is inferred. No alternative pair, offset, weight or oracle is selected. Train fourchoice profile permission is separately frozen before exposedVal outcomes are read.\n\n## Existing gate baselines\n\nAll4 fixed thresholds plus stock-default threshold1 are full-body offline readouts and cost4 actual body passes. Their estimatedprefix depth is not actual halting work or measured speed. Native1..4,6 oldcalearly and original .9/entropy.3 early output-only controls define the existing estimatedbudget frontier. Threshold selection uses only identity allTrain1094; no threshold is reselected for a subset or Validation.\n\n'+json.dumps(gate,default=default,indent=2)+'\n\n## Scope and reproducibility\n\nCPUsource freeze occurred before newvariant scientific outcome/label comparisons were read, after technical parity outputs already existed; the scientific definitions and unique pair were frozen earlier in the Astra order/E7protocol. No claim is made that the implementation hash predates every model outcome. Source/input/isolatedlabel/helper hashes, canonicalvariant checksums, allcoverage and native bridges are recorded. Each question remains one bootstrap block across alloffsets/depths. Ordinary and adjusted intervals describe empirical question resampling, not training seeds or fresh-dataset generalization.\n\nGold is used only for outcome/risk/gap summaries. The dynamic wrong margin is correctsemantic logit minus current strongestwrong, while top1−top2 is retained separately. IdentityK1 strongestwrong tracks the same originalcandidate across allviews; receiverK1 strongestwrong is fixed within thatview. These definitions never enter the inference candidate.\n\nThe practical research goal remains open until a frozen method improves a real measured accuracy–cost frontier and survives independent-data confirmation. A causal input-response effect cannot by itself identify a specific writer, pure label bias, or method success. P changes option causal context and S changes demo/query symbol tokens. H2 G1 remains failed and no E6C is authorized.\n'
 (out/(split.upper()+'_REPORT.md')).write_text(text)
if __name__=='__main__':main()
