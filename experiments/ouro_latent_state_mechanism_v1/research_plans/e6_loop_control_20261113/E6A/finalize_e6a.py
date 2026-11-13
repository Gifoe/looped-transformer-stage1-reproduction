"""Read-only postprocessing of frozen outputs; never selects or fits a method."""
import json, hashlib, collections, gzip, shutil
from pathlib import Path
import numpy as np
from E6A_METHODS import infer

ROOT=Path(__file__).resolve().parent
RUN=ROOT.parent
LABELS=Path('/root/autodl-tmp/ouro-e4-e5/inputs/labels.json')
def read(p): return json.loads(Path(p).read_text())
def write(p,x): Path(p).write_text(json.dumps(x,indent=2),encoding='utf8')
def text(p,s): Path(p).write_text(s,encoding='utf8')
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ci(d):
 d=np.asarray(d,dtype=float);rng=np.random.default_rng(20261113)
 return np.quantile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1)*100,[.025,.975]).tolist()
def table(rows):
 s='| Method | Correct / 1172 | Accuracy % | vs K4 pp | Rescue / damage | Paired 95% pp CI | Mean loops (replay) |\n|---|---:|---:|---:|---:|---|---:|\n'
 for r in rows:
  lo,hi=r['paired_bootstrap95_pp'];s+=f"| {r['method']} | {r['correct']} | {100*r['accuracy']:.3f} | {r['pp_vs_K4']:+.3f} | {r['rescue']} / {r['damage']} | [{lo:+.3f}, {hi:+.3f}] | {r['mean_loops']:.4f} |\n"
 return s
def captures(protocol,split):
 rr=[read(p) for p in sorted((RUN/protocol/split).glob('*.json'))]
 return [q for q in rr if isinstance(q,dict) and {'id','split','native'}<=set(q)]
def native_audit(protocol,labels):
 out={};records=[]
 for split in ['train','validation','test']:
  rr=[q for q in captures(protocol,split) if not labels[q['split']+':'+q['id']].get('is_demonstration',False)]
  pp=np.array([[n['prediction'] for n in q['native']] for q in rr]);gg=np.array([labels[q['split']+':'+q['id']]['gold'] for q in rr]);c=pp==gg[:,None]
  out[split]={'n':len(rr),'native_correct':c.sum(0).tolist(),'rescues':[int((~c[:,i]&c[:,i+1]).sum()) for i in range(3)],'damages':[int((c[:,i]&~c[:,i+1]).sum()) for i in range(3)],'oracle_any_correct':int(c.any(1).sum()),'K4_wrong_previously_correct':int((~c[:,3]&c[:,:3].any(1)).sum()),'all_wrong':int((~c.any(1)).sum()),'prediction_disagreement':int((pp!=pp[:,[0]]).any(1).sum()),'candidate_counts':dict(collections.Counter(len(q['native'][0]['scores']) for q in rr))}
  for q,correct in zip(rr,c):
   groups=[]
   for k in range(1,4):groups.append('beneficial' if not correct[k-1] and correct[k] else 'harmful' if correct[k-1] and not correct[k] else 'stable_correct' if correct[k] else 'stable_wrong')
   records.append({'id':q['id'],'split':split,'native_correct':correct.tolist(),'adjacent_utility_groups':groups,'first_correct_depth':int(np.argmax(correct)+1) if correct.any() else None,'oracle_evaluation_only':True})
 write(ROOT/('A0_'+protocol.upper()+'_NATIVE_ORACLE_AUDIT.json'),out)
 with gzip.open(ROOT/('A0_'+protocol.upper()+'_ORACLE_GROUPS.jsonl.gz'),'wt',encoding='utf8') as f:
  for r in records:f.write(json.dumps(r)+'\n')
 return out
def online(labels):
 rr=[json.loads(l) for l in (ROOT/'online_validation299/rows.jsonl').read_text().splitlines()];out={}
 for name in sorted({r['method'] for r in rr}):
  x=[r for r in rr if r['method']==name];first=x[0]['id'];warm=[r for r in x if r['id']!=first]
  out[name]={'n':len(x),'correct':sum(r['prediction']==labels[r['split']+':'+r['id']]['gold'] for r in x),'actual_mean_loops':float(np.mean([r['actual_executed_loops'] for r in x])),'mean_seconds':float(np.mean([r['seconds'] for r in x])),'median_seconds':float(np.median([r['seconds'] for r in x])),'mean_seconds_excluding_first_question':float(np.mean([r['seconds'] for r in warm])),'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in x),'actual_first_layer_calls_equal_depth':all(r['first_layer_calls']==r['actual_executed_loops'] for r in x)}
 return out
def resource(protocol):
 out={}
 for split in ['train','validation','test']:
  rr=captures(protocol,split);rece=read(RUN/protocol/split/'COMPLETE.json')
  body=len(rr) if protocol=='joint' else sum(len(q['native'][0]['scores']) for q in rr)
  out[split]={'n_including_demonstrations':len(rr),'actual_body_forwards':body,'actual_full_shaped_head_calls':4*body,'actual_physical_loop_applications':4*body,'complete_receipt':rece,'sum_instrumented_seconds':sum(q['wall_seconds'] for q in rr),'mean_instrumented_seconds':float(np.mean([q['wall_seconds'] for q in rr])),'feature_cpu_and_storage_seconds':sum(q.get('feature_cpu_and_storage_seconds',0) for q in rr),'saved_state_bytes':sum(q.get('state_storage_bytes',0) for q in rr),'peak_allocated_bytes':max(q['peak_allocated_bytes'] for q in rr)}
 return out
def main():
 labels=read(LABELS);joint=ROOT/'joint5090';fs=read(joint/'E6A_FROZEN_METHOD_SPEC.json');jt=read(joint/'TEST_SUMMARY.json');assert (joint/'TEST_COMPLETE.json').exists()
 fas=ROOT/'fas_char5090';assert (fas/'TEST_COMPLETE.json').exists();ft=read(fas/'TEST_SUMMARY.json');assert read(fas/'E6A_FROZEN_METHOD_SPEC.json')['primary_Cal']==fs['primary_Cal']
 # Preserve initial historical reports without editing their source or numbers.
 hist=ROOT/'historical_A6000';hist.mkdir(exist_ok=True)
 for p in list(ROOT.glob('E6A_*.md'))+list(ROOT.glob('E6A_RUNTIME_AND_MEMORY.json'))+list(ROOT.glob('EARLY_EXIT_PARETO.json')):
  if not (hist/p.name).exists():shutil.copy2(p,hist/p.name)
 for name in ['E6A_FROZEN_METHOD_SPEC.json','E6A_ALL_RESULTS.jsonl','DEVELOPMENT_SUMMARY.json','DEVELOPMENT_RESULTS.jsonl','TEST_SUMMARY.json','TEST_EVALUATION_RECEIPT.json','A0_NATIVE_ORACLE_AUDIT.json','E6A_DATA_AUDIT.json','build_reports.py','frozen_test_execution.log']:
  p=ROOT/name
  if p.exists() and not (hist/name).exists():shutil.copy2(p,hist/name)
 header='Fresh RTX5090 / BF16 / torch2.7.1+cu128, 1172 Test questions. Every method shares the same frozen model, prompt, candidate set and scoring protocol. Historical A6000 scores remain separate. All 95% intervals use 10000 paired question bootstraps, seed20261113, without multiplicity correction. The Test set had prior project exposure. No Test-driven refit or parameter revision was performed.\n\n'
 text(ROOT/'E6A_PROTOCOL.md','# E6-A completed execution protocol\n\n'+header+f"Primary ZL: `{fs['primary_ZL']}` fixed before development outcomes. Primary Cal: `{fs['primary_Cal']}`, chosen using Joint Validation among32 predefined core methods and2 fixed repository LoopCD controls, ties by lower loops/name. Frozen spec SHA256 `{digest(joint/'E6A_FROZEN_METHOD_SPEC.json')}`. Label-based selection is Cal even when selecting a fixed baseline. Full-Answer transfer reuses this exact choice without Full-Answer-label selection. Character-normalized continuation log likelihood is primary; raw sum and token mean are predeclared secondary protocols.\n\n"+
 'Train1119 includes25 demonstrations; analyses exclude those25 (1094), Validation299, Test1172. Candidate counts3/4/5 and original IDs are retained. Input, model, loaded source, collector and frozen evaluation hashes are recorded. Technical gates use full vocabulary endpoint byte hashes against independent unhooked forwards at each depth; cap0 equals nativeK4. Hooks save R-only natural states; no model weights change.\n\nFusion centers each loop across candidates. Candidate softmax confidence is not a calibrated correctness probability. Raw score and log probability fusion with identical weights differ only by a common offset. Rank ties are average ranks; prediction ties preserve first-index argmax. Exact and near duplicate audits are descriptive; rows were not removed after outcomes. No-tie sensitivity is in machine-readable tables. Macro-F1 over question-local answer positions is not a semantic-class statistic.\n\nAll K4-informed rollback/selection/fusion methods pay4 loops. Replay exit loop counts are legal-prefix cost models, not measured capture latency; full collection always executes4 loops. Real online Joint Validation exits and measured overhead are reported separately. Full-Answer online adaptive scheduling was not run; Full-Answer exit cost is replay only. Oracle uses gold for evaluation only and never controls inference. The old serialization failure was preserved; v2 only casts a metadata NumPy boolean, before fresh Test evaluation. Astra did not perform E6 analysis or execution. After all method freezes the user authorized Astra only to issue a separate next-step scientific directive; all data analysis and implementation remain with current Sol.\n')
 kinds=fs['methods'];fusion=lambda r:r['method'] in kinds and kinds[r['method']]['kind'] in ['native','score','probability','log_probability','rank']
 text(ROOT/'E6A_FUSION_RESULTS.md','# Frozen fusion results\n\n'+header+table([r for r in jt if fusion(r)])+'\nThe declared weighted4 primary does not improve NativeK4. No favorable exploratory Test row is promoted to primary.\n')
 text(ROOT/'E6A_DEPTH_SELECTION.md','# Frozen depth-selection results\n\n'+header+table([r for r in jt if r['method'] in kinds and kinds[r['method']]['kind'] not in ['score','probability','log_probability','rank','early']])+'\nSelecting an earlier endpoint after observing K4 consumes the full4-loop budget.\n')
 early=[r for r in jt if r['method']=='K4' or (r['method'] in kinds and kinds[r['method']]['kind']=='early')];front=[r for r in early if not any(s['accuracy']>=r['accuracy'] and s['mean_loops']<=r['mean_loops'] and (s['accuracy']>r['accuracy'] or s['mean_loops']<r['mean_loops']) for s in early)];write(ROOT/'EARLY_EXIT_PARETO.json',{'scope':'descriptive frozen-rule replay frontier, not Test-selected policy','rows':front})
 on=online(labels);write(ROOT/'E6A_ONLINE_VALIDATION_SUMMARY.json',on)
 text(ROOT/'E6A_EARLY_EXIT.md','# Early exits: replay and actual execution\n\n'+header+table(early)+'\nTest rows above are prefix replay. No actual Test latency is claimed. Cal early0.99 was a historical-development choice, while the fresh5090 Cal primary is LoopCD; these identities are kept separate.\n\nActual Joint Validation299 online forwards stop at the selected prefix via an exception before later loops; physical first-layer calls equal selected depth and full-vocabulary endpoint hashes equal independently captured native prefixes. Alternating policy order controls warm-order effects. Timings include instrumented identity/hash checks and extra intermediate heads, exclude model loading, and are not uninstrumented production throughput.\n\n'+json.dumps(on,indent=2)+'\n')
 baseline=[r for r in jt if r['method'] in ['K1','K2','K3','K4',fs['primary_ZL'],fs['primary_Cal'],'loopcd_adaptive_cap1','loopcd_fixed_0.5']]
 text(ROOT/'E6A_BASELINE_COMPARISON.md','# Matched baseline comparison\n\n'+header+table(baseline)+'\nRepository LoopCD-logits controls: referenceK1, adaptivecap1 and fixedstrength0.5. Adaptive strength uses the full-vocabulary top1/top2 probability gap, not candidate-only confidence; corrected full logits are normalized over the full vocabulary before answer scoring. Cap0 identity passed. Source SHA2563611a505185be5817876581f63b322b1d4c7785d25d5a3078546aba7fbc622bb; original protocol SHA25634041068b2fad3725166406f8360b21ea5cf7e3a91c9edcdabe5f7fd38e13246. This is a matched repository control, not a claim to reproduce every original paper condition. Historical candidate-only archives cannot supply the adaptive gap; that old comparison remains NOT RUN.\n')
 secondary=''
 for name in ['fas_sum5090','fas_token5090']:
  assert (ROOT/name/'TEST_COMPLETE.json').exists();secondary+='\n## Predeclared secondary '+name+'\n\n'+table(read(ROOT/name/'TEST_SUMMARY.json'))
 text(ROOT/'E6A_STANDARD_FULL_ANSWER_TRANSFER.md','# Standard Full-Answer transfer\n\n'+header+'Question-only25-shot prompts with complete correct demonstration answers and no offered choices in context. Each candidate full text is a continuation with a leading space; context+answer tokenization is checked at the boundary. No EOS score is added. Primary score is total full-vocabulary FP32 continuation log likelihood divided by answer character count excluding the leading space. Each native depth and LoopCD use this same protocol. Joint calibrated identity and thresholds are reused unchanged, with zero Full-Answer refitting/reselection.\n\n'+table(ft)+'\nThe tables disclose all frozen methods; only the original Joint-designated primary identities are confirmatory comparisons in this transfer.\n'+secondary)
 audits={p:native_audit(p,labels) for p in ['joint','full_answer']};runtime={'joint':resource('joint'),'full_answer':resource('full_answer'),'actual_online_Joint_Validation':on,'actual_online_Full_Answer':'NOT RUN; replay only','weights_modified':False,'Astra_used_for_E6_analysis_or_execution':False,'Astra_next_step_director_authorized':True,'hardware':'RTX5090','timing_scope':'Shared observers plus4 full-shaped heads and LoopCD controls; per-question CPU/storage time separately reported. Not each method standalone latency.'};write(ROOT/'E6A_RUNTIME_AND_MEMORY.json',runtime)
 fresh=captures('joint','test');old=np.array([[int(s==labels['test:'+q['id']]['gold']) for s in labels['test:'+q['id']]['historical_predictions']] for q in fresh]);new=np.array([[n['prediction']==labels['test:'+q['id']]['gold'] for n in q['native']] for q in fresh]);agreement=[sum(q['native'][k]['prediction']==labels['test:'+q['id']]['historical_predictions'][k] for q in fresh) for k in range(4)]
 write(ROOT/'HISTORICAL_REFRESH_AUDIT.json',{'n':len(fresh),'old_hardware':'RTX A6000 / torch2.5.1cu124','new_hardware':'RTX5090 / torch2.7.1cu128','old_correct':old.sum(0).tolist(),'new_correct':new.sum(0).tolist(),'prediction_agreement_by_depth':agreement,'interpretation':'Hardware/runtime numerical refresh, not a method gain. Method comparisons never mix historical scores with fresh native baselines.'})
 # Paired protocol changes and pairwise existing baseline controls, no method selection.
 paired={};results={}
 for name in ['joint5090','fas_char5090']:
  rr=[json.loads(l) for l in (ROOT/name/'E6A_ALL_RESULTS.jsonl').read_text().splitlines() if json.loads(l)['split']=='test'];results[name]={}
  for r in rr:results[name].setdefault(r['method'],{})[r['id']]=r
  for method in ['weighted4',fs['primary_Cal']]:
   for base in ['K4','loopcd_adaptive_cap1','loopcd_fixed_0.5']:
    a=results[name][method];b=results[name][base];keys=sorted(a);d=np.array([int(a[k]['correct'])-int(b[k]['correct']) for k in keys]);paired[name+':'+method+' vs '+base]={'n':len(d),'net_correct':int(d.sum()),'pp':100*float(d.mean()),'paired95_pp':ci(d)}
 for method in ['K4','weighted4',fs['primary_Cal']]:
  a=results['joint5090'][method];b=results['fas_char5090'][method];keys=sorted(a);assert set(a)==set(b);d=np.array([int(b[k]['correct'])-int(a[k]['correct']) for k in keys]);paired['FullAnswer minus Joint:'+method]={'n':len(d),'net_correct':int(d.sum()),'pp':100*float(d.mean()),'paired95_pp':ci(d),'interpretation':'Paired change of the complete scoring/prompt protocol, not a causal method gain'}
 write(ROOT/'PAIRED_PROTOCOL_AND_BASELINE_COMPARISONS.json',paired)
 equivalence={}
 for folder,protocol,agg in [('joint5090','joint','scores'),('fas_char5090','full_answer','scores'),('fas_sum5090','full_answer','raw_sum'),('fas_token5090','full_answer','mean_token')]:
  spec=read(ROOT/folder/'E6A_FROZEN_METHOD_SPEC.json');mismatch=[];maxerr=0.
  for q in captures(protocol,'test'):
   x=np.array([n[agg] for n in q['native']]);aa=infer(x,spec['methods']['uniform234']);bb=infer(x,spec['methods']['log_probability']);u=np.array(aa['scores']);v=np.array(bb['scores']);err=float(np.max(np.abs((u-u.mean())-(v-v.mean()))));maxerr=max(maxerr,err)
   if aa['prediction']!=bb['prediction']:
    su=np.sort(u);sv=np.sort(v);mismatch.append({'id':q['id'],'raw_prediction':aa['prediction'],'log_probability_prediction':bb['prediction'],'raw_top_gap':float(su[-1]-su[-2]),'log_probability_top_gap':float(sv[-1]-sv[-2]),'centered_vector_max_error':err})
  equivalence[folder]={'n':1172,'centered_fused_vector_max_abs_error':maxerr,'finite_precision_argmax_differences':mismatch,'conclusion':'Algebraically equivalent rules, not independent discoveries. Predeclared exact argmax outputs retained, including floating-point near-tie differences; no post-Test tie-rule edits.'}
 write(ROOT/'FUSION_EQUIVALENCE_AUDIT.json',equivalence)
 with (ROOT/'E6A_FUSION_RESULTS.md').open('a') as doc:doc.write('\nRaw uniform234 and log-probability averaging with the same weights are algebraically equivalent. Tiny floating-point perturbations can resolve an exactly/near tied fused winner differently; FUSION_EQUIVALENCE_AUDIT.json lists every discrepancy and score gap. Both frozen outputs are preserved, not treated as two independent method findings.\n')
 cases={};case_text='# Primary fusion failure analysis and counterexamples\n\nAll gold-dependent classifications below are offline evaluation only; they do not change methods or select a new Test winner.\n\n'
 for name,protocol in [('joint5090','joint'),('fas_char5090','full_answer')]:
  raw={q['id']:q for q in captures(protocol,'test')};rr=results[name]['weighted4'];groups=collections.Counter();examples=[]
  for key in sorted(rr):
   r=rr[key];q=raw[key];gold=labels['test:'+key]['gold'];correct=[n['prediction']==gold for n in q['native']];event='rescue' if r['correct'] and not correct[3] else 'damage' if not r['correct'] and correct[3] else 'unchanged';native_group='K4_only_correct' if correct[3] and not any(correct[:3]) else 'earlier_recoverable_K4_error' if not correct[3] and any(correct[:3]) else 'all_native_wrong' if not any(correct) else 'K4_and_earlier_correct';groups[event+':'+native_group]+=1
   if event!='unchanged':examples.append({'id':key,'event':event,'native_group':native_group,'native_predictions':[n['prediction'] for n in q['native']],'gold_offline_only':gold,'weighted4_prediction':r['prediction'],'native_scores':[n['scores'] for n in q['native']]})
  cases[name]={'counts':dict(groups),'all_changed_questions':examples};case_text+='## '+name+'\n\n'+json.dumps(dict(groups),indent=2)+'\n\n'
  for event in ['rescue','damage']:
   case_text+=event.title()+' examples (first5 by ID): '+', '.join([x['id'] for x in examples if x['event']==event][:5])+'\n\n'
 case_text+='Fusion changes decisions by mixing earlier, sometimes incorrect candidate competitions into K4. Native oracle headroom is not evidence that a label-free selector can identify those corrections. Full counts and every changed primary question are in E6A_PRIMARY_COUNTEREXAMPLES.json. Point estimates and intervals support failure to establish the declared primary gain; they do not imply that every possible fusion rule is impossible.\n'
 write(ROOT/'E6A_PRIMARY_COUNTEREXAMPLES.json',cases);text(ROOT/'E6A_COUNTEREXAMPLES.md',case_text)
 j={r['method']:r for r in jt};f={r['method']:r for r in ft}
 text(ROOT/'E6A_ASTRA_SCIENTIFIC_REVIEW.md','# Current-model scientific self-review\n\nThis is the current Sol self-review. Astra did not analyze E6 data or implement/run E6. After freezing and evaluating the Joint experiments, the user authorized Astra only to direct the next research step; that directive is a separate document. Required legacy filename retained for compatibility.\n\n'+f"Fresh Joint NativeK4={j['K4']['correct']}/1172; primaryZL weighted4={j['weighted4']['correct']}; Validation-selected Cal {fs['primary_Cal']}={j[fs['primary_Cal']]['correct']}. Full-Answer primary NativeK4={f['K4']['correct']}; transferred weighted4={f['weighted4']['correct']}; transferred Cal={f[fs['primary_Cal']]['correct']}. The declared primary comparisons, their paired intervals and full negative results are retained.\n\n"+'Output fusion has not established a robust improvement over K4 in Joint scoring. Early exit can preserve much of K4 accuracy with fewer replay loops, but deployment claims require actual measured online execution, reported only for Joint Validation. The Full-Answer result is evaluated independently; any protocol-dependent reversal is a limit on generalization. Existing Test exposure, limited299-item Validation, multiple exploratory rules, character-normalization dependence, near duplicate audit limitations and measured instrumentation overhead constrain claims. No E6-C experiments or unrelated benchmark expansion were performed.\n')
 for name in ['E6A_FROZEN_METHOD_SPEC.json','E6A_ALL_RESULTS.jsonl','DEVELOPMENT_SUMMARY.json','DEVELOPMENT_RESULTS.jsonl','TEST_SUMMARY.json']:
  shutil.copy2(joint/name,ROOT/name)
 shutil.copy2(joint/'TEST_COMPLETE.json',ROOT/'TEST_EVALUATION_RECEIPT.json')
 write(ROOT/'A0_NATIVE_ORACLE_AUDIT.json',audits['joint'])
 write(ROOT/'E6A_DATA_AUDIT.json',{'n':2590,'split_sizes_including_demonstrations':{'train':1119,'validation':299,'test':1172},'train_after_excluding_25_demonstrations':1094,'candidate_counts':{'4':2577,'3':8,'5':5},'hardware':'Fresh homogeneous RTX5090, historical A6000 separately preserved','native_identity':fs['native_identity'],'historical_input_and_label_identity_audit':'historical_A6000/E6A_DATA_AUDIT.json','near_duplicates':read(ROOT/'E6A_NEAR_DUPLICATE_AUDIT.json'),'row_exclusions':'Only the predefined25 demonstrations from Train analysis; no duplicate/Test-driven exclusions','fresh_native_audits':audits,'gold_separation':'Gold-free capture inputs; labels only in offline analysis and allowed Train diagnostic fitting / Validation calibration','source_and_score_hashes':'Per-question capture records and frozen dev/Test receipts','historical_exposure':'E3 Train1030/Validation299 informed mechanism choice; ARC Test repeatedly exposed, not external blind confirmation'})
 text(hist/'README.md','Historical A6000 CPU recovery and original frozen evaluation. These are separate from fresh5090 outcomes in the parent. Initial prose may contain PENDING refresh markers; it is preserved as provenance and superseded by completed parent reports. Original historical driver build_reports.py is retained only as archival source.\n')
 write(ROOT/'E6A_COMPLETION_RECEIPT.json',{'status':'COMPLETE','protocols':['Joint','FullAnswer characters primary','FullAnswer raw sum secondary','FullAnswer token mean secondary'],'joint_spec_sha256':digest(joint/'E6A_FROZEN_METHOD_SPEC.json'),'fas_spec_sha256':digest(fas/'E6A_FROZEN_METHOD_SPEC.json'),'joint_test':digest(joint/'TEST_COMPLETE.json'),'fas_test':digest(fas/'TEST_COMPLETE.json'),'Astra_used_for_E6_analysis_or_execution':False,'Astra_next_step_director_authorized':True,'weights_modified':False,'native_audits':audits})
 print('E6A_FINAL_REPORTS_COMPLETE',{'Joint_K4':j['K4']['correct'],'FullAnswer_K4':f['K4']['correct'],'online':on},flush=True)
if __name__=='__main__':main()
