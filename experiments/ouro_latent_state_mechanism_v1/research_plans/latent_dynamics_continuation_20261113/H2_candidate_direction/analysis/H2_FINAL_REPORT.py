"""Postprocess frozen H2 evidence; no estimator fit, policy selection or GPU."""
import json,hashlib,sys,importlib.util,gzip,shutil,datetime
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/ouro-e7');E=Path('/root/autodl-tmp/ouro-e6a');O=R/'analysis';P=R/'H2_analysis_release'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
spec=importlib.util.spec_from_file_location('original_train_diagnosis',E/'train_dev_diagnosis.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert read(O/'H2_TRAIN_COMPLETE.json')['status']=='COMPLETE' and read(O/'H2_VALIDATION_COMPLETE.json')['status']=='COMPLETE'
freeze=read(O/'H2_TRAIN_FROZEN_SPEC.json');assert sha(R/'H2_ANALYSIS.py')==freeze['hashes']['analysis_source'];assert sha(O/'H2_FINAL_TRAIN_MODELS.npz')==freeze['final_models_sha256'];pred=rows(O/'H2_OOF_PREDICTIONS.jsonl');folds=[];foldcomp=[]
for outer in range(5):
 for setting in ['retrospective','prospective']:
  for family in ['output','combined','unrelated','shuffled']:
   rs=[r for r in pred if r['outer_fold']==outer and r['setting']==setting and r['family']==family];p=np.asarray([r['probabilities'] for r in rs]);y=np.asarray([r['utility_class'] for r in rs]);folds.append({'outer_fold':outer,'setting':setting,'family':family,'n_questions':len({r['id'] for r in rs}),**m.metrics(p,y)})
  rs=[r for r in pred if r['outer_fold']==outer and r['setting']==setting];out=mappings={}
  for family in ['output','unrelated','shuffled']:
   target={(r['id'],r['depth']):r['brier'] for r in rs if r['family']=='combined'};control={(r['id'],r['depth']):r['brier'] for r in rs if r['family']==family};ids=sorted({k[0] for k in target});d=[np.mean([control[k]-target[k] for k in target if k[0]==i]) for i in ids];foldcomp.append({'outer_fold':outer,'setting':setting,'comparator':family,'n_questions':len(ids),'Brier_improvement':float(np.mean(d)),'paired_question_bootstrap95':m.bootstrap(d),'scope':'Descriptive fold stability, not an alternative G1 or selector'})
write(O/'H2_FOLD_METRICS.json',folds);write(O/'H2_FOLD_PAIRED_COMPARISONS.json',foldcomp)
matched=[]
for scope,file in [('OOF','H2_OOF_ACTIONS.jsonl'),('Validation','H2_VALIDATION_ACTIONS.jsonl')]:
 act=rows(O/file)
 for setting in ['retrospective','prospective']:
  for family in ['output','unrelated','shuffled']:
   hs={r['id']:r for r in act if r['setting']==setting and r['family']=='combined'};cs={r['id']:r for r in act if r['setting']==setting and r['family']==family};ids=sorted(hs);delta=[int(hs[k]['correct'])-int(cs[k]['correct']) for k in ids];matched.append({'scope':scope,'setting':setting,'comparator':family,'n':len(ids),'net_correct':sum(delta),'accuracy_improvement':float(np.mean(delta)),'paired_question_bootstrap95':m.bootstrap(delta),'mean_replay_loop_difference':float(np.mean([hs[k]['replay_loops']-cs[k]['replay_loops'] for k in ids]))})
write(O/'H2_MATCHED_FIXED_ACTION_COMPARISONS.json',matched)
complete=read(R/'natural/COMPLETE.json');capture=[]
for file in (R/'natural').glob('*.json'):
 r=read(file)
 if 'features' in r:capture.append(r)
assert len(capture)==1393
runtime={'analysis_source_sha256':sha(R/'H2_ANALYSIS.py'),'postprocessor_sha256':sha(__file__),'collector_complete':complete,'train_cpu_seconds':read(O/'H2_TRAIN_COMPLETE.json')['cpu_wall_seconds'],'analysis_new_GPU_forwards':0,'train_n':1094,'exposed_validation_n':299,'Test_examples_loaded':0,'CPU_peak_memory':'NOT_MEASURED','validation_CPU_seconds':'NOT_MEASURED_SEPARATELY','capture_cpu_lens_storage_seconds':sum(r['cpu_lens_storage_seconds'] for r in capture),'capture_state_bytes':sum(r['states_bytes'] for r in capture),'capture_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in capture),'independent_lens_QA':read(R/'H2_LENS_INDEPENDENT_QA.json'),'actual_online_policy_latency':'NOT_RUN_G1_FAILED','weights_modified':False};write(O/'H2_RUNTIME_AND_MEMORY.json',runtime)
foldtable='\n## Fixed five-fold prospective stability\n\nDescriptive only; the aggregate preregistered G1 remains unchanged.\n\n| Outer fold | Output Brier | H2 combined Brier | Low-layer Brier | Shuffle Brier |\n|---|---:|---:|---:|---:|\n'
for fold in range(5):
 vals={r['family']:r['brier'] for r in folds if r['setting']=='prospective' and r['outer_fold']==fold};foldtable+=f"| {fold} | {vals['output']:.6f} | {vals['combined']:.6f} | {vals['unrelated']:.6f} | {vals['shuffled']:.6f} |\n"
with (O/'H2_TRAIN_REPORT.md').open('a') as f:f.write(foldtable+'\nOOF output control reproduced all5470 original transition probabilities to maximum absolute difference4.44e-16. Raw records, frozen outer coefficients, alpha inner scores and exact group/fold manifest are preserved. The tiny difference is floating-point roundoff, not a changed baseline.\n')
md='''# H2 scientific assessment and current decision

All analysis, code and review here are by Sol. Astra is the director of the separately frozen research hypothesis; this file is not Astra-authored review. Original Ouro weights remain unchanged.

The primary prospective gate fails. Candidate-aligned H2 additions reduce aggregate output-only Brier by only0.0001181, with a question-bootstrap interval spanning zero; the matched low-layer block has lower Brier than the target block. Shuffle and old-geometry contrasts also span zero. Retrospective proper-loss differences and improved logloss do not replace the frozen prospective proper-loss criterion. This is failure to establish incremental future-update information under this fixed diagnostic, not proof that no hidden-state information exists.

The causal direction from prior E3 counterfactual contexts and these natural candidate-projected scalars are different objects. H2 specifically addressed vector orientation by observing the actual per-layer residual-add output and pre-state, applying checkpoint final RMSNorm and candidate head rows, centering their difference, then taking temporal changes. Endpoints matched all four cached native depths, and independent operator QA checked the numerical lens. Its failure cannot be attributed to substituting the old mean residual state. The prescribed summaries still compress7 candidate-vector directions into16 numbers, and final-normalization candidate projections at middle layers are a lens rather than the downstream final decision function.

Small nominal effects remain compatible with noise, redundancy with output features, generic update signal present in lower layers, and poor utility calibration of the fixed class-balanced ridge/softmaxT1 head. The estimator's limited probability calibration is visible in ECE/logloss, but changing calibration or model complexity after seeing H2 outcomes would be a new hypothesis. No threshold/model-family/policy search is performed to repair this result.

The fixed supervised class-to-action actions are offline diagnostic replays. Their correct counts, paired question uncertainty, native and matched output controls remain visible. They are not evidence of a training-free policy or true latency savings. As G1 failed, G2 online execution is not authorized by the frozen gate and was not run. No Test examples entered this stage. Validation299 is a once-after-freeze exposed review, not independent confirmation and not a route to select another method.

E3 Train1030 and Validation299 were used for mechanism selection; historical Test exposure and related prior diagnostics remain disclosed. Exact/known duplicate questions share fixed groups/folds, and all depths of each question are held together. Repeated cross-validation training sets are correlated, so the conditional question-bootstrap does not capture new-dataset or training-seed generalization.

The practical goal of improving final accuracy/compute over output-only and strong baselines remains unmet. Completing H2 capture/analysis is evidence completion, not goal success. Further direction belongs to Astra's next separately frozen instruction; no E6-C intervention, new Test use, external GPU benchmark, model training or alternate primary metric was launched by this analysis.
'''
(O/'H2_SCIENTIFIC_ASSESSMENT.md').write_text(md)
P.mkdir(exist_ok=True);names=[f for f in O.iterdir() if f.is_file()]
for f in names:
 if f.suffix=='.jsonl' and f.stat().st_size>500000:
  raw=f.read_bytes();data=gzip.compress(raw,compresslevel=9,mtime=0);assert gzip.decompress(data)==raw;(P/(f.name+'.gz')).write_bytes(data)
 else:shutil.copy2(f,P/f.name)
for f in [R/'H2_ANALYSIS.py',R/'H2_EXPORT_VALIDATION.py',Path(__file__),R/'H2_PRE_REGISTRATION.json',R/'H2_LENS_INDEPENDENT_QA.json',R/'H2_VALIDATION_EXPORT_RECEIPT.json']:shutil.copy2(f,P/f.name)
(P/'README.md').write_text('''# H2 supervised diagnostic evidence

G1 prospective Brier gate failed; no G2 online inference run. See H2_TRAIN_REPORT.md, H2_VALIDATION_REPORT.md and H2_SCIENTIFIC_ASSESSMENT.md. Sol performed all analysis/code/execution/self-review; Astra supplies scientific direction only.

H2_ANALYSIS_SOURCE_FREEZE.json fixes source, original helper hashes, exact Train fixture/folds, original output-feature implementation, and G1/permission before H2 results were analyzed. H2_TRAIN_FROZEN_SPEC.json freezes all final model choices and hashes before isolated Validation fixture export/read. H2_VALIDATION_COMPLETE.json attests zero refits/selections. Original Test data was never loaded.

`H2_OUTER_MODELS.npz`: forty OOF models, keys `{setting}_{family}_{outer}_{mean|scale|coef}`, setting retrospective/prospective, family output/combined/unrelated/shuffled, outer0..4. `H2_FINAL_TRAIN_MODELS.npz`: eight fullTrain models, same keys without outer. These are small diagnostic ridge parameters, not Ouro checkpoint weights.

Exact original prediction: `z = concatenate([(x-mean)/scale,1]) @ coef`; probabilities are stable softmax(z), temperature1; class=argmax. x is original10 `output_features(scores,depth)` plus current16 H2 target/control values. d1=None is unavailable, and no old index11/index5 gate is used. All coefficients/inner alpha scores and group folds are included. Old geometric OOF control is compared without refitting; all output-only OOF probabilities match the prior run to4.44e-16.

Large JSONL files are lossless gzip. Use Python gzip.open(file,'rt') for rows, or gzip.decompress(Path(file).read_bytes()) to restore the required original `.jsonl` names. ARTIFACT_MANIFEST.json records compressed and raw SHA/byte counts. Original Train-only fixture and base helper code are the pinned E6 next_step_train_diagnosis files identified by freeze hashes; H2 analysis imports their estimator and probability routines directly.

Runtime JSON retains producer counts, full1393 natural capture receipt, measured CPU lens/storage and forward wall time, GPU memory and independent native/lens QA. High-dimensional x/a/y/b states remain server-only; no full-token tensor or checkpoint is included. Diagnostic actions are replay, not measured online policy latency. No credentials or deployment helpers are bundled. No Git publication is performed by this analysis worker.
''')
files=[]
for f in sorted(P.rglob('*')):
 if f.is_file() and f.name!='ARTIFACT_MANIFEST.json':
  row={'path':f.relative_to(P).as_posix(),'bytes':f.stat().st_size,'sha256':sha(f)}
  if f.name.endswith('.jsonl.gz'):
   raw=gzip.decompress(f.read_bytes());row.update(uncompressed_path=f.name[:-3],uncompressed_bytes=len(raw),uncompressed_sha256=hashlib.sha256(raw).hexdigest())
  files.append(row)
write(P/'ARTIFACT_MANIFEST.json',{'status':'COMPLETE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'G1_pass':False,'online_GPU_runs':0,'postprocessor_sha256':sha(__file__)})
print('H2_RELEASE_COMPLETE',len(files),sum(r['bytes'] for r in files),sha(P/'ARTIFACT_MANIFEST.json'))
