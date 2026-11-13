"""Evidence-only final packaging; does not fit or select any method."""
import argparse,json,shutil,hashlib,statistics,subprocess,sys,gzip
from pathlib import Path
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.root);out=Path(a.out);src=root/'E6B';j=src/'joint5090'
 # Preserve measured scopes rather than treating replay depths as actual latency.
 runtime={'scope':'RTX5090 BF16 SDPA. Full collection includes hooks, four full-shaped heads, real LoopCD, endpoint hashes; not minimal inference. Online resource probe is independent Validation299, not Test latency.','capture':{},'technical':{}}
 runtime['analysis_execution_owner']='Current Sol model; Astra subsequently authorized only for next-step scientific direction, not E6 analysis/code/GPU execution or self-review.'
 runtime['cpu_analysis']={'joint_test_receipt':read(j/'E6B_TEST_RECEIPT.json'),'cpu_peak_memory':'NOT_MEASURED','development_wall_time':'NOT_MEASURED_SEPARATELY','analysis_device':'CPU NumPy; no model forward'}
 for protocol,base in [('joint',root/'joint'),('full_answer',root/'full_answer')]:
  for split in ['train','validation','test']:
   d=base/split
   if not (d/'COMPLETE.json').exists():continue
   rows=[]
   for f in d.glob('*.json'):
    r=read(f)
    if 'native' in r:rows.append(r)
   nums=lambda k:[r[k] for r in rows if isinstance(r.get(k),(int,float))]
   r={'receipt':read(d/'COMPLETE.json'),'question_count':len(rows)}
   for k in ['wall_seconds','feature_cpu_and_storage_seconds','state_storage_bytes','peak_allocated_bytes','peak_reserved_bytes']:
    vs=nums(k)
    if vs:r[k]={'sum':sum(vs),'mean':statistics.mean(vs),'median':statistics.median(vs),'max':max(vs)}
   runtime['capture'][protocol+'/'+split]=r
 for name in ['technical_joint_v3','long_joint_v3','technical_fas_v2','long_fas_v2']:
  d=root/name
  if (d/'COMPLETE.json').exists():
   runtime['technical'][name]=read(d/'COMPLETE.json');runtime['technical'][name]['resource_rows']=[]
   for f in d.glob('*.json'):
    r=read(f)
    if 'native' in r:runtime['technical'][name]['resource_rows'].append({k:r[k] for k in ['id','wall_seconds','feature_cpu_and_storage_seconds','state_storage_bytes','peak_allocated_bytes','peak_reserved_bytes'] if k in r})
 hp=src/'hook_overhead.json'
 if hp.exists():
  h=read(hp);runtime['hook_overhead_raw']=h;times={False:[],True:[]}
  for r in h['rows']:
   for m in r['measurements']:times[m['observed']].append(m['seconds'])
  runtime['hook_overhead_summary']={str(k):{'n':len(v),'mean_seconds':statistics.mean(v),'median_seconds':statistics.median(v)} for k,v in times.items()}
 op=src/'online_validation299'
 if (op/'COMPLETE.json').exists():
  runtime['online_validation_receipt']=read(op/'COMPLETE.json');rr=[json.loads(l) for l in (op/'rows.jsonl').read_text().splitlines()];runtime['online_validation_methods']={}
  for method in sorted({r['method'] for r in rr}):
   rs=[r for r in rr if r['method']==method];runtime['online_validation_methods'][method]={'n':len(rs),'raw_row_keys':list(rs[0]),'rows_file':'online_validation299/rows.jsonl'}
  shutil.copytree(op,out/'online_validation299',dirs_exist_ok=True)
 write(out/'E6B_RUNTIME_AND_MEMORY.json',runtime)
 audit=read(out/'E6B_STATE_AVAILABILITY_AUDIT.json');audit['new_capture']['technical_status']='PASS_JOINT3_AND_LONG_FAS3_AND_LONG';audit['new_capture']['formal_capture_status']={k:v['receipt']['status'] for k,v in runtime['capture'].items()};audit['new_capture']['state_storage']='Natural R-only float32 exact BF16 values in server NPZ; low-dimensional features released. No full-token tensors.';write(out/'E6B_STATE_AVAILABILITY_AUDIT.json',audit)
 for name in ['E6B_FEATURES.jsonl','E6B_GROUP_DESCRIPTIVES.json','E6B_MATCHED_ABLATIONS.json','E6B_ADDITIONAL_FROZEN_CONTROLS.json']:
  if (j/name).exists():shutil.copy2(j/name,out/name)
 labels={(r['split'],r['id']):r for r in map(json.loads,(src/'E6B_LABELS.jsonl').read_text().splitlines())}
 examples={g:[] for g in ['stable_wrong','rescue','damage','stable_correct']}
 for r in map(json.loads,(j/'E6B_FEATURES.jsonl').read_text().splitlines()):
  if r['split']!='test':continue
  gold=labels[(r['split'],r['id'])]['gold'];before=r['native_predictions'][0]==gold;after=r['native_predictions'][3]==gold
  g='stable_correct' if before and after else 'damage' if before else 'rescue' if after else 'stable_wrong'
  examples[g].append({'id':r['id'],'gold':gold,'predictions_K1_K4':r['native_predictions'],'delta_feature_K1_K4':[x[11] for x in r['latent']]})
 # Lexical examples are descriptive only; all group membership remains available.
 for g in examples:examples[g]=sorted(examples[g],key=lambda r:r['id'])
 write(out/'E6B_NATURAL_GROUP_MEMBERSHIP.json',examples)
 with (out/'E6B_COUNTEREXAMPLES.md').open('a',encoding='utf8') as f:
  f.write('\n## Natural K1 to K4 outcomes\n\nGroups compare endpoints, so intermediate transitions can differ. First three lexical Test IDs per group are displayed, with all members and all per-method incorrect decisions retained. Gold is used only for offline outcome classification.\n\n| Group | N | Example IDs |\n|---|---:|---|\n')
  for g,rs in examples.items():f.write('| '+g+' | '+str(len(rs))+' | '+', '.join(r['id'] for r in rs[:3])+' |\n')
 matched=read(j/'E6B_MATCHED_ABLATIONS.json')['comparisons'];text='\n## Direct paired latent-minus-output contrasts\n\n| Split | Rule / track | Net correct | Gain pp [95% CI] |\n|---|---|---:|---|\n'
 for r in matched:
  s=r['latent_spec'];lo,hi=r['paired_question_bootstrap95_pp'];text+=f"| {r['split']} | {s['strategy']} / {s['track']} | {r['net_gain']:+d} | {r['pp_gain']:+.3f} [{lo:+.3f},{hi:+.3f}] |\n"
 text+='\nThe agreement rule gains 15 correct decisions from latent gating over its matched output-only rule on Joint Test, while remaining six below native K4. This is an isolated fixed-rule contrast, not evidence of a superior final controller. Rollback ZL and calibrated latent rules have zero net incremental correct decisions over their matched output baselines. Early stopping produces the same Test correct count with more executed loops when the latent gate is enabled.\n'
 if (j/'E6B_ADDITIONAL_FROZEN_CONTROLS.json').exists():
  controls=read(j/'E6B_ADDITIONAL_FROZEN_CONTROLS.json')['comparisons'];text+='\n## Existing frozen E6A confidence/depth controls\n\nAll30 paired Test comparisons are retained in E6B_ADDITIONAL_FROZEN_CONTROLS.json. They apply the existing E6A spec directly, with no new rule or Test selection. Primary latent ZL rollback (1018/1172) compares as follows:\n\n| Existing control | Control correct | Latent net correct | Gain pp [95% CI] |\n|---|---:|---:|---|\n'
  for r in controls:
   if r['method_index']!=1:continue
   lo,hi=r['paired_question_bootstrap95_pp'];text+=f"| {r['comparator']} | {r['comparator_correct']} | {r['net_gain']:+d} | {r['pp_gain']:+.3f} [{lo:+.3f},{hi:+.3f}] |\n"
 with (out/'E6B_OUTPUT_VS_LATENT_ABLATION.md').open('a',encoding='utf8') as f:f.write(text)
 with (out/'E6B_ACCURACY_AND_COMPUTE.md').open('a',encoding='utf8') as f:f.write('\n## Actual online Validation resource check\n\nFrozen ZL early-exit p=.9, median Train delta threshold. All observed native prefix vocabulary hashes and final replay predictions/depths matched; actual layer0 executions equal policy depth. Validation299: K4 261 correct, 4 loops, mean .18108 s; output-only 260 correct, 3.12375 loops, .14744 s; latent 260 correct, 3.47492 loops, .16592 s. Latent is approximately 12.5% slower than the matched output-only controller, with equal accuracy. Timing includes callback/head work, and excludes disk export. Alternating method order and warm paired Train3 hooks are recorded in runtime JSON. These are Validation measurements; no online Test or FAS speedup is claimed.\n')
 with (out/'E6B_ACCURACY_AND_COMPUTE.md').open('a',encoding='utf8') as f:
  f.write('\n## Actual shared collection resources\n\nThese costs include data acquisition work and are separate from controller policy replay. Each Joint item uses one four-loop body and four full-shaped heads; FAS repeats that work for each candidate. Real LoopCD and observed-feature extraction are included. Production windows use synchronized wall-clock timers around forward/scoring/hash work, not isolated kernel timing.\n\n| Protocol / split | N | End-to-end collection s | Production forward/scoring s | Max allocated GiB | R-state storage MiB |\n|---|---:|---:|---:|---:|---:|\n')
  for key,r in runtime['capture'].items():
   cp=r['receipt'];f.write(f"| {key} | {r['question_count']} | {cp['wall_seconds']:.2f} | {cp['measured_seconds']:.2f} | {r.get('peak_allocated_bytes',{}).get('max',0)/2**30:.3f} | {r.get('state_storage_bytes',{}).get('sum',0)/2**20:.2f} |\n")
  f.write('\nModel initialization and technical gate overhead are retained in separate logs/receipts rather than represented as online per-question policy speed. CPU memory was not measured; unavailable metrics are marked in runtime JSON.\n')
 with (out/'E6C_GO_NO_GO_DECISION.md').open('a',encoding='utf8') as f:f.write('\nThe independently Validation-selected Cal latent rollback also loses four correct decisions versus native K4 (1018/1172 vs 1022/1172), ties matched calibrated output-only, and retains four-loop cost. Neither retrospective nor prospective combined diagnostic improves macro-F1 over outputs alone on Joint Test (.6227 vs .6320; .3740 vs .3825). The fixed agreement contrast deserves visibility, but does not establish an accuracy/compute frontier improvement. Do not launch E6-C from this evidence. Full-answer transfer is assessed separately with the same frozen parameters.\n')
 review='''# Current-model scientific self-review (not Astra)

All E6B data analysis, implementation, execution and this scientific self-review were performed by the current Sol model under the initial user instruction. After the E6 method freezes, the user separately authorized Astra to provide next-step scientific direction only. Astra did not perform E6 analysis, code, GPU work or this review. This is not independent human review or an Astra-authored result.

The central E6B claim is unsupported: the prescribed low-dimensional natural L10–16 attention-update features do not yield a controller that exceeds native K4 on Joint Test. Latent rollback ZL and Cal both score 1018/1172 against K4 1022/1172. They tie their matched output-only policies. Retrospective combined macro-F1 .6227 is below output .6320; prospective .3740 is below .3825. These diagnostic coefficients are explicitly supervised and do not qualify as zero-label policy learning.

The result is not uniformly null. The fixed agreement latent gate improves 15 decisions over its weak matched output-only comparator, paired95% CI +.427 to +2.218 percentage points, while remaining six below K4. This comparison is one of several frozen contrasts; the interval is unadjusted for multiplicity. It supports a local gating effect, not superiority to the native endpoint or a post-Test choice of a new primary method.

Real online Validation replay passed endpoint hashes and exact stop/depth matching. The latent gate retains the same accuracy while executing more loops and taking about12.5% longer than output-only. A small warmed Train3 hook microbenchmark isolates observation overhead but cannot characterize a population latency distribution. FAS speedups and online Test latency were not measured. Full-answer results, when complete, are independent no-refit transfer rather than a second opportunity to tune.

Protocol strengths relevant to interpretation are same5090 scores/features, real post-localnorm hooks that never mutate the forward, full-shaped heads and exact native identity gates, Train/Validation freezing before new Test capture, gold-free worker inputs, question-level intervals, and matched unrelated/shuffled controls. The frozen and receipt hashes retain these checks. Historical ARC/Test exposure means this remains a project-internal frozen evaluation rather than a new external benchmark.

A causal pathway from E3 counterfactual patching need not expose a useful natural scalar for selecting a correct output. The tested summary statistics discard vector orientation and answer-specific direction. The present failure does not establish that no latent information exists or that richer supervised features could never help. Such changes would define a new hypothesis requiring a new development/evaluation protocol. They cannot repair this Test result after the fact. Current evidence does not justify advancing E6-C or claiming an accuracy/compute frontier gain.

## Failure hypotheses and discriminating future checks

1. Redundancy with output confidence: supported by output-only diagnostic performance exceeding combined features and by null matched rollback/early accuracy effects. A future development-only conditional probe could residualize latent features against outputs, then assess incremental prediction on newly held-out data. It must compare matched dimensionality/regularization and cannot select a rule using this Test.
2. Loss of answer-specific direction: norms, cosines and concentration can vary without identifying which candidate is correct. A separately preregistered diagnostic could compare these scalars against a small candidate-aligned projection of the same natural R update, with equal parameter count and unrelated-layer controls. It would be supervised mechanism diagnosis, not a ZL success.
3. Counterfactual-to-natural mismatch: E3 patched N/P differences establish context-specific intervention effects; they do not guarantee a separable feature in ordinary trajectories. A future matched natural/counterfactual study on Train-only items could test whether the causal direction is actually activated in natural rescue versus damage. No E6-C intervention was executed here.
4. Conservative gate cost: the latent early criterion rejects output-confident stopping points without changing aggregate correct count. Per-prefix accept/reject outcomes and true loop counts already discriminate this explanation from merely slow hooks. Optimizing a different stopping loss now would require a new development protocol and fresh evaluation rather than retuning the present thresholds on Test.
5. Prompt/score-distribution shift: Joint-frozen quantiles and candidate-probability thresholds may transfer poorly to full-answer character means. The independent no-refit FAS transfer measures this limitation; no FAS calibration is permitted within the present transfer claim. Shape-dependent BF16 differences are separately bounded by same-protocol native identity gates, not erased by cross-protocol score comparisons.

The practical research goal is a method that improves accuracy/compute beyond output-only and strong frozen baselines. Completing the prescribed experiments is not attainment of that goal. The executed evidence is an evidence package and a current no-go decision; it is not a successful new inference method.
'''
 (out/'E6B_ASTRA_SCIENTIFIC_REVIEW.md').write_text(review,encoding='utf8')
 exposure='\nPrior mechanism selection used E3 Train1030 and Validation299. Those items are not independent new samples here; historical Test has been exposed repeatedly across earlier project experiments. The new Train1094 excludes25 demonstrations, and its current Train→Validation→frozen Test chronology prevents new-result tuning within this run but does not erase historical selection/exposure. All generalization and interval claims are conditional on this project-internal scope.\n'
 for name in ['E6B_PROTOCOL.md','E6B_ASTRA_SCIENTIFIC_REVIEW.md','E6B_UTILITY_PREDICTION_ANALYSIS.md','E6B_FULL_ANSWER_TRANSFER.md','E6C_GO_NO_GO_DECISION.md']:
  with (out/name).open('a',encoding='utf8') as f:f.write(exposure)
 protocol=(out/'E6B_PROTOCOL.md').read_text(encoding='utf8').replace('Execution owner: current model. No Astra agent is invoked; the requested scientific-review filename will contain an explicitly identified current-model self-review.','Execution and analysis owner: current Sol model. E6B scientific review is explicitly Sol self-review. Initial no-Astra instruction governed E6 execution; after method freezes the user separately authorized Astra for next-step scientific direction only, without E6 analysis/code/GPU execution.');(out/'E6B_PROTOCOL.md').write_text(protocol,encoding='utf8')
 definitions=(out/'E6B_FEATURE_DEFINITIONS.md').read_text(encoding='utf8').replace('Current-model execution; no Astra agent or Astra authorship.','Sol analysis and execution; no Astra authorship of these E6 features. Astra was subsequently authorized only for next-step scientific direction.');(out/'E6B_FEATURE_DEFINITIONS.md').write_text(definitions,encoding='utf8')
 shutil.copytree(src/'references',out/'references',dirs_exist_ok=True) if (src/'references').exists() else None
 for name in ['E6B_ONLINE_EARLY_EXIT.py','E6B_HOOK_OVERHEAD_PROBE.py','E6B_DESCRIPTIVES.py','E6B_MATCHED_ABLATION.py','E6B_MATCHED_TRANSFER_ABLATION.py','E6B_ADDITIONAL_FROZEN_CONTROLS.py','E6B_REPORTS.py','E6B_FINALIZE_ARTIFACTS.py','E6B_REPLAY_RELEASE.py','E6B_REPLAY_FAS_RELEASE.py','INPUT_FREEZE.json','SPLIT_INPUT_FREEZE.json','LONG_PROBE_FREEZE.json','E6B_LABELS.jsonl']:
  if (src/name).exists():shutil.copy2(src/name,out/name)
 for name in ['E6B_DEVELOPMENT_DECISIONS.jsonl']:
  if (j/name).exists():shutil.copy2(j/name,out/name)
 fas=src/'fas5090'
 if (fas/'E6B_FAS_TEST_RECEIPT.json').exists():
  for f in fas.iterdir():
   if f.is_file():
    name={'E6B_FEATURES.jsonl':'E6B_FAS_FEATURES.jsonl','E6B_GROUP_DESCRIPTIVES.json':'E6B_FAS_GROUP_DESCRIPTIVES.json','E6B_ADDITIONAL_FROZEN_CONTROLS.json':'E6B_FAS_ADDITIONAL_FROZEN_CONTROLS.json'}.get(f.name,f.name);shutil.copy2(f,out/name)
  ff=read(fas/'E6B_FAS_TRANSFER_FREEZE.json');digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert ff['hashes']['source_sha256']==digest(src/'E6B_TRANSFER_v2.py');assert ff['hashes']['controller_source_sha256']==digest(src/'E6B_TRAINING_FREE_CONTROLLER.py');assert ff['hashes']['joint_freeze_sha256']==digest(j/'E6B_FROZEN_SPEC.json');assert read(fas/'E6B_FAS_TEST_RECEIPT.json')['frozen_transfer_sha256']==digest(fas/'E6B_FAS_TRANSFER_FREEZE.json');write(out/'E6B_FAS_SOURCE_AND_RULE_QA.json',{'status':'PASS','transfer_freeze_sha256':digest(fas/'E6B_FAS_TRANSFER_FREEZE.json'),'hashes':ff['hashes'],'base_model_parameters_changed':False,'new_FAS_fits':0,'new_FAS_parameter_selections':0})
  ft=[r for r in read(fas/'E6B_FAS_TRANSFER_SUMMARY.json') if r['split']=='test'];primary=next(r for r in ft if r['method_index']==1);cal=next(r for r in ft if r['method_index']==9);native=primary['correct']-primary['net_gain'];para=f"\nIndependent FAS frozen Test: native K4 {native}/1172; latent ZL rollback {primary['correct']}/1172 (net {primary['net_gain']:+d}); Joint-selected Cal latent rollback {cal['correct']}/1172 (net {cal['net_gain']:+d}). Parameters, quantiles and supervised coefficients were transferred without refitting. These outcomes do not replace the Joint endpoint.\n"
  for name in ['E6B_ASTRA_SCIENTIFIC_REVIEW.md','E6C_GO_NO_GO_DECISION.md']:
   with (out/name).open('a',encoding='utf8') as f:f.write(para)
  with (out/'E6B_FULL_ANSWER_TRANSFER.md').open('a',encoding='utf8') as f:
   f.write(para+'\n## Supervised no-refit diagnostic transfer\n\n| Timing | Features | Test macro-F1 |\n|---|---|---:|\n')
   for r in read(fas/'E6B_FAS_DIAGNOSTICS_TRANSFER.json'):f.write(f"| {r['setting']} | {r['family']} | {r['test_macro_f1']:.4f} |\n")
   mp=fas/'E6B_FAS_MATCHED_ABLATIONS.json'
   if mp.exists():
    f.write('\n## Matched FAS Test latent-minus-output contrasts\n\n| Rule / track | Net correct | Gain pp [95% CI] |\n|---|---:|---|\n')
    for r in read(mp)['comparisons']:
     if r['split']!='test':continue
     s=r['latent_spec'];lo,hi=r['paired_question_bootstrap95_pp'];f.write(f"| {s['strategy']} / {s['track']} | {r['net_gain']:+d} | {r['pp_gain']:+.3f} [{lo:+.3f},{hi:+.3f}] |\n")
 write(out/'REPRODUCIBILITY_SCOPE.json',{'model_weights_modified':False,'diagnostic_parameters_are_base_model_parameters':False,'diagnostic_models':'Train-only class-balanced ridge coefficients and standardization arrays; Validation alpha selection; frozen before Test','fitting_receipt':'E6B_FROZEN_SPEC.json','raw_high_dimensional_states':'Server-only per-question compressed R-only NPZ, SHA referenced by capture JSON; release includes E6B_FEATURES.jsonl sufficient for frozen controller replay','labels':'E6B_LABELS.jsonl isolated from collectors and online controller; needed only for offline evaluation','online_test_latency':'NOT_RUN','online_full_answer_latency':'NOT_RUN','GPU_queue':'Parent-exclusive scheduler; no concurrent experiment GPU work','review_author':'Current Sol self-review; Astra later authorized only for next-step direction, not E6 analysis/implementation/execution'})
 (out/'README.md').write_text('''# E6-B: natural latent update utility

Joint frozen Test: native K4 1022/1172; primary latent ZL rollback and Validation-calibrated latent rollback 1018/1172. Both tie their matched output-only controls in correct count. Natural features did not improve the final accuracy/compute frontier. Full-answer no-refit transfer is reported separately. Sol performed all E6 analysis/execution/review; Astra was later authorized separately for next-step direction only.

## CPU replay

With Python3.10+ and NumPy, from this directory run:

```sh
python E6B_REPLAY_RELEASE.py .
```

This checks all11720 frozen Joint Test controller decisions (prediction, selected depth, executed loops) and all10 frozen supervised diagnostic macro-F1 values. It reads released low-dimensional features and Train-fit coefficient NPZ, performs zero fits, and needs neither CUDA nor model weights. Labels are isolated evaluation data. The coefficient NPZ modifies no Ouro model parameter.

## Evidence and source layout

`E6B_FROZEN_SPEC.json` retains Train/Validation source hashes, policy thresholds, separate ZL/Cal permissions, selected diagnostic alphas and coefficient hash. `E6B_TEST_RECEIPT.json` binds the single frozen Test evaluation. `E6B_FEATURES.jsonl` contains all2565 non-demonstration natural observations with candidate scores and legal prefix features. `E6B_CONTROLLER_RESULTS.jsonl` retains Test decisions; development decisions are separate. FAS files have the `E6B_FAS_` prefix and use an independent prompt/character-normalized scoring protocol.

The repository sibling `shared_capture/` contains the complete original Joint/FAS score-and-feature capture exports and their SHA manifests, assembled by the E6A coordinator. Compressed high-dimensional R-only NPZ states remain on the experiment server; no full-token states are released. The frozen CPU evaluators load per-question JSON plus `IDENTITY.json` from the original capture directory layout; released decision replay above needs only this E6B directory.

`E6B_NATURAL_FEATURE_EXTRACTION_v3.py` and `E6_SHARED_FULL_ANSWER_CAPTURE_v2.py` are the actual frozen GPU producers. The required unsuffixed extraction filename is a byte-identical alias. `E6B_ANALYSIS_v2.py` and `E6B_TRANSFER_v2.py` implement development/freeze and evaluation phases. `E6B_ANALYSIS.py` supplies frozen metrics/probe helpers; its old monolithic main is not the executed formal workflow. `E6B_ONLINE_EARLY_EXIT.py` and the hook probe contain independent measured Validation/Train resource checks. Runtime JSON records their exact scope and raw timings.

These are project-internal ARC evaluations after historical ARC/Test exposure. No independent external benchmark or E6-C intervention was launched. Scientific review is explicitly current-model self-review.
''',encoding='utf8')
 subprocess.run([sys.executable,str(out/'E6B_REPLAY_RELEASE.py'),str(out)],check=True)
 if (out/'E6B_FAS_FEATURES.jsonl').exists():subprocess.run([sys.executable,str(out/'E6B_REPLAY_FAS_RELEASE.py'),str(out)],check=True)
 compressed=[]
 for f in sorted(out.rglob('*.jsonl')):
  if f.stat().st_size<1_000_000:continue
  raw=f.read_bytes();data=gzip.compress(raw,compresslevel=9,mtime=0);assert gzip.decompress(data)==raw;target=f.with_name(f.name+'.gz');target.write_bytes(data);compressed.append({'path':target.relative_to(out).as_posix(),'uncompressed_path':f.relative_to(out).as_posix(),'uncompressed_bytes':len(raw),'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'gzip_bytes':len(data),'gzip_sha256':hashlib.sha256(data).hexdigest()});f.unlink()
 write(out/'LOSSLESS_JSONL_MANIFEST.json',{'compression':'gzip, mtime0; exact round-trip bytes verified','files':compressed})
 with (out/'README.md').open('a',encoding='utf8') as f:f.write('''
## Lossless JSONL compression

Large JSONL artifacts are published as `.jsonl.gz`. Both replay scripts automatically read gzip, so decompression is optional. `LOSSLESS_JSONL_MANIFEST.json` records each exact uncompressed filename, byte count and SHA256, plus compressed SHA256. To restore the required unsuffixed artifact names with Python:

```python
import gzip, pathlib, hashlib, json
root = pathlib.Path('.')
for row in json.loads((root / 'LOSSLESS_JSONL_MANIFEST.json').read_text())['files']:
    data = gzip.decompress((root / row['path']).read_bytes())
    assert hashlib.sha256(data).hexdigest() == row['uncompressed_sha256']
    (root / row['uncompressed_path']).write_bytes(data)
```

After decompression `E6B_FEATURES.jsonl` and `E6B_CONTROLLER_RESULTS.jsonl` have their required names and exact original content. `E6B_NATURAL_FEATURE_EXTRACTION.py` is already present as the byte-identical alias of actual frozen v3 producer. No deployment helper, credentials, wheel archive, model weights or high-dimensional states are bundled.
''')
 # Also test the published compressed form, not only the staging plain files.
 subprocess.run([sys.executable,str(out/'E6B_REPLAY_RELEASE.py'),str(out)],check=True)
 if (out/'E6B_FAS_FEATURES.jsonl.gz').exists():subprocess.run([sys.executable,str(out/'E6B_REPLAY_FAS_RELEASE.py'),str(out)],check=True)
 files=[]
 for f in sorted(out.rglob('*')):
  if f.is_file() and f.name!='ARTIFACT_MANIFEST.json':files.append({'path':f.relative_to(out).as_posix(),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 write(out/'ARTIFACT_MANIFEST.json',{'status':'COMPLETE','files':files,'packager_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()});print('FINAL_ARTIFACTS',len(files))
if __name__=='__main__':main()
