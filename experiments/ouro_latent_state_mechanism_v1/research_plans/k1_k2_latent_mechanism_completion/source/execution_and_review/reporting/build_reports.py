"""Evidence-bound final reports; refuse incomplete or unreviewed stages."""
import collections,datetime,gzip,hashlib,json,pathlib,statistics
from fractions import Fraction
R=pathlib.Path(__file__).resolve().parents[1]
C=pathlib.Path('/root/autodl-tmp/ouro-k1-k2-mechanism')
O=R/'publication/experiments/ouro_latent_state_mechanism_v1/research_plans/k1_k2_latent_mechanism_completion'
STAGES=('train64','train1030','validation299');HOLDS=('train1030','validation299')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(pathlib.Path(p).read_text())
rows=lambda p:[json.loads(s) for s in pathlib.Path(p).read_text().splitlines()]
def table(header,data):return '|'+ '|'.join(header)+'|\n|'+ '|'.join('---' for _ in header)+'|\n'+''.join('|'+ '|'.join(str(v) for v in row)+'|\n' for row in data)
def fmt(x):return 'undefined' if x is None else f'{x:.6f}'
def put(name,text):
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(text.strip()+'\n')
assert read(R/'B2_QUEUE_STATE.json')['status']=='COMPLETE'
assert read(R/'posthoc_review/QUEUE_STATE.json')['status']=='COMPLETE'
assert read(R/'parallel/ALL_LINEAGE_COMPLETE.json')['status']=='PASS'
assert read(R/'E2_ALL_STAGES_COMPLETE.json')['status']=='PASS'
AB={};B2={};BIN={};AR={};BR={};E2={};proof={}
for stage in STAGES:
 for family,directory,analysis in [('AB1',R/'data'/stage,R/'analysis_results'/stage),('B2',R/'b2_data/treatments'/stage,R/'b2_analysis'/stage)]:
  p=read(directory/'PENDING.json');a=read(directory/'INDEPENDENT_COVERAGE_AUDIT.json');c=read(analysis/'COMPLETE.json')
  assert p['status']=='COMPLETE' and a['status']=='PASS' and c['status']=='COMPLETE'
  assert a['pending_sha256']==c['pending_sha256']==sha(directory/'PENDING.json')
  assert c['audit_sha256']==sha(directory/'INDEPENDENT_COVERAGE_AUDIT.json')
  proof[family+'_'+stage]=dict(n=p['n'],pending_sha256=sha(directory/'PENDING.json'),audit_sha256=sha(directory/'INDEPENDENT_COVERAGE_AUDIT.json'),analysis_rows_sha256=sha(analysis/'rows.jsonl'),summary_sha256=sha(analysis/'summary.json'))
 AB[stage]=read(R/'analysis_results'/stage/'summary.json');B2[stage]=read(R/'b2_analysis'/stage/'summary.json')
 AR[stage]=rows(R/'analysis_results'/stage/'rows.jsonl');BR[stage]=rows(R/'b2_analysis'/stage/'rows.jsonl')
 assert [(r['id'],r['natural_group'],r['natural_no_tie']) for r in AR[stage]]==[(r['id'],r['natural_group'],r['natural_no_tie']) for r in BR[stage]]
 E2[stage]=read(R/'analysis_results'/stage/'E2_DESCRIPTIVE.json')
 assert read(R/'analysis_results'/stage/'E2_FROZEN_INDEPENDENT_AUDIT.json')['status']=='PASS'
 assert read(R/'analysis_results'/stage/'SELECTED_PAIR_VECTORS.json')['status']=='PASS'
 review=read(R/'posthoc_review'/stage/'INDEPENDENT_REVIEW.json');assert review['status']=='PASS'
 assert all(sha(R/k)==v for k,v in review['source_sha256'].items())
 assert review['script_sha256']==sha(R/'posthoc_review/independent_B2_review.py')
 assert read(R/'parallel'/(stage+'_LINEAGE.json'))['status']=='PASS'
 BIN[stage]=read(R/'posthoc_review'/stage/'BINARY_PATHS.json')
assert read(C/'analysis/TRAIN_COMPLETE.json')['status']==read(C/'analysis/VALIDATION_COMPLETE.json')['status']=='PASS'
num8=read(C/'evidence_packet/NUMERICAL8_INDEPENDENT_COMPARISON.json');assert num8['status']=='PASS'
assert not O.exists(),'Never overwrite a publication candidate'
O.mkdir(parents=True)
base_scope='All new scientific direction, implementation, execution and analysis are Sol. Historical Astra orders remain provenance only. This RTX4090 run is distinct from historical5090 C1 and A6000 E1/E2. No model/probe fitting, new Test, controller search or condition/site expansion was performed.'
contracts='Primary contrasts use fresh same-hardware native K1-wrong/K2-correct questions, including ties. The original natural-no-tie subset checks mean sign reversal; post-treatment ties never filter primary means. Production FP32 full-vocabulary log-probability gold-minus-mean-wrong gaps are primary; fixed K1 strongest-wrong, dynamic-wrong and exact BF16 target-logit Fraction margins remain available. B2 uses5000 paired bootstrap draws, seed20261114, separate one-sided Bonferroni families6/12/6 and ordinary95% intervals. No global FWER across all families is claimed. Train64 engineering, disjoint Train1030 and previously exposed Validation299 stay separate; these are within-project confirmations, not blind external tests.'
coh=table(['RTX4090 cohort','n','K1 correct','K2 correct','rescue','damage','both correct','both wrong'],[(s,B2[s]['all']['n'],len(B2[s]['all']['conditions']['K1']['correct_ids']),len(B2[s]['all']['conditions']['N']['correct_ids']),*[B2[s][g]['n'] for g in ('rescue','damage','stable_correct','stable_wrong')]) for s in STAGES])
fam_table=table(['Cohort','attention6 all pass','L18 branch12 all pass','MLP secondary6 all pass'],[(s,*[B2[s]['rescue']['families'][f]['all_positive_pass'] for f in ('attention','L18_branch','MLP_secondary')]) for s in HOLDS])
replicated={f:all(B2[s]['rescue']['families'][f]['all_positive_pass'] for s in HOLDS) for f in ('attention','L18_branch','MLP_secondary')}
boundary_gate=table(['Cohort','AB1 Q route','AB1 R route','AB1 Q9 path'],[(s,*[AB[s]['PRIMARY_PREDECLARED'][k] for k in ('Q_route_pass','R_route_pass','Q9_path_pass')]) for s in HOLDS])
def contrast_table(family):
 data=[]
 for s in HOLDS:
  for name,c in B2[s]['rescue']['families'][family]['contrasts'].items():data.append((s,name,str(c['n_defined'])+'/'+str(c['n_total']),fmt(c['mean']),'['+', '.join(fmt(x) for x in c['ci95'])+']',fmt(c['lower']),c['positive_pass']))
 return table(['Cohort','contrast','defined/rescue','mean gap effect','ordinary95%CI','adjusted one-sided lower','declared pass'],data)
def transition_table():
 data=[]
 for s in HOLDS:
  for g in ('rescue','damage','stable_correct','stable_wrong'):
   v=BIN[s][g]
   for name in ('N->T','T->TX','T->TA','N->TA','N->TX','U->U_plus'):
    c=v['transitions'][name];data.append((s,g,name,c['n_defined'],len(c['wrong_to_correct_ids']),len(c['correct_to_wrong_ids']),len(c['before_tie_ids']),len(c['after_tie_ids'])))
 return table(['Cohort','natural group','before to after','n','wrong to correct','correct to wrong','before ties','after ties'],data)
bridge_description='''N is actual natural K2=AB1 H11. U is AB1 H01: the actual K2 boundary question/options Q uses E, while fixed demonstration prefix P and final readout R retain actual end-of-K1 normalized B. At physical L10-L16 R, d_a=a_N-a_U and d_m=b_N-b_U use actual normalized attention and MLP increments in FP64. These same-question boundary-conditioned reference trajectory differences include feedback. They are neither natural K2-K1 differences nor the historical Q9-conditioned C1 difference.

T starts from N and subtracts fixed d_a from each actually computed current attention increment, then casts once to BF16. U_plus adds d_a on U. M and U_M_plus perform separate MLP-only operations. No entire trajectory is clamped to its saved reference. At L18, T attention/QKV executes before composition: TX=x_N+a_T, TA=x_T+a_N and TXA=x_N+a_N. The same composed y enters both MLP input and retained outer skip; the complete MLP and remaining suffix execute. Both-branch TXA=N is a verified structural identity, not a mediation fraction.

All25 endpoints are fixed: N,U,T; TX,TA,TXA,U_plus,upstream_norm,upstream_unrelated,U_plus_norm_preserve,U_plus_unrelated; T/N x X/A x norm/unrelated eight branch controls; M,U_M_plus,M_norm_preserve,M_unrelated,U_M_plus_norm_preserve,U_M_plus_unrelated. Upstream norm-preserving controls normalize the current proposal to the receiver increment norm. Branch norm-only controls scale the current base direction to target norm. Unrelated donors are next lexicographic IDs cyclic within original full Train1094 or Validation299, without labels/resampling. Per-site/branch doses are matched before BF16 rounding; actual scales/doses/casts remain. Undefined controls require independently proved witnesses and invalidate full-defined positive coverage rather than being dropped.'''
heads=E2['validation299']['primary_head_keys']
probe_table=table(['Cohort','same original frozen head','K1 accuracy','K2 accuracy','K2-K1'],[(s,h,fmt(E2[s]['summary']['all']['K1|'+h]['accuracy']),fmt(E2[s]['summary']['all']['H11|'+h]['accuracy']),fmt(E2[s]['summary']['all']['H11|'+h]['accuracy']-E2[s]['summary']['all']['K1|'+h]['accuracy'])) for s in HOLDS for h in heads])
secondary=table(['Cohort','secondary TrainK2 head','K1 accuracy','K2 accuracy'],[(s,h,fmt(v['accuracy']),fmt(E2[s]['summary']['all']['H11|'+h]['accuracy'])) for s in HOLDS for key,v in sorted(E2[s]['summary']['all'].items()) if key.startswith('K1|') and 'trainK2' in key and (h:=key.split('|',1)[1])])
geometry=[];earliest=[]
for s in HOLDS:
 g=rows(R/'analysis_results'/s/'A_ANSWER_LABEL_GEOMETRY.jsonl');buckets=collections.defaultdict(list)
 for item in g:buckets[(item['site'],item['head'],item['family'])].append(item)
 for (site,head,family),rr in sorted(buckets.items()):geometry.append((s,site,head,family,len(rr),statistics.median(x['rank'] for x in rr),fmt(statistics.median(x['answer_label_axis_fraction_squared'] for x in rr if x['answer_label_axis_fraction_squared'] is not None))))
 e=read(R/'analysis_results'/s/'SELECTED_PAIR_VECTORS.json');groups={x['id']:x['natural_group'] for x in AR[s]};counts=collections.Counter((groups[x['id']],x['edge'],x['earliest_observed_selected_site']) for x in e['earliest'])
 for (group,edge,site),n in sorted(counts.items(),key=lambda x:str(x[0])):earliest.append((s,group,edge,site,n))
geometry_table=table(['Cohort','site','frozen raw head','definition','n','median rank','median delta energy fraction'],geometry)
earliest_table=table(['Cohort','natural group','original pair','earliest differing SAVED R site','questions'],earliest)
counter={};strict=[];examples=[]
for s in STAGES:
 counter[s]={}
 for group in ('rescue','damage','stable_correct','stable_wrong'):
  rr=[r for r in BR[s] if r['natural_group']==group]
  kinds={
   'N_correct_T_wrong_TX_correct':lambda r:r['conditions']['N']['correct'] and not r['conditions']['T']['correct'] and r['conditions']['TX']['correct'],
   'N_correct_T_wrong_TA_correct':lambda r:r['conditions']['N']['correct'] and not r['conditions']['T']['correct'] and r['conditions']['TA']['correct'],
   'N_correct_T_wrong_neither_single_recovers':lambda r:r['conditions']['N']['correct'] and not r['conditions']['T']['correct'] and not r['conditions']['TX']['correct'] and not r['conditions']['TA']['correct'],
   'N_correct_not_discretely_blocked':lambda r:r['conditions']['N']['correct'] and r['conditions']['T']['correct'],
   'T_correct_TX_wrong':lambda r:r['conditions']['T']['correct'] and not r['conditions']['TX']['correct'],
   'T_correct_TA_wrong':lambda r:r['conditions']['T']['correct'] and not r['conditions']['TA']['correct'],
   'any_endpoint_tie':lambda r:any(c['status']=='OK' and len(c['ties'])>1 for c in r['conditions'].values())}
  records={name:sorted(r['id'] for r in rr if predicate(r)) for name,predicate in kinds.items()}
  for family,cells in B2[s]['all']['families'].items():
   for contrast in cells['contrasts']:records['negative_exact_'+family+'_'+contrast]=sorted(r['id'] for r in rr if r['families'][family][contrast] is not None and Fraction(r['families'][family][contrast]['fraction']['meanwrong'])<0)
  for branch in ('X','A'):
   keys=[k for k in B2[s]['all']['families']['L18_branch']['contrasts'] if k.startswith(branch+'_')]
   used={'N','T','TX','TA'}|{f'{d}_{branch}_{k}' for d in ('N','T') for k in ('norm','unrelated')}
   ids=[r['id'] for r in rr if all(r['conditions'][k]['status']=='OK' and len(r['conditions'][k]['ties'])==1 for k in used) and all(r['families']['L18_branch'][k] is not None and Fraction(r['families']['L18_branch'][k]['fraction']['meanwrong'])>0 for k in keys)]
   records['descriptive_strict_'+branch]=sorted(ids);strict.append((s,group,branch,len(rr),len(ids)))
  counter[s][group]=records
  for category,ids in records.items():
   for ident in ids[:2]:
    row=next(r for r in rr if r['id']==ident);examples.append(dict(stage=s,group=group,category=category,id=ident,gold=row['gold'],conditions=row['conditions'],families=row['families']))
put('results/COUNTEREXAMPLE_IDS.json',json.dumps(counter,indent=2))
put('results/COUNTEREXAMPLE_EXAMPLES.jsonl','\n'.join(json.dumps(x) for x in examples))
summary_counter=table(['Cohort','natural rescue population','blocked by T','X restores among blocked','A restores among blocked','neither single restores','not discretely blocked'],[(s,BIN[s]['rescue']['n'],len(BIN[s]['rescue']['binary_paths']['N_correct__T_wrong__TX_correct']['susceptible_ids']),len(BIN[s]['rescue']['binary_paths']['N_correct__T_wrong__TX_correct']['recovered_ids']),len(BIN[s]['rescue']['binary_paths']['N_correct__T_wrong__TA_correct']['recovered_ids']),len(counter[s]['rescue']['N_correct_T_wrong_neither_single_recovers']),len(counter[s]['rescue']['N_correct_not_discretely_blocked'])) for s in HOLDS])
strict_table=table(['Cohort','native group','branch','all group n','positive6 exact contrasts and no involved ties'],strict)
conclusion='A reproducible conditional, distributed boundary-to-late-update-to-readout path is supported. K2 makes answer-relevant states more accessible to the original fixed readouts. Boundary-conditioned late attention differences participate in natural K2 corrections; downstream effects can be tested through separate L18 residual and attention branches. These interventions establish computational participation on specified receivers, not semantic information origin, a single writer, universal necessity/sufficiency or a percentage of all reasoning explained.'
if not replicated['attention'] or not replicated['L18_branch']:conclusion='The tested operators and effects are reproducible, but the complete declared boundary-induced attention-to-L18 route is not positively confirmed in both cohorts. Failed contrasts and the remaining edge are retained; local effects do not complete a universal K1-to-K2 explanation.'
mlp_sentence='The separate MLP bundle passes all six declared directions in both confirmation cohorts.' if replicated['MLP_secondary'] else 'The separate MLP bundle does not pass every declared direction in both cohorts; exact failures stay visible. This is not proof that MLP computation is absent or unimportant.'
put('RESEARCH_HYPOTHESES.md',f'''# Research hypotheses and discriminating evidence

{base_scope}

H1: recurrence improves access/readout of information already encoded after K1. H2: processing integrates information into newly usable representations. H3: the gain depends on interactions among residual, attention, MLP and downstream paths. These hypotheses overlap. Finite probes and region transplants do not make H1/H2 mutually exclusive.

|Hypothesis|Discriminating evidence|Supported scope|Remaining distinction|
|---|---|---|---|
|H1 accessibility|Identical original frozen E2 heads at actual K1/K2|Readout compatibility increases under a fixed transform|Decoding success is not native-use proof; failure is not information absence|
|H2 newly usable integration|True-boundary Q/R factorial, Q9 crosses and induced late updates|Further recurrence changes intermediate computation and decisions|No exhaustive whole-sequence information-origin assay|
|H3 distributed computation|Same-receiver upstream removal, actual L18 X/A composition and reciprocals|Conditional component interactions and incomplete single-branch recovery|No unique layer, head or writer is isolated|

{fam_table}

Final conditions, primary signs, donors and criteria were frozen before B2 scientific capture. Final descriptive case checklists do not alter them. Historical E3/C1 Q9-conditioned reference operators and current B2 boundary-conditioned operators remain distinct.

{contracts}
''')
put('INFORMATION_AVAILABILITY.md',f'''# Information availability and fixed-readout accessibility

The actual natural comparison is K1 versus H11=N=K2. H00 retains normalized K1 prefix B but resets Q/R to E; it is a hybrid, never native K1. No new probe was fitted. Original NPZ SHA204a3788301c7aa826a198b9e40ce495ef81284b6cd41a4758dfd942f8650890; all16 frozen heads across1393 questions/334320 applications were independently arithmetically audited.

{probe_table}

## Secondary frozen TrainK2 heads

{secondary}

TrainK1 identifies coefficient-training states. Its original pooled Train coordinate scaler includes both loops, so the entire transform was not learned from K1 alone. Raw/unit-L2, real/permuted and L18/post-norm stay separate. Unit-L2 is followed by coordinate standardization and is not a pure angular invariant. Train1030 probe accuracy is in-sample relative to original Train1094 head/scaler fitting. Probe capacity/lambda are frozen; partially exposed Validation was historically used for selection. Complete E2 descriptive outputs retain gold-index/candidate-count strata and both probe/native disagreement directions. Global token order is(330,340,389,414,422), i.e.A,C,B,E,D; the adapter remaps actual offered token order. Rare E has no positive Train gold examples. Candidate-conditioned permutation changes820/1094 Train labels; singleton candidate sets cannot be shuffled.

## Candidate-relative geometry

{geometry_table}

These centered raw W/std label-axis projections use actual K1-to-K2 deltas, rank<=4 for global5 and rank<=m-1 for offered candidates. They are distinct definitions; an energy fraction is not native causal use or information content. The extra E-axis confound is retained. Separate embedding-alignment geometry is explicitly named embedding geometry. Actual doses and norm controls are reported instead of ranking raw effects by projection norm.

{earliest_table}

Q-only boundary pairs have identical R at L1. Earliest means among SAVED selected R sites; unsaved L2-L8 prevent a first-physical-writer claim. Successful decoding establishes accessibility to this readout. Failed decoding establishes only failure of finite decoders. Which semantic information was absent from all K1 states remains unidentified.
''')
put('K1_K2_STATE_TRANSITION.md',f'''# True recurrent boundary and state transition

{coh}

The actual normalized full sequence after K1 physical L24 becomes K2 L1 input. E is actual first-loop embedding input, B actual end-of-K1 RMSNorm output. P is fixed25-shot demonstrations, Q the current question/options/format except final R, R the final readout row. AB1 holds P=B_P: H00=(E_Q,E_R), H10=(B_Q,E_R), H01=(E_Q,B_R), H11=(B_Q,B_R)=N. E_all to native K1 and H11 to native K2 were verified on the4090. H00 is not a K1 replay.

Four edges: H10-H00(Q_R_E), H11-H01(Q_R_B), H01-H00(R_Q_E), H11-H10(R_Q_B). Each has receiver-specific norm-matched boundary donor and fixed Gaussian displacement controls; two Q9 crosses give14 endpoints. All per-question Q/R interactions remain. H11-H00 and H00-K1 must be separated: hybrid reset combines geometry, scale and context effects and is not a natural loop transition.

{boundary_gate}

Strong gate failure does not imply zero effect. Raw positive Q9 restoration/rollback is not a complete specific boundary route when boundary controls fail; the Q9 crosses also lack dedicated Q9 direction-matched nulls. B2 tests a narrower induced-update path without relabeling the failed AB1 specificity criterion.

{earliest_table}
''')
put('UPSTREAM_CAUSAL_TRACING.md',f'''# Boundary-conditioned upstream causal tracing

{bridge_description}

{fam_table}

## Attention update family

{contrast_table('attention')}

## Separate MLP family

{contrast_table('MLP_secondary')}

{mlp_sentence}

Full-bundle interventions establish only specified contrasts. Reference d includes N/U trajectory feedback. Norm and unrelated-direction controls address the tested alternatives, not every perturbation. Attention/MLP raw magnitudes have different doses and contexts and cannot rank intrinsic importance. No unique writer or additional site/head/token sweep is asserted.

{contracts}
''')
put('NATURAL_MEDIATION_RESULTS.md',f'''# Same-receiver natural correction mediation

{bridge_description}

{summary_counter}

These are deterministic answer transitions on fresh4090 natural groups. X/A recovery sets overlap. Counts, unions and TXA identity are not percentages of all reasoning explained. Not-discretely-blocked questions may still have continuous margin effects. Restoration is conditional to this intervention and receiver, not universal necessity or sufficiency.

## Original L18 positive contrast family

{contrast_table('L18_branch')}

## All four natural groups and ties

{transition_table()}

Complete records retain every offered raw BF16 target logit, production score, actual prediction, top ties, all three margin definitions and exact Fraction counterparts; undefined controls remain explicit. Every endpoint's accuracy/ties and both recovery/damage directions are exported. Current native-rescue populations are334/1030 and103/299, not historical C1 membership.

## Historical C1 is separate

C1 used a Q9-conditioned reference on the earlier5090, with342/1030 and102/299 natural rescues. Its independent audits and fixed8 FP32/math sensitivity remain under historical_C1. Both aggregate X/A route gates passed there, but strict per-case controls/ties are sparse and do not establish universal mediation. C1/B2 operators and groups are not concatenated by transitivity or merged as identical runs.

{contracts}
''')
put('CAUSAL_EVIDENCE_MAP.md',f'''# Causal evidence map

|Edge|Evidence and receiver|Supported scope|Remaining limit|
|---|---|---|---|
|K1 to readable K2 state|Same original16 frozen E2 heads at actual K1/K2; mapping/permuted/L2 controls|Fixed-readout accessibility|Not native use, or absence in K1|
|K1 boundary to K2 trajectory|Actual full normalized B; AB1 Q/R factorial and Q9 crosses; receiver norm/Gaussian controls|Defined hybrid interventions and interactions|Original strong specificity failures retained|
|Boundary-conditioned trajectory to late attention|N/U-defined d_a, bidirectional T/U_plus, next-ID donor/norm controls|Specified conditional update participation|Reference-feedback difference, not unique writer|
|Late attention through L18 X/A to decision|Actual T attention then separate TX/TA and reciprocals; shared MLP y/skip and full suffix|Controlled branch-mediated effects|Cross-world composition, not universal mediation percentage|
|Boundary-conditioned MLP to decision|Separate d_m removal/restoration and six declared controls|Separate MLP contrasts|Raw attention/MLP sizes not intrinsic ranking|
|Natural correction participation|Fresh native scores; all4 groups and all ties; independent reconstruction|Identified tested corrections and failures|Other paths/nonblocked cases remain|
|Mechanism to inference method|No new controller/utility evaluation; historical E6/H2 retained|Design principle only|No validated label-free control signal or accuracy gain|

{fam_table}

Every scalar claim binds original source/coverage/tensor audits, independent score/bootstrap review and execution lineage. Technical48-layer replay and parallel equivalence are numerical gates, not mechanism success labels.
''')
put('COUNTEREXAMPLES.md',f'''# Complete counterexample and correction sets

{summary_counter}

Every stage and all four natural groups are retained in results/COUNTEREXAMPLE_IDS.json. Full condition score/prediction/tie examples and exact contrasts are in results/COUNTEREXAMPLE_EXAMPLES.jsonl. Examples use first two lexicographic IDs per explicit category; they do not replace denominators or select a favorable mechanism. Negative exact signs remain for every attention, branch and MLP contrast; complete original rows contain all remaining question evidence.

## Additional descriptive strict checklist

{strict_table}

This final reporting checklist requires all six already-defined branch contrasts to have positive exact BF16 Fraction signs and all involved endpoints unique top predictions. It is not a preregistered additional test, per-case confidence interval or explanatory percentage, and never filters primary means. Failure to block an answer is not proof of no continuous path effect. Failure of one branch does not show other downstream paths are irrelevant.

Historical C1 case sets and numerical sign reversals remain separately reported. Probe-correct/native-wrong and native-correct/probe-wrong IDs, gold/candidate strata and rare-E cases remain in the complete frozen E2 descriptive outputs.
''')
put('NUMERICAL_AND_SOURCE_AUDIT.md',f'''# Numerical, source and coverage audit

{base_scope}

Weights SHA58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af; modeling source SHA c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413. Original BF16/SDPA, torch2.7.1+cu128, transformers4.55.0, eval/no cache, two loops and Joint-Option remain. The full49152-shaped head/vocabulary normalizer executes. Native4090 anchors are established per question; historical A6000/5090 predictions are not borrowed.

AB1 freeze33637fcb85c5a14e0dfd47be2ce62bc37de82da46f3e90d5bd0a9aecb35b86f8. B2 freeze50092820c417c5f55de9d3e5d9180cb7fa13e0a7a5cd5820f769fcff6eaa40a9. Actual loaded-source/checkpoint/config hashes, cohorts and fullsplit donors remain. Complete technical3 explicit48-layer independent replays passed before science. B2 technical v1 observed Q9 after instead of before L9; a retained separate v2 corrected only observation before scientific capture. Failed sources and chronology remain.

Lossless CAS checks archive/raw tensor hashes and reconstructed canonical safetensors hash. Prefix sharing uses actual content, never assumed equality across BF16 sequence shapes. Every row passed original independent full saved-tensor/site/cast/control/score arithmetic. CPU long-double norm comparison uses the prospectively declared FP64 reduction envelope; proposals, casts, branch/skip composition, model/logit identities and exact Fraction signs remain strict. Undefined controls need mathematical witnesses and cannot be silently dropped as positive coverage.

Execution-only amendments first overlap a CPU writer, then isolate two model/hook processes. No original frozen file/scientific contract changed.619 prior Train1030 records keep original metadata/tensor hashes. Independent lineage checks original, async and parallel variants. Per-worker CAS prevents file-creation races and rebased manifests reconstruct the same canonical tensors. The matched6 empty-GPU benchmark was64.004s versus40.517s, observed1.58x throughput. Only object IDs/allocation addresses are excluded from cross-process numeric equality, while their schemas, original in-process alias assertions and raw recorded values stay. Earlier comparator and paused-context teardown failures are retained, never relabeled PASS. Cleanup is bounded only after every capture, writer and source check has a verified PASS report. All original full audits remain mandatory.

Historical C1 fixed8 independent numerical status {num8['status']}; max FP32 score error versus FP64 {num8['max_FP32_score_error_vs_FP64']}. This combines dtype/backend/deterministic settings, not isolated dtype sensitivity, seed replication or B2 FP32 validation. Small branch effects/interactions can change sign. Full historical receipts preserve them. No B2 FP32 sweep was added.

Only verified unused obsolete wheel installers were removed, freeing4,140,592,140 bytes. Models, installed environments, inputs and raw scientific evidence remain. Duplicate private manifests are losslessly gzip archived with receipts; no final scientific root manifest/referenced tensor chunk is deleted. Public compact evidence does not include all raw tensor archives: full raw evidence remains on the authorized server with hashes.

{contracts}

Reports and package receive their own SHA manifest. Publication is complete only after actual GitHub branch commit/tree verification.
''')
put('FINAL_MECHANISTIC_MODEL.md',f'''# Evidence-supported mechanistic model

{conclusion}

{fam_table}

The source-constrained path is: normalized K1 full-sequence B becomes K2 input; question/options context and readout state jointly alter the trajectory; boundary-conditioned L10-L16 R attention differences alter actual L18 residual and attention computations; branch recomposition and the recomputed shared MLP/suffix change answer selection on identified natural corrections. {mlp_sentence}

{summary_counter}

H1 reorganization and H2 newly usable integration remain observationally compatible: finite probes at selected R sites are not an exhaustive whole-K1-sequence assay. The resolved edge is the specified controlled update/mediator intervention, not semantic origin or unique writer. Nonblocked cases, failed single-branch recoveries, negative exact contrasts and ties delimit the account. AB1 boundary specificity failures remain failures despite narrower B2 effects.

An inference-time principle is to preserve compatibility between recurrent context updates and both downstream readout branches, and test path-sensitive behavior instead of hidden-distance alone. A label-free natural utility/control signal is speculative. Offline gold-stratified causal analysis neither authorizes label-dependent inference nor establishes an accuracy-improving controller. Historical E6/H2 limitations remain; no E6-C or optimization search was launched.
''')
put('FINAL_RESEARCH_REPORT.md',f'''# K1 to K2 latent mechanism investigation

{conclusion}

{base_scope}

## Population and native behavior

{coh}

## What is represented after K1?

Original frozen heads decode answer-relevant information at selected K1 R sites. The same heads generally read more effectively from actual K2 states. INFORMATION_AVAILABILITY.md contains all matched-head controls, mapping/scale/rare-E and pooled-scaler constraints. Neither finite-probe failure nor mixed-boundary reset proves information absence across all K1 states.

## What changes during K2, and which computations cause it?

Actual full normalized K1 output is K2 input. AB1 factors Q/R inheritance and observes selected intermediate changes without claiming a first physical writer. Its strong direction-specific gates remain:

{boundary_gate}

B2 tests exact boundary-conditioned late updates on the same receivers, bridging their actual effects through separate L18 residual/attention composition and full remaining computation:

{fam_table}

{mlp_sentence}

## How do changes affect answers, and which corrections are attributed?

{summary_counter}

These conditional deterministic recovery sets overlap and are not fractions of all reasoning. Ordinary/corrected intervals, reciprocal losses, norm/unrelated controls, all4 groups and ties remain. Detailed effects: UPSTREAM_CAUSAL_TRACING.md and NATURAL_MEDIATION_RESULTS.md. Failures/IDs: COUNTEREXAMPLES.md. Earlier5090 C1 is separately audited context, never substituted for this same4090 receiver bridge.

## Remaining limits

Exhaustive H1-versus-H2 information origin, a unique layer/head writer, universal necessity/sufficiency and validated label-free inference utility remain unresolved. The exact tested path and control failures are independently audited; neither correlation, TXA identity nor probe improvement is a complete theory of reasoning. The investigation ends with a reproducible scoped causal account and explicit limits rather than an unconstrained additional sweep.

## Reproducibility

NUMERICAL_AND_SOURCE_AUDIT.md binds actual runtime/source, full coverage and independent gates; CAUSAL_EVIDENCE_MAP.md binds arrows to operators/receivers. All10 report entrypoints, original code/orders, compact per-question results and audits remain. Full raw tensor CAS stays on the server; public evidence retains scores, predictions, ties, controls and raw/canonical/source hashes, not model weights or every raw archive.

{contracts}

Generated UTC {datetime.datetime.now(datetime.timezone.utc).isoformat()}. Report generation is separate from verified GitHub publication.
''')
put('results/REQUIREMENT_EVIDENCE_LEDGER.json',json.dumps(dict(status='EVIDENCE_BOUND_REPORTS_GENERATED_NOT_YET_PUBLISHED',all10_entrypoints=['RESEARCH_HYPOTHESES.md','INFORMATION_AVAILABILITY.md','K1_K2_STATE_TRANSITION.md','UPSTREAM_CAUSAL_TRACING.md','NATURAL_MEDIATION_RESULTS.md','CAUSAL_EVIDENCE_MAP.md','COUNTEREXAMPLES.md','NUMERICAL_AND_SOURCE_AUDIT.md','FINAL_MECHANISTIC_MODEL.md','FINAL_RESEARCH_REPORT.md'],proof=proof,replicated_declared_families=replicated,remaining_scientific_limits=['H1/H2 not exhaustively identified','semantic origin and unique writer','universal necessity/sufficiency','label-free inference utility'],remaining_delivery=['figure QA','evidence packaging','verified GitHub publication']),indent=2))
print('REPORTS_GENERATED',str(O),json.dumps(replicated))
