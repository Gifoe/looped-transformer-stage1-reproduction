import pathlib,json,gzip,datetime,statistics
R=pathlib.Path('/root/ouro-computation-sol-20261116');O=R/'reports';O.mkdir(exist_ok=True)
a=json.loads((R/'analysis/A_SUMMARY.json').read_text());b=json.loads((R/'analysis/B_SUMMARY.json').read_text());c=json.loads((R/'analysis/C_SUMMARY.json').read_text())
def write(name,s):(O/name).write_text(s.strip()+'\n')
def interval(d):return f"{d['mean']:.4f} [{d['interval'][0]:.4f}, {d['interval'][1]:.4f}]"
def atable(cohort):
 q=a[cohort]['groups']['all'];s=f"\n### {cohort} (n={q['n']})\n\n| Condition | Correct | Accuracy | Native correct lost / wrong rescued | Gold-margin delta vs native (interval) |\n|---|---:|---:|---:|---|\n"
 for name,v in q['conditions'].items():s+=f"| {name} | {v['correct']}/{v['n']} | {v['accuracy']['mean']:.3%} | {v['native_correct_lost']} / {v['native_wrong_rescued']} | {interval(v['paired_margin_vs_native'])} |\n"
 return s
q=b['heldout'];bt='| Condition | Crossed /512 | Original | Donor | Other | Cross margin (95% cluster interval) |\n|---|---:|---:|---:|---:|---|\n'
for n,v in q['conditions'].items():
 cc=v['outcome_categories'];bt+=f"| {n} | {v['crossed']} | {cc.get('original',0)} | {cc.get('donor',0)} | {cc.get('other',0)} | {interval(v['cross_margin'])} |\n"
ct='| Routing intervention | Crossed /512 | Loss of crossed margin from source patch (interval) |\n|---|---:|---|\n'
for n,v in c['all']['conditions'].items():ct+=f"| {n} | {v['crossed']} | {interval(v['source_minus_treatment_margin'])} |\n"
write('K1_CONTEXT_REUSE_RESULTS.md',f'''# K1 context reuse

Complete same-device Train1094 (development64 + confirmation1030) and historically exposed Validation299. Every condition keeps R's real attention, MLP and residual computation. Non-R/all, Q/current question-options or P/few-shot prefix are reset to actual final-normalized K1 B at every K2 physical layer input, including L1 where native already equals B. Frozen values pass through the real layer-specific norm, projections, RoPE and causal mask. No original embeddings or future states serve as frozen sources.

Norm controls rescale each B row to that row's current receiver norm. Dose controls feature-permute B-current displacement, preserving per-row FP32 requested L2 dose before BF16 cast; realized dose and norms are recorded per layer and question. Neither restores all compatibility relations in the residual stream. These controls distinguish some magnitude/noise effects, not all hybrid-state incompatibility.

{atable('train1030')}{atable('validation299')}{atable('development64')}{atable('train1094')}

## Natural corrections and counterexamples

'''+ '\n'.join(f"{cohort}: "+'; '.join(f"{g} n={v['n']}, all-freeze correct={v['conditions'].get('freeze_all',{}).get('correct',0)}" for g,v in a[cohort]['groups'].items() if g!='all') for cohort in ['train1030','validation299'])+'''

The full four-group tables, fixed IDs, every candidate score, tie and individual transitions are in analysis/A_SUMMARY.json and data/A_QUESTION_RESULTS.jsonl.gz. Groups are reconstructed from this device's K1/K2, never imported from historical GPU results. K1 is the genuine first-pass readout, not an E/B hybrid.

Continued R processing can retain a subset of natural corrections from fixed K1 context. The full native K2 behavior is not reproduced by that intervention. Losses establish dependence under this precise frozen-context intervention; they do not uniquely establish a missing semantic operation, because B is reused at layers where native states have different compatibility and scale. Norm and dose results must be read alongside losses. No survival fraction is a universal fraction of reasoning explained.

Intervals use5000 paired question bootstrap replicates, seed20261116. Three primary freeze-vs-native margins use98.333% intervals; other intervals are descriptive95%. Family correction is not global inference across all displayed cohorts/groups/controls. Exposed Validation is not a fresh held-out benchmark.
''')
write('CROSS_WORLD_DATASET_SPEC.md',f'''# Cross-world dataset

Seed20261116;160 deterministic base pairs:32 development,128 heldout. Each has both swap directions and normal/reversed label order, yielding128 development and512 heldout directed units. All128 heldout pairs use new object entities T/U/V/W, distinct from development P/Q/R/S. The heldout population includes64 new combinations of the development template and64 pairs in an independently reserved second template. No examples were selected by native correctness. Choices contain all four distinct original, crossed, donor and other cabinets. Cabinet-to-role and option-to-label assignments are permuted reproducibly. Frozen exact IDs, semantic graph, predictions, text, token IDs, offsets and hashes are in inputs/B_INPUTS.jsonl.gz, inputs/B_UNITS.jsonl.gz and freeze/B_FREEZE.json.

Original object->boxA, A->cabinet a, B->cabinet b. Donor object->boxB, A->cabinet c, B->cabinet d. Transfer donor box-source variable to original predicts b, reverse predicts c. All a,b,c,d are distinct. Predictions were frozen before synthetic model execution. The source is the Box+entity token span in the first clause; all later mappings are causally inaccessible. Actual SDPA calls are causal with no sliding exclusion, and changing only future mappings leaves this source B bitwise identical in every unit. Same-variable donor nulls preserve all49152 logits in every unit.

Four text worlds are evaluated natively: original, donor, first-hop-edited original, first-hop-edited donor. Two fixed demonstrations precede the query. Native Joint Option scoring uses a full49152 vocabulary head and the offered label tokens A/B/C/D=[330,389,340,422], first offered label breaks ties. Both loops run natively with cache false.

Heldout native K1 correct={q['conditions']['K1']['semantic_native_or_text_correct']}/512, K2={q['conditions']['native']['semantic_native_or_text_correct']}/512; paired K2-minus-K1 accuracy {interval(q['native_K2_minus_K1_accuracy'])}. The synthetic family therefore does show additional-loop behavioral gain. This does not by itself explain ARC gains.

The bootstrap unit is the160/128 base world-pair, clustering directions and label reversals. All four units per base pair remain together. Resampling128 heldout pairs5000 times does not treat512 correlated units as512 independent examples. The tested templates/entities are a finite family, not arbitrary relational reasoning.
''')
write('INTERMEDIATE_VARIABLE_PATCHING.md',f'''# Intermediate-variable exchange

Patch only the donor's matched causal source span at the true normalized K1->K2 boundary; retain receiver's other K1 states and R. Then execute complete native K2. This is a semantic cross-world prediction, not a donor-answer target. Source cannot have read donor mappings or answer options.

{bt}

Four preregistered heldout cross-margin specificity contrasts (source minus control; Bonferroni98.75% base-pair intervals):

'''+ '\n'.join(f"- {n}: {interval(v)}" for n,v in q['source_margin_specificity'].items())+f'''

The frozen strong abstraction gate is {b['frozen_gate']}. All four specificity comparisons pass; the majority-crossed/positive-margin criterion across templates, directions and labels fails. There is reproducible, semantically directed causal influence. Of the86 crossed outputs,68 differ from BOTH actual original and donor native predictions; the corresponding strictly novel counts are0same-variable,11unrelated,7dose and2readout-copy. The86 total includes18 cases where one native input already predicted the crossed label, so86 is not the number of genuinely new crossed outputs. The localized source replacement is not a reliable substitute for the entire first-hop variable across this dataset. Examples and failures are both retained in analysis/B_EXAMPLES.json and all unit records.

The one preregistered alternative replaces receiver B[source_start:R] with the receiver's textual-firsthop-counterfactual states, keeping original R. It gives {q['conditions']['descendant_edit']['crossed']}/512 crossed answers, compared with {q['conditions']['text_edit_K2']['crossed']}/512 for the full text edit. This is a distributed-descendant compatibility diagnostic. Those downstream K1 states can already encode relations or an answer, so this result cannot locate a single intermediate variable or prove K2 alone computes composition. No alternative site sweep was run.
''')
write('COMPUTATIONAL_ABSTRACTION.md',f'''# Computational abstraction and bounded routing

The candidate abstract program is Retrieve(object,first-hop)->Compose(box,receiver mapping)->Bind(cabinet,options). It predicts all textual edits and exact crossed answers before neural execution. Native K2 and textual edits improve on K1 for this family, but a compact boundary-source substitution implements the program only incompletely. The strong substitution gate fails. No diagnostic probe is fitted and no probe output is substituted for native computation.

Because the four source-vs-control semantic specificity comparisons passed, a bounded prospective routing extension was frozen after B. It uses all512 already-exposed heldout units, not only the86 successes. C is a new intervention on an exposed cohort, not fresh confirmation. Sites are fixed by prior research: earlyL2 R input, actual normalized attention updates at L10-16, lateL18 R input. Seven physical layers are one predefined interval, not a sweep.

Attention restoration subtracts the source-patch-minus-native reference update in FP64 from each actually computed R attention increment, then writes BF16 once. Real attention, MLP and residual computation continues; non-R is preserved. Norm and dose/permutation controls are measured. Early/late R restorations replace only that actual layer's R input with its native receiver R. Native and source-patch full49152 reference hashes must match B before every C unit is accepted. Technical zero-delta controls are exact identities.

{ct}

Raw-restoration specificity diagnostics (97.5% intervals; norm preservation is also a compatibility diagnostic):

'''+ '\n'.join(f"- {n}: {interval(v)}" for n,v in c['all']['raw_restoration_specificity'].items())+'''

The primary is the full512-unit paired effect. Correct-source and wrong-source strata in C_SUMMARY are post-intervention descriptive subsets; they cannot replace the primary gate. Retained R states at L1/L9/L10/L16/L18/L24 are trajectory observations, not independent evidence of a unique writer. The tested restorations can identify a conditioned routing dependence, not the entire symbolic algorithm or an exclusively responsible layer interval. All errors, option classes, labels and ties remain visible.
''')
write('CAUSAL_CONTROLS_AND_COUNTEREXAMPLES.md',f'''# Controls and counterexamples

A includes per-region norm-preserving B and displacement-dose feature permutations at every K2 layer. Positive rescaling can largely cancel in each layer's input RMSNorm, so norm preservation does not rebuild all contextual compatibility and is not an independent natural-state rescue. Large drops under both meaningful and unrelated interventions constrain causal interpretation. Both losses and surviving natural rescues are reported, not selected successes.

B controls are same-intermediate donor (exact full-vocabulary identity for all640 units), unrelated cabinet-position exchange, norm/dose-matched unrelated source displacement, final-R donor exchange, explicit textual edits, and reversed answer-label permutations. The unrelated-position control is a later cabinet-value token and has a different accessibility/contextual role; it is not asserted to be the identical neural operation. The per-feature permutation preserves the source displacement's requested row L2 norm before BF16 cast; realized dose is stored. Both swap directions are primary. Crossed and donor answer labels are distinct by symbolic construction.

C compares the semantic source effect with native-attention restoration, norm-preserving restoration and equal-dose permuted restoration, plus early and late R restorations. Norm-preserved restoration has a similar loss to raw restoration; raw-minus-norm superiority is not established. The effect is direction-sensitive relative to the unrelated permutation, rather than simply superiority over every displayed control. The negative early-R restoration result limits a hypothesis that all causal influence has already reached R immediately after L1.

All four B outcome categories, ties, every failing case and complete candidate scores are retained. Illustrative positive and negative records are the first8 in fixed order in each category; they are not independent validation. Source-correct/source-wrong C subgroups are post-treatment descriptive only. No outcome subset replaces the primary population.

Source-span cross correctness is{q['conditions']['source_boundary']['crossed']}/512; most units do not follow the compact-variable interchange. The broad descendant alternative is not a localized-variable control: it changes downstream representations that may already carry an answer. Thus distributed encoding, residual contextual incompatibility and alternate causal routes remain possible.
''')
arc=json.loads((R/'ARC_STRUCTURE_FREEZE.json').read_text());arows=[json.loads(l) for l in gzip.open(R/'analysis/A_QUESTION_RESULTS.jsonl.gz','rt')];am={x['id']:x for x in arows};at='| ID | Independently reviewed structure | Native K1 correct | K2 correct | All-freeze correct |\n|---|---|---:|---:|---:|\n'
for x in arc['cases']:
 z=am[x['id']];rr={v['condition']:v for v in z['records']};at+=f"| {x['id']} | {x['rationale']} | {int(rr['K1']['prediction']==z['gold'])} | {int(rr['native']['prediction']==z['gold'])} | {int(rr['freeze_all']['prediction']==z['gold'])} |\n"
write('ARC_MECHANISM_TRANSFER.md',f'''# ARC structural transfer audit

Eight Train1030 examples were selected from the first32 SHA256-ordered lexical candidates by manual review of explicit relations/comparisons, before opening these examples' outcomes. Exact screening text, IDs, rationale and selection timestamp are frozen. No ARC Test was used. This is exploratory structural selection, not an independently annotated benchmark or blind confirmation.

{at}

A's same-device native/frozen-context results are available for all eight. Correctness is not a criterion for structural inclusion, nor proof of the two-hop neural mechanism. Their structures include causal sign chains, geometric composition, conservation and temperature comparisons; several also require scientific knowledge absent from the synthetic task. There is no justified one-to-one mapping from these distinct structures to a single Box variable.

A validated compact interchange did not generalize strongly even within the synthetic family. Therefore no claim of a shared ARC neural composition algorithm is made. A does establish intervention-specific contextual dependence on these examples, but no ARC intermediate-variable cross-world intervention was run. This transfer claim remains unresolved. Conducting an unrestricted search for a favorable ARC example was explicitly avoided. Prior Validation exposure and historical Test exposure elsewhere in the project do not become independent confirmation in this study.
''')
source=json.loads((R/'SOURCE_RUNTIME.json').read_text());audit=json.loads((R/'INDEPENDENT_CPU_AUDIT.json').read_text());caudit=json.loads((R/'C_INDEPENDENT_CPU_AUDIT.json').read_text())
write('SOURCE_AND_NUMERICAL_AUDIT.md',f'''# Source and numerical audit

Checkpoint/source verified before loading and after initialization; actually imported dynamic model/config files match the expected hashes. Loader is copied from the starting commit's prior package. Source/model artifacts are unchanged.

```json
{json.dumps(source,indent=2)}
```

Device is a separate RTX4090 from the13580 server. Runtime Torch2.5.1+cu124 differs from previous Torch2.7.1+cu128; no prior endpoint is pooled as a native reference. Current Train1030 K1/K2 counts558/857 and Validation148/243 differ slightly from historical560/859 and149/244. These are distinct numerical cohorts, not unexplained identity claims.

Technical gates: all three ARC full49152 native repeats, official forcedK1/K2 full-shaped interface endpoints, and per-layer native non-R replay agree bitwise. B repeats the short-sequence replay/official gate. Actual SDPA call observation verifies causal=True, mask=None, no sliding exclusion. All640 B future-only mapping changes preserve source B bytes, and all640 same-variable exchanges preserve the full final vocabulary vector. C requires original B native/source endpoints to match bitwise before its route interventions; zero attention delta is exact on four fixed technical units. Replay executes complete original full-shaped layers while overwriting non-R inputs; it is an identity check, not a speedup or mechanism result.

Independent retained-tensor CPU audit: {audit['audited_full49152_endpoints']} A/B endpoints and{caudit['full49152_endpoints']} C endpoints. Full logits hashes, candidate logits, predictions and tie lists are exact. GPU FP32 log-probabilities agree with FP64 reference with maximum error{audit['maximum_FP64_reference_vs_GPU_FP32_logprob_absolute_error']:.9g}. The initial CPU FP32 reference had a2.38e-5 reduction error; LOGSOFTMAX_AUDIT_AMENDMENT.md records why it was replaced with FP64. The exact-head gate was never relaxed. Main margins use differences of raw BF16 candidate logits and are independent of log normalization.

A record field originally called stage_R_sha256 always captures K2 stage vectors, even beside a K1 score record. Public scalar records rename it captured_K2_stage_R_sha256; K1 endpoint logits and true B are separately preserved. This is a metadata clarification, not changing a forward. Source versions and transformations are retained.

Native B serialization initially reused world/unit keys. The original tensor files were preserved; missing native boundary states were recollected from frozen inputs, requiring original K1/K2 fullendpoint SHA equality in every one of640 restorations. B_NATIVE_SERIALIZATION_REPAIR.json documents this repair, and the CPU audit cross-checks the recovered raw tensors. No scientific samples, outputs or criteria changed. A complete additional archive replication using the corrected serializer subsequently reproduces every one of6400 native/intervention endpoint records, candidate scores, predictions, ties and captured K2 R-stage hashes exactly; all full native B and patch tensors are retained in data/B_replicated. B_FULL_ARCHIVE_REPLICATION_PASS.json is the authoritative receipt. This is audit work beyond the primary science-cost envelope.

Large raw B/history/full-vocabulary tensors stay on this server under /root/ouro-computation-sol-20261116/data and technical; hash/size manifests accompany the scalar release. Model weights, private connection material, and historical raw packages are excluded from GitHub. Code, frozen inputs, per-question candidate scores/predictions/ties, numerical receipts and scalar CPU reproduction are published.
''')
# Precisely distinguish summed instrumented forward wall time from exclusive GPU-active time.
bs=[json.loads(l) for l in gzip.open(R/'analysis/B_QUESTION_RESULTS.jsonl.gz','rt')];cs=[json.loads(l) for l in gzip.open(R/'analysis/C_QUESTION_RESULTS.jsonl.gz','rt')];atime=sum(r['seconds'] for x in arows for r in x['records'] if r['condition']!='K1');btime=sum(r['seconds'] for u in bs for r in u['records'] if r['condition'] not in ('K1','text_edit_K1'));ctime=sum(r['seconds'] for u in cs for r in [u['source_reference'],u['native_reference'],*u['records']])
start=datetime.datetime.fromisoformat(json.loads((R/'A_EXPANSION_FREEZE.json').read_text())['utc']);ends=[datetime.datetime.fromisoformat(json.loads((R/f).read_text())['utc']) for f in ['A_validation_COMPLETE.json','C_COMPLETE.json']];end=max(ends);tele=[json.loads(l) for l in (R/'RESOURCE_SAMPLES.jsonl').read_text().splitlines()]
cost={'science_body_forwards':{'A':1393*10,'B':160*32,'C':512*7},'decoder_layers_per_body_forward':48,'summed_instrumented_forward_seconds':{'A':atime,'B':btime,'C_route_treatments_only':sum(r['seconds'] for u in cs for r in u['records'])},'science_start_utc':start.isoformat(),'last_science_complete_utc':end.isoformat(),'science_wall_envelope_seconds':(end-start).total_seconds(),'caveat':'Forward seconds are synchronized wall time including hook overhead/contending processes; not exclusive GPU-active time. C native/source reference timer values in metadata refer to B, so only fresh C treatment durations are summed here. GPU initialization, technical gates, serialization audit recollection and CPU/reporting are additional.','telemetry_partial_window':{'samples':len(tele),'mean_util_percent':statistics.mean(x['util_percent'] for x in tele),'max_memory_MiB':max(x['memory_MiB'] for x in tele),'first':tele[0]['utc'],'last':tele[-1]['utc'],'note':'Monitoring began during science and ended after A/B; final C-tail sampling may not be included.'}}
(R/'GPU_COSTS.json').write_text(json.dumps(cost,indent=2))
write('FINAL_MECHANISM_REPORT.md',f'''# K1 to K2 computational investigation

**Status: bounded experiments A, B and the justified limited C extension completed; the complete K1->K2 mechanism remains unresolved.** Sol performed implementation, execution, analysis and audits. No Astra agent, fitting, model training, controller optimization or unrestricted layer search was used. Starting commit e7da5c78676bf18ef5bd74a967788ad9875e9ea9.

## Established

- The actual normalized full-sequence K1 state feeds K2. Native layer-history replay is bitwise exact on the technical cases. This validates the intervention interface.
- A covers all Train1094 and Validation299, with all/Q/P freeze and matched controls. Same-device Validation native K1/K2 are148/243 correct; all/Q/P frozen K1 context gives78/76/80. R continues the full real computation. Fixed K1 context is not a drop-in replacement for the native context trajectory. Norm/dose controls and mixed-state incompatibility prevent a unique missing-semantic-operation attribution.
- On the new heldout two-hop family, native K1/K2 are217/446 correct out of512; explicit first-hop text edits produce448 crossed answers. Source span is provably unable to access later mappings. Local donor-variable exchange produces86 crossed answers, compared with15 same-variable,14 unrelated-position,21 equal-dose unrelated and13 final-R donor exchanges. All four paired margin specificity comparisons pass their98.75% intervals. Of these86 cases,68 predict the crossed label that neither original nor donor native execution produced; the other18 are explicitly excluded from this stronger novelty statement. The matched strictly novel counts are0/11/7/2. This is partial controlled semantic influence toward a predicted receiver-composed answer, beyond a generic output change or donor-answer target.
- The source-induced crossed-margin influence is carried partly through the previously specified L10-16 attention route: restoring native receiver updates reduces it by{c['all']['conditions']['restore_attention']['source_minus_treatment_margin']['mean']:.4f} logit units ({interval(c['all']['conditions']['restore_attention']['source_minus_treatment_margin'])}). Equal-dose feature permutation gives no comparable loss. Norm-preserving restoration has a similar effect. L18 R restoration also attenuates the effect; L2 R restoration does not show such attenuation. This is conditioned routing evidence, not a unique-layer claim.

## Rejected

The strong claim that the small causal source span is a sufficient modular representation implementing Retrieve->Compose->Bind across this family fails the preregistered majority/positive-margin gate. Most replacements do not produce the predicted crossed answer. The data do not justify claiming a complete interpretable algorithm from a selected set of positive examples. Raw attention restoration is not superior to its norm-preserved version; both exhibit the same directional effect.

## Unresolved

The source clearly carries causally relevant first-hop information but is not a complete interchangeable variable. The two-box task and fixed mapping-clause role order also do not distinguish a general symbolic program from lexical matching or simpler retrieval strategies; graph/answer assignments are permuted, but arbitrary relation structures and presentation orders were not tested. Broader descendant replacement produces441 crossed answers while retaining original R, suggesting distributed information/compatibility is important; because descendants may already encode an answer, this does not separate redundant variable encoding, contextual incompatibility and precomputed answer representation. A cannot by itself separate missing context computation from hybrid-state incompatibility. No unique writer, complete symbolic abstraction, general relational algorithm or shared ARC mechanism is established. The reviewed8 ARC structural cases provide contextual-dependency descriptions only; no validated ARC cross-world interchange was run.

## Coverage, inference and cost

A all1393 questions, B all160 base pairs/640 directed label units, C all512 heldout units plus4 technical units. B's independent combinations/entities/templates were frozen before execution. C was frozen after B and reuses exposed B examples; it is an extension, not independent confirmation. Prior ARC Validation is exposed; ARC Test was not accessed. All candidate logits/scores, decisions, ties, controls and counterexamples are retained. Intervals use5000 question/base-pair paired bootstrap replicates with family-specific corrections disclosed in the freeze; no global family-wise assertion is made.

Science wall envelope after the A throughput freeze: {cost['science_wall_envelope_seconds']/60:.1f} minutes. Synchronized forward-time sums and partial resource observations are in GPU_COSTS.json; concurrency means these sums are not exclusive GPU-hours. Total science body forwards22634, each48 decoder-layer applications, plus numerical/serialization audit forwards. The five-hour initial GPU budget was not exhausted. All owned GPU workers exited and model memory was released after execution.

## Artifacts

The nine required reports, frozen IDs/specifications, executable scripts, input token sequences, full question-level scalar results, all audit receipts, raw tensor manifests and CPU reproduction are packaged in this namespace. Full raw tensors/checkpoint remain on the server; exact full-vocabulary hashes and candidate scalars are in the release. See README.md for commands. A negative strong gate is reported as a limit, not relabeled as completion of the entire research goal.
''')
write('EXPERIMENT_FREEZE.md',(R/'EXPERIMENT_FREEZE.md').read_text()+'''\n## Execution amendments\n\nA_EXPANSION_FREEZE locks all1393 ARC cases on measured throughput before full scientific analysis. B_FREEZE fixes640 units and semantic predictions. C_FREEZE records the later justified limited routing extension on previously exposed synthetic units. CONCURRENCY_AMENDMENT allows two owned workers with separate exclusive worker locks; science kernels are unchanged. Original source versions, the B serialization repair and the FP64 log-softmax reference correction are retained transparently. No endpoint/sample/criterion was replaced after outcome inspection.\n''')
