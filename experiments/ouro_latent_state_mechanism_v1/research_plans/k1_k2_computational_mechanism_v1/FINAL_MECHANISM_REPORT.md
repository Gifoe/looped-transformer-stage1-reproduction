# K1 to K2 computational investigation

**Status: bounded experiments A, B and the justified limited C extension completed; the complete K1->K2 mechanism remains unresolved.** Sol performed implementation, execution, analysis and audits. No Astra agent, fitting, model training, controller optimization or unrestricted layer search was used. Starting commit e7da5c78676bf18ef5bd74a967788ad9875e9ea9.

## Established

- The actual normalized full-sequence K1 state feeds K2. Native layer-history replay is bitwise exact on the technical cases. This validates the intervention interface.
- A covers all Train1094 and Validation299, with all/Q/P freeze and matched controls. Same-device Validation native K1/K2 are148/243 correct; all/Q/P frozen K1 context gives78/76/80. R continues the full real computation. Fixed K1 context is not a drop-in replacement for the native context trajectory. Norm/dose controls and mixed-state incompatibility prevent a unique missing-semantic-operation attribution.
- On the new heldout two-hop family, native K1/K2 are217/446 correct out of512; explicit first-hop text edits produce448 crossed answers. Source span is provably unable to access later mappings. Local donor-variable exchange produces86 crossed answers, compared with15 same-variable,14 unrelated-position,21 equal-dose unrelated and13 final-R donor exchanges. All four paired margin specificity comparisons pass their98.75% intervals. Of these86 cases,68 predict the crossed label that neither original nor donor native execution produced; the other18 are explicitly excluded from this stronger novelty statement. The matched strictly novel counts are0/11/7/2. This is partial controlled semantic influence toward a predicted receiver-composed answer, beyond a generic output change or donor-answer target.
- The source-induced crossed-margin influence is carried partly through the previously specified L10-16 attention route: restoring native receiver updates reduces it by0.3912 logit units (0.3912 [0.3603, 0.4201]). Equal-dose feature permutation gives no comparable loss. Norm-preserving restoration has a similar effect. L18 R restoration also attenuates the effect; L2 R restoration does not show such attenuation. This is conditioned routing evidence, not a unique-layer claim.

## Rejected

The strong claim that the small causal source span is a sufficient modular representation implementing Retrieve->Compose->Bind across this family fails the preregistered majority/positive-margin gate. Most replacements do not produce the predicted crossed answer. The data do not justify claiming a complete interpretable algorithm from a selected set of positive examples. Raw attention restoration is not superior to its norm-preserved version; both exhibit the same directional effect.

## Unresolved

The source clearly carries causally relevant first-hop information but is not a complete interchangeable variable. The two-box task and fixed mapping-clause role order also do not distinguish a general symbolic program from lexical matching or simpler retrieval strategies; graph/answer assignments are permuted, but arbitrary relation structures and presentation orders were not tested. Broader descendant replacement produces441 crossed answers while retaining original R, suggesting distributed information/compatibility is important; because descendants may already encode an answer, this does not separate redundant variable encoding, contextual incompatibility and precomputed answer representation. A cannot by itself separate missing context computation from hybrid-state incompatibility. No unique writer, complete symbolic abstraction, general relational algorithm or shared ARC mechanism is established. The reviewed8 ARC structural cases provide contextual-dependency descriptions only; no validated ARC cross-world interchange was run.

## Coverage, inference and cost

A all1393 questions, B all160 base pairs/640 directed label units, C all512 heldout units plus4 technical units. B's independent combinations/entities/templates were frozen before execution. C was frozen after B and reuses exposed B examples; it is an extension, not independent confirmation. Prior ARC Validation is exposed; ARC Test was not accessed. All candidate logits/scores, decisions, ties, controls and counterexamples are retained. Intervals use5000 question/base-pair paired bootstrap replicates with family-specific corrections disclosed in the freeze; no global family-wise assertion is made.

Science wall envelope after the A throughput freeze: 33.2 minutes. Synchronized forward-time sums and partial resource observations are in GPU_COSTS.json; concurrency means these sums are not exclusive GPU-hours. Total science body forwards22634, each48 decoder-layer applications, plus numerical/serialization audit forwards. The five-hour initial GPU budget was not exhausted. All owned GPU workers exited and model memory was released after execution.

## Artifacts

The nine required reports, frozen IDs/specifications, executable scripts, input token sequences, full question-level scalar results, all audit receipts, raw tensor manifests and CPU reproduction are packaged in this namespace. Full raw tensors/checkpoint remain on the server; exact full-vocabulary hashes and candidate scalars are in the release. See README.md for commands. A negative strong gate is reported as a limit, not relabeled as completion of the entire research goal.
