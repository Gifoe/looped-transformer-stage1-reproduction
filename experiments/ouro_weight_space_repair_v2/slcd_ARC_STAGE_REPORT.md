# ARC stage snapshot — cross-benchmark evaluation in progress

Frozen primary: A1 gamma=1, `S_fixed(c) + S4(c) - S5(c)`, using character-normalized complete candidate likelihood scores. Original unedited Ouro-1.4B; same global fixed25 demonstrations, seed2021.

Selection used train discovery384, selection192, fresh train confirmation517 and one official validation299 comparison. All47 registered configurations and five iterations are preserved. Validation selected A1 at173/299 vs Fixed168. No test-based reselection.

Complete ARC test1172: NativeK4 702, Fixed717, Adaptive716, selected726 (61.945%). Relative to Fixed:18 rescues,9 regressions,net+9 (+0.768pp). Paired50,000-bootstrap95%CI[-0.085,1.621]pp; exactMcNemar p=0.1221. The point estimate passes the requested>717 execution gate, but is not statistically reliable evidence of superiority.

The same A1gamma1 configuration had a historically exposed retrospective726; this is a previously used regression benchmark and this rerun is not independent blind confirmation or a novel discovery. Current selection was documented using train/validation, but prior exposure cannot be removed.

Full1172 original baseline outputs were reused after full prompt/candidate/character-length/token-count, recipe/source and checkpoint/environment checks. This stage is CPU rescoring of original saved scores; new test GPU replay coverage is0. Training576 and validation299 had exact eight-baseline replay. The mathematical method needsK5 (+1loop,+25% recurrent work and3 vocabulary-head readouts vs2); the collection code used sharedK6 and supports no runtime speed claim.

The original6h additional GPU-worker budget remains enforced. ARC passed the gate; full HellaSwag validation10042 and optional OpenBookQA test500 are proceeding with the frozen method, no coefficient changes. Initial server network download failures are preserved and charged; pinned official parquet bytes were downloaded locally and transferred with SHA verification.

Final cross-task results, figures, raw score archive and Chinese report will be committed to this same branch when complete. See FINAL_METHOD_LOCK.json and results/slcd_ARC_TEST_COMPLETE.json for the completed ARC stage.
