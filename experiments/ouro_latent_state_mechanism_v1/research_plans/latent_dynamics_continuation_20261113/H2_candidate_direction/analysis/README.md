# H2 supervised diagnostic evidence

G1 prospective Brier gate failed; no G2 online inference run. See H2_TRAIN_REPORT.md, H2_VALIDATION_REPORT.md and H2_SCIENTIFIC_ASSESSMENT.md. Sol performed all analysis/code/execution/self-review; Astra supplies scientific direction only.

H2_ANALYSIS_SOURCE_FREEZE.json fixes source, original helper hashes, exact Train fixture/folds, original output-feature implementation, and G1/permission before H2 results were analyzed. H2_TRAIN_FROZEN_SPEC.json freezes all final model choices and hashes before isolated Validation fixture export/read. H2_VALIDATION_COMPLETE.json attests zero refits/selections. Original Test data was never loaded.

`H2_OUTER_MODELS.npz`: forty OOF models, keys `{setting}_{family}_{outer}_{mean|scale|coef}`, setting retrospective/prospective, family output/combined/unrelated/shuffled, outer0..4. `H2_FINAL_TRAIN_MODELS.npz`: eight fullTrain models, same keys without outer. These are small diagnostic ridge parameters, not Ouro checkpoint weights.

Exact original prediction: `z = concatenate([(x-mean)/scale,1]) @ coef`; probabilities are stable softmax(z), temperature1; class=argmax. x is original10 `output_features(scores,depth)` plus current16 H2 target/control values. d1=None is unavailable, and no old index11/index5 gate is used. All coefficients/inner alpha scores and group folds are included. Old geometric OOF control is compared without refitting; all output-only OOF probabilities match the prior run to4.44e-16.

Large JSONL files are lossless gzip. Use Python gzip.open(file,'rt') for rows, or gzip.decompress(Path(file).read_bytes()) to restore the required original `.jsonl` names. ARTIFACT_MANIFEST.json records compressed and raw SHA/byte counts. Original Train-only fixture and base helper code are the pinned E6 next_step_train_diagnosis files identified by freeze hashes; H2 analysis imports their estimator and probability routines directly.

Runtime JSON retains producer counts, full1393 natural capture receipt, measured CPU lens/storage and forward wall time, GPU memory and independent native/lens QA. High-dimensional x/a/y/b states remain server-only; no full-token tensor or checkpoint is included. Diagnostic actions are replay, not measured online policy latency. No credentials or deployment helpers are bundled. No Git publication is performed by this analysis worker.
