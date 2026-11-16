# Ouro E6-A / E6-B: frozen output and natural-latent control

Complete ARC-Challenge Train1119 (1094 after excluding25 demonstrations), Validation299 and Test1172 in each scoring protocol. Same fixed Ouro-1.4B checkpoint, fresh RTX5090/BF16/SDPA/torch2.7.1cu128. Source base commit d9c3fbde171cbb886c4663fe3a1ddad742212041. All experimental analysis, implementation, capture and audits were performed by current Sol. Astra was subsequently authorized only to issue a separate next-step research directive. No original weights were updated.

| Frozen identity | Joint correct /1172 | Full-Answer characters correct /1172 |
|---|---:|---:|
| K1 | 633 | 438 |
| K2 | 954 | 651 |
| K3 | 999 | 697 |
| K4 | 1022 | 705 |
| weighted4 | 1016 | 693 |
| loopcd_adaptive_cap1 | 1014 | 721 |
| B zero-label output rollback | 1018 | 702 |
| B zero-label output+latent rollback | 1018 | 700 |

The Joint primary weighted fusion and Validation-selected Cal method do not exceed nativeK4. B has not demonstrated general incremental utility over matched output-only rules: its principal rollback and calibrated strategies tie output-only and remain belowK4. One predeclared agreement rule benefits from latent features but remains belowK4; this local result is retained rather than promoted posthoc to the primary. Full-Answer results are independent matched-protocol evaluations, not reused letter-scoring values. See paired bootstrap intervals, rescue/damage and every frozen rule in the linked reports.

## Evidence and code

- [E6-A protocol and permission/freeze ledger](E6A/E6A_PROTOCOL.md)
- [E6-A all fresh Joint results](E6A/joint5090/TEST_SUMMARY.json) and [complete per-question decisions](E6A/joint5090/E6A_ALL_RESULTS.jsonl.gz)
- [Full-Answer primary and secondary aggregation results](E6A/E6A_STANDARD_FULL_ANSWER_TRANSFER.md)
- [True Joint early-exit execution versus replay](E6A/E6A_EARLY_EXIT.md)
- [E6-B protocol](E6B/E6B_PROTOCOL.md), [output/latent ablation](E6B/E6B_OUTPUT_VS_LATENT_ABLATION.md), [full-answer transfer](E6B/E6B_FULL_ANSWER_TRANSFER.md)
- [E6-B scientific review](E6B/E6B_ASTRA_SCIENTIFIC_REVIEW.md) and [E6-C Go/No-Go](E6B/E6C_GO_NO_GO_DECISION.md)
- [All original scores and derived features](shared_capture/) and [retained activation array SHA manifest](shared_capture/ACTIVATION_ARRAY_MANIFEST.json)
- [Numerical identity gates and runtime environment](execution/)
- [Historical A6000 appendix](E6A/historical_A6000/) and [hardware refresh audit](E6A/HISTORICAL_REFRESH_AUDIT.json)
- [Frozen-result figure](E6A/figures/E6A_FROZEN_RESULTS.png)

## Limits and interpretation

Historical ARC Test was repeatedly exposed; E3 Train1030/Validation299 helped select mechanisms. Current source/parameter freeze does not restore independent confirmation. Bootstrap intervals are question paired,10000 resamples, seed20261113, without multiple-comparison correction. Thirty-two A core rules, two repository LoopCD controls, and ten B frozen controller configurations are disclosed; the best posthoc Test row is not a selected deployment policy. Supervised diagnostics use Train labels and are separate from zero-label and Validation-calibrated controllers.

Fixed-budget selection/rollback always pays four loops. Full capture also always pays four loops, plus candidate-wise passes under Full-Answer; replay mean loops is not measured capture cost. Actual online stopping was validated only on Joint Validation299, with physical calls equal depth and full-vocabulary endpoint hash equality. Latency includes validation instrumentation; resource probes use a small fixed Train sample. Full-Answer online adaptive scheduling and independent external-benchmark confirmation were NOT RUN. Large R-only state arrays remain on the server; complete scores/features and all small Train-fit diagnostic coefficients are included here. These are negative/limited method results, not achievement of the ultimate research objective.

## Reproduce without a GPU

Decompress any .jsonl.gz needed by a legacy script with Python gzip; decompression is byte-lossless. The raw shared score/feature archives already contain native predictions, full candidate scores at all four depths, source/hardware identity, LoopCD scores and derived natural-state features for every question. Diagnostic coefficients are frozen artifacts, not fitted during release replay.

```bash
python E6A/replay_e6a_release.py --release . --out E6A_REPLAY.json
# See E6B/E6B_REPLAY_RELEASE.py --help for B controller/model replay.
```

The A replay verifies every decision, selected endpoint and loop cost for all four fresh scoring aggregations; the B replay verifies frozen controllers and diagnostics with zero fitting or GPU forwards. Receipts are included. To re-run GPU capture, restore the pinned checkpoint externally, recreate the recorded environment from the official hashed wheel manifest, decompress frozen inputs, and use the retained collector CLIs and execution shell scripts after adapting only artifact paths. Model weights are excluded. Full-vocabulary technical endpoint hashes must pass before any new study. Evaluation code refuses to overwrite a frozen Test receipt; reproducibility runs require a fresh output directory and the original frozen spec.

## Next research step

[Scientific-director instructions](ASTRA_NEXT_STEP_DIRECTIVE.md) are planning only. They do not change E6 parameters, outcomes or scientific conclusions. Raw-data analysis and execution remain with Sol.

The first director-ordered Train-only diagnosis has also been executed by Sol: [report](next_step_train_diagnosis/TRAIN_DEV_FAILURE_DIAGNOSIS.md), [folds/freeze](next_step_train_diagnosis/DIAGNOSIS_FREEZE.json), [proper-loss paired controls](next_step_train_diagnosis/PAIRED_LATENT_VS_CONTROLS.json). It does not establish geometry-specific incremental utility. The new candidate-direction H2 capture is NOT RUN; missing per-layer x cannot be replaced by cached layer means.
