# Parent resolution of release review

The independent static review in slcd_R6_RELEASE_PIPELINE_REVIEW.md describes the pre-fix snapshot and remains preserved. Before the full external jobs finished, the parent corrected both findings:

1. Q3 now distinguishes R2/R4 failures from R3's +1 train point estimate that did not justify a new test.
2. Public export and verify both bind the current external/final locks, each COMPLETE byte SHA and raw-score SHA to the combined passed independent acceptance. CSV artifacts are also matched to the acceptance artifact map. Public verification checks that the exporter source is the same source recorded during export.

These are report/receipt-chain corrections, not changes to the frozen inference worker, ranking rule, model, data, or coefficients. Full real-data acceptance and end-to-end export/restore remain pending while the GPU worker runs; static review alone does not claim completion.

Post-fix source hashes:

- scripts/slcd_r6_public_scores.py: `ef469b56d5fc7e4c034563c40e583278c2f6ff790547f6c227c8143335b97716`
- scripts/slcd_r6_make_deliverables.py: `222f69f2925f7268a058d22acff2ca3d43b0ba340e47c1bc042755aca260b3a1`
