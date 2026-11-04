# Parent resolution of release review

The independent static review in slcd_R6_RELEASE_PIPELINE_REVIEW.md describes the pre-fix snapshot and remains preserved. Before the full external jobs finished, the parent corrected both findings:

1. Q3 now distinguishes R2/R4 failures from R3's +1 train point estimate that did not justify a new test.
2. Public export and verify both bind the current external/final locks, each COMPLETE byte SHA and raw-score SHA to the combined passed independent acceptance. CSV artifacts are also matched to the acceptance artifact map. Public verification checks that the exporter source is the same source recorded during export.

These are report/receipt-chain corrections, not changes to the frozen inference worker, ranking rule, model, data, or coefficients. Full real-data acceptance and end-to-end export/restore remain pending while the GPU worker runs; static review alone does not claim completion.

Post-fix source hashes:

- scripts/slcd_r6_public_scores.py: `ef469b56d5fc7e4c034563c40e583278c2f6ff790547f6c227c8143335b97716`
- scripts/slcd_r6_make_deliverables.py: `76b1a9b0a85cbe337ce052ab904a5896d6d10908b757ac6d56784ba4b9f31e14`

## Read-only model symlink audit fix

The server reuses models/Ouro-1.4B through a read-only link to the original V1 model directory. The independent audit initially rejected that resolved path with its general experiment-root artifact guard. A separate read_path now permits exactly the ten pinned model asset basenames for read-only SHA/tokenizer access. General artifact/output paths remain inside the V2 root. The exception covers bind, tokenizer loading and final rehash; each file must still match the frozen byte SHA. Server verification passed the full model SHA, nine metadata SHAs, tokenizer-only load and rejection of unexpected/read-as-output paths. Non-model release folders contain no symlinks. No inference worker, ranker, coefficient, source manifest or model parameter changed. The initial shell validation command had a quoting error and was corrected; it did not start a GPU/model job.

Audit source SHA256: `7b71f7f6b10a146b5a5304a512b238c2f3ff3b19556dc3d34aea4f7f5b678ad2`.
