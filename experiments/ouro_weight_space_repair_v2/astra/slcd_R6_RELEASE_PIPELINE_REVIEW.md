# R6 release pipeline: independent static review

Reviewed 2026-11-04. Read-only review of `slcd_r6_public_scores.py`, `slcd_r6_make_deliverables.py`, `slcd_r6_package_records.py`, `slcd_r6_restore_records.py`, and `slcd_r6_cpu_finalize.py`. No worker, ranking implementation, lock, or release script was modified. No partial HellaSwag/OpenBookQA outcomes were opened; no model/GPU job was run.

## Findings requiring correction

1. **Report contradicts the known development evidence.** `slcd_r6_make_deliverables.py`, Q3 (line 174 in reviewed version), says R2–R4 did not exceed A1. R3 matched-norm plus score-late reached 691 versus 690. The later table/paragraph acknowledges that +1, so the report contradicts itself. Say that R2/R4 did not exceed A1, while R3's +1 was insufficient to justify another full test. Do not replace this with a claim that R3 demonstrated no effect.

2. **The public export has an incomplete acceptance-to-artifact hash link.** `slcd_r6_public_scores.py` checks that an acceptance exists and passes, and checks current scores against current COMPLETE receipts. It does not check that those current receipts/scores/lock are the exact files accepted by that audit. A changed, internally consistent score+COMPLETE pair after the audit could therefore receive an export advertised as independently accepted. In both export and verify, require combined passed acceptance; match its `external_lock_sha256` to the current lock; match `tasks[task].complete_sha256` to the actual COMPLETE bytes and `tasks[task].score_sha256` to the score SHA. Verification can compare score SHA via the export/COMPLETE without requiring the omitted raw JSONL. Also require the export's recorded `source_sha256` to match this exporter/verifier source. This is a binding defect, not evidence that any actual output changed.

No deterministic runtime/schema blocker was found in the five reviewed versions. The two findings above were sent to the parent before completion. Parent owns any fixes and upload; this review does not alter the running process.

## Checks completed and their limits

- All five sources parse successfully with Python AST. The package filter was exercised on existing path metadata only: HellaSwag raw parquet excluded, its source receipt included, ARC prior cache included, frozen ranker included. No excluded raw file content was read.
- Numeric export uses float64 arrays for the sixteen raw LL vectors and five ranking vectors, not rounded CSV reconstruction. Verification recomputes character normalization, frozen selected R6 only, five predictions, exact McNemar, and the matching seeded 50,000-draw paired multinomial bootstrap. It does not call `all_scores` or select a method.
- The named HellaSwag raw manifest, both old/new HellaSwag raw score JSONLs, external prior cache, and raw parquet are excluded. Public numeric metadata contains hashes/IDs/cache references, not target token IDs or source text. The whole-file exclusion policy is not a content scanner: I did not inspect arbitrary logs for accidentally embedded source text. Current worker source prints progress/aggregate records, not normal per-question text.
- The active `slcd_r6_cpu_finalize*` stdout/stderr logs are excluded by prefix; finalizer COMPLETE is intentionally written after packaging and excluded. Other external worker logs should be closed before chain COMPLETE. This depends on the actual launch log basename following that excluded prefix.
- The archive is re-opened and every member is checked against individual hashes. Release index binds archive, ordered chunks, and the separately distributed contents index. Restore checks all chunk hashes, full archive hash, member order and bytes; it refuses a different existing destination and paths outside the experiment root. `--verify-only` verifies archive content, and compares existing local files; it deliberately does not require absent files to already exist. It still requires their parent directories to exist, as documented for the existing experiment layout.
- Packaging includes the historical root `FINAL_METHOD_LOCK.json` explicitly and does not overwrite it. The R6 lock remains separately named. The old-files preservation audit accurately claims size/mtime preservation for 671 files and a full-byte SHA for original model weights; it must not be described as 671 independent content hashes.
- The documented public numeric verify and `make_deliverables --public-records` path do not need HellaSwag raw text, cache, tokenizer, model, or torch load. They do need the packaged COMPLETE/acceptance/export receipts, numeric artifacts, CSVs, budget/development/overlap/case receipts, NumPy/SciPy, and plotting dependencies. Those named dependencies are selected by the package rules. The full server input audit is correctly distinguished from this public numeric reproduction.
- Final report correctly distinguishes ARC target attainment from the non-significant comparison with A1, old contextual-score reuse from fresh auxiliary-prior computation, three replayed HellaSwag cases from 1956 reused rows, logical depth from actual K6 collection, and exposed benchmark reuse from a blind test. I did not validate external numerical results, since neither task's outcomes were read during this review.
- I did not execute an end-to-end final package/report or public-record regeneration: both require the completed external task evidence. Thus AST/dependency review is not a claim that the real finalizer has passed.

## Reviewed source hashes

| Script | SHA256 |
|---|---|
| slcd_r6_public_scores.py | 07a36a9f324a6e1941a17dd44cce83a0222fb2fa39315f57bbd479dc4c076dbf |
| slcd_r6_make_deliverables.py | 6b2bef7e5fb145bcb5f8fdab6eb3ac75daea29bcec3a5c0ad0c3f91ae58bc579 |
| slcd_r6_package_records.py | 9430a30172a3049f19b4e6bf501e0f7bd60313787fea0ec3fd51e82212fb2f15 |
| slcd_r6_restore_records.py | be5974572626068fb39b290228c46209f38153a4bd70e2c60e7f52b3544626e6 |
| slcd_r6_cpu_finalize.py | 988a001f2279fecc21745c2e5b0747d41a5d8426ed82b651357d7696aacbfbb7 |
