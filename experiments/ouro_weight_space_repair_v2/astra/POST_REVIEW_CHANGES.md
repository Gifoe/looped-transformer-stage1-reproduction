# Root changes after independent source reviews

The independent reviews record the source hashes they inspected. Root subsequently made these bounded integrity/documentation changes, without changing candidate construction, scoring, selection rules, statistics formulas, or frozen weights:

- analyze_v2.py verifies original baseline rows against the SHA256 of the complete copied input manifest, including candidate continuations; also requires uncached, fixed25 metadata.
- assemble_report.py displays readable method names and explicitly discloses normalized question-text overlap, if any; it describes all-negative depth point estimates directly.
- unpack_artifacts.py explicitly verifies the resolved temporary directory remains inside V2 before automatic cleanup.
- package_v2.py publishes the ARC data attribution/CC BY-SA notice directly as well as inside the records archive; README links the notice.

Report clarification also states that prediction/gold choice indices start at0.

Final code hashes for these changes:

- `scripts/analyze_v2.py`: `0e52cbec437d29eeb6eeedf5c17915ca707492e89c66df0ead886008e1ad2a22`
- `scripts/assemble_report.py`: `c2e2dd8321c174f64ce043b66f368eb781ce73ee291833a69eab12157162aa6c`
- `scripts/unpack_artifacts.py`: `1b409b2cee70083c89ea9086b8daceae114e9617b2c8c6199425704a28644c8d`
- `scripts/package_v2.py`: `7165bcbc130e4ccfbe25d8beaa8e2869a13a1d1bf0bf34ea24d7839ce6656fcc`

## Recorded selected-checkpoint gate repair

The first real selected-checkpoint gate registered read-only observation hooks after the scorer fingerprint was taken. The fingerprint guard correctly rejected this; no final evaluation scores or weights were modified. Root added scorer.invalidate() immediately after observer registration and after removal, then ran the same training-only check under selected_checkpoint_gate_retry1. The failed attempt and its GPU charge are preserved. The stable recovery receipt is logs/FINISHER_RECOVERY.json.

- corrected `scripts/verify_selected_checkpoint.py` SHA256: `6c3a820f77d3ede22a74c95a32802fbaa547f322740c3d2cb114a2fe397fc17f`

Final visual QA moved the Fixed LoopCD note below its star markers to remove overlap; no values, statistics, or frozen weights changed. Direct-file indices are refreshed after archive completion. Current analyze_v2.py SHA256: `bee960951c2bd30fca39b4479d3f8b6c291453b48a2d6424f631e7f8f1f8ab56`.

Final evidence review requested explicit non-significance language for the two negative primary point estimates. The report generator now says point-estimate decrease, non-significant, no established improvement; numerical values and tests remain unchanged. The original report-assembly receipt stays in the records archive; revised text uses results/REPORT_TEXT_QA.json. Current assemble_report.py SHA256: `6952341775af1fbcdf7d3df99bdb7151b0fd63d8c1d1c438b71717a907f20805`.
