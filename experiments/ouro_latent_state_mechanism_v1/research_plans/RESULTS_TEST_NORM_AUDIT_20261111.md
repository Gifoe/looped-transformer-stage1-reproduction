# ROUND_2 test: BF16 norm controls

| Control | C max absolute error | R mean absolute error | R median absolute error | R max absolute error | R max relative error |
|---|---:|---:|---:|---:|---:|
| K1 direction / K2 target norm | 0.107819 | 0.0005334 | 0.0004127 | 0.0046496 | 0.07144% |
| K2 direction / K1 target norm | 0.081558 | 0.0004417 | 0.0003645 | 0.0042615 | 0.08410% |

All 1,172 test questions and 2,233,719 input positions were audited. Per-question C mean absolute error, averaged equally over questions, is 0.0008732 / 0.0006639. Every full-sequence maximum falls at token 0 / token 1, respectively, both within the demonstration prefix. R actual-minus-target signs are 560 positive, 612 negative, zero 0 / 567 positive, 605 negative, zero 0. R maximum-error examples are Mercury_7218348 (actual 7.204539, target 7.209188) and Mercury_7018270 (actual 5.071645, target 5.067383).

The values are numerical diagnostics for approximate BF16 target-norm preservation, not proof of a direction-only natural mechanism or negligible effect on predictions. Arrays do not include a post-rounding direction-angle audit. Original exporter NameError and the independent exact recovery are documented in DISCOVERY_LOG_TEST_NORM_AUDIT_20261111.md.

Evidence SHA256: full norm gzip e37bde984a27638ce436672e729c92bbfa83716fdb708e0e5b0fcaaa025898c4; recovery receipt c5e34b0414583a9be43acdae765d4b1ae404cee0add5dc7dde1783fd1bc103b7; coverage 1cdee583defb6fd580001b532e9fefd11de8f46b7943707ebdb9a30c3d4da4a7; readout ledger 051f11ea05f984e98408aa69659deda24c12c40e09a8d60e758ca1aaec80785e.
