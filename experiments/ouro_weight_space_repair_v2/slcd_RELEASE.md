# Final release: frozen R6 LoopCD

All experiments and independent acceptance are complete. Original Ouro-1.4B
weights were used throughout; no training or weight editing occurred.

| Benchmark | Protocol | Fixed | R6 | R6 accuracy | Net vs Fixed |
|---|---|---:|---:|---:|---:|
| ARC-Challenge | fixed 25-shot, full test | 717/1172 | 736/1172 | 62.7986% | +19 |
| HellaSwag | 0-shot, full validation | 7231/10042 | 7259/10042 | 72.2864% | +28 |
| OpenBookQA | 0-shot, full test | 211/500 | 219/500 | 43.8000% | +8 |

ARC exceeds the required 62.5%. Versus Fixed, ARC rescue/damage is 35/16,
exact McNemar p=.01097. Versus A1=726, ARC improvement is not significant
(p=.13250). Both external point estimates improve, but their paired confidence
intervals cross zero (p=.17910 and .11532). ARC was historically exposed;
these results do not establish broad superiority or a new causal mechanism.

Additional GPU-worker use: 13,565.214 seconds (3.7681 hours), including failed
attempts, within the original six-hour ledger. Historical cache generation is
excluded from this incremental budget. This run's GPU workers have finished.

- [Full Chinese report](results/slcd_REPORT_ZH.md)
- [Experiment README and all five controls](slcd_README.md)
- [Current frozen R6 lock](configs/slcd_R6_FINAL_LOCK.json)
- [Independent full-input acceptance](results/slcd_R6_EXTERNAL_ACCEPTANCE.json)
- [Lossless record release index](artifacts/slcd_R6_RELEASE_INDEX.json)
- [Post-package portable reproduction check](results/slcd_R6_PUBLIC_REPRODUCTION_CHECK.json)

The canonical archive contains 329 records in six ordered parts (24,394,996
compressed bytes), each with SHA256. Full original inputs/caches remain on the
server. Public HellaSwag records contain complete numeric readouts and input
hashes, rather than source prose or reversible token IDs. Models and state
tensors are not distributed. Source/licensing and the precise release scope
are recorded in the README and archive index.

## Portable public replay

Restore the six parts using `python -B scripts/slcd_r6_restore_records.py`.
On x86 or another CPU architecture, run:

```sh
python -B scripts/slcd_r6_public_scores_portable.py
python -B scripts/slcd_r6_make_deliverables.py --public-records
```

The canonical `slcd_r6_public_scores.py verify` uses bitwise comparisons and
passed on the original ARM server. Local x86 replay found arithmetic differences
up to 8.89e-16, with zero changed predictions among all 10,542 external questions.
The portable verifier allows absolute arithmetic roundoff <=1e-13, relative
tolerance zero, and requires every argmax, count, paired transition and statistic
to remain unchanged. Exact serialized scores, CSV values and file hashes are
still checked exactly. This is a verification tolerance, not an inference
parameter or a modified method.

The portable verifier, its diagnostic and the post-package checker are separate
additions alongside the immutable canonical archive. The latter replayed the
numeric results and regenerated identical report text/tables without a model,
raw HellaSwag manifest, text-containing score files or prior cache, then restored
and rehashed all 329 canonical records. See
[portability diagnostic](results/slcd_R6_PUBLIC_PORTABILITY_DIAGNOSTIC.json) and
[portable acceptance](results/slcd_R6_PORTABLE_NUMERIC_ACCEPTANCE.json).

Ordinary report regeneration writes new PDF metadata/timestamps. Use
`scripts/slcd_r6_public_reproduction_check.py` in a restored public-only copy if
the canonical server bytes must be preserved during the check.
