# R6 ARC independent CPU evidence audit

Audit date: 2026-11-04. Scope: the local frozen R6 ARC artifacts and scoring source. No GPU execution, method changes, external-task outcomes, or external-worker changes.

**Finding: the local evidence reproduces 736/1172 (62.7986%), exceeding the required 733. No numerical discrepancy or double normalization was found. The improvement over incumbent A1 is not statistically established at 0.05.**

| Comparison | Rescues / regressions | Net correct | Difference, percentage points | Exact two-sided McNemar p | Paired 95% bootstrap interval, pp |
|---|---:|---:|---:|---:|---:|
| R6 vs Fixed | 35 / 16 | +19 | +1.621160 | 0.010973563 | [0.426621, 2.815700] |
| R6 vs incumbent A1 | 23 / 13 | +10 | +0.853242 | 0.132498164 | [-0.170648, 1.877133] |

## Independently verified

- Parsed all 5,860 transition rows: five methods, each with the same 1,172 unique IDs. Gold and both comparator predictions agree across method rows. Correctness flags match predictions. Counts are native4=702, Fixed=717, Adaptive=716, A1=726, R6=736.
- Recomputed all ten aggregate comparisons from the transition rows, including rescues, regressions, accuracy, exact McNemar p-values and the seeded 50,000-replicate paired multinomial bootstrap. All values match the completion JSON and results CSV within 1e-12. The multinomial bootstrap resamples the paired outcome categories and is equivalent to question resampling for the accuracy difference. These are raw comparison p-values and marginal intervals, not a correction for the adaptive research history.
- Read both local contextual and auxiliary-prior score JSONLs. Each contains 1,172 distinct matching IDs. Their gold, prompt SHA, character denominators and target-token counts agree. Independently checked 18,752 native/Fixed/Adaptive readouts against raw LL sums, character normalization and stored predictions. There are 4,687 candidate answers.
- Independently rebuilt only the selected R6 formula from those raw score artifacts. Its predictions match all 1,172 selected-method CSV predictions, yielding 736 correct. No unselected R6 rule was evaluated on test during this audit.
- Verified 19 local hash bindings: Final, Dev, ranking source, method config, R6/R2/R5 evaluators, acceptance controller, ARC completion, prior completion, first-round final lock, development result, data notice, and six inference helpers. The original contextual-score JSONL, prior-score JSONL and baseline-reuse audit also match their locked/receipted hashes. Final SHA is `6de605b885b4fb9ff72b273ecc0c0484e9fe72fc5d51d793b22367db879b7da7`.

### Follow-up: downloaded manifest and cache verified

The initial audit lacked the test manifest and ARC prior cache. Both are now local, and the earlier verification above remains valid.

- The full 1,172-row manifest matches the Final lock's SHA. Recomputed all prompt SHA values; each candidate request uses its corresponding complete prompt. Manifest, contextual-score rows and prior-score rows have exactly matching ID sets, gold, prompt SHA and character denominators.
- The cache contains 4,408 unique entries, all referenced by the 4,687 candidate occurrences; no unreferenced entries were found. Reconstructed its global signature from the recorded model/source/version/prefix settings. Every cache key recomputes exactly from its inputs, and every per-row cache-entry digest matches the complete entry bytes under canonical JSON hashing. All 37,496 referenced prior LL values (4,687 choices x eight readouts) match the cache exactly.
- Cache file SHA256: `b9b67ed92a8c6004fae788f27fb7fa43c08accd9ea5adcb301a6297c377e8e05`. This newly measured whole-file digest supplements the original per-entry bindings; it is not claimed to have been frozen before inference.
- Located existing tokenizer assets in the read-only V1 model directory. All five files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt`) match the R6 Final lock's recorded hashes. No package installation or model loading was needed.
- Retrieved the official [lm-evaluation-harness v0.4.9 source](https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/v0.4.9/lm_eval/api/model.py) and extracted only `TemplateLM._encode_pair`. Its SHA is `76ddb572bbdd56cacba955743155140fe1315935bcdb3f3d201efe51406372f0`, exactly matching the original source-audit receipt. Used that function's causal-LM branch with the local tokenizer backend and `add_special_tokens=False`.
- All 4,408 auxiliary prefix/target token-ID pairs re-encode exactly. Re-encoding each of the 4,687 full contextual prompt/continuation pairs produces target IDs exactly equal to the referenced auxiliary target IDs and stored token counts. Continuation bytes match the correct candidate position throughout. This closes the missing manifest/cache/target-boundary checks for the audited artifacts.

Boundary of the follow-up: local `tokenizers` is 0.22.2, whereas the recorded server version is 0.21.4. Exact agreement was checked on all these concrete inputs; this is not an exact reproduction of the server Python environment. Full contextual prefix token arrays were not persisted in the score rows, so their byte-for-byte equality to the historical runtime arrays cannot be checked directly. No GPU likelihoods were rerun.

## Formula and provenance interpretation

Every input S is the complete-answer token LL sum divided once by the original answer character count. F is the normalized-token Fixed LoopCD readout, not a linear combination of native sequence scores. With A=F+(S4-S5), g=F-S4 and e=F_prior-S4_prior, the selected score is A-alpha*e. Alpha is clip(dot(center(g),center(e))/dot(center(e),center(e)),0,1), with zero denominator giving zero. Centering is across the current choices. The ranking kernel consumes already normalized scores and performs no second length division.

The locked method uses original unedited-model contextual scores from the same fixed 25-shot protocol. R6 did **not** rerun the original test contexts on GPU: contextual replay coverage is explicitly zero. The new auxiliary calculation uses the original model with fixed `Answer:` and no demos/question. Its receipt covers all 1,172 question rows, with 4,408 new candidate forwards and 279 cache hits, totaling 4,687 answers. Its reported elapsed time is 529.7498 seconds; this is a worker receipt, not an independently checked ledger charge. The historical independent prior-reference coverage is zero, as disclosed.

Logical dependencies are context K5 with heads 1/4/5, and prior K4 with heads 1/4. Actual collection used K6 trajectories. These facts support neither a standalone speedup nor a fixed total-runtime ratio: short prior sequences and cache reuse have different costs from the contextual pass.

## Actual gaps and claim limits

1. **The initial missing-artifact gap is resolved.** The newly downloaded manifest and cache pass the follow-up checks above. Include them, with their checksums, in the reproducibility package or document their separate availability. This audit did not rerun GPU inference or perform a new train/validation/test semantic-overlap study.
2. **Public pre-test publication was not independently verified by this reviewer.** The stage report cites commit `496602d2416f8114c5227720235c26e5c14e488d`. The parent reports that the GitHub raw pre-test lock blob matches that commit and that stage results were published in commit beginning `2e127`; those publication checks were not independently repeated here. Local lock/hash verification and the parent's reported remote verification should be distinguished. Supply the full verified public commit links in the final report.
3. The report correctly describes ARC as a reused regression benchmark and all 1,093 training questions/folds as previously used for development. Known historical A1=726 influenced the research setting. One newly selected test variant does not turn the benchmark into a fresh blind evaluation. Raw paired significance does not eliminate that limitation.
4. The numerical target is reached, but “significantly better than A1,” “stable general superiority,” and causal claims that harmful priors were identified/removed are unsupported. The coefficient is label-free at inference, but the method was selected using labeled development evidence. It depends on the supplied choice set; clipped component removal need not be strictly orthogonal. No weight-space repair or newly invented projection principle was demonstrated here.
5. The stage report's four relative artifact links resolve locally. Add direct links to the frozen ranking source, method config, raw contextual/prior scores and prior receipt for full auditability. A derived complete-candidate R6 score/alpha table would be useful, but is not needed to reproduce the present predictions from the available raw LL files. External HellaSwag/OpenBookQA results and the current total GPU budget were not assessed; their completion must be reported separately.

## Local evidence links

- [ARC stage report](../slcd_R6_ARC_STAGE_REPORT.md)
- [Frozen final lock](../configs/slcd_R6_FINAL_LOCK.json)
- [Frozen ranking source](../scripts/slcd_r6_scores.py) and [four-method specification](../configs/slcd_R6_METHODS.json)
- [ARC completion](../results/slcd_R6_ARC_COMPLETE.json) and [acceptance](../results/slcd_R6_TEST_ACCEPTANCE.json)
- [Aggregate CSV](../results/slcd_r6_arc_full_test_results.csv) and [paired transitions](../results/slcd_r6_paired_transitions.csv)
- [Contextual raw scores](../results/slcd_original_test.jsonl)
- [Auxiliary-prior raw scores](../results/slcd_r6_priors_test.jsonl) and [prior completion receipt](../results/slcd_r6_priors_test_COMPLETE.json)
- [Locked test manifest](../data/slcd_test.jsonl) and [authenticated prior cache](../results/slcd_r6_prior_cache.jsonl)

This audit supports the reported ARC count, paired arithmetic, selected ranking implementation, hash chain, concrete manifest/candidate correspondence and authenticated cache/token-ID bindings. It does not substitute for GPU rerunning, exact server-environment reproduction, independent publication-timing verification, or evaluation on unseen benchmarks.
