# Completion requirements and evidence map

This is a live evidence map, not a completion certificate. The final
`results/completion_audit.json`, delivered file hashes and observed runtime exits
must substantiate the claims below. A script merely existing does not prove that
its experiment ran. The original specification is the sealed Stage1 `TASK.md`.

## Original study, retained separately

The original Stage1 package contains its five requested figure families,
Qwen/Ouro/generation/halting raw outputs, twelve-question FINDINGS report and
175-file manifest. All 175 hashes were rechecked during the extension's seven
protocol tests. These historical outputs use their own protocol and must not be
pooled with the extension's disjoint few-shot questions.

Direct official Qwen-patch parity remains **unverified**: the author repository,
API, raw file, Git transport and connected GitHub query return 404. Search-cache
README text and independent reproduction repositories do not pass this gate.
An overall report must preserve this limitation and must not claim successful
direct author-code reproduction.

## Extension requirements

| Requirement | Evidence needed | Observed state at this checkpoint |
| --- | --- | --- |
| Storage isolation, preserve prior work | Listed server folders, new-only writes, original175 hashes | User authorized other disks: code/results/env on D; extension-added Ouro2.6 snapshot on E through its original D-path junction; all12 copy hashes passed. UTC19:32 inventory: all task folders/caches/own env16.75GB, junction counted once; free RAM50.89GiB; STORAGE_LAYOUT.md and logs/storage_inventory_20261026.json |
| Frozen pretrained weights, BF16, no training/quantization | Loading metadata, source review, checkpoint hashes | Three checkpoint loads/parity runs passed; no training code invoked |
| Qwen3-4B-Base, Ouro1.4 plus second trained-loop scale | Model revisions/configs/loading info | All three MC workers completed; Ouro2.6 added at pinned revision |
| Source identity and method correctness | Pinned DoLa/HFLM sources; real-logit oracle; native gate/cache/identity checks | All three MC parity reports passed; author-patch limitation above remains |
| Correct few-shot prompts and official task scoring | ConfigurableTask requests, HFLM encode_pair, independent likelihood parity | ARC25/OpenBook0/MMLU5 checked for all three checkpoints |
| Validation/test separation and no test tuning | Frozen manifests, question/ID exclusions, independent selection recomputation | Data tests passed; final all-stage selection audit still pending |
| Qwen A0..A4 plus two DoLa pools | 250 validation and600 test records per method, aligned IDs | All14 files completed; independent raw recovery gate passed |
| Ouro R1..4, fixed/adaptive Logits, Hidden, temperature and projection controls | 23 readouts per question on250 validation/300 test per checkpoint | All4 formal files completed and raw recovery gate passed |
| Dynamic halting and matched compute controls | 13 signals,51 quantiles/100 positive gate thresholds, validation-selected budgets,10k within-task matched-depth random assignments | MC statistics produced; final deterministic rebuild/audit pending |
| Real stopping rather than only simulated savings | Native norm/gate stop hooks, actual layer counts, all-choice likelihood/exit parity,30 balanced questions×5 methods×5 repeats | Implemented; independent CPU audit now requires exact750 ID/method/repeat keys plus150 measured layer-parity cases per model and recomputes exits/scores/quality/history. Nine synthetic auditor contract tests plus seven existing protocol tests passed locally. Actual GPU execution remains pending; fixtures are not measurements |
| Autoregressive recurrent cache correctness | HF.generate vs custom2/3/4 loops, zero guidance and tracing identity | Both checkpoints passed actual GPU parity |
| Full math validation including negative settings | 100 official-training questions×9 conditions/checkpoint; all-grid CSV | Ouro1.4 all900 completed and token/trace/recovery-lineage checks passed. Ouro2.6 STOPPED at207/900 after native access violations; complete prefix independently checked on both runtimes. A boundary probe passed but formal resume failed before adding a row, invalidating that gate. Controlled diagnostics remain separate from formal outputs; combined all-grid analysis pending |
| Held-out math baseline/control set | Aligned fresh test prefix, eight conditions, strict and Decimal EM, raw text/token IDs, cap/truncation/reuse flags | Pending; count will be frozen from runtime before any test generation |
| Generation trajectory semantics | Baseline all-depth token diagnostics; distinguish agreement from gold-token correctness and separately generated fixed-depth paths | Implemented and token-preserving parity passed; formal traces pending |
| Predictive corrections/degradations and failures | All four endpoint classes, reversals, Wilson intervals, trajectory-attributed LoopCD effects, raw/normalized scoring sensitivity, cases and offline oracle | MC CSV/JSON produced and downloaded; generation analysis pending |
| Current literature/novelty scope | Primary-paper comparison with prior mechanisms; no paper numbers substituted for our measurements | RELATED_WORK.md includes DoLa/LoopCD/AdaptiveDepth/fixed-point/FlashLoop/LoopSpec; contextual methods explicitly not rerun |
| Resource hard limits and runtime evidence | Original+new ledger<50h, per-token guard, own-process timeout,5GiB free reserve, method-specific measured estimates | Guards active; final budget/count/storage proof pending |
| Complete statistics and figures | Paired10k question bootstrap, correct denominators, all methods/strata; figures6..10 | MC figures6/7/8 visually checked; math/actual-latency figures pending |
| Research judgment answering original Q1..Q12 | Measured evidence, source limitations, causal claim limits, Go/No-Go and minimum next intervention if warranted | RESEARCH_REVIEW_CHECKPOINT.md maps current MC/closed1.4 validation evidence and pending requirements to all12 questions. Final judgment pending all experiments; no conclusion precommitted |
| Reproducible final delivery | Scripts/configs/raw JSONL/logs/dependencies/report, ZIP+manifest SHA checks, extracted CPU rebuild and visual QA | Local CPU audit passed18 MC files/31,250 readouts;15 rebuilt MC tables/8,000 rows and both selected configs match server under identical source; self-contained frozen MC references added; tokenizer/data7 checks passed; full math/latency/archive/rebuild/seal remain pending |

## Known execution failure, preserved

The initial MC CPU analysis exited with native status `0xC0000409` after every
GPU scoring worker had passed. A retry of the same calculations with diagnostic
progress/faulthandler completed. The root cause is unestablished. The separate
18-file raw recovery gate checked counts, alignment, code/protocol fingerprints,
finite scores, predictions and parity reports before generation resumed. The
original supervisor failure/log is retained. This does not constitute a failed
model configuration or justify rerunning/changing the already complete tests.

The first long Ouro1.4 validation worker later exited with native status
0xC0000409 after835 complete records. WER identified ucrtbase.dll; its underlying
cause remains unestablished. The failed stage interval was conservatively charged,
and the complete prefix passed CPU checks. Three boundary cases and the next
unfinished configuration were replayed. A controller captured an actual zero
child exit, including numeric scoring/serialization checks, before resuming.
Full900 validation then passed: the old835 records are byte-identical and the
next formal record matches the probe. Both diagnostic runs and the successful
resume are retained/charged; original failure logs and snapshots are preserved.

## Final audit procedure

1. Observe successful terminal exits for the current generation/halting workflow;
   inspect live process handles before deciding anything stopped.
2. Rebuild all MC, math, trajectory-attribution and actual-latency statistics from
   raw records. Recompute validation choices independently and compare them to
   the configurations actually executed.
3. Inspect all measured evidence, then write the twelve-question research
   decision. Do not interpret AUC<.5 as absence of information, a CI crossing0 as
   proof of equivalence, or output stability as causal state reusability.
4. Generate the report and figures, inspect rendering, list failures and missing
   gates explicitly, then package only this extension. Never run the sealed
   Stage1 packager.
5. Verify downloaded ZIP/file hashes before rebuilding. Rebuild in the original
   Stage1 directory layout and compare numeric tables/selections; recheck the
   original seal and all scoped completion requirements.
