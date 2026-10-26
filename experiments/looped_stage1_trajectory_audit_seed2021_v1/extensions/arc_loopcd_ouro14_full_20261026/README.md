# Full ARC-Challenge LoopCD reproduction

This experiment uses the complete1172-question official ARC-Challenge test split,
the pinned official ByteDance/Ouro-1.4B checkpoint,25-shot prompts, BF16, four
recurrent iterations and the first iteration as the reference. It evaluates
three precommitted configurations from LoopCD arXiv2610.02185 Table2/A4:

| Method | Setting | Paper ARC-C accuracy |
| --- | --- | ---: |
| Unguided baseline | final recurrent logits |60.41% |
| Fixed LoopCD-Logits | z4 +0.5(z4-z1) |62.29% |
| Adaptive LoopCD-Logits | omega=1-(p_top1-p_top2), cap1.0 |61.69% |

The dataset source revisions, canonical question order, rendered contexts,
candidate continuations and their hashes are frozen before formal prediction.
Official lm-eval0.4.9 task definitions supply likelihood requests; acc_norm is
the primary metric and unnormalized accuracy is reported separately. The paper
does not disclose its exact harness version or few-shot RNG state; seed2021 and
the complete rendered prompt manifest make this run explicit and reproducible.
Previously evaluated public test questions remain included in this complete
benchmark. No test output is used to choose strengths, thresholds or prompts.

All three MC readouts share one four-loop trajectory for each candidate because
they have identical teacher-forced prefixes. The complete candidate likelihoods
and predictions for all three methods are retained. This is not a generation
speed claim; later GSM8K methods must generate their own separate prefixes.

The precommitted ARC acceptance gate requires all three absolute accuracies to
be within1pp of the reported scores, both guided net-correction counts positive,
both paired deltas within1pp of their reported improvements, and the reported
deltas inside the paired bootstrap95% intervals. McNemar exact two-sided tests
and whether the intervals exclude zero are reported separately. The gate is an
operational replication rule, not a substitute for statistical evidence.
Original gross correction/degradation counts cannot be inferred from published
aggregate scores. The implied net counts are about22 for Fixed and15 for Adaptive.

Only after this gate passes may the full1319-question GSM8K test proceed, with
three-shot greedy generation, fixed omega0.3 and adaptive cap0.5. A shared1024-token
cap is fixed in advance because the paper does not specify its exact generation
limit; that limitation must accompany numerical comparisons to its GSM8K table.

`scripts/arc_controller.py` records actual worker affinity and native exit status,
preserves loaded source bytes, conservatively charges native failures, then runs
the independent complete-set/score/paired audit. Only this task's process and
children receive CPU mask0xffff0000; no unrelated process or system setting is
changed. The shared cumulative50-GPU-hour ledger includes the sealed Stage1,
previous extension, failures and this experiment. The D-drive free-space reserve
is5GiB. All original175 sealed Stage1 files remain read-only.

Artifacts: `data/FULL_MANIFEST_AUDIT.json`, `results/ARC_PARITY.json`,
`trajectories/arc_three_methods.jsonl`, `results/arc_summary.csv`,
`results/arc_paired_transitions.csv`, `results/ARC_REPLICATION_GATE.json`, and
`logs/arc_controller/` with captured sources and actual process/exit evidence.
The presence of scripts is not proof that a run completed.
