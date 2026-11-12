# E4/E5 execution and evidence log

This is the 2026-11-13 (Asia/Shanghai) E4/E5 snapshot from the RTX4090 instance. Earlier E1–E3 producer tracing is tracked separately; a specific generating writer remains unresolved. All timestamps below are UTC.

- 2026-11-12 13:19: audited the empty instance, reused an existing fixed checkpoint, and built an isolated matching inference environment. Original model weights and historical runs were preserved.
- 13:41–14:08: completed the scoped KV technical gate, fixed Train64 discovery and full Validation299 matched KV interventions across K1–K4. Reconstructed actual complete-shape SDPA and changed only the answer-readout output row.
- 14:16–14:28: completed natural-K2 Q × joint-KV factorial and head-norm controls, reusing the verified KV cell and current-machine native baselines.
- 14:41–14:54: completed true input-R-state rollback, actual A(U), mediator restoration and matched controls. Native, blocked and clamped node identities and all non-R guards passed.
- 15:04–15:30: completed identical actual-state/attention mediation at K3←K2 and K4←K3, including frozen precision checks and all299 questions per receiver.
- 15:15:49: froze128 remaining Train questions by label-independent ID hash, excluding demonstrations, both discovery sets and exact input-hash overlap. This was before any new128 forward. Historical/probe exposure is retained as a limitation.
- 15:39 onward: completed technical gates, then the fixed native4/reduced-KV, Q, R2, R3 and R4 confirmation sequence. All23 execution namespaces are complete; all independent audits and scope/precision closure checks passed. No experiment follows Order006.

The project contains491 unique questions, with repeated technical and operator conditions recorded separately. There are28,722 measured model-forward windows; these are not independent sampling replicates. Detailed timing, coverage, source identities and measurement limits are in [RUN_INDEX.json](RUN_INDEX.json), [EXECUTION_TOTALS.json](EXECUTION_TOTALS.json) and [SOL_EXECUTION_LEDGER.md](SOL_EXECUTION_LEDGER.md).

Read [FINAL_RESEARCH_REPORT.md](FINAL_RESEARCH_REPORT.md) for the consolidated scientific judgment, [FINAL_MECHANISTIC_MODEL.md](FINAL_MECHANISTIC_MODEL.md) for the bounded mechanism and remaining gaps, and [REPRODUCE.md](REPRODUCE.md) for reproduction. All original numeric records are retained. GitHub publication is separately authorized and excludes model weights, wheels, temporary archives and the large tensor collection.
