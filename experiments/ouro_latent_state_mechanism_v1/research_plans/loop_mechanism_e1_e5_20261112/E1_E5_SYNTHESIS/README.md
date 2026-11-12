# E1–E5 research release and execution status

This directory is the cross-program entry point. Reports are separated by experiment and preserve their original evidence scope.

- [E1/E2 audited results, code and compact rows](../E1_E2_A6000/README.md) — completed E1 Train/Validation/Test and E2 frozen Test and fixed discovery studies. The E1 Train damage q1 pre-frozen sign fails; the E2 probe and precision limitations are retained.
- [E3 4090 writer and source interventions](../E3_WRITER_4090/README.md) — parent full-attention bundle Train/Validation, E/L discovery, and L-only nested Train/Validation all completed; L passes all six frozen criteria in both splits. L9 E failed two qualification criteria. No new L9 key/value Stage B or routing test is authorized by that failure.
- [E4/E5 previously published research](../E4_E5_4090/README.md) — independently scoped 299 and 128 controlled state/attention and depth experiments, including strict KV confirmation failure. It does not establish mediation of the new E3 L intervention.
- [E1–E5 mechanistic model](FINAL_MECHANISTIC_MODEL.md), [research report](FINAL_RESEARCH_REPORT.md), [causal evidence map](CAUSAL_EVIDENCE_MAP.md), and [counterexamples](COUNTEREXAMPLES.md) — integrated synthesis with explicit scope and limitations.

## Reproduction scope

The E3 package contains a SHA manifest of its copied site-specific source and immutable receipt references. Its README specifies the checkpoint revision and environment, historical inputs and audit bindings. E4/E5's [reproduction guide](../E4_E5_4090/REPRODUCE.md) documents its own inputs and executable code. E1/E2 source and cohort ledgers are in its source and historical research-plan tree. The model checkpoint and large original intervention tensors are not hosted in this small GitHub release; full forward reproduction therefore also requires access to those separately retained sources and the exact source/target/offset ledgers named in the receipts.

Read-only verification of archived frozen artifacts is distinct from a fresh experimental rerun. Historical receipts bind exact paths, code, weights, runtime and tensor hashes. Moving files or editing hard-coded paths changes those bindings: a new run requires an isolated namespace, fresh identity/technical checks and new receipts, rather than declaring historical hashes passed. Cross-GPU baselines must be re-established on the target device. An earlier authorized cleanup removed some raw natural shards; current E1/E2 provenance distinguishes historical SHA references from actual rereads of surviving branch tensors and recovered metadata. The paused Hugging Face bulk upload is not resumed by this release.

Historical `PENDING.json` is the immutable post-collection/pre-audit receipt, not a live status flag; a successful linked CUDA replay, CPU arithmetic audit and final analysis establish completed current status. See `../E3_WRITER_4090/README.md` for the full rule.
