# Ouro E4/E5: code and complete small results

2026-11-13 E4/E5 snapshot, RTX4090, fixed Ouro-1.4B. This delivery is isolated from the peer E1–E3 work; its specific generating writer remains pending.

The K×V transplant result does not establish the main natural K1→K2 mechanism. In the preregistered128-question check, strict control-specific, tie-free KV necessity cases are0. A conditional R-state→attention restoration path has2 strict cases, while most continuous residual-path harm remains after restoring native attention. Later-loop effects do not support a simple convergence or monotone-decay claim. These are bounded local causal findings, not a complete explanation of all natural gains.

- [Final research report](FINAL_RESEARCH_REPORT.md) · [Mechanistic model and gaps](FINAL_MECHANISTIC_MODEL.md)
- [E4 necessity](E4/E4_NECESSITY_RESULTS.md) · [E4 mediation](E4/E4_MEDIATION_ANALYSIS.md) · [E5 causal depth comparison](E5/E5_CAUSAL_DEPTH_COMPARISON.md)
- [Complete result index](results/RAW_RESULTS_INDEX.md) · [Figures](figures/) · [Independent audits](results/)
- [Execution code](results/code/) · [Analysis code](exec/) · [Reproduction commands](REPRODUCE.md)
- [Research log](MASTER_RESEARCH_LOG.md) · [Execution/source index](RUN_INDEX.json) · [Resource totals](EXECUTION_TOTALS.json)

Train64 is discovery; later Q/R/late299 are exposed-cohort follow-ups. The128 questions were not used to choose these new causal operators, but have historical/eight-corner and possible probe exposure. They are not globally unseen or blind. Cohorts and repeated conditions are not pooled as independent samples.

All per-question native/intervention JSON records are published losslessly as JSONL.gz, including original relative paths, original JSON text and SHA256. Restore them with `python results/code/recover_small_results.py --release . --output recovered`, then recompute scalar scores/gaps/counts with the supplied code. See the [raw index](results/RAW_RESULTS_INDEX.md).

Large activation tensors stay on the authorized server. [Tensor manifest](results/TENSOR_MANIFEST.json) lists exact paths, sizes and hashes. Exact tensor audits require obtaining those files from the owner or regenerating them using the fixed inputs, model and code; Git alone supports full scalar-result reanalysis. Model revision: `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`; checkpoint SHA256: `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.
