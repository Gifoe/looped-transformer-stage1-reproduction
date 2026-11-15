import pathlib,json
R=pathlib.Path(__file__).resolve().parents[1]
O=R/'publication/experiments/ouro_latent_state_mechanism_v1/research_plans/k1_k2_latent_mechanism_completion'
assert (O/'FINAL_RESEARCH_REPORT.md').exists()
text='''# K1 to K2 mechanism: audited boundary-to-readout investigation

This package completes the fixed A/B/C investigation with a conditional causal account and explicit unresolved limits. All resumed work is Sol. Start with [FINAL_RESEARCH_REPORT.md](FINAL_RESEARCH_REPORT.md) and [CAUSAL_EVIDENCE_MAP.md](CAUSAL_EVIDENCE_MAP.md).

|Question|Report|
|---|---|
|Competing hypotheses and decisions|[RESEARCH_HYPOTHESES.md](RESEARCH_HYPOTHESES.md)|
|Frozen-readout accessibility versus native use|[INFORMATION_AVAILABILITY.md](INFORMATION_AVAILABILITY.md)|
|Actual recurrent boundary and selected state changes|[K1_K2_STATE_TRANSITION.md](K1_K2_STATE_TRANSITION.md)|
|Boundary-conditioned late attention/MLP operators|[UPSTREAM_CAUSAL_TRACING.md](UPSTREAM_CAUSAL_TRACING.md)|
|L18 residual/attention effects on natural corrections|[NATURAL_MEDIATION_RESULTS.md](NATURAL_MEDIATION_RESULTS.md)|
|Exact failures, ties and case sets|[COUNTEREXAMPLES.md](COUNTEREXAMPLES.md)|
|Numerics, source hashes and coverage|[NUMERICAL_AND_SOURCE_AUDIT.md](NUMERICAL_AND_SOURCE_AUDIT.md)|
|Resulting scoped mechanism and remaining edges|[FINAL_MECHANISTIC_MODEL.md](FINAL_MECHANISTIC_MODEL.md)|

## Code and evidence layout

- `source/original_tree/`: exact original science-frozen execution tree and freezes; restore these relative paths when reconstructing the original run. Original source versions are retained, not silently overwritten.
- `source/execution_and_review/`: versioned asynchronous/isolated-worker execution, independent reviews, source/format lineage, reporting, plotting and public scalar reproduction.
- `results/AB1/{train64,train1030,validation299}/`: all original14 endpoint question rows, original16 frozen E2 outputs, candidate-relative geometry, selected-vector checks and complete summaries.
- `results/B2/{train64,train1030,validation299}/`: all25 endpoint scalar evidence, every offered candidate score/prediction/tie, exact BF16 margins, controls, all natural groups, primary tables and independent scalar review.
- `receipts/`: coverage, full saved-tensor audits, technical gates, science/execution freezes and retained failed engineering attempts.
- `historical_C1/`: separate earlier5090 Q9-conditioned bridge and fixed8 numerical sensitivity. Its groups/operators are not merged with new4090 results.
- `figures/`: PNG/PDF/editable SVG, plotted source hashes and visual QA.
- `PACKAGE_MANIFEST.json`: SHA256/size and Git blob SHA for every physical public artifact. `SHARDED_ARTIFACTS.json` describes byte-exact parts of larger logical files.

## Reproduce published scalar tables

Use Python with NumPy1.26.4, the recorded analysis version. No model execution or probe fitting is needed for this check:

```bash
python source/execution_and_review/reporting/recompute_published.py --root .
python source/execution_and_review/reporting/assemble_shards.py --root . --verify-only
```

The scalar reproducer loads shards transparently and verifies package file hashes, all per-question candidate predictions/ties, three production/exact margin definitions, original B2 contrasts, all4 natural groups, original5000-bootstrap intervals/gates and frozen E2 scalar accuracy tables. It is not a substitute for the original raw-tensor/full-vocabulary model audit or reapplication of heads to unpublished raw features; those completed independent server audit receipts are retained.

To materialize larger logical files for another tool, use `assemble_shards.py --root . --output /your/scratch/reassembled`. Parts retain the complete compressed file or original source bytes, not reduced data. The original server keeps the full raw tensor CAS and canonical hashes. Model weights and every raw tensor archive are not uploaded to this Git repository.

Full GPU reproduction uses the recorded checkpoint/source/runtime, original gold-free model inputs, original frozen heads and the original technical identity/full48-layer replay gates. A different GPU/backend needs a declared new execution identity and fresh native baselines; do not disable source/coverage guards or mix historical predictions. Runtime: BF16/SDPA, torch2.7.1+cu128, transformers4.55.0, no cache, eval, two loops; figure runtime matplotlib3.9.0. Full source files and orders identify actual producers.

## Earlier evidence retained in this branch

[Prior E1-E5 synthesis](../loop_mechanism_e1_e5_20261112/E1_E5_SYNTHESIS/FINAL_RESEARCH_REPORT.md), [E6 control/utility investigation](../e6_loop_control_20261113/README.md), and [H2 Validation report](../latent_dynamics_continuation_20261113/H2_candidate_direction/analysis/H2_VALIDATION_REPORT.md) remain unchanged. Current results do not claim a validated label-free inference selector, an improved controller or a fraction of all reasoning explained.
'''
(O/'README.md').write_text(text)
print('README_WRITTEN')
