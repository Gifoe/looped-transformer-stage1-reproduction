# Validation evidence

The final delivery retains executed model checks, rather than substituting unit
stubs for GPU forward comparisons.

| Check | Executable entry point | Evidence |
| --- | --- | --- |
| HF / Qwen wrapper identity, K=1, beta=1, canonical cache and decode bypass | `scripts/qwen_audit.py` | `results/qwen_parity.json` |
| Ouro four-depth readout and normalization | `scripts/ouro_audit.py --stage parity` | `results/ouro_parity.json` |
| Autoregressive omega-zero identity against HF.generate | `scripts/math_generation.py` | `results/generation_parity.json` |
| True truncated depth and 24/48/72 cache levels | `scripts/math_depth_audit.py` | `results/math_depth_parity.json` |
| Independent complete teacher-forcing likelihood | `scripts/scoring_and_timing.py` | `results/scoring_parity.json` |
| Same-reference mathematical four-depth readout | `scripts/math_teacherforced_probe.py`, invoked by scoring script | `results/math_teacherforced_parity.json` |
| Released gate CDF against native threshold readout | `scripts/native_gate_probe.py`, invoked by scoring script | `results/native_gate_parity.json` |
| All model weight hashes | `scripts/verify_snapshots.py` | `results/checkpoint_integrity.json` |
| Halting causality, KL's first valid depth, bootstrap pairing, split alignment, final-endpoint oscillation | `python -m unittest discover -s tests -v` | `logs/final_protocol_tests.log` |
| Counts, unique IDs, cross-method alignment, parity gates, budget | `scripts/package_artifacts.py` | `results/completion_audit.json` |

The inaccessible author Qwen patch remains a separate, unpassed source gate.
Passing HF identity checks does not establish parity with that patch.
