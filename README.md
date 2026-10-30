# Neural Pairing: nl33w ARC Hidden-State Audit

This branch publishes the 2026-10-30 audit. The existing main branch is unchanged.

- [Full report](experiments/neural_pairing_hidden_state_audit_v1/REPORT.md)
- [Case studies](experiments/neural_pairing_hidden_state_audit_v1/CASE_STUDIES.md)
- [Code, results and figures](experiments/neural_pairing_hidden_state_audit_v1/)
- [Release: complete code/source package and raw state data](https://github.com/Gifoe/looped-transformer-stage1-reproduction/releases/tag/neural-pairing-nl33w-arc-audit-20261030)

Complete ARC-Challenge1165 analysis,95-question eight-layer intervention study and16 new target validation. Norm-controlled K/V substitution improved margin on13/16 new cases, but0 answer repairs passed both native BF16 and FP64 output-dot checks. FP64 refers only to the readout, not a higher-precision backbone. No new semantic failure mechanism or Loop justification is established.
