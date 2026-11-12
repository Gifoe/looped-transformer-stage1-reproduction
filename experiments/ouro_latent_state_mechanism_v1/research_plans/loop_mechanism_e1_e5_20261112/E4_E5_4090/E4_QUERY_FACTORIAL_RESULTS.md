# Natural K2 Query-factorial results

Final confirmation 128: Q-only rescue/damage 2/4, joint0/5; each has3 strict natural-correction losses. Q-direction matchedheadnorm meanwrong−0.092611, Q-only−0.095825, magnitude-only+0.001790. All four metrics/groups are retained in ROOT_CONFIRMATION128_ALL_METRICS.json. Earlier299 is exposed analysis for operator selection, not independent confirmation.

All 64 frozen Train questions and all 299 Validation questions completed, with exact current-hardware baseline/source reuse and independent CPU audits. This follow-up adds Q-only and joint QKV rollback; it does not rerun the original 26 conditions. Current native K2 has 243/299 correct.

| Rollback or control | Rescue / damage | Mean gold–meanwrong change | Dynamic margin change |
|---|---:|---:|---:|
| Q only | 2 / 10 | −0.134685 | −0.150293 |
| Q1 direction, Q2 per-head norm | 2 / 10 | −0.134964 | −0.151547 |
| Q2 direction, Q1 per-head norm | 2 / 1 | −0.005086 | −0.005644 |
| Q-only endpoint norm control | 0 / 0 | +0.000070 | −0.005226 |
| Q-only unrelated output-delta control | 4 / 2 | −0.048147 | −0.078804 |
| Joint QKV | 4 / 19 | −0.484114 | −0.544732 |
| Joint endpoint norm control | 0 / 2 | +0.003066 | +0.000209 |
| Joint unrelated output-delta control | 7 / 9 | −0.051404 | −0.176212 |

Q-only loses ten of 104 natural corrections. Seven are control-specific and free of native/target endpoint ties. Joint QKV loses eighteen natural corrections; eleven are control-specific and tie-free. Eighty-six natural corrections survive complete L18 attention-output rollback. Therefore Query contributes to some natural decisions, but retaining Query alone does not explain the broader robustness to KV rollback. Direction-preserving norm controls differ substantially from donor-direction rollback, limiting an explanation based solely on Query head magnitude.

The new finite difference is **Q × joint-KV in a natural K2 receiver**: F(Q1K1V1)−F(Q1K2V2)−F(Q2K1V1)+F(Q2K2V2). Validation mean gold–meanwrong interaction is −0.021147, with exact raw-logit signs 132 positive/147 negative/20 zero. Fixed-wrong mean is +0.000418 and dynamic mean −0.029891. This differs both in factors and receiver from the historical fixed-K1 K×V interaction. It neither directly refutes that historical result nor measures a three-way Q×K×V interaction or mediated percentage.

The enriched pilot again has only one damaged natural correction, a tied control-sensitive case; retain that negative pilot result alongside the full Validation result. One of eight pilot Q-only FP32-SDPA/BF16-writeback sensitivity conditions changes prediction. No full-model FP32 claim is made.

Detailed independent audit: results/ASTRA_QUERY_VALIDATION299_REVIEW_INDEPENDENT.json. Figure: figures/validation299_query_factorial.png, with PDF/SVG exports. These findings motivate the separately frozen single-site residual-to-attention mediation test in Order 004; that later test is not included in this table.
