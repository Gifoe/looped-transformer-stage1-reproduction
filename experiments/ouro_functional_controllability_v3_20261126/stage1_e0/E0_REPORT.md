# E0 development results

| Arm | External macro | Historical macro | MMLU-Pro parseable | Final cos(op, T1) | Final cos(op, Mixed) |
|---|---:|---:|---:|---:|---:|
| native | 60.74% | 51.81% | 62.71% | — | — |
| t1 | 66.31% | 54.00% | 100.00% | — | — |
| t2 | 66.11% | 56.47% | 100.00% | — | — |
| mixed | 60.12% | 60.67% | 45.76% | — | — |
| core.cold.arc.rank.840.26112601 | 68.60% | 57.55% | 100.00% | 0.1864 | 0.0554 |
| core.cold.arc.rank.840.26112602 | 67.53% | 58.79% | 100.00% | 0.2310 | 0.0597 |
| core.cold.arc.nll_rank.840.26112601 | 59.20% | 56.68% | 22.03% | 0.1201 | 0.1490 |
| core.cold.arc.nll_rank.840.26112602 | 59.18% | 56.77% | 22.03% | 0.1079 | 0.1516 |
| core.cold.arc_csqa.rank.840.26112601 | 68.33% | 54.81% | 100.00% | 0.1594 | 0.0661 |
| core.cold.arc_csqa.rank.840.26112602 | 68.16% | 54.64% | 93.22% | 0.1252 | 0.0870 |
| core.cold.arc_csqa.nll_rank.840.26112601 | 62.50% | 57.10% | 44.07% | 0.0657 | 0.2070 |
| core.cold.arc_csqa.nll_rank.840.26112602 | 58.91% | 55.83% | 25.42% | 0.0501 | 0.2313 |
| core.spec_warm.arc.rank.840.26112601 | 68.08% | 54.21% | 98.31% | 0.6455 | 0.0329 |
| core.spec_warm.arc.rank.840.26112602 | 69.19% | 59.26% | 100.00% | 0.6374 | 0.0326 |
| core.spec_warm.arc.nll_rank.840.26112601 | 61.77% | 59.62% | 40.68% | 0.5336 | 0.0581 |
| core.spec_warm.arc.nll_rank.840.26112602 | 61.90% | 58.38% | 42.37% | 0.5173 | 0.0788 |
| core.spec_warm.arc_csqa.rank.840.26112601 | 68.20% | 54.23% | 100.00% | 0.6456 | 0.0373 |
| core.spec_warm.arc_csqa.rank.840.26112602 | 67.76% | 52.40% | 98.31% | 0.6423 | 0.0471 |
| core.spec_warm.arc_csqa.nll_rank.840.26112601 | 64.11% | 56.31% | 64.41% | 0.5129 | 0.0742 |
| core.spec_warm.arc_csqa.nll_rank.840.26112602 | 61.68% | 55.76% | 45.76% | 0.5169 | 0.0820 |
| norm.norm_rand.arc.rank.840.26112601 | 67.79% | 57.58% | 98.31% | 0.0783 | 0.0273 |
| norm.norm_rand.arc.rank.840.26112602 | 67.59% | 58.28% | 96.61% | 0.0778 | 0.0177 |
| norm.norm_rand.arc_csqa.rank.840.26112601 | 67.04% | 56.64% | 100.00% | 0.0561 | 0.0349 |
| norm.norm_rand.arc_csqa.rank.840.26112602 | 67.07% | 53.77% | 100.00% | 0.0638 | 0.0446 |
| bridge.cold.mixed4.nll_rank.840.26112601 | 58.38% | 62.95% | 22.03% | 0.0265 | 0.3767 |
| bridge.cold.mixed4.nll_rank.840.26112602 | 57.60% | 62.42% | 15.25% | 0.0326 | 0.3694 |
| duration.cold.arc.rank.280.26112601 | 65.56% | 55.42% | 94.92% | 0.2218 | 0.0571 |
| duration.spec_warm.arc.rank.280.26112601 | 67.57% | 55.22% | 98.31% | 0.8362 | 0.0325 |
| historical_init_bridge.historical_t1_axis.arc.rank.840.26112601 | 68.61% | 54.75% | 100.00% | 0.6505 | 0.0340 |
| historical_init_bridge.historical_t1_axis.arc.rank.840.26112602 | 68.57% | 61.27% | 100.00% | 0.6355 | 0.0354 |

The primary macro averages SciQ, ARC-Easy, CosmosQA, MMLU and MMLU-Pro equally.
The official MMLU auxiliary development items retained in the primary panel also occur in the historical comparison.

Matched Rank-only spectral direction effect +0.94 pp. Initialization scale effect -0.78 pp.

Joint objective effect -7.08 pp. ARC+CSQA data effect +0.52 pp.

Cold viability rule

Cold exceeds Native on both 840-step seeds and retains at least 50% of any positive matched spectral gain.
