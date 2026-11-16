# Directional response

Only Native, V+0.1, V-0.1, P+0.1 and P-0.1 at K2 R, physical layers 10-16, were eligible. All original 25-shot prompts, candidate order and full four-pass execution were reused unchanged. These are same-GPU interventions from the completed CATC pilot; no new coefficient search or model fitting occurred.

The deterministic development cohort contains 256 questions. An input-hash audit found one identical prompt across the ID-based development/confirmation split. The confirmation duplicate remains in raw data and original512 statistics, but is excluded from the corrected511 inference population. This repair occurred after initial aggregates were read; all Train outcomes also had prior project exposure. It is not a blind replication. The prospective split and selection rules are in FREEZE.json.

## K2 development

| Setting | Correct/256 | Mean candidate CE | CE delta vs native | CE improves/harms/neutral | Same-depth rescue/damage |
|---|---:|---:|---:|---:|---:|
| Native | 214 | 0.402008621 | +0.000000000 | 0/0/256 | 0/0 |
| V+ | 213 | 0.399848052 | -0.002160569 | 174/49/33 | 0/1 |
| V- | 212 | 0.405752900 | +0.003744280 | 51/169/36 | 0/2 |
| P+ | 214 | 0.402975704 | +0.000967083 | 128/92/36 | 1/1 |
| P- | 214 | 0.402840094 | +0.000831473 | 82/132/42 | 1/1 |

| Component | Prefer positive | Prefer negative | No unique beneficial direction |
|---|---:|---:|---:|
| V | 163/256 | 25/256 | 68/256 |
| P | 105/256 | 56/256 | 95/256 |

The gold-dependent best-setting accuracy oracle is 215/256; mean best CE is 0.386598571. It is an offline upper bound, never an inference-time controller.

## K4 development

| Setting | Correct/256 | Mean candidate CE | CE delta vs native | CE improves/harms/neutral | Same-depth rescue/damage |
|---|---:|---:|---:|---:|---:|
| Native | 229 | 0.283655754 | +0.000000000 | 0/0/256 | 0/0 |
| V+ | 229 | 0.284581021 | +0.000925268 | 111/67/78 | 0/0 |
| V- | 229 | 0.284690375 | +0.001034621 | 58/114/84 | 0/0 |
| P+ | 229 | 0.285279578 | +0.001623825 | 87/82/87 | 0/0 |
| P- | 229 | 0.283432637 | -0.000223116 | 73/110/73 | 0/0 |

| Component | Prefer positive | Prefer negative | No unique beneficial direction |
|---|---:|---:|---:|
| V | 92/256 | 32/256 | 132/256 |
| P | 57/256 | 47/256 | 152/256 |

The gold-dependent best-setting accuracy oracle is 229/256; mean best CE is 0.275750899. It is an offline upper bound, never an inference-time controller.

Neutral means no CE improvement beyond 1e-8 or a positive/negative tie within 1e-8; exact changes are preserved. Beneficial-direction preference is a CE diagnostic, not an accuracy gain.

## Fixed subset stability

| Development subset | N | K4 CE delta V+ | V- | P+ | P- |
|---|---:|---:|---:|---:|---:|
| halves 0 | 128 | +0.0047317 | +0.0048614 | +0.0000954 | +0.0014735 |
| halves 1 | 128 | -0.0028811 | -0.0027921 | +0.0031523 | -0.0019197 |
| quarters 0 | 64 | +0.0031453 | +0.0051479 | -0.0007366 | -0.0003732 |
| quarters 1 | 64 | +0.0063181 | +0.0045748 | +0.0009274 | +0.0033202 |
| quarters 2 | 64 | -0.0008839 | -0.0034228 | +0.0056605 | -0.0008555 |
| quarters 3 | 64 | -0.0048784 | -0.0021615 | +0.0006440 | -0.0029840 |

Selected **P-**, alphaV=0, alphaP=-0.1, by minimum mean K4 candidate CE among the four nonzero settings. Native remains a comparator. Exact ties follow V+, V-, P+, P-. This is supervised development calibration, not zero-label selection. Selection was frozen before reading new matched-control or Validation outputs. All per-question candidate logits, losses, preferences and ties are in QUESTION_LEVEL_RESULTS.jsonl (losslessly compressed in the GitHub package).

Both decompositions exactly reconstruct the same attention-output difference on 21 fixed technical sites. The symmetric convention moves half the interaction term between Value and Routing. Therefore component labels are not uniquely identifiable causal mechanisms. See TECHNICAL_GATE.json for reconstruction errors, component norms, cosines and witness hashes.
