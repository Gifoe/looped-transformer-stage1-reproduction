# Independent source/input audit

Status: all 2,590 original question metadata records passed coverage-linked SHA256 verification; all historical prediction, gold, and exact top-tie flags agree with existing audited behavior files. This is a data/provenance result, not a new GPU result.

The preservation ZIP contains 448 removed metadata files; the remaining 2,142 records were read from their original files. No old activation tensors were regenerated. The portable collector inputs contain original token IDs, readout position, target IDs, metadata hashes and historical scores; labels are stored separately.

| Cohort | N | K1/K2/K3/K4 correct | Top ties |
|---|---:|---|---|
| train | 1119 | 609 / 919 / 985 / 990 | 46 / 30 / 13 / 9 |
| validation | 299 | 150 / 244 / 257 / 262 | 11 / 1 / 2 / 2 |
| test | 1172 | 626 / 953 / 1000 / 1020 | 37 / 28 / 22 / 18 |
| train_non_demonstration | 1094 | 589 / 894 / 960 / 965 | 46 / 30 / 13 / 9 |

| Test transition | Rescue / wrong at start | Conditional rescue | Damage / correct at start | Conditional damage |
|---|---:|---:|---:|---:|
| K1→K2 | 378 / 546 | 69.2308% | 51 / 626 | 8.1470% |
| K2→K3 | 81 / 219 | 36.9863% | 34 / 953 | 3.5677% |
| K3→K4 | 33 / 172 | 19.1860% | 13 / 1000 | 1.3000% |

The shrinking pool of wrong answers explains part of the smaller absolute gains. Conditional rescue rates also decrease, but the risk sets contain different questions and increasing residual difficulty; these rates alone do not show that the same internal computation weakens. Later native improvements and harms remain real. Hidden-state convergence requires direct state measurements.

The 64-question train pilot was frozen before new interventions: 16 per historical rescue, damage, stable-correct and stable-wrong group, excluding all 25 demonstrations, ordered by SHA256(ouro-e4-v1|ID). This enriched pilot is not a population accuracy estimate. Validation uses all 299 questions. Test has prior exposure.

Historical scores were recomputed in Python floating-point arithmetic from the saved FP32 log probabilities. This audit does not claim a full FP32 model forward. The new 4090 baseline must be paired within the new run and checked separately against the A6000 history.

Reproduction: `exec/root_build_portable_inputs.py`, `inputs/EXPORT_MANIFEST.json`, `inputs/historical_per_question_audit.jsonl`, and `results/ROOT_INPUT_AUDIT.json`.
