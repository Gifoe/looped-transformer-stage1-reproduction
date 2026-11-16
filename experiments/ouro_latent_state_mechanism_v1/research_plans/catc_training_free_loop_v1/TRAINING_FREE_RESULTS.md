# Fixed training-free results

Complete frozen Train1094, no correctness filtering. This is development evidence, not independent generalization. All six structured rows and all24 matchedcontrols are retained.

| Method | Correct/1094 | Accuracy | pp vs K4 | Rescue/damage vs K4 | 95% paired accuracy delta |
|---|---:|---:|---:|---:|---|
| T1_structured_K4 | 967 | 88.391% | 0.091 | 2/1 | [-0.0018281535648994515, 0.003656307129798903] |
| T2_structured_K4 | 969 | 88.574% | 0.274 | 3/0 | [0.0, 0.006398537477148081] |
| T3_structured_K4 | 966 | 88.300% | 0.000 | 1/1 | [-0.002742230347349177, 0.002742230347349177] |
| T4_structured_K4 | 967 | 88.391% | 0.091 | 1/0 | [0.0, 0.002742230347349177] |
| T5_structured_K4 | 966 | 88.300% | 0.000 | 0/0 | [0.0, 0.0] |
| T6_structured_K4 | 967 | 88.391% | 0.091 | 2/1 | [-0.0018281535648994515, 0.004570383912248629] |


K2 scores and same-depth rescues/damages, raw margins, ties, norm/clipping and all question IDs are in SUMMARY.json and TRAINING_FREE_ALL_ROWS.jsonl.95% intervals are descriptive; gates use one-sided Bonferroni30 lower bounds within their respective accuracy/margin families. No global family-wise5% claim is made across both gate routes.

Frozen gates:

```json
{
  "T1_structured_K4": {
    "primary_pass": false,
    "mechanism_specific_secondary_pass": true,
    "gate_to_C": true
  },
  "T2_structured_K4": {
    "primary_pass": false,
    "mechanism_specific_secondary_pass": false,
    "gate_to_C": false
  },
  "T3_structured_K4": {
    "primary_pass": false,
    "mechanism_specific_secondary_pass": false,
    "gate_to_C": false
  },
  "T4_structured_K4": {
    "primary_pass": false,
    "mechanism_specific_secondary_pass": false,
    "gate_to_C": false
  },
  "T5_structured_K4": {
    "primary_pass": false,
    "mechanism_specific_secondary_pass": true,
    "gate_to_C": true
  },
  "T6_structured_K4": {
    "primary_pass": false,
    "mechanism_specific_secondary_pass": false,
    "gate_to_C": false
  }
}
```
