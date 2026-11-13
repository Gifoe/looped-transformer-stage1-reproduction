# Early exits: replay and actual execution

Fresh RTX5090 / BF16 / torch2.7.1+cu128, 1172 Test questions. Every method shares the same frozen model, prompt, candidate set and scoring protocol. Historical A6000 scores remain separate. All 95% intervals use 10000 paired question bootstraps, seed20261113, without multiplicity correction. The Test set had prior project exposure. No Test-driven refit or parameter revision was performed.

| Method | Correct / 1172 | Accuracy % | vs K4 pp | Rescue / damage | Paired 95% pp CI | Mean loops (replay) |
|---|---:|---:|---:|---:|---|---:|
| K4 | 1022 | 87.201 | +0.000 | 0 / 0 | [+0.000, +0.000] | 4.0000 |
| early_fixed | 1021 | 87.116 | -0.085 | 0 / 1 | [-0.256, +0.000] | 3.1084 |
| cal_early_0.6 | 1017 | 86.775 | -0.427 | 0 / 5 | [-0.853, -0.085] | 2.8908 |
| cal_early_0.7 | 1017 | 86.775 | -0.427 | 0 / 5 | [-0.853, -0.085] | 2.8951 |
| cal_early_0.8 | 1019 | 86.945 | -0.256 | 0 / 3 | [-0.597, +0.000] | 2.9292 |
| cal_early_0.9 | 1021 | 87.116 | -0.085 | 0 / 1 | [-0.256, +0.000] | 3.1049 |
| cal_early_0.95 | 1021 | 87.116 | -0.085 | 0 / 1 | [-0.256, +0.000] | 3.2662 |
| cal_early_0.99 | 1022 | 87.201 | +0.000 | 0 / 0 | [+0.000, +0.000] | 3.6886 |

Test rows above are prefix replay. No actual Test latency is claimed. Cal early0.99 was a historical-development choice, while the fresh5090 Cal primary is LoopCD; these identities are kept separate.

Actual Joint Validation299 online forwards stop at the selected prefix via an exception before later loops; physical first-layer calls equal selected depth and full-vocabulary endpoint hashes equal independently captured native prefixes. Alternating policy order controls warm-order effects. Timings include instrumented identity/hash checks and extra intermediate heads, exclude model loading, and are not uninstrumented production throughput.

{
  "cal_early_0.99": {
    "n": 299,
    "correct": 261,
    "actual_mean_loops": 3.7123745819397995,
    "mean_seconds": 0.17315298591718625,
    "median_seconds": 0.18115328717976809,
    "mean_seconds_excluding_first_question": 0.17313233124804778,
    "peak_allocated_bytes": 3130357248,
    "actual_first_layer_calls_equal_depth": true
  },
  "early_fixed": {
    "n": 299,
    "correct": 260,
    "actual_mean_loops": 3.1270903010033444,
    "mean_seconds": 0.1466519699609499,
    "median_seconds": 0.13771618250757456,
    "mean_seconds_excluding_first_question": 0.1465405304916203,
    "peak_allocated_bytes": 3130322432,
    "actual_first_layer_calls_equal_depth": true
  },
  "native4": {
    "n": 299,
    "correct": 261,
    "actual_mean_loops": 4.0,
    "mean_seconds": 0.1817139349597503,
    "median_seconds": 0.17487889807671309,
    "mean_seconds_excluding_first_question": 0.17991304550719942,
    "peak_allocated_bytes": 3108288000,
    "actual_first_layer_calls_equal_depth": true
  }
}
