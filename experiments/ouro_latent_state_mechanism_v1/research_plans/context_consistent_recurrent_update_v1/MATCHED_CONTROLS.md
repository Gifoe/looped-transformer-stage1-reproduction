# Matched controls

Selected Q-. Confirmation uses frozen same-sign Q/R/P/random conditions.

| Condition | Depth | Correct/N | Rescue/damage | Strict rescue/damage | Tied questions | Strict delta 95% (pp) |
|---|---:|---:|---:|---:|---:|---|
| N | K2 | 412/512 | 0/0 | 0/0 | 11 | [+0.000,+0.000] |
| N | K3 | 443/512 | 0/0 | 0/0 | 4 | [+0.000,+0.000] |
| N | K4 | 448/512 | 0/0 | 0/0 | 4 | [+0.000,+0.000] |
| Q- | K2 | 412/512 | 5/5 | 3/1 | 13 | [-0.391,+1.172] |
| Q- | K3 | 444/512 | 2/1 | 1/0 | 7 | [+0.000,+0.586] |
| Q- | K4 | 448/512 | 1/1 | 0/0 | 6 | [+0.000,+0.000] |
| R- | K2 | 415/512 | 4/1 | 0/1 | 9 | [-0.586,+0.000] |
| R- | K3 | 443/512 | 1/1 | 0/0 | 2 | [+0.000,+0.000] |
| R- | K4 | 448/512 | 0/0 | 0/0 | 3 | [+0.000,+0.000] |
| P- | K2 | 415/512 | 3/0 | 0/0 | 9 | [+0.000,+0.000] |
| P- | K3 | 441/512 | 0/2 | 0/0 | 5 | [+0.000,+0.000] |
| P- | K4 | 450/512 | 2/0 | 0/0 | 6 | [+0.000,+0.000] |
| Random- | K2 | 414/512 | 2/0 | 0/0 | 8 | [+0.000,+0.000] |
| Random- | K3 | 444/512 | 2/1 | 1/0 | 2 | [+0.000,+0.586] |
| Random- | K4 | 450/512 | 2/0 | 1/0 | 3 | [+0.000,+0.586] |

## Primary paired contrasts

{
  "N": {
    "mean": 0.0,
    "ci95": [
      0.0,
      0.0
    ],
    "lower_bonferroni4": 0.0,
    "n": 512
  },
  "R-": {
    "mean": 0.0,
    "ci95": [
      0.0,
      0.0
    ],
    "lower_bonferroni4": 0.0,
    "n": 512
  },
  "P-": {
    "mean": 0.0,
    "ci95": [
      0.0,
      0.0
    ],
    "lower_bonferroni4": 0.0,
    "n": 512
  },
  "Random-": {
    "mean": -0.001953125,
    "ci95": [
      -0.005859375,
      0.0
    ],
    "lower_bonferroni4": -0.0078125,
    "n": 512
  }
}

Each contrast uses question-paired strict utilities relative to native: 5,000 bootstrap samples, seed20261117, four one-sided Bonferroni bounds. Ordinary accuracy must also improve. Gate pass: False.

## Doses and cost

{
  "development:N": {
    "questions": 256,
    "token_sites": 0,
    "total_L2_mean": 0.0,
    "per_token_actual_L2_mean": 0,
    "max_actual_relative": 0,
    "clipped_tokens": 0,
    "random_matched_sites": 0,
    "random_sites": 0,
    "sum_capture_seconds": 50.74200733844191,
    "peak_allocated_bytes": 3708565504
  },
  "development:Q+": {
    "questions": 256,
    "token_sites": 16109,
    "total_L2_mean": 3.0687847077332995,
    "per_token_actual_L2_mean": 0.3894784694504139,
    "max_actual_relative": 0.09999908415957003,
    "clipped_tokens": 88,
    "random_matched_sites": 0,
    "random_sites": 0,
    "sum_capture_seconds": 50.85316266864538,
    "peak_allocated_bytes": 3708565504
  },
  "development:Q-": {
    "questions": 256,
    "token_sites": 16109,
    "total_L2_mean": 3.0687789182108873,
    "per_token_actual_L2_mean": 0.3894774065554364,
    "max_actual_relative": 0.09999608832634738,
    "clipped_tokens": 88,
    "random_matched_sites": 0,
    "random_sites": 0,
    "sum_capture_seconds": 51.03840446472168,
    "peak_allocated_bytes": 3708565504
  },
  "development:R+": {
    "questions": 256,
    "token_sites": 256,
    "total_L2_mean": 0.3848541243911476,
    "per_token_actual_L2_mean": 0.3848541243911476,
    "max_actual_relative": 0.0791348650166178,
    "clipped_tokens": 0,
    "random_matched_sites": 0,
    "random_sites": 0,
    "sum_capture_seconds": 50.59294935222715,
    "peak_allocated_bytes": 3708565504
  },
  "development:R-": {
    "questions": 256,
    "token_sites": 256,
    "total_L2_mean": 0.3848842118006779,
    "per_token_actual_L2_mean": 0.3848842118006779,
    "max_actual_relative": 0.07914212241728176,
    "clipped_tokens": 0,
    "random_matched_sites": 0,
    "random_sites": 0,
    "sum_capture_seconds": 50.58917786180973,
    "peak_allocated_bytes": 3708565504
  },
  "development:P+": {
    "questions": 256,
    "token_sites": 470784,
    "total_L2_mean": 19.025540362466455,
    "per_token_actual_L2_mean": 0.3942422660610398,
    "max_actual_relative": 0.09999929163583218,
    "clipped_tokens": 2560,
    "random_matched_sites": 0,
    "random_sites": 0,
    "sum_capture_seconds": 56.836333331651986,
    "peak_allocated_bytes": 3708565504
  },
  "development:P-": {
    "questions": 256,
    "token_sites": 470784,
    "total_L2_mean": 19.02580330290433,
    "per_token_actual_L2_mean": 0.39424551202999425,
    "max_actual_relative": 0.0999905362989143,
    "clipped_tokens": 2560,
    "random_matched_sites": 0,
    "random_sites": 0,
    "sum_capture_seconds": 56.318287127651274,
    "peak_allocated_bytes": 3708565504
  },
  "development:Random+": {
    "questions": 256,
    "token_sites": 16109,
    "total_L2_mean": 3.068786730543518,
    "per_token_actual_L2_mean": 0.3894786652803782,
    "max_actual_relative": 0.09999959861207858,
    "clipped_tokens": 88,
    "random_matched_sites": 16109,
    "random_sites": 16109,
    "sum_capture_seconds": 53.86786629538983,
    "peak_allocated_bytes": 3708565504
  },
  "development:Random-": {
    "questions": 256,
    "token_sites": 16109,
    "total_L2_mean": 3.068779098695552,
    "per_token_actual_L2_mean": 0.3894773997246711,
    "max_actual_relative": 0.09999581708590737,
    "clipped_tokens": 88,
    "random_matched_sites": 16109,
    "random_sites": 16109,
    "sum_capture_seconds": 54.149536958895624,
    "peak_allocated_bytes": 3708565504
  },
  "confirmation:N": {
    "questions": 512,
    "token_sites": 0,
    "total_L2_mean": 0.0,
    "per_token_actual_L2_mean": 0,
    "max_actual_relative": 0,
    "clipped_tokens": 0,
    "random_matched_sites": 0,
    "random_sites": 0,
    "sum_capture_seconds": 100.8162295203656,
    "peak_allocated_bytes": 3701143552
  },
  "confirmation:Q-": {
    "questions": 512,
    "token_sites": 32939,
    "total_L2_mean": 3.0997470682103563,
    "per_token_actual_L2_mean": 0.38862884026500993,
    "max_actual_relative": 0.09999973671814139,
    "clipped_tokens": 173,
    "random_matched_sites": 0,
    "random_sites": 0,
    "sum_capture_seconds": 101.78970886114985,
    "peak_allocated_bytes": 3701143552
  },
  "confirmation:R-": {
    "questions": 512,
    "token_sites": 512,
    "total_L2_mean": 0.38479548602589697,
    "per_token_actual_L2_mean": 0.38479548602589697,
    "max_actual_relative": 0.0813149931839291,
    "clipped_tokens": 0,
    "random_matched_sites": 0,
    "random_sites": 0,
    "sum_capture_seconds": 101.32291625905782,
    "peak_allocated_bytes": 3701143552
  },
  "confirmation:P-": {
    "questions": 512,
    "token_sites": 941568,
    "total_L2_mean": 19.02568675048741,
    "per_token_actual_L2_mean": 0.39424223193383395,
    "max_actual_relative": 0.0999905362989143,
    "clipped_tokens": 5120,
    "random_matched_sites": 0,
    "random_sites": 0,
    "sum_capture_seconds": 112.83751954045147,
    "peak_allocated_bytes": 3701143552
  },
  "confirmation:Random-": {
    "questions": 512,
    "token_sites": 32939,
    "total_L2_mean": 3.0997458067547354,
    "per_token_actual_L2_mean": 0.38862868290892605,
    "max_actual_relative": 0.09999954703217874,
    "clipped_tokens": 173,
    "random_matched_sites": 32939,
    "random_sites": 32939,
    "sum_capture_seconds": 107.85240783728659,
    "peak_allocated_bytes": 3701143552
  }
}

P edits all prefix tokens, R edits one token, Q edits the current question/options excluding R. Thus regional total doses differ systematically: this comparison does not identify a unique context-semantics mechanism. Random directions match the actual structured-Q displacement per token to within the reported precision; all unmatched cases are retained. Capture timings include four full-shaped readout heads and observation hooks in every condition, not an optimized deployment claim.
