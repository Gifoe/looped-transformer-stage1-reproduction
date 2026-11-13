# H2 Train OOF

All analysis/implementation: Sol; Astra provided the separately frozen scientific directive. SUPERVISED_DIAGNOSTIC, not training-free or a deployable method. No Test examples, no new GPU forwards, no Ouro parameter changes.

Primary G1 Train prospective four-class Brier: FAIL. G1 requires positive lower confidence limits against all three prescribed controls. Secondary endpoints do not change this decision.

| Timing | Family | Brier | Logloss | Macro-F1 | ECE10 |
|---|---|---:|---:|---:|---:|
| retrospective | output | 0.535799 | 1.006307 | 0.6256 | 0.3859 |
| retrospective | combined | 0.534041 | 1.003404 | 0.6214 | 0.3817 |
| retrospective | unrelated | 0.534976 | 1.004735 | 0.6296 | 0.3878 |
| retrospective | shuffled | 0.534678 | 1.004330 | 0.6320 | 0.3885 |
| prospective | output | 0.574897 | 1.085014 | 0.3908 | 0.3351 |
| prospective | combined | 0.574779 | 1.081724 | 0.3575 | 0.3269 |
| prospective | unrelated | 0.574272 | 1.081634 | 0.3721 | 0.3095 |
| prospective | shuffled | 0.576053 | 1.084409 | 0.3722 | 0.3276 |

## Paired proper-loss contrasts

Positive means H2 combined improves proper Brier loss. All depths stay within the question; lexical IDs fix bootstrap draw alignment;10000 samples, seed20261113. Intervals are conditional on the correlated cross-validation fits, not independent training-seed uncertainty.

| Timing | Comparator | Improvement | 95% CI |
|---|---|---:|---|
| retrospective | output | +0.001758 | [+0.001022, +0.002496] |
| retrospective | unrelated | +0.000935 | [+0.000056, +0.001804] |
| retrospective | shuffled | +0.000637 | [-0.000408, +0.001648] |
| retrospective | old_geometry | +0.000620 | [-0.000563, +0.001778] |
| prospective | output | +0.000118 | [-0.001276, +0.001524] |
| prospective | unrelated | -0.000507 | [-0.002151, +0.001112] |
| prospective | shuffled | +0.001274 | [-0.000545, +0.003083] |
| prospective | old_geometry | +0.000582 | [-0.001303, +0.002432] |

## Fixed supervised class-to-action replay

Retrospective completedK4 class2 selects K3; otherwise K4, with cost4. Prospective class2 or3 stops currentK2/K3; otherwise continues. No threshold/strategy search. OOF/held-out replay depths are not measured latency.

[
  {
    "setting": "retrospective",
    "family": "output",
    "n": 1094,
    "correct": 959,
    "native_K4_correct": 966,
    "net_correct": -7,
    "rescue": 11,
    "damage": 18,
    "mean_replay_loops": 4.0,
    "paired_question95_net_accuracy": [
      -0.016453382084095063,
      0.002742230347349177
    ],
    "scope": "Fixed preregistered class-to-action mapping, no policy/threshold search; supervised diagnostic, not deployable or training-free."
  },
  {
    "setting": "retrospective",
    "family": "combined",
    "n": 1094,
    "correct": 958,
    "native_K4_correct": 966,
    "net_correct": -8,
    "rescue": 11,
    "damage": 19,
    "mean_replay_loops": 4.0,
    "paired_question95_net_accuracy": [
      -0.01736745886654479,
      0.0018281535648994515
    ],
    "scope": "Fixed preregistered class-to-action mapping, no policy/threshold search; supervised diagnostic, not deployable or training-free."
  },
  {
    "setting": "retrospective",
    "family": "unrelated",
    "n": 1094,
    "correct": 957,
    "native_K4_correct": 966,
    "net_correct": -9,
    "rescue": 10,
    "damage": 19,
    "mean_replay_loops": 4.0,
    "paired_question95_net_accuracy": [
      -0.018281535648994516,
      0.0009140767824497258
    ],
    "scope": "Fixed preregistered class-to-action mapping, no policy/threshold search; supervised diagnostic, not deployable or training-free."
  },
  {
    "setting": "retrospective",
    "family": "shuffled",
    "n": 1094,
    "correct": 958,
    "native_K4_correct": 966,
    "net_correct": -8,
    "rescue": 10,
    "damage": 18,
    "mean_replay_loops": 4.0,
    "paired_question95_net_accuracy": [
      -0.016453382084095063,
      0.0018281535648994515
    ],
    "scope": "Fixed preregistered class-to-action mapping, no policy/threshold search; supervised diagnostic, not deployable or training-free."
  },
  {
    "setting": "prospective",
    "family": "output",
    "n": 1094,
    "correct": 948,
    "native_K4_correct": 966,
    "net_correct": -18,
    "rescue": 6,
    "damage": 24,
    "mean_replay_loops": 2.4725776965265083,
    "paired_question95_net_accuracy": [
      -0.026508226691042046,
      -0.007312614259597806
    ],
    "scope": "Fixed preregistered class-to-action mapping, no policy/threshold search; supervised diagnostic, not deployable or training-free."
  },
  {
    "setting": "prospective",
    "family": "combined",
    "n": 1094,
    "correct": 943,
    "native_K4_correct": 966,
    "net_correct": -23,
    "rescue": 10,
    "damage": 33,
    "mean_replay_loops": 2.4250457038391224,
    "paired_question95_net_accuracy": [
      -0.03290676416819013,
      -0.009140767824497258
    ],
    "scope": "Fixed preregistered class-to-action mapping, no policy/threshold search; supervised diagnostic, not deployable or training-free."
  },
  {
    "setting": "prospective",
    "family": "unrelated",
    "n": 1094,
    "correct": 944,
    "native_K4_correct": 966,
    "net_correct": -22,
    "rescue": 8,
    "damage": 30,
    "mean_replay_loops": 2.419561243144424,
    "paired_question95_net_accuracy": [
      -0.031078610603290677,
      -0.010054844606946984
    ],
    "scope": "Fixed preregistered class-to-action mapping, no policy/threshold search; supervised diagnostic, not deployable or training-free."
  },
  {
    "setting": "prospective",
    "family": "shuffled",
    "n": 1094,
    "correct": 938,
    "native_K4_correct": 966,
    "net_correct": -28,
    "rescue": 9,
    "damage": 37,
    "mean_replay_loops": 2.4232175502742233,
    "paired_question95_net_accuracy": [
      -0.037477148080438755,
      -0.013711151736745886
    ],
    "scope": "Fixed preregistered class-to-action mapping, no policy/threshold search; supervised diagnostic, not deployable or training-free."
  }
]

## Interpretation and scope

The 16-dimensional H2 block is candidate-aligned natural lens geometry, replacing the previous16 geometry coordinates. It is never inserted into old index11/index5 gates. d1=None is unavailable, not a zero measurement; all observations use d2+. Original10 LP-output features and original class-balanced ridge/T1/seeds/folds are unchanged. Output-only OOF equality is an explicit technical control. Extra-feature families have26 dimensions; output-only10 is not falsely called equal parameter count. No base-model weights were trained.

E3 Train1030/Validation299 guided earlier mechanism selection and historical Test was repeatedly exposed. Current Train-only group cross-validation and frozen exposed-Val review do not create independent benchmark confirmation. A G1 pass permits consideration of a supervised inference experiment, not a claim that the practical research goal or a deployable accuracy/compute improvement has been achieved. A G1 failure leaves future direction to the separately authorized Astra scientific directive; neither primary metric nor mapping is changed after results.

## Fixed five-fold prospective stability

Descriptive only; the aggregate preregistered G1 remains unchanged.

| Outer fold | Output Brier | H2 combined Brier | Low-layer Brier | Shuffle Brier |
|---|---:|---:|---:|---:|
| 0 | 0.583351 | 0.582912 | 0.580193 | 0.584616 |
| 1 | 0.557755 | 0.556331 | 0.558079 | 0.558229 |
| 2 | 0.593132 | 0.594550 | 0.591668 | 0.593339 |
| 3 | 0.568262 | 0.567466 | 0.569477 | 0.569907 |
| 4 | 0.569945 | 0.570492 | 0.569996 | 0.572132 |

OOF output control reproduced all5470 original transition probabilities to maximum absolute difference4.44e-16. Raw records, frozen outer coefficients, alpha inner scores and exact group/fold manifest are preserved. The tiny difference is floating-point roundoff, not a changed baseline.
