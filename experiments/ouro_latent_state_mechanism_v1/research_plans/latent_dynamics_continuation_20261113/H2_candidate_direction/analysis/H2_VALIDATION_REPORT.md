# H2 Exposed Validation review

All analysis/implementation: Sol; Astra provided the separately frozen scientific directive. SUPERVISED_DIAGNOSTIC, not training-free or a deployable method. No Test examples, no new GPU forwards, no Ouro parameter changes.

Primary G1 Train prospective four-class Brier: FAIL. G1 requires positive lower confidence limits against all three prescribed controls. Secondary endpoints do not change this decision.

| Timing | Family | Brier | Logloss | Macro-F1 | ECE10 |
|---|---|---:|---:|---:|---:|
| retrospective | output | 0.556348 | 1.039064 | 0.5755 | 0.3358 |
| retrospective | combined | 0.554285 | 1.035722 | 0.5844 | 0.3437 |
| retrospective | unrelated | 0.555863 | 1.037876 | 0.5849 | 0.3439 |
| retrospective | shuffled | 0.555173 | 1.036589 | 0.5852 | 0.3402 |
| prospective | output | 0.598275 | 1.124739 | 0.3434 | 0.2993 |
| prospective | combined | 0.596868 | 1.119679 | 0.3248 | 0.3078 |
| prospective | unrelated | 0.599911 | 1.125607 | 0.3394 | 0.2834 |
| prospective | shuffled | 0.598892 | 1.124322 | 0.3220 | 0.2689 |

## Paired proper-loss contrasts

Positive means H2 combined improves proper Brier loss. All depths stay within the question; lexical IDs fix bootstrap draw alignment;10000 samples, seed20261113. Intervals are conditional on the correlated cross-validation fits, not independent training-seed uncertainty.

| Timing | Comparator | Improvement | 95% CI |
|---|---|---:|---|
| retrospective | output | +0.002063 | [+0.000610, +0.003534] |
| retrospective | unrelated | +0.001578 | [+0.000010, +0.003176] |
| retrospective | shuffled | +0.000888 | [-0.001023, +0.002800] |
| prospective | output | +0.001408 | [-0.001208, +0.004099] |
| prospective | unrelated | +0.003043 | [-0.000088, +0.006218] |
| prospective | shuffled | +0.002025 | [-0.001254, +0.005338] |

## Fixed supervised class-to-action replay

Retrospective completedK4 class2 selects K3; otherwise K4, with cost4. Prospective class2 or3 stops currentK2/K3; otherwise continues. No threshold/strategy search. OOF/held-out replay depths are not measured latency.

[
  {
    "setting": "retrospective",
    "family": "output",
    "n": 299,
    "correct": 261,
    "native_K4_correct": 261,
    "net_correct": 0,
    "rescue": 4,
    "damage": 4,
    "mean_replay_loops": 4.0,
    "paired_question95_net_accuracy": [
      -0.020066889632107024,
      0.016722408026755852
    ],
    "scope": "Fixed preregistered class-to-action mapping, no policy/threshold search; supervised diagnostic, not deployable or training-free."
  },
  {
    "setting": "retrospective",
    "family": "combined",
    "n": 299,
    "correct": 260,
    "native_K4_correct": 261,
    "net_correct": -1,
    "rescue": 4,
    "damage": 5,
    "mean_replay_loops": 4.0,
    "paired_question95_net_accuracy": [
      -0.023411371237458192,
      0.016722408026755852
    ],
    "scope": "Fixed preregistered class-to-action mapping, no policy/threshold search; supervised diagnostic, not deployable or training-free."
  },
  {
    "setting": "retrospective",
    "family": "unrelated",
    "n": 299,
    "correct": 261,
    "native_K4_correct": 261,
    "net_correct": 0,
    "rescue": 4,
    "damage": 4,
    "mean_replay_loops": 4.0,
    "paired_question95_net_accuracy": [
      -0.020066889632107024,
      0.016722408026755852
    ],
    "scope": "Fixed preregistered class-to-action mapping, no policy/threshold search; supervised diagnostic, not deployable or training-free."
  },
  {
    "setting": "retrospective",
    "family": "shuffled",
    "n": 299,
    "correct": 261,
    "native_K4_correct": 261,
    "net_correct": 0,
    "rescue": 4,
    "damage": 4,
    "mean_replay_loops": 4.0,
    "paired_question95_net_accuracy": [
      -0.020066889632107024,
      0.016722408026755852
    ],
    "scope": "Fixed preregistered class-to-action mapping, no policy/threshold search; supervised diagnostic, not deployable or training-free."
  },
  {
    "setting": "prospective",
    "family": "output",
    "n": 299,
    "correct": 256,
    "native_K4_correct": 261,
    "net_correct": -5,
    "rescue": 1,
    "damage": 6,
    "mean_replay_loops": 2.5150501672240804,
    "paired_question95_net_accuracy": [
      -0.03678929765886288,
      0.0
    ],
    "scope": "Fixed preregistered class-to-action mapping, no policy/threshold search; supervised diagnostic, not deployable or training-free."
  },
  {
    "setting": "prospective",
    "family": "combined",
    "n": 299,
    "correct": 253,
    "native_K4_correct": 261,
    "net_correct": -8,
    "rescue": 2,
    "damage": 10,
    "mean_replay_loops": 2.488294314381271,
    "paired_question95_net_accuracy": [
      -0.05016722408026756,
      -0.006688963210702341
    ],
    "scope": "Fixed preregistered class-to-action mapping, no policy/threshold search; supervised diagnostic, not deployable or training-free."
  },
  {
    "setting": "prospective",
    "family": "unrelated",
    "n": 299,
    "correct": 258,
    "native_K4_correct": 261,
    "net_correct": -3,
    "rescue": 4,
    "damage": 7,
    "mean_replay_loops": 2.491638795986622,
    "paired_question95_net_accuracy": [
      -0.033444816053511704,
      0.010033444816053512
    ],
    "scope": "Fixed preregistered class-to-action mapping, no policy/threshold search; supervised diagnostic, not deployable or training-free."
  },
  {
    "setting": "prospective",
    "family": "shuffled",
    "n": 299,
    "correct": 254,
    "native_K4_correct": 261,
    "net_correct": -7,
    "rescue": 0,
    "damage": 7,
    "mean_replay_loops": 2.4749163879598663,
    "paired_question95_net_accuracy": [
      -0.043478260869565216,
      -0.006688963210702341
    ],
    "scope": "Fixed preregistered class-to-action mapping, no policy/threshold search; supervised diagnostic, not deployable or training-free."
  }
]

## Interpretation and scope

The 16-dimensional H2 block is candidate-aligned natural lens geometry, replacing the previous16 geometry coordinates. It is never inserted into old index11/index5 gates. d1=None is unavailable, not a zero measurement; all observations use d2+. Original10 LP-output features and original class-balanced ridge/T1/seeds/folds are unchanged. Output-only OOF equality is an explicit technical control. Extra-feature families have26 dimensions; output-only10 is not falsely called equal parameter count. No base-model weights were trained.

E3 Train1030/Validation299 guided earlier mechanism selection and historical Test was repeatedly exposed. Current Train-only group cross-validation and frozen exposed-Val review do not create independent benchmark confirmation. A G1 pass permits consideration of a supervised inference experiment, not a claim that the practical research goal or a deployable accuracy/compute improvement has been achieved. A G1 failure leaves future direction to the separately authorized Astra scientific directive; neither primary metric nor mapping is changed after results.
