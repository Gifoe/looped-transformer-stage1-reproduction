# E7 train frozen natural-reencoding analysis

Scientific direction: Astra. All CPU implementation, analysis and review: Sol. Natural inputs only; no latent intervention, new model fit or Test data. Historically exposed ARC/Val is not independent benchmark confirmation.

## Main four-choice K1→K2 gain contrasts

P/S nonzerooffsets average within each question before question averaging; all16 cells retained. Estimates and intervals are percentage points. Primary support uses three-family Bonferroni98.333333% interval; ordinary95% retained.

| Effect | Estimate pp | 95% | 98.333333% |
|---|---:|---|---|
| P | +1.0302 | [-0.9081196581196573, 2.9382249694749656] | [-1.3507326007326008, 3.31959706959707] |
| S | +2.0986 | [0.7020757020757016, 3.4645909645909696] | [0.3764117826617826, 3.7774725274725314] |
| interaction | -4.3244 | [-7.885632885632877, -0.7529507529507538] | [-8.662494912494907, -0.040700040700042156] |

All secondary endpoints, both tie rules, alloffset heterogeneity and separate3/5-choice extensions are retained in machine-readable tables. No-related-tie subset excludes an entire question if any K1/K2 top tie occurs anywhere in its grid; its coverage is explicit. Nonsignificance is not invariance: practical equivalence requires every relevant primary interval within±1pp. A consistent semantic prediction can be consistently wrong; perdepth allcorrect/allsamewrong/mixedcorrectness and exact top-set/Jaccard summaries are separate. Full-view correctness oracle is offline gold only. The p=s query-symbol-preserving diagonal still changes demo symbols, so it is not a pure whole-prompt position-only contrast.

## Unique fixed pair versus frozen controls

| Choices | Comparator | Pair correct/N | Net | Gain95% pp |
|---|---|---:|---:|---|
| 3 | identity_K4 | 1/1 | +0 | [0.0, 0.0] |
| 3 | E6_primary_ZL_weighted4 | 1/1 | +0 | [0.0, 0.0] |
| 3 | E6_primary_Cal_loopcd_adaptive_cap1 | 1/1 | +0 | [0.0, 0.0] |
| 3 | strong_output_cal_early_0.6 | 1/1 | +0 | [0.0, 0.0] |
| 4 | identity_K4 | 936/1092 | -28 | [-4.395604395604396, -0.7326007326007326] |
| 4 | E6_primary_ZL_weighted4 | 936/1092 | -21 | [-3.571428571428571, -0.27472527472527475] |
| 4 | E6_primary_Cal_loopcd_adaptive_cap1 | 936/1092 | -31 | [-4.761904761904762, -0.9157509157509158] |
| 4 | strong_output_cal_early_0.6 | 936/1092 | -27 | [-4.304029304029304, -0.7326007326007326] |
| 5 | identity_K4 | 1/1 | +0 | [0.0, 0.0] |
| 5 | E6_primary_ZL_weighted4 | 1/1 | +0 | [0.0, 0.0] |
| 5 | E6_primary_Cal_loopcd_adaptive_cap1 | 1/1 | +0 | [0.0, 0.0] |
| 5 | strong_output_cal_early_0.6 | 1/1 | +0 | [0.0, 0.0] |

Two K2 views nominally total4 loop passes but execute two prompts/forwards/heads. Token-loop units and full method records are retained; no equality of real latency is inferred. No alternative pair, offset, weight or oracle is selected. Train fourchoice profile permission is separately frozen before exposedVal outcomes are read.

## Existing gate baselines

All4 fixed thresholds plus stock-default threshold1 are full-body offline readouts and cost4 actual body passes. Their estimatedprefix depth is not actual halting work or measured speed. Native1..4,6 oldcalearly and original .9/entropy.3 early output-only controls define the existing estimatedbudget frontier. Threshold selection uses only identity allTrain1094; no threshold is reselected for a subset or Validation.

{
  "selection_population": "All original identity Train1094 for Train selection; all exposedVal299 only frozen review",
  "all_fixed_thresholds": [
    {
      "threshold": 0.25,
      "correct": 962,
      "n": 1094,
      "accuracy": 0.8793418647166362,
      "mean_estimated_prefix_depth": 2.790676416819013,
      "offline_actual_full_body_depth": 4,
      "source": "Actual BF16 R gate PDF/CDF after full four-loop capture; technical stock-default full-sequence parity validated separately",
      "frontier_eligible": true,
      "dominated": false,
      "exact_existing_point": false
    },
    {
      "threshold": 0.5,
      "correct": 966,
      "n": 1094,
      "accuracy": 0.8829981718464351,
      "mean_estimated_prefix_depth": 3.4661791590493602,
      "offline_actual_full_body_depth": 4,
      "source": "Actual BF16 R gate PDF/CDF after full four-loop capture; technical stock-default full-sequence parity validated separately",
      "frontier_eligible": false,
      "dominated": true,
      "exact_existing_point": false
    },
    {
      "threshold": 0.75,
      "correct": 966,
      "n": 1094,
      "accuracy": 0.8829981718464351,
      "mean_estimated_prefix_depth": 4.0,
      "offline_actual_full_body_depth": 4,
      "source": "Actual BF16 R gate PDF/CDF after full four-loop capture; technical stock-default full-sequence parity validated separately",
      "frontier_eligible": false,
      "dominated": true,
      "exact_existing_point": false
    },
    {
      "threshold": 1.0,
      "correct": 966,
      "n": 1094,
      "accuracy": 0.8829981718464351,
      "mean_estimated_prefix_depth": 4.0,
      "offline_actual_full_body_depth": 4,
      "source": "Actual BF16 R gate PDF/CDF after full four-loop capture; technical stock-default full-sequence parity validated separately",
      "frontier_eligible": false,
      "dominated": true,
      "exact_existing_point": false
    }
  ],
  "stock_default": {
    "threshold": 1.0,
    "correct": 966,
    "n": 1094,
    "accuracy": 0.8829981718464351,
    "mean_estimated_prefix_depth": 4.0,
    "offline_actual_full_body_depth": 4,
    "source": "Actual BF16 R gate PDF/CDF after full four-loop capture; technical stock-default full-sequence parity validated separately",
    "frontier_eligible": false,
    "dominated": true,
    "exact_existing_point": false,
    "name": "stock_default_threshold1_full_body",
    "scope": "R readout corresponds to per-token public default at R; all4 body passes executed; not sample-level runtime halting"
  },
  "existing_output_controls": [
    {
      "name": "K1",
      "correct": 583,
      "n": 1094,
      "mean_estimated_prefix_depth": 1.0,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "K2",
      "correct": 889,
      "n": 1094,
      "mean_estimated_prefix_depth": 2.0,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "K3",
      "correct": 958,
      "n": 1094,
      "mean_estimated_prefix_depth": 3.0,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "K4",
      "correct": 966,
      "n": 1094,
      "mean_estimated_prefix_depth": 4.0,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "early_fixed",
      "correct": 968,
      "n": 1094,
      "mean_estimated_prefix_depth": 3.046617915904936,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "cal_early_0.6",
      "correct": 965,
      "n": 1094,
      "mean_estimated_prefix_depth": 2.827239488117002,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "cal_early_0.7",
      "correct": 965,
      "n": 1094,
      "mean_estimated_prefix_depth": 2.829981718464351,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "cal_early_0.8",
      "correct": 967,
      "n": 1094,
      "mean_estimated_prefix_depth": 2.8692870201096894,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "cal_early_0.9",
      "correct": 968,
      "n": 1094,
      "mean_estimated_prefix_depth": 3.0402193784277878,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "cal_early_0.95",
      "correct": 966,
      "n": 1094,
      "mean_estimated_prefix_depth": 3.206581352833638,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "cal_early_0.99",
      "correct": 966,
      "n": 1094,
      "mean_estimated_prefix_depth": 3.63436928702011,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    }
  ],
  "existing_output_frontier": [
    {
      "name": "K1",
      "correct": 583,
      "n": 1094,
      "mean_estimated_prefix_depth": 1.0,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "K2",
      "correct": 889,
      "n": 1094,
      "mean_estimated_prefix_depth": 2.0,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "cal_early_0.6",
      "correct": 965,
      "n": 1094,
      "mean_estimated_prefix_depth": 2.827239488117002,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "cal_early_0.8",
      "correct": 967,
      "n": 1094,
      "mean_estimated_prefix_depth": 2.8692870201096894,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "cal_early_0.9",
      "correct": 968,
      "n": 1094,
      "mean_estimated_prefix_depth": 3.0402193784277878,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    }
  ],
  "selected_threshold": 0.25,
  "selection": "Train only: nondominated and not an exact existing point; correct desc, prefixdepth asc, numericthreshold asc. Validation never reselects.",
  "actual_online_latency": "NOT_RUN"
}

## Scope and reproducibility

CPUsource freeze occurred before newvariant scientific outcome/label comparisons were read, after technical parity outputs already existed; the scientific definitions and unique pair were frozen earlier in the Astra order/E7protocol. No claim is made that the implementation hash predates every model outcome. Source/input/isolatedlabel/helper hashes, canonicalvariant checksums, allcoverage and native bridges are recorded. Each question remains one bootstrap block across alloffsets/depths. Ordinary and adjusted intervals describe empirical question resampling, not training seeds or fresh-dataset generalization.

Gold is used only for outcome/risk/gap summaries. The dynamic wrong margin is correctsemantic logit minus current strongestwrong, while top1−top2 is retained separately. IdentityK1 strongestwrong tracks the same originalcandidate across allviews; receiverK1 strongestwrong is fixed within thatview. These definitions never enter the inference candidate.

The practical research goal remains open until a frozen method improves a real measured accuracy–cost frontier and survives independent-data confirmation. A causal input-response effect cannot by itself identify a specific writer, pure label bias, or method success. P changes option causal context and S changes demo/query symbol tokens. H2 G1 remains failed and no E6C is authorized.
