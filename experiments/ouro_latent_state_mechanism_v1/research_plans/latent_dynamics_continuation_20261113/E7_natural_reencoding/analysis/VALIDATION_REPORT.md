# E7 validation frozen natural-reencoding analysis

Scientific direction: Astra. All CPU implementation, analysis and review: Sol. Natural inputs only; no latent intervention, new model fit or Test data. Historically exposed ARC/Val is not independent benchmark confirmation.

## Main four-choice K1→K2 gain contrasts

P/S nonzerooffsets average within each question before question averaging; all16 cells retained. Estimates and intervals are percentage points. Primary support uses three-family Bonferroni98.333333% interval; ordinary95% retained.

| Effect | Estimate pp | 95% | 98.333333% |
|---|---:|---|---|
| P | +0.4237 | [-2.9096045197740112, 3.7860169491525317] | [-3.634887005649717, 4.491525423728815] |
| S | +0.0847 | [-2.740112994350283, 2.8531073446327664] | [-3.446327683615819, 3.4180790960451963] |
| interaction | -3.6535 | [-10.395480225988697, 3.1261770244821103] | [-12.015065913370996, 4.808851224105502] |

All secondary endpoints, both tie rules, alloffset heterogeneity and separate3/5-choice extensions are retained in machine-readable tables. No-related-tie subset excludes an entire question if any K1/K2 top tie occurs anywhere in its grid; its coverage is explicit. Nonsignificance is not invariance: practical equivalence requires every relevant primary interval within±1pp. A consistent semantic prediction can be consistently wrong; perdepth allcorrect/allsamewrong/mixedcorrectness and exact top-set/Jaccard summaries are separate. Full-view correctness oracle is offline gold only. The p=s query-symbol-preserving diagonal still changes demo symbols, so it is not a pure whole-prompt position-only contrast.

## Unique fixed pair versus frozen controls

| Choices | Comparator | Pair correct/N | Net | Gain95% pp |
|---|---|---:|---:|---|
| 3 | identity_K4 | 2/3 | -1 | [-100.0, 0.0] |
| 3 | E6_primary_ZL_weighted4 | 2/3 | +0 | [0.0, 0.0] |
| 3 | E6_primary_Cal_loopcd_adaptive_cap1 | 2/3 | -1 | [-100.0, 0.0] |
| 3 | strong_output_cal_early_0.6 | 2/3 | -1 | [-100.0, 0.0] |
| 4 | identity_K4 | 244/295 | -13 | [-7.796610169491526, -1.0169491525423728] |
| 4 | E6_primary_ZL_weighted4 | 244/295 | -12 | [-7.118644067796611, -1.0169491525423728] |
| 4 | E6_primary_Cal_loopcd_adaptive_cap1 | 244/295 | -15 | [-8.813559322033898, -1.694915254237288] |
| 4 | strong_output_cal_early_0.6 | 244/295 | -12 | [-7.457627118644068, -0.6779661016949152] |
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
      "correct": 261,
      "n": 299,
      "accuracy": 0.8729096989966555,
      "mean_estimated_prefix_depth": 2.8461538461538463,
      "offline_actual_full_body_depth": 4,
      "source": "Actual BF16 R gate PDF/CDF after full four-loop capture; technical stock-default full-sequence parity validated separately",
      "dominated": false,
      "exact_existing_point": false,
      "descriptive_nondominated_Val_only": true,
      "online_eligibility_decided_on_Train_only": true
    },
    {
      "threshold": 0.5,
      "correct": 261,
      "n": 299,
      "accuracy": 0.8729096989966555,
      "mean_estimated_prefix_depth": 3.488294314381271,
      "offline_actual_full_body_depth": 4,
      "source": "Actual BF16 R gate PDF/CDF after full four-loop capture; technical stock-default full-sequence parity validated separately",
      "dominated": false,
      "exact_existing_point": false,
      "descriptive_nondominated_Val_only": true,
      "online_eligibility_decided_on_Train_only": true
    },
    {
      "threshold": 0.75,
      "correct": 261,
      "n": 299,
      "accuracy": 0.8729096989966555,
      "mean_estimated_prefix_depth": 4.0,
      "offline_actual_full_body_depth": 4,
      "source": "Actual BF16 R gate PDF/CDF after full four-loop capture; technical stock-default full-sequence parity validated separately",
      "dominated": true,
      "exact_existing_point": false,
      "descriptive_nondominated_Val_only": false,
      "online_eligibility_decided_on_Train_only": true
    },
    {
      "threshold": 1.0,
      "correct": 261,
      "n": 299,
      "accuracy": 0.8729096989966555,
      "mean_estimated_prefix_depth": 4.0,
      "offline_actual_full_body_depth": 4,
      "source": "Actual BF16 R gate PDF/CDF after full four-loop capture; technical stock-default full-sequence parity validated separately",
      "dominated": true,
      "exact_existing_point": false,
      "descriptive_nondominated_Val_only": false,
      "online_eligibility_decided_on_Train_only": true
    }
  ],
  "stock_default": {
    "threshold": 1.0,
    "correct": 261,
    "n": 299,
    "accuracy": 0.8729096989966555,
    "mean_estimated_prefix_depth": 4.0,
    "offline_actual_full_body_depth": 4,
    "source": "Actual BF16 R gate PDF/CDF after full four-loop capture; technical stock-default full-sequence parity validated separately",
    "dominated": true,
    "exact_existing_point": false,
    "descriptive_nondominated_Val_only": false,
    "online_eligibility_decided_on_Train_only": true,
    "name": "stock_default_threshold1_full_body",
    "scope": "R readout corresponds to per-token public default at R; all4 body passes executed; not sample-level runtime halting"
  },
  "existing_output_controls": [
    {
      "name": "K1",
      "correct": 149,
      "n": 299,
      "mean_estimated_prefix_depth": 1.0,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "K2",
      "correct": 243,
      "n": 299,
      "mean_estimated_prefix_depth": 2.0,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "K3",
      "correct": 259,
      "n": 299,
      "mean_estimated_prefix_depth": 3.0,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "K4",
      "correct": 261,
      "n": 299,
      "mean_estimated_prefix_depth": 4.0,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "early_fixed",
      "correct": 260,
      "n": 299,
      "mean_estimated_prefix_depth": 3.1270903010033444,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "cal_early_0.6",
      "correct": 260,
      "n": 299,
      "mean_estimated_prefix_depth": 2.8762541806020065,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "cal_early_0.7",
      "correct": 260,
      "n": 299,
      "mean_estimated_prefix_depth": 2.882943143812709,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "cal_early_0.8",
      "correct": 260,
      "n": 299,
      "mean_estimated_prefix_depth": 2.9096989966555182,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "cal_early_0.9",
      "correct": 260,
      "n": 299,
      "mean_estimated_prefix_depth": 3.1237458193979935,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "cal_early_0.95",
      "correct": 260,
      "n": 299,
      "mean_estimated_prefix_depth": 3.311036789297659,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "cal_early_0.99",
      "correct": 261,
      "n": 299,
      "mean_estimated_prefix_depth": 3.7123745819397995,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    }
  ],
  "existing_output_frontier": [
    {
      "name": "K1",
      "correct": 149,
      "n": 299,
      "mean_estimated_prefix_depth": 1.0,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "K2",
      "correct": 243,
      "n": 299,
      "mean_estimated_prefix_depth": 2.0,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "cal_early_0.6",
      "correct": 260,
      "n": 299,
      "mean_estimated_prefix_depth": 2.8762541806020065,
      "source": "Existing E6 native1..4/6calearly/.9entropy.3early; no new parameter grid"
    },
    {
      "name": "cal_early_0.99",
      "correct": 261,
      "n": 299,
      "mean_estimated_prefix_depth": 3.7123745819397995,
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
