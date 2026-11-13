# Train-only failure diagnosis

All analysis and code: current Sol. Astra supplies only the separately frozen directive. No GPU forwards; no Validation/Test examples loaded by the fit process. Existing exposed Train is not independent benchmark confirmation. Exact inputs share a fold; the known Train near-duplicate pair is exact and grouped. All depths of each question stay together.

## Cache limitation

Natural attention increments are stored per layer, but residual x is averaged across seven layers. The per-layer final-normalization lens in the directive is therefore NOT COMPUTABLE from the current cache. Mean-x is not substituted. This analysis does not evaluate H2 or imply its failure.

## Train action upper bounds (offline gold only)

{
  "n": 1094,
  "native_correct": [
    583,
    889,
    958,
    966
  ],
  "oracle_any_depth_correct": 1014,
  "K4_wrong_previous_correct": 48,
  "K4_correct_K2_wrong": 95,
  "all_native_wrong": 80,
  "oracle_is_not_a_method": true
}

## Conditional information

[
  {
    "setting": "retrospective",
    "family": "combined",
    "n_questions": 1094,
    "Brier_improvement_vs_output": 0.0011373612336865022,
    "paired_question_bootstrap95": [
      0.00011171127264590163,
      0.0021674920227177145
    ]
  },
  {
    "setting": "retrospective",
    "family": "unrelated",
    "n_questions": 1094,
    "Brier_improvement_vs_output": 0.0018524293431156451,
    "paired_question_bootstrap95": [
      0.0008741324817027003,
      0.0028499425036258375
    ]
  },
  {
    "setting": "retrospective",
    "family": "shuffled",
    "n_questions": 1094,
    "Brier_improvement_vs_output": 0.0011411155308947503,
    "paired_question_bootstrap95": [
      0.0003439808240650327,
      0.0019429674761409837
    ]
  },
  {
    "setting": "prospective",
    "family": "combined",
    "n_questions": 1094,
    "Brier_improvement_vs_output": -0.00046385642410705645,
    "paired_question_bootstrap95": [
      -0.0016894668588922504,
      0.0007709061051015014
    ]
  },
  {
    "setting": "prospective",
    "family": "unrelated",
    "n_questions": 1094,
    "Brier_improvement_vs_output": 0.0010428132900108715,
    "paired_question_bootstrap95": [
      -0.0002695269794792941,
      0.002320936383884076
    ]
  },
  {
    "setting": "prospective",
    "family": "shuffled",
    "n_questions": 1094,
    "Brier_improvement_vs_output": -0.0002533532494186621,
    "paired_question_bootstrap95": [
      -0.0016624310262906995,
      0.001185323255902974
    ]
  }
]

A positive Brier improvement means adding the feature block reduces proper predictive loss. Intervals resample whole questions, with every depth together. Inner/outer fits never share a question group; alpha selection uses inner Brier only. Same ridge head, same3 alpha choices, fixed temperature1; input dimensions are10 versus26, with same-dimension unrelated/shuffle controls. Probability maps are uncalibrated ridge-softmax; logloss and ECE are retained. Correlated CV training and historical mechanism selection limit generalization claims. No controller is tuned from these OOF outcomes.

Fixed supervised action mappings and all OOF predictions are included, but they do not establish a deployable early-exit algorithm or latency benefit. The H2 family remains a separately preregistered hypothesis requiring true per-layer x/a and native endpoint validation; E6C remains NO-GO.

## Sol interpretation of H0/H1

The natural selector oracle leaves48 recoverable K4 errors in1094 Train questions, so lack of headroom alone does not explain failure. Existing geometry gives only a small retrospective proper-loss change; unrelated-layer and shuffled additions also improve the baseline, and prospective combined does not improve. Direct same-question latent-versus-control differences are retained below; no mechanism-specific information gain is established merely because one baseline interval is positive. Retrospective diagnostic action correctness is959 for both output and combined, belowK4=966; prospective combined943 is below output948. This does not validate hidden-state control.

[
  {
    "setting": "retrospective",
    "comparison": "combined relative output",
    "n_questions": 1094,
    "Brier_improvement": 0.0011373612336865022,
    "paired_question95": [
      0.0001117112726459004,
      0.002167492022717715
    ]
  },
  {
    "setting": "retrospective",
    "comparison": "combined relative unrelated",
    "n_questions": 1094,
    "Brier_improvement": -0.0007150681094291428,
    "paired_question95": [
      -0.001909726830759199,
      0.00046854359446347815
    ]
  },
  {
    "setting": "retrospective",
    "comparison": "combined relative shuffled",
    "n_questions": 1094,
    "Brier_improvement": -3.754297208246317e-06,
    "paired_question95": [
      -0.0012697657652073569,
      0.0012699501883463864
    ]
  },
  {
    "setting": "prospective",
    "comparison": "combined relative output",
    "n_questions": 1094,
    "Brier_improvement": -0.00046385642410705493,
    "paired_question95": [
      -0.0016894668588922491,
      0.0007709061051015034
    ]
  },
  {
    "setting": "prospective",
    "comparison": "combined relative unrelated",
    "n_questions": 1094,
    "Brier_improvement": -0.0015066697141179293,
    "paired_question95": [
      -0.003088613714594145,
      6.559449574672042e-05
    ]
  },
  {
    "setting": "prospective",
    "comparison": "combined relative shuffled",
    "n_questions": 1094,
    "Brier_improvement": -0.00021050317468839468,
    "paired_question95": [
      -0.002092585687018452,
      0.001642079540261558
    ]
  }
]

H2 is NOT RUN. It requires per-layer pre-attention residual states not retained by E6B; current cache contains only layer-averaged x. The next permitted experiment is the single frozen candidate-direction lens family under the director technical and30-GPU-minute budget gates. Do not substitute mean-x, tune old geometry thresholds, or use the current Test for feature selection. E6C remains NO-GO.
