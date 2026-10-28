# Reproduce the prioritized1172 +600 delivery

## Environment

CPU only, tested with Python3.10, numpy2.2.6, scipy1.15.3, scikit-learn1.7.2, pandas2.3.3 and matplotlib3.10.7. No model weights, CUDA or SSH access is required to rebuild these results.

From this experiment directory:

```powershell
python -X utf8 -u scripts/analyze_discovery.py
python -X utf8 scripts/render_figures.py
python -X utf8 -m unittest discover -s tests -v
python -X utf8 scripts/recheck_arc1172.py
```

The scripts resolve their own directory and the sibling `looped_stage1_trajectory_audit_seed2021_v1`. Keep the repository layout. Install the exact CPU package versions listed in `results/COMPLETION_AUDIT.json` if necessary. There is no LLM training or new inference.

## Inputs and immutability

Read the sealed historical `data/qwen_test.jsonl`, `results/qwen_A0_test.jsonl`, `results/qwen_A3_test.jsonl`, `scripts/qwen_recipe.py`, `scripts/qwen_audit.py` and `configs/protocol.json`. The original175-file SHA256/length manifest is checked before and after analysis. Every sample is joined by unique ID and checked for identical question, candidate order, gold, revision and scoring protocol. Prediction is rebuilt from argmax of candidate scores, not trusted from the correctness flag.

`data/discovery_features.csv` contains IDs and inference-visible features only. Gold, correctness and g are separated into `discovery_labels.csv`. The extractor has an explicit field allowlist and tests that changing gold/correctness/gold-NLL/oracle fields leaves features identical.

The archived full-vocabulary KL is missing for235/600 questions, due to the historical runner retaining baseline vocabulary vectors only within a process. Missing values are left missing, never filled with measured zeros. This optional feature is analyzed on its available subset, not included in the predeclared20-feature linear model. Choice KL/JS/TV have600/600 coverage. No hidden vectors were archived; only scalar window summaries can be reconstructed.

## Validation

`configs/analysis_plan.json` fixes the primary endpoint,20 linear features, gates, seeds, grid, fold design, statistics and current scope. The primary comparison is nested-CV linear versus Always Euler across all600 questions. Ridge predicts g on all changed training samples, including g=0 both-wrong changes. It never trains on held-out fold samples. Alpha and decision threshold are selected by four-fold inner validation. Median/scaling are fitted separately inside each training fold. No class reweighting is used. Threshold-only gates learn direction and quantile only from development rows. Same-data fitted results are explicitly exploratory.

Five outer folds produce one out-of-fold prediction per question and policy. Leave-task-out fits on two of ARC/OpenBookQA/MMLU and predicts the third. MMLU pools the original five mathematics/logic subjects. Neither form is a fresh independent dataset. `fitted_models_and_selection.json` saves training IDs, preprocessing coefficients, all inner splits, regularization/threshold scores and final parameters.

The17 reported policies comprise Always Original, Always Euler, three simple gates, the linear predictor, Oracle and controls: original confidence, Euler confidence, original-only linear, random feature, shuffled training labels, task prior and four acceptance-count-matched random gates. Oracle is never used in fitting or model selection. Policy actions are retained even for unchanged answers, so Always Euler has acceptance rate1. Matched random controls preserve both overall and changed-answer acceptance counts within each task;10,000 randomizations evaluate the accuracy-relevant changed-answer choices. These are offline action rates, not measured compute savings.

All final accuracy/transition metrics include600 samples. Binary AUC/AUPRC instead compare32 beneficial against15 harmful examples. Their AP prevalence reference is32/47=0.681, not0.5. Univariate raw signs are fixed as “larger feature predicts beneficial”; direction-optimized AUC is separately marked exploratory. No sign is reversed after cross-validation. Pooled OOF scores originate from different fitted models; `foldwise_predictability.csv` exposes each fold separately.

Paired bootstrap uses10,000 question-level resamples, seed2021. Exact two-sided McNemar uses discordant correctness counts. Bootstrap intervals condition on saved policy decisions/OOF scores and omit retraining/model-selection uncertainty. Holm correction covers the60 univariate association tests; gate comparisons beyond the primary endpoint are exploratory. Degenerate zero policy differences can produce[0,0] intervals; that describes this sample's identical decisions and is not proof of population equivalence.

## ARC1172 recheck

`recheck_arc1172.py` independently computes candidate-length-normalized predictions from the sealed full ARC score vectors, joins prompts by ID, reconstructs1172×3 correctness flags and all paired counts/CIs/McNemar values, and compares them with the previously uploaded summary. It imports neither the inference producer nor its original auditor.

## Cost and scope

CPU predictor and feature overhead are measured live. A0/A3 model costs are the historical per-question synchronized timings, summed for a serial two-output implementation. This is not a new latency benchmark or a measured shared-compute optimization. No acceleration claim is made. Rerun timing fields and rendering metadata may change; IDs, features, fitted decisions and statistics should reproduce deterministically with the pinned packages.

No independent new Qwen hold-out was created or evaluated after the user narrowed this turn to1172 +600. Existing extension samples are inventoried as previously exposed, not relabeled as unseen. A future continuation must refresh all exposure manifests and freeze rules before new correctness results are accessed.
