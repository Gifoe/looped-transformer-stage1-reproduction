# Locked development intervention study

This follows the eight-question implementation pilot at commit f6d9ec685b88ecd2a6f59ebcb49f075dfc055c7c. The user's clarified priority is an informative intermediate variable that can control continuation or feedback, with independent accuracy/compute superiority still required for algorithm success. This is an exploratory mechanism study on all 400 already-exposed OpenBookQA discovery questions. It is not independent confirmation. Eight pilot outcomes have already been observed; no parameter is changed in response.

## Fixed mechanism

Import the sealed pilot's `choose` and `Intervention` functions without changes. One decision after depth 2; post-normalization h2 is replaced by h2+a*(h2-h1), a in [0,-1,-0.5,+0.5]. Continue the original remaining two loops and use identical Adaptive LoopCD final logits. No backbone training, learned head, new thresholds, sign flip, or architecture search. Input-token group A proposes an action; group B only vetoes it, threshold 0.001 nat/token. Fewer than four probe tokens means baseline. A/B are correlated within a question, not independent truth certificates.

## Comparators and recorded counterfactuals

Evaluate all four fixed actions for every question, before attaching its gold. Record full candidate log-likelihood and character-normalized prediction. The probe controller uses one prompt-only decision shared by all answer candidates.

1. Official four-loop Adaptive (a=0); scores must match sealed discovery within 1e-5 absolute tolerance.
2. Each fixed state action (-1,-0.5,+0.5).
3. A confidence controller: choose the action maximizing full-vocabulary top-two probability margin at the last prompt position after depth 2; ties follow [0,-1,-0.5,+0.5]. This uses the same pre-answer decision point and action set.
4. A terminal-only perturbation control: on the unmodified trajectory, replace only the final readout state with h4+a*(h2-h1), then apply the same Adaptive logit rule. It uses the probe-selected a and the same perturbation vector, but does not propagate it through the recurrence. This is an ablation, not a claimed reproduction of another method.
5. Shuffled probe decisions: 10,000 fixed-seed permutations of the selected action assignments among questions with sufficient probes. Exact action counts, nonzero-action rate and probe availability are preserved. Scores are evaluated from recorded counterfactuals. This tests selection information; it is also a matched-action-frequency random baseline.
6. Oracle selection from the four state actions, explicitly an unattainable upper bound.

The intervention worker records 4-depth state differences at the last prompt position, input-probe action responses and per-question phase timing. The zero-action pass also computes the terminal-only controls; this diagnostic overhead is separately timed. Controller deployment cost is estimated from prompt decision time plus the selected state branch time, with terminal diagnostic time removed from a=0. It is an indicative serial replay estimate, not an optimized serving benchmark. Whole experimental worker time counts against GPU budget.

## Locked analysis

Primary mechanism report: actual paired corrections/degradations versus Adaptive; actual selected actions; whether beneficial/harmful actions are distinguishable. For each nonzero action, define its proxy value as min(L_A(0)-L_A(a), L_B(0)-L_B(a)); evaluate AUC only on nonzero accuracy-change events, report event counts and stratified bootstrap CI. No sign reversal. Spearman association with normalized gold candidate log-weight change is secondary and is not accuracy evidence.

Use 10,000 paired empirical bootstrap resamples and exact two-sided McNemar tests. Apply Holm to six comparisons of the probe controller versus Adaptive, three fixed state actions, confidence controller and selected terminal-only control. Report the permutation distribution and upper-tail p=(1+#random_correct>=observed_correct)/(10001). All inference remains exploratory because discovery data are exposed. Report full 400 and, as a predefined sensitivity check, exclude the first eight implementation-pilot questions; never select between these summaries.

Evidence sufficient to justify a new confirmation protocol requires positive lower 95% paired CI and Holm p<0.05 against all six comparators, plus permutation p<0.05. AUC evidence alone may justify a separately bounded supervised value-head study on true state-intervention targets; it cannot establish a working controller. No test of these new policies on ARC1172, Qwen600, SciQ1000 or PIQA1802 is allowed. A failed or inconclusive development result must be reported as such without threshold tuning.

## Scope and budget

Exactly 400 question IDs from the original discovery manifest. No new external data acquisition. One worker with a hard 3,600-second cap and explicit process receipt, enough below the remaining 22,479-second original research allowance. Preserve all historical and pilot files. No early stop or compute-saving claim: rolling back an executed update does not recover its cost. The pilot and this directory remain distinct.
