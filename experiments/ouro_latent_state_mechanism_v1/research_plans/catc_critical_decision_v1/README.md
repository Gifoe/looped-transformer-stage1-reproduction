# CATC critical decision study

Principal recommendation: **D. Stop the tested CATC route.** All three authorized experiments are complete. Sol performed all code, execution, analysis and review; the requested ASTRA_FINAL_DECISION filename is retained.

Frozen development selection chose routing -0.1 at K2 R, physical layers 10-16. Corrected confirmation: CATC/native/scaling 446/511; Gaussian random control 449/511. Both accuracy and CE-specificity gates fail. Remaining Train: CATC/native 291/326 versus 290/326. Exposed Validation: both 261/299, below the E6 conservative and LoopCD1 counts of 263.

An ID-only split initially missed one identical tokenized prompt crossing development and confirmation. The input-hash audit found it after initial aggregates were read. The duplicate confirmation item was excluded from the corrected 511 inference population; original512 data and statistics remain visible. Selection and coefficients were not changed. This is a retrospective protocol repair, not blind independent confirmation. See COHORT_DUPLICATE_REPAIR.json.

Read DIRECTIONAL_RESPONSE.md, MATCHED_CONTROL_RESULTS.md, FINAL_K4_COMPARISON.md, EXECUTION_AND_NUMERICAL_AUDIT.md and ASTRA_FINAL_DECISION.md. Full question-level logits, losses, norms, ties and output baselines are in QUESTION_LEVEL_RESULTS.jsonl.gz. Archives may be losslessly sharded; the replay assembles them automatically.

CPU replay (Python 3.10, NumPy 1.26.4):

```
python code/reproduce_release.py --release .
```

This recomputes all paired summaries and dose audits exactly, verifies input hashes, candidate scores/predictions/ties, correction bounds and frozen development selection. It does not rerun GPU forwards. GPU scripts preserve the external checkpoint/source identity and require adapting paths to a fresh output root before rerunning. Original weights and previous experiments are preserved.

The completed parent CATC pilot is commit 17c957cbca8b3a883ff7e4ca5d3b97200c3aa9ff. The critical study reused its same-device full 25-shot captures instead of repeating them. No Test, model training, LoRA, new Gate, head/layer/coefficient search, or conditional-controller fitting was performed. Original/symmetric decomposition and fresh full-vocabulary numerical gates are retained.
