# Complete counterexample and correction sets

|Cohort|natural rescue population|blocked by T|X restores among blocked|A restores among blocked|neither single restores|not discretely blocked|
|---|---|---|---|---|---|---|
|train1030|334|161|143|114|12|173|
|validation299|103|50|39|33|6|53|


Every stage and all four natural groups are retained in results/COUNTEREXAMPLE_IDS.json. Full condition score/prediction/tie examples and exact contrasts are in results/COUNTEREXAMPLE_EXAMPLES.jsonl. Examples use first two lexicographic IDs per explicit category; they do not replace denominators or select a favorable mechanism. Negative exact signs remain for every attention, branch and MLP contrast; complete original rows contain all remaining question evidence.

## Additional descriptive strict checklist

|Cohort|native group|branch|all group n|positive6 exact contrasts and no involved ties|
|---|---|---|---|---|
|train64|rescue|X|16|9|
|train64|rescue|A|16|9|
|train64|damage|X|15|1|
|train64|damage|A|15|1|
|train64|stable_correct|X|15|11|
|train64|stable_correct|A|15|8|
|train64|stable_wrong|X|18|4|
|train64|stable_wrong|A|18|4|
|train1030|rescue|X|334|210|
|train1030|rescue|A|334|211|
|train1030|damage|X|35|7|
|train1030|damage|A|35|5|
|train1030|stable_correct|X|525|382|
|train1030|stable_correct|A|525|263|
|train1030|stable_wrong|X|136|31|
|train1030|stable_wrong|A|136|26|
|validation299|rescue|X|103|55|
|validation299|rescue|A|103|64|
|validation299|damage|X|8|2|
|validation299|damage|A|8|1|
|validation299|stable_correct|X|141|98|
|validation299|stable_correct|A|141|79|
|validation299|stable_wrong|X|47|12|
|validation299|stable_wrong|A|47|7|


This final reporting checklist requires all six already-defined branch contrasts to have positive exact BF16 Fraction signs and all involved endpoints unique top predictions. It is not a preregistered additional test, per-case confidence interval or explanatory percentage, and never filters primary means. Failure to block an answer is not proof of no continuous path effect. Failure of one branch does not show other downstream paths are irrelevant.

Historical C1 case sets and numerical sign reversals remain separately reported. Probe-correct/native-wrong and native-correct/probe-wrong IDs, gold/candidate strata and rare-E cases remain in the complete frozen E2 descriptive outputs.
