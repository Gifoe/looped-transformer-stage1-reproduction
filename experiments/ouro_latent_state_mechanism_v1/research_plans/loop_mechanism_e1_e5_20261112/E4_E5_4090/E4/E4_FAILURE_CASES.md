# E4 counterexamples and limitations

Final closed review: all fixed 299/128 experiments are complete. Confirmation failures below limit the mechanism claim; earlier case records remain intact.

Complete machine-readable counterexamples, all candidate values and selection reasons are retained in results/ASTRA_PILOT64_REVIEW_INDEPENDENT.json and results/ASTRA_VALIDATION299_REVIEW_INDEPENDENT.json. Examples below illustrate specific boundaries rather than a selected success narrative.

- CSZ_2008_5_CSZ10022: the pilot's only lost native rescue is a native top tie, also damaged by unrelated control, with historical/current K2 prediction drift. It is not clean target-specific necessity evidence.
- Mercury_7032340: Validation's sole natural-rescue question satisfying both forward restoration and reverse damage has a tied rollback endpoint. It does not support a tie-robust necessary-and-sufficient claim.
- MDSA_2010_5_18, NCEOGA_2013_8_4 and Mercury_192168: target rollback damages a native rescue, but unrelated control also damages it; these are not target-exclusive losses.
- Mercury_184293: FP32 SDPA followed by BF16 writeback changes the K2→K3 restore prediction in the frozen sensitivity subset. This is numerical dependence under the stated scope, not a full-model FP32 comparison.
- Natural K2 KV rollback rescues four Validation questions while damaging fourteen. A negative mean gap cannot replace reporting those opposite-direction decisions.
- Native K3→K4 corrects seven Validation questions; the matched late KV rollback damages none. This local candidate does not account for all later corrections.

Historical/current native predictions differ for seven pilot IDs and eleven Validation IDs. New effects and groups use the same-run current baseline; historical differences are preserved. Pilot unrelated donors follow the actual pre-frozen selection-hash-order cycle, yielding 32/64 same historical-group donors. The formal frozen map was followed, but the initial draft description of ID sorting was inaccurate and the donor distribution is not independent of sampling strata. Validation uses a separately frozen ID-sorted map. These records are retained without retroactive donor replacement.

Output-norm controls do not eliminate Query/Key softmax-temperature mechanisms. Cross-depth interventions also shrink in norm, so smaller later effects cannot be assigned solely to reduced receiver sensitivity.


The completed R/attention four corners add boundaries beyond the original KV candidate. ACTAAP_2007_7_25 loses the correction under R-state rollback and remains wrong after exact native-attention restoration, although both state controls retain correctness. NCEOGA_2013_8_4 remains correct under the state block but becomes wrong only when native attention is restored. Such new damage prevents describing restoration as a universally beneficial repair. Seven natural corrections lost under the block remain wrong after restoration; one new natural correction is damaged. Raw values and all 299 classifications are in ASTRA_MEDIATION_DECISION_CASES.json.

At the same time,15 natural corrections meet tie-free mediator-restoration specificity and 11 also survive both input-state controls. These positive cases must not be suppressed by a narrative based only on the large remaining mean gap. Continuous and decision outcomes address different questions. The pilot had no discrete recovery uniquely better than both mediator controls; full Validation changed the evidence and is reported as such.


## Fixed128 and late-depth counterexamples

- KV target rollback damages 3 versus unrelated5. Of2 lost natural corrections,only 1 survives both controls and that case has a relevant tie;strict KV necessity count0. The299 nine strict cases are not a strong replicated decision result.
- ConfirmationK2 U/restored-U/native correct counts 98/105/103 coexist with meanwrong effects−1.394368/−1.353923/0. A large continuous loss does not imply a comparable accuracy loss. Restoration meanwrong+0.040446 has question-bootstrap interval[-0.0160,+0.0970],so its population-average positivity is uncertain.
- Two clean R2 conditional recoveries remain:Mercury_SC_400986 andTIMSS_2011_8_pg100. Preserve these alongside the failures;they meet the predefined no-tie and all-four-control criterion.
- In299,late natural rescues 20/7 lose9/5 under state rollback and recover 4/1;strict four-control counts 2/1. In128,late rescues 9/1 lose2/1 and recover 0/0. The same fixed test does not demonstrate late recovery in the cross-question cohort.
- Both cohorts have negative K3/K4 average native-attention restoration effects. Smaller positive early behavior is not an adequate description of late computation.
- Confirmation native69→103→110→107 includes a K4 net loss 3 (1 rescue,4 damages). Accuracy need not improve monotonically with depth.
- Peer coarse producer cuts have unequal target-state perturbations and no same-cut matched norm/unrelated controls. Their real effects do not identify a writer, hidden-state regeneration or layer importance.

Full5152 confirmation endpoint rows and 208 four-metric/group summaries are retained in ROOT_CONFIRMATION128_ALL_METRICS.json. All R damaged/restored/persistent cases are retained in the R2/R3/R4 audits;KV/Query strict classifications are in ROOT_CONFIRMATION128_DECISION_CASES.json. No adverse result triggered operator replacement or a new scan.
