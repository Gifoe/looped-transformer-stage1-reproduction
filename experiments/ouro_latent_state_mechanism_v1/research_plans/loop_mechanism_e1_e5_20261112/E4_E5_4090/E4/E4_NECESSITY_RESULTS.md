# E4 necessity results

Final cross-question result: confirmation 128 KV strict natural necessity count is 0 (target damages 3 versus unrelated5;2 natural losses,1 control-specific but tied). Thus299's9 strict cases do not strongly reproduce. Query-only and joint each retain 3 strict cases. The R-state block/restoration is a separate controlled mechanism; see E4_MEDIATION_ANALYSIS.md.

Completed: frozen train 64 local pilot and all 299 Validation questions, with independent row and compact-tensor audits. The generating-layer pathway is not established; the later controlled R-state/attention mediation result is detailed in E4_MEDIATION_ANALYSIS.md. The completed distinct Query-factorial follow-up is reported in E4_QUERY_FACTORIAL_RESULTS.md; this file covers the KV candidate.

On the current RTX4090 Validation baseline, K1/K2 correct counts are 148/243; there are 104 natural rescues and 9 natural damages. Replacing K2 L18 KV with same-question K1 KV while preserving Q2 and the K2 receiver gives 233 correct: 4 rescues and 14 damages relative to natural K2. All 14 damages occur in the natural-rescue group. Mean gold–meanwrong change is −0.328282; dynamic strongest-wrong margin changes −0.364548. Norm-only gives 1 rescue/0 damage; unrelated matched delta gives 7/6.

Of the 14 lost natural corrections, 11 retain correctness under both controls. Ten of those have no native K1/K2 top tie; nine additionally have no target-intervention endpoint tie. These are intervention-defined, control-specific local dependence examples. They are not proof that the entire upstream KV-generating path is necessary, nor a fraction of total reasoning explained. Ninety of 104 natural corrections remain correct under this rollback.

Example without native or target ties: LEAP_2012_8_10441, gold index 2, native K2 raw logits [19.25,16.625,20.0,17.125], rollback [17.875,15.5625,17.625,15.875]. Gold loses its lead under target rollback while both controls remain correct. Full necessity cases are in results/ASTRA_VALIDATION_NECESSITY_CASES.json; full scores, fixed-wrong and dynamic metrics, ties and groups remain in the raw rows and independent audit.

The enriched train 64 pilot differs: only one natural correction is lost, and that case is a native tie also harmed by unrelated control. The full Validation evidence therefore supports local dependence for a subset that the small pilot did not establish. Do not erase either outcome or treat the two differently composed cohorts as directly comparable population estimates.
