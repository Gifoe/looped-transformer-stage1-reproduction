# True recurrent boundary and state transition

|RTX4090 cohort|n|K1 correct|K2 correct|rescue|damage|both correct|both wrong|
|---|---|---|---|---|---|---|---|
|train64|64|30|31|16|15|15|18|
|train1030|1030|560|859|334|35|525|136|
|validation299|299|149|244|103|8|141|47|


The actual normalized full sequence after K1 physical L24 becomes K2 L1 input. E is actual first-loop embedding input, B actual end-of-K1 RMSNorm output. P is fixed25-shot demonstrations, Q the current question/options/format except final R, R the final readout row. AB1 holds P=B_P: H00=(E_Q,E_R), H10=(B_Q,E_R), H01=(E_Q,B_R), H11=(B_Q,B_R)=N. E_all to native K1 and H11 to native K2 were verified on the4090. H00 is not a K1 replay.

Four edges: H10-H00(Q_R_E), H11-H01(Q_R_B), H01-H00(R_Q_E), H11-H10(R_Q_B). Each has receiver-specific norm-matched boundary donor and fixed Gaussian displacement controls; two Q9 crosses give14 endpoints. All per-question Q/R interactions remain. H11-H00 and H00-K1 must be separated: hybrid reset combines geometry, scale and context effects and is not a natural loop transition.

|Cohort|AB1 Q route|AB1 R route|AB1 Q9 path|
|---|---|---|---|
|train1030|False|False|False|
|validation299|False|False|False|


Strong gate failure does not imply zero effect. Raw positive Q9 restoration/rollback is not a complete specific boundary route when boundary controls fail; the Q9 crosses also lack dedicated Q9 direction-matched nulls. B2 tests a narrower induced-update path without relabeling the failed AB1 specificity criterion.

|Cohort|natural group|original pair|earliest differing SAVED R site|questions|
|---|---|---|---|---|
|train1030|damage|Q9_restoration|L10_a_R|35|
|train1030|damage|Q9_transfer_loss|L10_a_R|35|
|train1030|damage|Q_R_B|L9_R|35|
|train1030|damage|Q_R_E|L9_R|35|
|train1030|damage|R_Q_B|L1_R|35|
|train1030|damage|R_Q_E|L1_R|35|
|train1030|rescue|Q9_restoration|L10_a_R|334|
|train1030|rescue|Q9_transfer_loss|L10_a_R|334|
|train1030|rescue|Q_R_B|L9_R|334|
|train1030|rescue|Q_R_E|L9_R|334|
|train1030|rescue|R_Q_B|L1_R|334|
|train1030|rescue|R_Q_E|L1_R|334|
|train1030|stable_correct|Q9_restoration|L10_a_R|525|
|train1030|stable_correct|Q9_transfer_loss|L10_a_R|525|
|train1030|stable_correct|Q_R_B|L9_R|525|
|train1030|stable_correct|Q_R_E|L9_R|525|
|train1030|stable_correct|R_Q_B|L1_R|525|
|train1030|stable_correct|R_Q_E|L1_R|525|
|train1030|stable_wrong|Q9_restoration|L10_a_R|136|
|train1030|stable_wrong|Q9_transfer_loss|L10_a_R|136|
|train1030|stable_wrong|Q_R_B|L9_R|136|
|train1030|stable_wrong|Q_R_E|L9_R|136|
|train1030|stable_wrong|R_Q_B|L1_R|136|
|train1030|stable_wrong|R_Q_E|L1_R|136|
|validation299|damage|Q9_restoration|L10_a_R|8|
|validation299|damage|Q9_transfer_loss|L10_a_R|8|
|validation299|damage|Q_R_B|L9_R|8|
|validation299|damage|Q_R_E|L9_R|8|
|validation299|damage|R_Q_B|L1_R|8|
|validation299|damage|R_Q_E|L1_R|8|
|validation299|rescue|Q9_restoration|L10_a_R|103|
|validation299|rescue|Q9_transfer_loss|L10_a_R|103|
|validation299|rescue|Q_R_B|L9_R|103|
|validation299|rescue|Q_R_E|L9_R|103|
|validation299|rescue|R_Q_B|L1_R|103|
|validation299|rescue|R_Q_E|L1_R|103|
|validation299|stable_correct|Q9_restoration|L10_a_R|141|
|validation299|stable_correct|Q9_transfer_loss|L10_a_R|141|
|validation299|stable_correct|Q_R_B|L9_R|141|
|validation299|stable_correct|Q_R_E|L9_R|141|
|validation299|stable_correct|R_Q_B|L1_R|141|
|validation299|stable_correct|R_Q_E|L1_R|141|
|validation299|stable_wrong|Q9_restoration|L10_a_R|47|
|validation299|stable_wrong|Q9_transfer_loss|L10_a_R|47|
|validation299|stable_wrong|Q_R_B|L9_R|47|
|validation299|stable_wrong|Q_R_E|L9_R|47|
|validation299|stable_wrong|R_Q_B|L1_R|47|
|validation299|stable_wrong|R_Q_E|L1_R|47|
