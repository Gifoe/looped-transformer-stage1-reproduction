# Information availability and fixed-readout accessibility

The actual natural comparison is K1 versus H11=N=K2. H00 retains normalized K1 prefix B but resets Q/R to E; it is a hybrid, never native K1. No new probe was fitted. Original NPZ SHA204a3788301c7aa826a198b9e40ce495ef81284b6cd41a4758dfd942f8650890; all16 frozen heads across1393 questions/334320 applications were independently arithmetically audited.

|Cohort|same original frozen head|K1 accuracy|K2 accuracy|K2-K1|
|---|---|---|---|---|
|train1030|l18_input_raw_trainK1_permuted|0.241748|0.202913|-0.038835|
|train1030|l18_input_raw_trainK1_real|0.684466|0.766990|0.082524|
|train1030|l18_input_unit_l2_trainK1_permuted|0.239806|0.207767|-0.032039|
|train1030|l18_input_unit_l2_trainK1_real|0.692233|0.769903|0.077670|
|train1030|post_norm_raw_trainK1_permuted|0.222330|0.212621|-0.009709|
|train1030|post_norm_raw_trainK1_real|0.678641|0.839806|0.161165|
|train1030|post_norm_unit_l2_trainK1_permuted|0.223301|0.210680|-0.012621|
|train1030|post_norm_unit_l2_trainK1_real|0.694175|0.841748|0.147573|
|validation299|l18_input_raw_trainK1_permuted|0.207358|0.220736|0.013378|
|validation299|l18_input_raw_trainK1_real|0.394649|0.652174|0.257525|
|validation299|l18_input_unit_l2_trainK1_permuted|0.204013|0.220736|0.016722|
|validation299|l18_input_unit_l2_trainK1_real|0.401338|0.655518|0.254181|
|validation299|post_norm_raw_trainK1_permuted|0.227425|0.217391|-0.010033|
|validation299|post_norm_raw_trainK1_real|0.515050|0.799331|0.284281|
|validation299|post_norm_unit_l2_trainK1_permuted|0.210702|0.217391|0.006689|
|validation299|post_norm_unit_l2_trainK1_real|0.508361|0.792642|0.284281|


## Secondary frozen TrainK2 heads

|Cohort|secondary TrainK2 head|K1 accuracy|K2 accuracy|
|---|---|---|---|
|train1030|l18_input_raw_trainK2_permuted|0.207767|0.279612|
|train1030|l18_input_raw_trainK2_real|0.422330|0.889320|
|train1030|l18_input_unit_l2_trainK2_permuted|0.206796|0.269903|
|train1030|l18_input_unit_l2_trainK2_real|0.424272|0.887379|
|train1030|post_norm_raw_trainK2_permuted|0.211650|0.208738|
|train1030|post_norm_raw_trainK2_real|0.577670|0.903883|
|train1030|post_norm_unit_l2_trainK2_permuted|0.204854|0.196117|
|train1030|post_norm_unit_l2_trainK2_real|0.575728|0.901942|
|validation299|l18_input_raw_trainK2_permuted|0.214047|0.220736|
|validation299|l18_input_raw_trainK2_real|0.371237|0.739130|
|validation299|l18_input_unit_l2_trainK2_permuted|0.214047|0.217391|
|validation299|l18_input_unit_l2_trainK2_real|0.371237|0.742475|
|validation299|post_norm_raw_trainK2_permuted|0.210702|0.190635|
|validation299|post_norm_raw_trainK2_real|0.478261|0.799331|
|validation299|post_norm_unit_l2_trainK2_permuted|0.210702|0.187291|
|validation299|post_norm_unit_l2_trainK2_real|0.461538|0.806020|


TrainK1 identifies coefficient-training states. Its original pooled Train coordinate scaler includes both loops, so the entire transform was not learned from K1 alone. Raw/unit-L2, real/permuted and L18/post-norm stay separate. Unit-L2 is followed by coordinate standardization and is not a pure angular invariant. Train1030 probe accuracy is in-sample relative to original Train1094 head/scaler fitting. Probe capacity/lambda are frozen; partially exposed Validation was historically used for selection. Complete E2 descriptive outputs retain gold-index/candidate-count strata and both probe/native disagreement directions. Global token order is(330,340,389,414,422), i.e.A,C,B,E,D; the adapter remaps actual offered token order. Rare E has no positive Train gold examples. Candidate-conditioned permutation changes820/1094 Train labels; singleton candidate sets cannot be shuffled.

## Candidate-relative geometry

|Cohort|site|frozen raw head|definition|n|median rank|median delta energy fraction|
|---|---|---|---|---|---|---|
|train1030|l18_input|l18_input_raw_trainK1_permuted|global5|1030|4.0|0.244938|
|train1030|l18_input|l18_input_raw_trainK1_permuted|offered|1030|3.0|0.202639|
|train1030|l18_input|l18_input_raw_trainK1_real|global5|1030|4.0|0.104375|
|train1030|l18_input|l18_input_raw_trainK1_real|offered|1030|3.0|0.004161|
|train1030|l18_input|l18_input_raw_trainK2_permuted|global5|1030|4.0|0.284950|
|train1030|l18_input|l18_input_raw_trainK2_permuted|offered|1030|3.0|0.276340|
|train1030|l18_input|l18_input_raw_trainK2_real|global5|1030|4.0|0.036849|
|train1030|l18_input|l18_input_raw_trainK2_real|offered|1030|3.0|0.013703|
|train1030|post_norm|post_norm_raw_trainK1_permuted|global5|1030|4.0|0.195695|
|train1030|post_norm|post_norm_raw_trainK1_permuted|offered|1030|3.0|0.190069|
|train1030|post_norm|post_norm_raw_trainK1_real|global5|1030|4.0|0.031080|
|train1030|post_norm|post_norm_raw_trainK1_real|offered|1030|3.0|0.012316|
|train1030|post_norm|post_norm_raw_trainK2_permuted|global5|1030|4.0|0.162679|
|train1030|post_norm|post_norm_raw_trainK2_permuted|offered|1030|3.0|0.160712|
|train1030|post_norm|post_norm_raw_trainK2_real|global5|1030|4.0|0.031574|
|train1030|post_norm|post_norm_raw_trainK2_real|offered|1030|3.0|0.019380|
|validation299|l18_input|l18_input_raw_trainK1_permuted|global5|299|4|0.245214|
|validation299|l18_input|l18_input_raw_trainK1_permuted|offered|299|3|0.200881|
|validation299|l18_input|l18_input_raw_trainK1_real|global5|299|4|0.105134|
|validation299|l18_input|l18_input_raw_trainK1_real|offered|299|3|0.004864|
|validation299|l18_input|l18_input_raw_trainK2_permuted|global5|299|4|0.282314|
|validation299|l18_input|l18_input_raw_trainK2_permuted|offered|299|3|0.274460|
|validation299|l18_input|l18_input_raw_trainK2_real|global5|299|4|0.035373|
|validation299|l18_input|l18_input_raw_trainK2_real|offered|299|3|0.013162|
|validation299|post_norm|post_norm_raw_trainK1_permuted|global5|299|4|0.198322|
|validation299|post_norm|post_norm_raw_trainK1_permuted|offered|299|3|0.193461|
|validation299|post_norm|post_norm_raw_trainK1_real|global5|299|4|0.031416|
|validation299|post_norm|post_norm_raw_trainK1_real|offered|299|3|0.013492|
|validation299|post_norm|post_norm_raw_trainK2_permuted|global5|299|4|0.163642|
|validation299|post_norm|post_norm_raw_trainK2_permuted|offered|299|3|0.162131|
|validation299|post_norm|post_norm_raw_trainK2_real|global5|299|4|0.031336|
|validation299|post_norm|post_norm_raw_trainK2_real|offered|299|3|0.018793|


These centered raw W/std label-axis projections use actual K1-to-K2 deltas, rank<=4 for global5 and rank<=m-1 for offered candidates. They are distinct definitions; an energy fraction is not native causal use or information content. The extra E-axis confound is retained. Separate embedding-alignment geometry is explicitly named embedding geometry. Actual doses and norm controls are reported instead of ranking raw effects by projection norm.

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


Q-only boundary pairs have identical R at L1. Earliest means among SAVED selected R sites; unobserved layer outputs between the L1 input and L9 input prevent a first-physical-writer claim. Successful decoding establishes accessibility to this readout. Failed decoding establishes only failure of finite decoders. Which semantic information was absent from all K1 states remains unidentified.
