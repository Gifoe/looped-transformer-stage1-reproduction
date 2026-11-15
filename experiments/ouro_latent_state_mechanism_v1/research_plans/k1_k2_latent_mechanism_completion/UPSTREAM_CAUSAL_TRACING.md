# Boundary-conditioned upstream causal tracing

N is actual natural K2=AB1 H11. U is AB1 H01: the actual K2 boundary question/options Q uses E, while fixed demonstration prefix P and final readout R retain actual end-of-K1 normalized B. At physical L10-L16 R, d_a=a_N-a_U and d_m=b_N-b_U use actual normalized attention and MLP increments in FP64. These same-question boundary-conditioned reference trajectory differences include feedback. They are neither natural K2-K1 differences nor the historical Q9-conditioned C1 difference.

T starts from N and subtracts fixed d_a from each actually computed current attention increment, then casts once to BF16. U_plus adds d_a on U. M and U_M_plus perform separate MLP-only operations. No entire trajectory is clamped to its saved reference. At L18, T attention/QKV executes before composition: TX=x_N+a_T, TA=x_T+a_N and TXA=x_N+a_N. The same composed y enters both MLP input and retained outer skip; the complete MLP and remaining suffix execute. Both-branch TXA=N is a verified structural identity, not a mediation fraction.

All25 endpoints are fixed: N,U,T; TX,TA,TXA,U_plus,upstream_norm,upstream_unrelated,U_plus_norm_preserve,U_plus_unrelated; T/N x X/A x norm/unrelated eight branch controls; M,U_M_plus,M_norm_preserve,M_unrelated,U_M_plus_norm_preserve,U_M_plus_unrelated. Upstream norm-preserving controls normalize the current proposal to the receiver increment norm. Branch norm-only controls scale the current base direction to target norm. Unrelated donors are next lexicographic IDs cyclic within original full Train1094 or Validation299, without labels/resampling. Per-site/branch doses are matched before BF16 rounding; actual scales/doses/casts remain. Undefined controls require independently proved witnesses and invalidate full-defined positive coverage rather than being dropped.

|Cohort|attention6 all pass|L18 branch12 all pass|MLP secondary6 all pass|
|---|---|---|---|
|train1030|True|True|True|
|validation299|True|True|False|


## Attention update family

|Cohort|contrast|defined/rescue|mean gap effect|ordinary95%CI|adjusted one-sided lower|declared pass|
|---|---|---|---|---|---|---|
|train1030|loss|334/334|2.652289|[2.491942, 2.809261]|2.455337|True|
|train1030|restore|334/334|0.765843|[0.697728, 0.832214]|0.683193|True|
|train1030|norm_loss|334/334|2.770615|[2.602918, 2.940122]|2.569989|True|
|train1030|norm_restore|334/334|0.574289|[0.525508, 0.625125]|0.512910|True|
|train1030|specific_loss|334/334|1.203655|[1.059815, 1.348681]|1.031746|True|
|train1030|specific_restore|334/334|0.743700|[0.645511, 0.840446]|0.617160|True|
|validation299|loss|103/103|2.406351|[2.124793, 2.678403]|2.061617|True|
|validation299|restore|103/103|0.681735|[0.555309, 0.810485]|0.520591|True|
|validation299|norm_loss|103/103|2.510214|[2.231159, 2.793330]|2.156317|True|
|validation299|norm_restore|103/103|0.525485|[0.417061, 0.631478]|0.386327|True|
|validation299|specific_loss|103/103|1.149272|[0.858207, 1.450040]|0.805016|True|
|validation299|specific_restore|103/103|0.678095|[0.515777, 0.841133]|0.476300|True|


## Separate MLP family

|Cohort|contrast|defined/rescue|mean gap effect|ordinary95%CI|adjusted one-sided lower|declared pass|
|---|---|---|---|---|---|---|
|train1030|loss|334/334|1.511228|[1.370069, 1.647089]|1.342724|True|
|train1030|restore|334/334|0.111371|[0.074379, 0.147646]|0.065784|True|
|train1030|norm_loss|334/334|1.474364|[1.338811, 1.612551]|1.309974|True|
|train1030|norm_restore|334/334|0.087700|[0.054329, 0.121073]|0.046969|True|
|train1030|specific_loss|334/334|0.109749|[0.033462, 0.184011]|0.016488|True|
|train1030|specific_restore|334/334|0.105539|[0.061876, 0.149576]|0.051366|True|
|validation299|loss|103/103|1.385316|[1.166864, 1.604986]|1.116095|True|
|validation299|restore|103/103|0.033374|[-0.035599, 0.103964]|-0.050098|False|
|validation299|norm_loss|103/103|1.349110|[1.136934, 1.571809]|1.095795|True|
|validation299|norm_restore|103/103|0.018507|[-0.042779, 0.079796]|-0.057411|False|
|validation299|specific_loss|103/103|0.031958|[-0.108419, 0.172740]|-0.136667|False|
|validation299|specific_restore|103/103|0.042981|[-0.046521, 0.139773]|-0.068637|False|


The separate MLP bundle does not pass every declared direction in both cohorts; exact failures stay visible. This is not proof that MLP computation is absent or unimportant.

Full-bundle interventions establish only specified contrasts. Reference d includes N/U trajectory feedback. Norm and unrelated-direction controls address the tested alternatives, not every perturbation. Attention/MLP raw magnitudes have different doses and contexts and cannot rank intrinsic importance. No unique writer or additional site/head/token sweep is asserted.

Primary contrasts use fresh same-hardware native K1-wrong/K2-correct questions, including ties. The original natural-no-tie subset checks mean sign reversal; post-treatment ties never filter primary means. Production FP32 full-vocabulary log-probability gold-minus-mean-wrong gaps are primary; fixed K1 strongest-wrong, dynamic-wrong and exact BF16 target-logit Fraction margins remain available. B2 uses5000 paired bootstrap draws, seed20261114, separate one-sided Bonferroni families6/12/6 and ordinary95% intervals. No global FWER across all families is claimed. Train64 engineering, disjoint Train1030 and previously exposed Validation299 stay separate; these are within-project confirmations, not blind external tests.

The endpoint name `unrelated` denotes the prospectively fixed next-lexicographic different-question direction. This is a tested other-question directional control, not a certificate of semantic independence. No labels or outcomes choose/resample the donor.
