# Final E3 scientific decision packet

**Evidence status:** same-site 4090 parent full-attention bundle and L-only nested Train1030/Validation299 both completed independent bare-model forward and CPU arithmetic audits. Independently recomputed saved scores, candidate BF16 Fraction contrasts and question-paired 5000-draw bootstrap endpoints have zero discrepancy. These two holdouts had already been used for parent bundle validation: L is a project-internal nested confirmation, not an unseen blind replication. The fixed discovery64 was stratified on historical A6000 groups; actual 4090 groups/no-tie define the analyses. No Stage B, routing or additional Test intervention ran.

## Six preregistered L-only criteria

Rest = F(P+L)−F(P); loss = F(N)−F(N−L) in that natural receiver under the specified subtraction. Norm and unrelated are the frozen controls; `specific` is target effect minus unrelated effect. F is per-question gold-versus-mean-nongold score gap; CI is paired question bootstrap, 5000, seed20261112, percentile 95%.

| Claim | Train rescue337 mean [95% CI] | Validation rescue104 mean [95% CI] |
|---|---|---|
| rest | 1.5544 [1.4301, 1.6787] | 1.5345 [1.3025, 1.7674] |
| loss | 1.7027 [1.5828, 1.8240] | 1.7471 [1.5315, 1.9559] |
| rest_norm | 1.4840 [1.3693, 1.5985] | 1.4611 [1.2436, 1.6747] |
| loss_norm | 1.7013 [1.5822, 1.8224] | 1.7214 [1.5082, 1.9281] |
| rest_specific | 1.5732 [1.4261, 1.7241] | 1.6567 [1.3722, 1.9485] |
| loss_specific | 1.2667 [1.1398, 1.3989] | 1.3629 [1.1553, 1.5662] |

All six lower bounds are positive; rescue natural-no-tie mean checks pass separately (Train306, Validation100). Exact Fraction means and all candidate scores/predictions are in the linked machine-readable rows. Specified subtraction measures a controlled natural-context contrast, not necessity of the complete natural path.

## Group heterogeneity and paired cases

| Split/group | N / no-tie | mean rest / loss | both gap effects >0 | same-ID P→P+L rescue and N→N−L damage |
|---|---:|---:|---:|---:|
| Train rescue | 337 / 306 | +1.5544 / +1.7027 | 309 | 29 |
| Train damage | 38 / 31 | −0.1946 / −0.3043 | 8 | 0 |
| Train stable correct | 520 / 507 | +1.3715 / +1.3464 | 456 | 2 |
| Train stable wrong | 135 / 113 | +0.0917 / +0.1352 | 62 | 0 |
| Validation rescue | 104 / 100 | +1.5345 / +1.7471 | 98 | 10 |
| Validation damage | 9 / 8 | −0.5787 / −0.7292 | 1 | 0 |
| Validation stable correct | 139 / 136 | +1.3892 / +1.3373 | 115 | 2 |
| Validation stable wrong | 47 / 44 | +0.0860 / +0.0811 | 19 | 0 |

These same-ID counts are descriptive, not claims that all questions follow the means. Rescue no-tie examples with at least one nonpositive factor/control contrast include Train `AKDE&ED_2012_8_9` and Validation `MCAS_2006_9_33`. Four-group rows, ties and all opposite-sign IDs remain in the source results. The discovery E=L9 factor failed loss −.01111 and norm-loss −.04167 in fresh rescue15, despite successful true full-shape L9 Q-KV endpoint bridges; E failure and its implications remain alongside positive L results.

## L18 relation and remaining scientific gap

In the completed discovery64, for rescue15 the median L18 x/a/SDPA R distance to N under P+L was 1.3105/1.1950/2.5816; under N−L it was 3.6947/.8741/2.2799. These are state distances in different spaces, not contribution fractions or a mediation demonstration. Earlier parent full-bundle conditions P, PK, P+A and P+A+key did perform an actual L18 KQ→R edge intervention. Posthoc rescue Train337 key-at-P/key-at-P+A/interaction were +.494189/+.610318/+.116129; Val104 +.437600/+.569912/+.132312. That genuine parent edge manipulation cannot identify the downstream mediator for this newer L-only bundle.

Prior E4/E5 studies use different operators and independently show conditional state-to-attention mediation for a subset, with important negative replication and depth results. They do not validate the new L path or prove it persists at K3/K4. The remaining gap is which downstream computation mediates L-only effects and where the relevant Q-state information originally arose; no head/token/layer scan or additional intervention is implied. Q is the full current input after the fixed demonstration prefix and before R, including choices and format.

Source and audits: `../E3_WRITER_4090/results/conditional_l_holdout/` for 1329 rows, summaries and collector/bare/math receipts; independent saved-score/Fraction/bootstrap checks are included there. E/L discovery: `../E3_WRITER_4090/results/factor_discovery64/`; parent full-bundle × L18 edge posthoc summary: `../E3_WRITER_4090/results/parent_edge_factorial/`; original E4/E5 `../E4_E5_4090/FINAL_RESEARCH_REPORT.md`. Independent saved-score checks do not claim a new model forward, whereas the separate bare receipts explicitly document real same-site model-forward replay.

**Precision of individual counterexample counts:** Among Train rescue/no-tie306, at least one of the six contrasts is nonpositive for 63 questions under original production scores and 65 under exact BF16 target-logit Fraction arithmetic. The two extra exact-zero loss_specific cases are AKDE&ED_2012_4_21 (production +7.450580596923828e−8, exact 0) and Mercury_SC_415540 (+2.9802322387695312e−8, exact 0). Validation has 14/100 in both definitions. This does not change the pre-frozen mean/CI rules; the separate exact-sign receipt is `../E3_WRITER_4090/results/conditional_l_holdout/INDEPENDENT_FRACTION_COUNTEREXAMPLES.json`.
