# Component and control comparisons

All 1094 Train questions are retained, including negative and zero effects. Margins are gold logit minus the highest non-gold candidate logit; positive/negative signs below use exact saved FP32 candidate scores, with no post-hoc significance threshold. Rescue/damage is relative to the native endpoint at the same depth.

| Setting | K2 correct | K2 rescue/damage | K4 correct | K4 rescue/damage | K4 margin positive/negative/zero |
|---|---:|---:|---:|---:|---:|
| T1_structured | 891 | 4/1 | 967 | 2/1 | 272/176/646 |
| T2_structured | 889 | 6/5 | 969 | 3/0 | 232/216/646 |
| T3_structured | 890 | 5/3 | 966 | 1/1 | 159/300/635 |
| T4_structured | 891 | 7/4 | 967 | 1/0 | 183/250/661 |
| T5_structured | 888 | 6/6 | 966 | 0/0 | 322/174/598 |
| T6_structured | 890 | 5/3 | 967 | 2/1 | 140/351/603 |

Every nonzero setting has unrelated-layer, randomized-direction, prefix-only and signed output-scaling controls. Their complete results and paired accuracy/margin contrasts are in SUMMARY.json and GATES.json. Randomization preserves requested basis norm; it does not guarantee exact equality of realized post-BF16 doses after different upstream trajectories. Actual per-site norms and clipping are retained, and all actual writebacks obey the 10% bound.

The two 30-contrast accuracy and gold-margin gate families are separate; no global 5% family-wise claim across both routes is made. None passes the accuracy gate; T1 and T5 pass only the secondary score gate.

# Static-coefficient fitting

All Ouro parameters remained frozen. The four runs used the same projected SPSA schedule, initialized at zero: learning rate .05/(t+1)^.602; perturbation .05/(t+1)^.101; L2 penalty .01; coefficient bounds [-.25,.25]. The actual post-BF16 correction was bounded at 10% per site. Each run allowed one shuffled Train epoch (35 minibatches of up to 32; seed 20261116). Validation candidate CE was checked at steps 0/12/24/35; earliest minimum won; two consecutive non-improving checks stopped a run.

| Policy | Active coefficients | Steps | Distinct Train IDs queried | Train forwards | Selected step | Validation CE |
|---|---:|---:|---:|---:|---:|---:|
| catc14 | 14 | 24 | 768 | 1536 | 0 | 0.374441628 |
| unstructured14 | 14 | 24 | 768 | 1536 | 0 | 0.374441628 |
| value7 | 7 | 24 | 768 | 1536 | 0 | 0.374441628 |
| routing7 | 7 | 35 | 1094 | 2188 | 12 | 0.373952531 |

The primary catc14, unstructured14 and value7 select the all-zero identity checkpoint. Routing7 selects step 12, trained on its first 384 shuffled Train examples. Later Train and Validation queries still count toward the selection cost. Exact coefficients are in phaseC/SELECTED_COEFFICIENTS.json, with every SPSA step and checkpoint row preserved.

The secondary structured family selects routing7 using Joint Validation CE before SciQ or FullAnswer outputs. This is supervised development calibration. The primary catc14 hypothesis is retained separately. No independent-set coefficient search occurred.

Derivative-free fitting was chosen prospectively to avoid storing full backbone backward state within this pilot. A backpropagation out-of-memory failure was not measured, and convergence is not established. Fourteen parameters do not remove the cost of the 6,796 training forward queries or the additional Validation checks. Actual wall envelopes, physical forwards, serial deployment memory and latency are recorded separately in phaseC/COMPUTE_COSTS.json and phaseC/SELECTED_INFERENCE_PROFILE.json.
