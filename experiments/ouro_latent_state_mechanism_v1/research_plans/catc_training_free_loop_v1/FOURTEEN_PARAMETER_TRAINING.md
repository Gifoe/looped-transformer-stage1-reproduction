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
