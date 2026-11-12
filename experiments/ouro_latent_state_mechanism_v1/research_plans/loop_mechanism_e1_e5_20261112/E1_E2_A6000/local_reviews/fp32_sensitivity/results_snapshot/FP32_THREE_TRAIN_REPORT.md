# Three-Train full-model FP32 sensitivity

Three pre-frozen discovery64 Train IDs, the lexicographically first three: one natural damage, one stable-correct, one stable-wrong. **No natural rescue item is included**, so this cannot establish precision stability of rescue-group effects. The sample was selected independently of its effects but is not a blind or population evaluation.

All 269 BF16 checkpoint tensors were verified individually equal to their FP32 model values after exact promotion; full model weights and activations were FP32. TF32 was disabled for matmul and cuDNN, highest matmul precision set, torch 2.5.1+cu124 on RTX A6000. First case actual profiler recorded `aten::_scaled_dot_product_efficient_attention`; the allowed SDPA backends were enabled, not forced to match BF16 dispatch. Original causal mask, scale and GQA/no-cache protocol remained in force. Unrelated donor inputs used the exact previously selected donor IDs and were fully re-forwarded in FP32. No checkpoint was modified.

| ID / natural group | K1 max candidate score difference | K2 max candidate score difference | Natural K1/K2 predictions BF16 → FP32 | Q1 interaction BF16 → FP32 | Q2 interaction BF16 → FP32 | Changed corner predictions (of 8) | Changed comparable control predictions |
|---|---:|---:|---|---:|---:|---:|---:|
| ACTAAP_2010_7_7 / damage | 0.027009 | 0.042106 | 3/2 → 3/2 | 0.083333 → 0.080802 | -0.062500 → -0.024318 | 1 | 0 |
| AKDE&ED_2008_4_41 / stable_correct | 0.084545 | 0.130515 | 3/3 → 3/3 | 0.500000 → 0.395580 | 0.312500 → 0.310801 | 0 | 0 |
| AKDE&ED_2008_8_32 / stable_wrong | 0.020263 | 0.148938 | 1/1 → 1/1 | 0.812500 → 0.780216 | 0.687500 → 0.645852 | 0 | 0 |

Each of the 24 corners and the specified ID/ADD/norm/unrelated/restore controls was freshly computed in FP32. ID and restore exactly matched the FP32 corner endpoints; G111 matched no-hook FP32 natural K1. Same-ID donor FP32 states came from separate fresh full-model FP32 forwards. The second-query interaction on the damage example remains negative but moves from −0.0625 to −0.0243; this is not a stability conclusion for other items. Output values, candidate scores, IDs, all controls and original dtype comparisons are in FP32_BF16_COMPARISON.json and per-question metadata.

Independent CPU saved-output verification checked the stored full FP32 logits hashes, identities, restore endpoints and 54 scored outputs against a separate FP64 logsumexp reference; maximum absolute score difference was 4.54×10⁻⁷. This did not independently replay the FP32 model on GPU. Three case CUDA-synchronized computation times excluding saving were 12.61, 13.01, 13.90 seconds; PENDING.json contains runtime/peak resource evidence. Treat the observed numbers as technical sensitivity, not causal validation or rescue-group inference.

Backend interpretation: the first FP32 natural forward was profiled as efficient attention. The BF16 execution was not profiled in this FP32 run, so its actual SDPA kernel is not asserted here. These are paired reference-protocol versus FP32-run differences; precision and backend changes are not separately identified, and the difference cannot be attributed to dtype alone. One of 24 corner predictions differed across the two runs. Natural predictions matched on all six loop evaluations in this selected sample.
