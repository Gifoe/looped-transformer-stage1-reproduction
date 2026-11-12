# E5 local computation by depth

Final status: matched KV and R-state/actual-attention experiments at K2–K4 are complete on full 299 and preregistered 128. Late native-attention restoration has negative mean effects in both cohorts; natural perturbation magnitudes also shrink. The following geometry and intervention tables support a local depth comparison, not a complete upstream mechanism or convergence claim.

Matched adjacent-loop KV-derived SDPA_R interventions were executed at the same physical L18 site, using native recipient Q, full attention shapes and the same downstream recipient computation. This is an exploratory local causal comparison, not propagation of a fully established E3 mechanism.

On all 299 Validation questions, mean native relative R-state change remains nonzero:

| State | K1→K2 | K2→K3 | K3→K4 |
|---|---:|---:|---:|
| L18 input residual | 0.565157 | 0.394546 | 0.288904 |
| SDPA_R | 0.754918 | 0.555237 | 0.432226 |
| Local-normalized attention increment | 0.669577 | 0.485715 | 0.362677 |
| Final-normalized R state | 0.626760 | 0.370021 | 0.271401 |

Relative change is ||H_later−H_earlier||/||H_earlier|| per question, then averaged. These measurements cover selected readout states, not the full sequence state. They show smaller but continuing changes; they do not prove global convergence.

Mean target perturbation norm also declines: restore 2.6796→1.7520→1.0286; rollback 2.5686→1.3629→0.8565. Consequently, the smaller later causal effects combine changing update magnitude, direction and recipient compatibility. Current controls match within each depth; they do not test equal-amplitude cross-depth receiver sensitivity.


## Completed state/actual-attention comparison

| Cohort | Receiver | Native correct | U correct | Restored-U correct | A-only correct | Natural rescues | Lost under U | Recovered by A | Strict four-control recoveries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 299 | K2 | 243 | 220 | 240 | 235 | 104 | 29 | 22 | 11 |
| 299 | K3 | 258 | 251 | 256 | 253 | 20 | 9 | 4 | 2 |
| 299 | K4 | 261 | 256 | 256 | 263 | 7 | 5 | 1 | 1 |
| 128 | K2 | 103 | 98 | 105 | 101 | 40 | 9 | 6 | 2 |
| 128 | K3 | 110 | 109 | 109 | 110 | 9 | 2 | 0 | 0 |
| 128 | K4 | 107 | 109 | 108 | 107 | 1 | 1 | 0 | 0 |

| Cohort | Receiver | Total meanwrong | Direct at native A | Restore effect | Restore 95% interval |
|---|---:|---:|---:|---:|---|
| 299 | K2 | -1.453247 | -1.375505 | +0.077742 | [+0.0375, +0.1203] |
| 299 | K3 | -0.467740 | -0.569363 | -0.101623 | [-0.1303, -0.0708] |
| 299 | K4 | -0.067917 | -0.115646 | -0.047729 | [-0.0628, -0.0317] |
| 128 | K2 | -1.394368 | -1.353923 | +0.040446 | [-0.0160, +0.0970] |
| 128 | K3 | -0.363932 | -0.506673 | -0.142741 | [-0.1813, -0.1032] |
| 128 | K4 | -0.032471 | -0.075399 | -0.042928 | [-0.0618, -0.0234] |

Mean state rollback delta norms on 299 are 4.3055/3.2560/2.5072, and on 128 are 4.2490/3.1886/2.4887. Actual-attention delta norms are 1.9645/1.8772/1.2526 and 1.9089/1.8317/1.1759 respectively. These decreasing natural perturbations confound any equal-size sensitivity interpretation. Nonzero late state changes coexist with weak or reversed decision effects. The same local controlled operator is now measured; a complete upstream generating mechanism remains unresolved.
