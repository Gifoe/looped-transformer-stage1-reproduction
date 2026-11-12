# Final matched causal comparison by depth

All fixed 299 and 128 interventions and independent audits are complete. The existing six-direction KV experiment is preserved; the matched R/actual-attention experiment now directly compares the locally supported mechanism across depth.

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

The changing natural-rescue denominators are explicit. Strict recoveries require no relevant ties plus specificity against both state and both mediator controls. Counts are not mediated percentages. Later restoration can rescue individual299 decisions while lowering the average margin;128 has no late recovery. ConfirmationK4 native107 is belowK3's110. Direct and restoration effects therefore cannot be summarized as uniform positive mechanism saturation.

The original299 KV target rescue/damage results remain: K1←K2 57/3;K2←K1 4/14;K2←K3 6/4;K3←K2 2/2;K3←K4 2/1;K4←K3 1/0. Their meanwrong changes are+0.494635,−0.328282,+0.036545,+0.013500,−0.027487,+0.024561. These recipient-specific effects are distinct from the state/attention four corners.

The question-bootstrap intervals use10,000 resamples, seed 20261112 and retain metric-specific differences. They are not seed replication or multiplicity-adjusted causal discovery. Complete controls, fixedwrong/global-fixedwrong/dynamicmargin, groups/ties and all cases are in the independent review JSONs and ROOT_CONFIRMATION128_ALL_METRICS.json. Figures validation299_state_mediation_depth and confirmation 128_state_mediation_depth are available asPNG/PDF/SVG. Natural delta sizes differ across depths; equal-amplitude sensitivity was not measured.
