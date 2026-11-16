# Final scientific review

**Outcome C: no demonstrated utility gain.** The primary objective of outperforming native K4 and strong matched output-only baselines was not met.

On full Train1094, native K4 gets 966 correct. The best fixed structured row, routing +0.1, gets 969, tying E6 conservative. No structured configuration passes the predeclared accuracy/control gate. Value +0.1 and joint positive correction pass only the separate secondary gold-margin gate; that reproducible score effect authorized the small static-coefficient fit.

The selected primary 14-scalar policy is exactly native. The Validation-selected routing-only policy gets 261/299, equal to native; E6 conservative gets 263. On the independently selected SciQ997 population, native and fixed value +0.1 get 924, while routing-only gets 922. Neither primary nor secondary passes its independent benefit gate.

FullAnswer is a separate protocol: native K4 gets 157/299, fixed value +0.1 gets 162/299 (5 rescues, 0 damages; descriptive paired 95% delta [+0.334, +3.344] pp), and the Validation-selected routing-only policy gets 161/299. Matched LoopCD 1 gets 173/299. Thus the fixed correction has a protocol-specific gain over native while remaining below the strong output-only baseline; it does not establish the failed Joint-Option and independent-benchmark objective.

The evidence covers one bounded coefficient grid and one fixed SPSA configuration, not every possible CATC optimizer or information-flow controller. No additional head, layer, threshold, coefficient or Gate search was launched to rescue the result. ARC Test was not justified and was not run.

## Interpretation and numerical scope

The value/routing decomposition is algebraic and non-unique, so component score effects do not establish causal attribution to accuracy. Later K2 sites use the actual trajectory after upstream edits, while K1 is natural: this is the explicitly frozen sequential feedback variant. It is not a separate unmodified K2 donor trajectory. Probabilities are FP64 reconstructions from actual post-RoPE Q/K, rather than bitwise extraction of fused SDPA internal probabilities. Original SDPA, projection, RMSNorm, residual, MLP, K3 and K4 operations remain in place.

Zero coefficients match full-vocabulary outputs bitwise on three fixed technical cases; all intended direct write sites and all four passes are audited. FP64-to-FP32 decomposition sensitivity on the same three cases changes 0/48 endpoint predictions and zero tie sets; that is not full-cohort or cross-hardware robustness. The exact sources, prospective freezes, numerical repairs and source-version reconstruction receipts are retained.

Validation was exposed and used for selection. SciQ is independent within this study, with no claim of absence from pretraining or exclusion of every semantic overlap. Its three exclusions are duplicate options; no additional ARC overlap was found by the stated exact/Jaccard screen. SciQ attribution and CC-BY-NC 3.0 terms accompany the inputs.

Full raw heads and Q/K/V witnesses remain on the server under hash manifests. Published scalar rows reproduce summaries and statistical decisions on CPU; they do not substitute for rerunning GPU forwards.

| Policy | Median seconds | Relative to native | Peak allocated GiB |
|---|---:|---:|---:|
| native | 0.1860 | 1.000 | 3.082 |
| fixedVplus | 0.1923 | 1.034 | 3.082 |
| catc14 | 0.1882 | 1.012 | 3.082 |
| unstructured14 | 0.1885 | 1.014 | 3.082 |
| value7 | 0.1891 | 1.017 | 3.082 |
| routing7 | 0.1943 | 1.044 | 3.082 |
| validation_selected_structured | 0.1942 | 1.044 | 3.082 |
