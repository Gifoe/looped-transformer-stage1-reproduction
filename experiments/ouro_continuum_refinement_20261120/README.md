# Ouro-1.4B continuum refinement

[中文报告](ODE_REFINEMENT_ZH.md) · [数值收敛](NUMERICAL_CONVERGENCE.md) · [离散与连续](DISCRETE_VS_CONTINUOUS.md) · [FP32 审计](FP32_PRECISION_AUDIT.md) · [动力学](DYNAMICS_ANALYSIS.md)

This experiment extends repository commit `ddf430e228f80c816a50d8c8df15896ae59bd788`. It uses the inherited prompt manifest, ARC transition strata, checkpoint loader and native context capture. The four iterations cover native parity and short refinement, higher resolution and RK4, precision and operator diagnostics, and a held-out prompt verification. All model weights are frozen.

| Figure | PDF |
| --- | --- |
| 1. Trajectories by step size | [Halving trajectories](results/figures/figure_1_halving_trajectories.pdf) |
| 2. Euler and RK4 | [Convergence by method and precision](results/figures/figure_2_euler_rk4.pdf) |
| 3. Error across 48 branches | [Same-start and propagated errors](results/figures/figure_3_branch_errors.pdf) |
| 4. Native endpoint comparisons | [Distances to K1 and K2](results/figures/figure_4_native_and_continuous_endpoints.pdf) |
| 5. Difference between dynamics | [Full-loop, branch and native path differences](results/figures/figure_5_dynamical_disagreement.pdf) |
| 6. Local trajectory geometry | [Speed, turn and curvature](results/figures/figure_6_speed_turn_curvature.pdf) |
| 7. Norm, precision and finite updates | [Mechanism controls](results/figures/figure_7_norm_precision_finite_steps.pdf) |

## Reproduction

Check out the complete research branch with its inherited experiment directories. Use Python 3.12 and the packages in [environment.txt](environment.txt). Download the nine locked files from [Ouro-1.4B Base revision 574fa66](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1). ARC and synthetic inputs for this study are already encoded in the inherited token manifest. The analysis uses the existing ARC validation panel.

```bash
cd experiments/ouro_continuum_refinement_20261120
bash reproduce.sh /absolute/path/to/pinned-model /absolute/path/to/new-run/results
sha256sum -c ARTIFACTS.sha256
```

The released measurement files contain per-case full-state errors, per-token normalized errors, candidate logits, trajectories at the fixed grid, matched-start branch diagnostics, numerical precision probes and synchronized timing receipts. Coordinates are derived from full-width activations in memory. The released data store measurement values rather than all intermediate activation tensors.

File families in [results](results/) correspond to the 12 primary prompts, 8 separate confirmation prompts, and three singleton precision traces using the same input and batch layouts across dtypes. Results are grouped by tableau, two-hop and ARC. The manifests include prompt IDs, token IDs, candidate ordering and the inherited ARC stratum label. The dtype audit retains the frozen parameter hashes and actual activation dtypes.
