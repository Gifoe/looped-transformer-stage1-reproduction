# Reproduce the fixed-boundary lookup experiment

Use the exact checkpoint and software versions in `results/VERIFICATION.json`. From this experiment directory in a checkout of the release branch, run:

```sh
sha256sum -c ARTIFACTS.sha256
python reproduce.py /path/to/Ouro-1.4B /path/to/new-output --verify-deck
python reproduce.py /path/to/Ouro-1.4B /path/to/new-output
```

The first Python command verifies the fixed deck and prompt history without loading the model. The full run reads its dependencies from the adjacent archived experiments, verifies the checkpoint hash, generates unrestricted integer answers and repeats the physical replay audit on CUDA. It writes results and a report under the chosen output directory. `report.py` checks both domains, prompted inputs, eligibility, exact complete answers and intervals. The server launchers preserve the exact production commands.

For receiving phase \(k\) and raw layer-input state \(x_{k,\ell}\) at the selected answer position, exchanging a donor state \(x_{d,\ell}\) while holding suffix attention and MLP writes fixed gives
\[
z^\mathrm{edited}_{k}=z^\mathrm{native}_{k}+x_{d,\ell}-x_{k,\ell}.
\]
Here \(z\) is the final residual before the native RMSNorm. Applying the checkpoint's actual RMSNorm and vocabulary head gives the intervened next token. The physical audit applies the exact boundary perturbation using a layer-input hook and restores each native normalized residual write at all later layers. It compares both the resulting residual and its selected vocabulary token. Earlier positions cannot read that output-token position in the masked forward.

For geometry controls the displacement norm and radial projection match at the intervention site. The unrelated-row control matches the authentic table displacement in norm. The paired table donor is in K2. In the phase experiment the table stays unchanged, and the donor is the same question in the other phase. Neither test inserts any candidate token. The same emitted prefix conditions both donor and receiver at each step.

The first successful boundary is a statement under this tested readout and fixed suffix. A confidence upper bound on the preceding boundary is required to claim a tested interval.
