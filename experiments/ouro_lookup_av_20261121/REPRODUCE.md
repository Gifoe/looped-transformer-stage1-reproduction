# Reproduce the attention-factor experiment

Use the frozen model and runtime versions in `results/VERIFICATION.json`. From this directory in the release checkout, run:

```sh
sha256sum -c ARTIFACTS.sha256
python reproduce.py /path/to/Ouro-1.4B /path/to/new-output --verify-deck
python reproduce.py /path/to/Ouro-1.4B /path/to/new-output
```

The first command checks the archive, the second rebuilds the fixed deck and verifies prompt history without loading the model, and the third reproduces native greedy generation, the full intervention, the physical audit, and the report. The prior lookup directories in the same checkout supply the native execution helper and historical prompt manifests. The archived GPU logs give the production order of checks.

Let $P$ be the frozen output projection, including its bias if present. In the targeted layer, let $A_r,V_r$ and $A_d,V_d$ be the observed native receiving and donating operands conditioned on the arm's emitted prefix. Route only installs $P(A_d V_r)$, value only installs $P(A_r V_d)$, and joint installs $P(A_d V_d)$. The interaction displacement uses the linear part of $P$ on $(A_d-A_r)(V_d-V_r)$. Consequently joint minus interaction is exactly $P(A_d V_r + A_r V_d - A_r V_r)$ up to floating-point roundoff. Matched subtraction preserves the norm and radial component of the same displacement at each query position. The architecture's normal post-projection RMSNorm and every MLP run on the edited values.
