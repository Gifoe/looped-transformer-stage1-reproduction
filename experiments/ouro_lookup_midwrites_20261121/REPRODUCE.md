# Reproduce native middle-layer replay

Use the checkpoint and software versions in `results/VERIFICATION.json`. From this directory, with the adjacent archived experiments present in the same checkout, run:

```sh
sha256sum -c ARTIFACTS.sha256
python reproduce.py /path/to/Ouro-1.4B /path/to/new-output --verify-deck
python reproduce.py /path/to/Ouro-1.4B /path/to/new-output
```

The deck check reconstructs both domain manifests from the locked seed and verifies the history overlap guard. The full run checks the model hash, generates the unmodified K1/K2 answers, and runs and verifies the replay if both cohorts qualify. `report.py` reconstructs the exact answer decisions, checks all arms and the replay identity controls, recomputes the statistical decisions, and writes a figure. The archived `native.log` and `mid.log` record the production run.

For branch $b$ in receiving phase $r$ at valid position $i$, the authentic arm installs the native donor write $W_{1-r,\ell,b,i}$ in place of the computed write. The fixed geometry and unrelated controls are built against the recipient's clean native write, at each generation step's actual input. All uninstalled branches are computed on the current state, including every write from physical layer 18 onwards. `mid_test.py` directly recomputes that suffix layer by layer from its recorded input in the fixed identity and replay audit.
