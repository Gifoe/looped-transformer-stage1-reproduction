# Reproduce native K2 interaction deletion

Use the frozen model and software versions in `results/VERIFICATION.json`, with the adjacent earlier experiment directories from the release checkout available. From this experiment directory run:

```sh
sha256sum -c ARTIFACTS.sha256
python reproduce.py /path/to/Ouro-1.4B /path/to/new-output --verify-deck
python reproduce.py /path/to/Ouro-1.4B /path/to/new-output
```

The deck check runs without loading model weights. The full run repeats greedy native scoring, clean-margin calibration, fixed matching, live K2 deletions, independently replayed suffix audits, and the statistical report. `results/MARGINS.json` records all eligible foil scores and exact token comparisons. `results/COHORT.json` binds the matching to the native and margin hashes before any edited generation.

For valid positions, the intervention is `O_live − D`, with `D = P_linear[(A2−A1)(V2−V1)]` computed from the two clean phases on the same emitted prefix. The runtime hook subtracts the displacement from the **actual current** projection after upstream effects of earlier deletions. It does not install the clean K2 projected state. Matching is performed against the clean K2 projection. The actual attention output, native post-projection normalization, MLP and residual update still use the current edited forward. All later layers run normally. Zero subtraction is checked against native full generation for every case.
