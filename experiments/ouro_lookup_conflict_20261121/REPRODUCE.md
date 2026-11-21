# Reproduce the target-row and residual conflict

Use the frozen model and software versions in `results/VERIFICATION.json`, with the adjacent research directories from the release checkout. From this directory run:

```sh
sha256sum -c ARTIFACTS.sha256
python reproduce.py /path/to/Ouro-1.4B /path/to/new-output --verify-deck
python reproduce.py /path/to/Ouro-1.4B /path/to/new-output
```

The first Python command reconstructs the held-out deck without model weights. The full run performs native K1/K2 scores, all paired-world K2 scores, token-span validation, edited greedy generation and independent audit. `PAIRS.json` stores the four native table worlds and their target-digit offsets. `COHORT.json` freezes eligible cases and audited questions before edits. Both directions and each independent control retain complete generated strings and token IDs.

K2 uses its normal cache-free computation. On every arm's emitted prefix, the residual transplant changes one position in the normalized first-pass output. At the value site, the experiment changes selected outputs of each K2 `v_proj` before the model's own grouped-head reshape and attention multiplication. Both Q and K remain computed from the running state. The exact position audit verifies equality of every non-target value-vector entry before and after hooking, and reconstructs the actual next-token-position attention product from the model's live weights and edited V. The finished verification also independently repeats every tokenizer span and the paired statistic calculation.
