# Reproduce the same-debit destination test

Use the checkpoint and runtime versions in results/VERIFICATION.json. From the archived experiment directory run:

    sha256sum -c ARTIFACTS.sha256
    python reproduce.py /path/to/Ouro-1.4B /path/to/new-output --verify-deck
    python reproduce.py /path/to/Ouro-1.4B /path/to/new-output

The deck check runs without model weights. Full reproduction calculates native answers, decision margins, stable pairing, and OV-reference scores for both candidate third rows before generating any edited answers. PROJECTION.json records the raw candidate scores and chosen ratios. COHORT.json hashes the fixed choices and matched cases.

At a fixed prefix the true and third-row probability perturbations are respectively -u at the same source plus w at the native destination, and -u plus the exact vector w at the chosen extra row. Their moved probability and sparse L1 and L2 norms are equal. The shared reference is from the four digit rows of the clean recipient, with other live weights proportionally scaled. V and downstream updates stay live. A full-live diagnostic preserves the original protocol's use of the running two-row mass rather than the clean reference. The first-prefix OV matching metric is a prospective check on projected reference dose, not a claim about equality of every downstream activation after generation branches.
