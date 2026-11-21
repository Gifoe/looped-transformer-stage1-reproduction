# Reproduce native two-row routing and content exchange

Use the checkpoint and runtime versions in results/VERIFICATION.json. Run from this experiment directory in a checkout that preserves adjacent experiment directories:

    sha256sum -c ARTIFACTS.sha256
    python reproduce.py /path/to/Ouro-1.4B /path/to/new-output --verify-deck
    python reproduce.py /path/to/Ouro-1.4B /path/to/new-output

The first Python command checks the deck without model weights. The full run performs native greedy generation, matching, attention and value edits, physical audits, and paired analysis. MARGINS.json stores clean decision margins and COHORT.json hashes the inputs and matching.

For the two selected rows, let L and R be their live summed weights, and let p be the naturally observed donor left fraction. An unbounded route edit would request L' = (L+R)p with R' = L+R-L'. In the matched arms a signed displacement calculated in the native references is capped by its feasible mass at both pairs, then by the running source mass. Row-internal weights are scaled uniformly, and the nonedited weight entries remain unchanged. The full diagnostics use the unbounded redistribution formula on live total mass. ROUTE_RAW.json records requested, common and realized dose by format, cohort, direction, arm and physical layer. The V edit uses only aligned digit sites, and the model multiplies the actual edited weights and values before the native output projection. No subsequent hidden state is copied.
