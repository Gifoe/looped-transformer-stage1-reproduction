# Native Up mediation and fixed-readout context transfer

Based on the preceding natural route/content study, commit `b1be92fc00c231533ffa69c9c69d81d5d43f3d01`. The full model is ByteDance/Ouro-1.4B Base revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. No weights were changed.

[中文报告](REPORT_ZH.md) · [Up preregistration](PREREGISTRATION.md) · [Independent context protocol](BODY_PROTOCOL.md) · [Figures](results/figures) · [Release verification](results/RELEASE_VERIFICATION.json)

The first experiment tests whether an actual K1 L1 attention write mediates K2 evidence through the Up component of L1 SwiGLU. Each recipient uses its real Gate and unchanged native suffix. A separate test swaps only the earlier part of the K1 residual stream before K2, preserving every K1 answer-scoring position bitwise. All raw paired effects, token likelihoods, output strings and input-deck hashes are included.

The upstream, context and matched-orientation cohorts contain 768, 384 and 288 non-overlapping questions. The last holds each token norm and angular distance constant. Position support and unrestricted greedy generation use fixed subsets of the first cohort. The open-text protocol differs from full-candidate likelihood. ARC IDs had whole-split historical project exposure, and the new assignment excludes all known earlier intervention decks without making a training-exposure claim.

## Reproduction

Place both inherited experiment directories in the same repository checkout. The frozen manifests in `results/` can be used directly. Raw source builders are available separately; selected OpenBookQA and ARC sources are hashed in the manifests. The model snapshot path, Python executable and a fresh output directory are the only arguments for replay.

```bash
bash reproduce.sh /path/to/python /path/to/pinned/model /path/to/new/output
sha256sum -c ARTIFACTS.sha256
```

