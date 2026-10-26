# Literature checked on 2026-10-23

- [Training-Free Looped Transformers](https://arxiv.org/html/2605.23872v1):
  controlled frozen-window reapplication, damped updates, and a separate canonical
  KV write pass. Prefill bypass differs from full autoregressive looping. Our
  fixed recipe follows the public author README, while the author code is 404.
- [LoopCD, October paper](https://arxiv.org/html/2610.02185v1): combines early and
  final logits, or pre-readout hidden states. Uses completed first-loop references.
  ARC uses 25 demonstrations for Ouro; our zero-shot audit is supplementary and
  cannot reproduce the paper's absolute table values. It already studies
  sharpening, close decisions, and harmful extrapolation.
- [Adaptive Depth](https://arxiv.org/html/2607.20519v1): separates trajectory
  formation from exit readout, comparing fixed depths and confidence/gate policies
  on independent threshold-selection splits. This substantially overlaps basic
  confidence-versus-depth diagnostics.
- [Per-Token Fixed-Point Convergence](https://arxiv.org/html/2607.14427v1): reads
  successive vocabulary KL for per-token allocation in a smaller trained model.
  Sample-level multiple-choice stopping is a different unit of adaptation.
- [TaH2](https://arxiv.org/html/2609.35748v1): lookahead depth supervision explicitly
  targets whether extra iteration improves prediction. A proposal merely to
  predict continuation benefit must be distinguished from this existing method.
- [Enabling Dynamic Computation](https://arxiv.org/html/2610.09013v1): addresses
  missing cache levels during actual token-level early exit with a best-available
  cache scheme. Offline depth savings alone do not settle that deployment problem.
- [Monitoring latent reasoning](https://arxiv.org/html/2610.18127v1): directly
  compares state settling, answer commitment, and which iteration reads out best.
  A new claim based only on their mismatch requires a more specific causal test.
- [FlashLoop](https://arxiv.org/abs/2609.29812): reduces redundant recurrent work
  through sparse updates/attention and cache compression. It is relevant to real
  systems efficiency, not evidence that an early answer is correct.

These are literature claims, not observations from this project's experiments.
