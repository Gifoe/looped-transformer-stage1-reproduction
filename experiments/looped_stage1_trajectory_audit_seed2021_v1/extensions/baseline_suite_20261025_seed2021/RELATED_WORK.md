# Literature scope update (2026-10-25)

These are primary-paper claims and scope comparisons, not our experimental
measurements. The frozen experiment suite is not expanded after inspecting its
test predictions.

| Primary source | Relevant overlap | Consequence for our inference |
| --- | --- | --- |
| [Training-Free Looped Transformers, 2605.23872](https://arxiv.org/html/2605.23872v1) | Damped/anchored inference-time reuse of a frozen middle-layer block | Merely observing naive-loop degradation or better damping does not establish a new problem. Author-source availability still limits our direct reproduction. |
| [DoLa, 2309.03883](https://arxiv.org/abs/2309.03883), [official source](https://github.com/voidism/DoLa/tree/805230e57e63ca561cb759994681b122ff6f81f0) | Contrasts mature/premature layer predictions | A useful established training-free decoder control. Our Qwen adaptation is tested; it does not reproduce the original LLaMA benchmark suite. |
| [LoopCD, 2610.02185](https://arxiv.org/html/2610.02185v1) | Contrasts final and early recurrent predictions, including adaptive guidance | Finding that readout changes improve accuracy is already covered by decoding controls. A continuation-aware claim must survive those controls rather than rename them. |
| [Adaptive Depth, 2607.20519](https://arxiv.org/html/2607.20519v1) | Validation-calibrated stopping signals and accuracy/compute tradeoffs | Imperfect confidence/KL stopping is not by itself novel. Our policy grids and actual-exit probes address its overlap within the chosen checkpoints. |
| [Per-Token Fixed-Point Convergence, 2607.14427](https://arxiv.org/html/2607.14427v1) | Reports heterogeneous token convergence and training-free output-stability allocation in a depth-elastic 135M-class model | The paper restricts scale/seed and distinguishes average-depth proxies from realized speed. Its results cannot simply be transferred to four-loop Ouro or used to equate small KL with correctness. |
| [FlashLoop, 2609.29812](https://arxiv.org/html/2609.29812v1) | Training-free token-sparse updates, sparse attention and cross-loop KV residual quantization exploit recurrent redundancy | Low state change and reusable cross-loop computation are already explicit research objects. Our dense BF16 stopping measurements do not evaluate its sparse kernels/KV compression. It is contextual prior art, not a reproduced baseline in this suite. |
| [LoopSpec, 2609.17184](https://arxiv.org/html/2609.17184v1) | Uses early recurrent states to draft tokens, pipelines target verification and adds selective deeper proposals; claims lossless greedy/sampling decoding | Intermediate-state usefulness for drafting is already exploited while retaining target verification. Our no-cache MC stopping does not reproduce speculative generation or establish lossless decoding. This is contextual prior art only. |

The final Go/No-Go assessment must separate observations of reversals, uncertainty
signals and approximate computational redundancy from a causal claim about
preserving future corrections. A new continuation module is not justified just
because one of these established mechanisms fails on a small benchmark subset.
The empirical evidence and proposed minimum intervention belong in FINDINGS.md,
after all configured experiments and the independent completion audit finish.
