# Literature and scope

## Training-free readouts

- LoopCD-Logits uses completed K1 and terminal logits to extrapolate, optionally using the top-two token probability margin. LoopCD-Hidden extrapolates before the final norm and projects once. Ouro-1.4B's reported ARC-C LoopCD experiment uses 25 demonstrations, while this study uses zero-shot. We implement the published equations as run-local baselines. [Decoding Looped Transformers Better for (Almost) Free, §3, B.2, and F.2](https://arxiv.org/html/2610.02185v1)
- Chen et al. loop a contiguous segment of a frozen non-recurrent model with damped updates. Their official code provides a different transformation than adapting the readout of Ouro's native recurrence. [Training-Free Looped Transformers](https://arxiv.org/abs/2605.23872) · [Official code](https://github.com/L-z-Chen/Training-Free-Looped-Transformer)
- A secant readout is a generic fixed-point extrapolation diagnostic. For consecutive increments (d_0,d_1), estimate their directional contraction by their inner product divided by the squared norm of (d_0). Clip the ratio to [0, 0.5], then add at most one increment to the terminal pre-norm state. The final vocabulary projection executes once. We test a same-size last-step variant and published LoopCD; a stable advantage would need independent confirmation. Fixed-point acceleration itself predates this study. [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377)

## Training and learned adaptation

- Loop Dropout studies stochastic loop-level regularization during training. LoopOPD uses on-policy self-distillation from deeper computations. Ordinary same-data output distillation is the relevant control for any future mechanism-supervised version. [Loop Dropout](https://arxiv.org/abs/2609.34218) · [LoopOPD](https://arxiv.org/abs/2610.10623)
- Gated Attention adds an input-dependent gating mechanism downstream of attention and studies training and sinks. Test-Time Training treats sequence-dependent updates as hidden state; applying test-time parameter updates would introduce compute outside the fixed-loop comparison. [Gated Attention](https://arxiv.org/abs/2505.06708) · [Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620)

## Reduced execution and mechanism

- FlashLoop studies token-sparse recurrent execution and cache optimization. A smaller nominal depth or a replayed routing budget still needs measured wall time on an actual execution path. [FlashLoop](https://arxiv.org/abs/2609.29812)
- Cross-Loop Query Handoff recovers a later reading operation on controlled two-hop tasks. Its evidence motivates a Q intervention and the separate K/V tests, while ARC naturally lacks a unique annotated source-to-mapping token path. This trial plants no answer-bearing token beyond each scoring position. Its K4 teacher is an offline oracle. [Cross-Loop Query Handoff](https://arxiv.org/abs/2610.34639)
- Mechanistic analyses and the checkpoint lineage supply controlled causal evidence. The new ARC experiment measures behavioral transfer to full choice text under a standard benchmark protocol. [Ouro](https://arxiv.org/abs/2510.25741) · [A Mechanistic Analysis of Looped Reasoning Language Models](https://arxiv.org/abs/2604.11791) · [Official code](https://github.com/HughBlayney/looped-mechanistic-analysis)

## Experimental provenance

- [Pinned repository parent](https://github.com/Gifoe/looped-transformer-stage1-reproduction/tree/f525239970b70c7ce28fe9fe91d95859653096cc/experiments)
- [Pinned AllenAI ARC split revision](https://huggingface.co/datasets/allenai/ai2_arc/tree/fdb98c79578d61bdf502e2ade3253c71154dd11f/ARC-Challenge)
- [lm-evaluation-harness ARC-C config](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/tasks/arc/arc_challenge.yaml) · [Inherited prompt templates](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/tasks/arc/arc_easy.yaml) · [Metric source](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.13/lm_eval/api/task.py#L1379-L1399)
