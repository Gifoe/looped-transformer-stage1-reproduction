# Primary-source check —28 October2026

This is a scoped novelty check, not a complete literature survey. The selector experiments do not reproduce these papers or compare their reported scores across incompatible protocols.

| Work | Verified primary source | Relevant overlap | Difference from this experiment |
|---|---|---|---|
| Think-at-Hard (TaH), v4,28 August2026 | [Paper](https://arxiv.org/abs/2511.08577v4), [author code](https://github.com/thu-nics/TaH) | Identifies harmful extra latent iterations and learns when to iterate. | Uses a neural decider, depth-aware LoRA and duo-causal attention. Here the Qwen weights and K6 intervention stay fixed; a small supervised linear selector sees completed A0/A3 outputs. |
| TaH2,28 September2026 | [Paper](https://arxiv.org/abs/2609.35748v1), [author project](https://fuvty.github.io/thinking_yard_project_page/projects/tah2/) | Directly asks whether more iteration improves a prediction; lookahead depth supervision labels iteration gains. | Jointly post-trains backbone and iteration decider. Here only a CPU linear rule is fitted on development labels, with no backbone training and no measured online compute sharing. |
| PoLar, v2,8 August2026 | [Paper](https://arxiv.org/abs/2606.06574v2), [author code](https://github.com/tianyi-lab/PoLar) | Learns input-dependent programs that skip or repeat pretrained layers; aims for quality/compute gains. | Learns routing programs over a richer execution space. Here the only choice is fixed Original versus fixed Euler after evaluating both. |

The broad idea “predict whether extra recurrent computation helps” is already established. Freezing the backbone and reducing the predictor to a linear model define a constrained empirical question, not sufficient novelty by themselves. No current result demonstrates a favorable frontier over these methods. Their end-to-end training, checkpoints, token-level decisions and generation benchmarks differ from this600-question choice-scoring audit; numerical head-to-head claims would be invalid.
