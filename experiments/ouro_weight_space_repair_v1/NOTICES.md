# Source and data notices

- Ouro-1.4B checkpoint and custom code: ByteDance/Ouro-1.4B, revision574fa66cb8bf5abdc979642d01cf2b79b16bfab1. Hugging Face model metadata declares Apache-2.0. The original weights are preserved. [Model](https://huggingface.co/ByteDance/Ouro-1.4B)
- ARC-Challenge: Allen Institute for AI, allenai/ai2_arc, revision210d026faf9955653af8916fad021475a3f00453. Dataset metadata declares CC-BY-SA-4.0. Question text and labels retain their original attribution/license. [Dataset](https://huggingface.co/datasets/allenai/ai2_arc)
- Evaluation construction follows EleutherAI lm-evaluation-harness0.4.9. The new fixed25-demonstration policy is explicitly user-authorized and differs from the previous advancing per-query sampler. [Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.9)
- Existing LoopCD scoring logic is reused from the user's reproduction repository; mathematical formulas, precision and candidate scoring are recorded in the report. [Repository](https://github.com/Gifoe/looped-transformer-stage1-reproduction)

Metadata snapshots and pinned LFS hashes are in sources/. Scientific output does not redistribute an edited model unless an edit passes the stated causal and held-out gates.
