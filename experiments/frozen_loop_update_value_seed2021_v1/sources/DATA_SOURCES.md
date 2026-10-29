# Data and model attribution

All questions, answer candidates and support passages remain attributed to their upstream datasets. This experiment adds fixed splits, rendered evaluation requests and model measurements; it does not claim ownership of benchmark text or change upstream licensing.

| Dataset | Pinned source | Split used |
|---|---|---|
| OpenBookQA | [allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa/tree/388097ea7776314e93a529163e0fea805b8a6454) | Official train; disjoint discovery/head-training/validation |
| SciQ | [allenai/sciq](https://huggingface.co/datasets/allenai/sciq/tree/2c94ad3e1aafab77146f384e23536f97a4849815) | Official test with its supplied support |
| PIQA | [baber/piqa](https://huggingface.co/datasets/baber/piqa/tree/142f6d7367fd9877f0fb3b5734ea6a545f54cdd1) | Official validation, as configured by lm-evaluation-harness0.4.9 |

Dataset metadata is preserved beside this file. SciQ's supplied metadata identifies CC-BY-NC3.0. The other saved cards do not provide a specific license value. The historical and new manifests record exact revisions, parquet hashes, question exclusions and task templates.

Checkpoint: [ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1). No model weights are redistributed here. Its pinned blob/LFS metadata and the independently computed hashes of nine checkpoint/tokenizer/code files are retained. Because the primary Hugging Face endpoint timed out from this host, the blob metadata was retrieved from hf-mirror.com at the same pinned revision; that transport is recorded explicitly.
