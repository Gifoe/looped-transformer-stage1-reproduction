# External benchmark attribution

HellaSwag: Zellers, Holtzman, Bisk, Farhadi and Choi (ACL 2019),
[HellaSwag: Can a Machine Really Finish Your Sentence?](https://aclanthology.org/P19-1472/).
Source: [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag),
revision `218ec52e09a7e7462a5400043bb9a69a41d06b76`, full validation split 10042.
The pinned [dataset card](https://huggingface.co/datasets/Rowan/hellaswag/blob/218ec52e09a7e7462a5400043bb9a69a41d06b76/README.md)
identifies MIT licensing. On 2026-11-04 the GitHub connector returned HTTP 451
for the upstream repository, identifying a DMCA access block (dated 2026-09-14).
The release therefore does not duplicate HellaSwag question/continuation text,
raw parquet, rendered inputs, or reversible token IDs. Full originals remain on
the experiment server. Published derived scores include stable IDs, lengths,
input hashes and fixed source revision for exact data alignment. This records
the observed access status, without making a legal determination about the dataset.

OpenBookQA: Mihaylov, Clark, Khot and Sabharwal (EMNLP 2018),
[Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering](https://aclanthology.org/D18-1260/).
Source: [allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa),
revision `388097ea7776314e93a529163e0fea805b8a6454`, main configuration, full test split 500.
The [upstream repository license](https://github.com/allenai/OpenBookQA/blob/main/LICENSE)
is Apache 2.0; an unchanged copy is retained as `slcd_openbookqa_LICENSE.txt`.

Changes: official lm-eval 0.4.9 task preprocessing and zero-shot rendering;
addition of stable IDs, prompt/input/token hashes, complete candidate likelihoods,
character-normalized ranking scores, and paired-result metadata. No question,
continuation or label is intentionally altered beyond official preprocessing.
Pinned raw parquet/source SHA receipts and official YAML/utils SHA values are in
`configs/slcd_R6_EXTERNAL_DATA.json` and `data/slcd_*_raw_SOURCE.json`.

The benchmark licenses concern their respective data, not the Ouro model or
other code. Model weights are not redistributed. ARC inputs/adaptations retain
the separate CC BY-SA 4.0 attribution in `DATA_SOURCE_NOTICE.txt`.
