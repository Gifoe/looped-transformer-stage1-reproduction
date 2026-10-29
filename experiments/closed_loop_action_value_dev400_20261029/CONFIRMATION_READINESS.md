# Future confirmation candidates — not selected or evaluated

Only source documentation was consulted on 2026-10-29. No new benchmark examples, labels or model outcomes were loaded for this note. It does not alter the current 400-question protocol or authorize treating old final tasks as untouched data.

- [AllenAI ARC dataset card](https://huggingface.co/datasets/allenai/ai2_arc/blob/main/README.md) lists 2,376 ARC-Easy test questions. ARC-Challenge test is already exposed. ARC-Easy could be considered only after question-ID and normalized-text overlap checks against all historical manifests, with explicit exclusion counts and pinned dataset revision.
- [Tel Aviv University CommonsenseQA dataset card](https://huggingface.co/datasets/tau/commonsense_qa/blob/main/README.md) lists 1,221 validation questions. That split is a possible independent benchmark only after exposure and scoring-configuration audits. Its public test split must not be assumed to have usable gold labels.

`scripts/build_exposure_index.py` builds a question-only hash index from the original experiment's historical exposure scan and all five first-round manifests. It never accesses gold fields or new examples. Exact normalized equality is not semantic deduplication and cannot establish freedom from model pretraining contamination.

Before any future confirmation, lock one controller, the official task scoring conventions, baseline implementations, paired hypotheses, compute accounting and all exclusion rules. The appropriate sample size depends on observed paired event rates; nominal benchmark size alone is not a power justification. These source counts are capacity planning, not a commitment to run a candidate that fails development.
