# Complete small-result index

Every per-question native and intervention JSON is retained byte-exactly, including source/input/model hashes, scores/logits, control donors and runtime metadata. Activation arrays remain on the server; see [tensor manifest](TENSOR_MANIFEST.json).

| Run | Native records | Intervention records |
|---|---:|---:|
| confirmation_base_full128_v1 | [128](raw_jsonl/confirmation_base_full128_v1.native.jsonl.gz) | [128](raw_jsonl/confirmation_base_full128_v1.results.jsonl.gz) |
| confirmation_base_technical2_v1 | [2](raw_jsonl/confirmation_base_technical2_v1.native.jsonl.gz) | [2](raw_jsonl/confirmation_base_technical2_v1.results.jsonl.gz) |
| confirmation_query_full128_v1 | [128](raw_jsonl/confirmation_query_full128_v1.native.jsonl.gz) | [128](raw_jsonl/confirmation_query_full128_v1.results.jsonl.gz) |
| confirmation_query_technical2_v1 | [2](raw_jsonl/confirmation_query_technical2_v1.native.jsonl.gz) | [2](raw_jsonl/confirmation_query_technical2_v1.results.jsonl.gz) |
| confirmation_r2_full128_v1 | [128](raw_jsonl/confirmation_r2_full128_v1.native.jsonl.gz) | [128](raw_jsonl/confirmation_r2_full128_v1.results.jsonl.gz) |
| confirmation_r2_technical2_v1 | [2](raw_jsonl/confirmation_r2_technical2_v1.native.jsonl.gz) | [2](raw_jsonl/confirmation_r2_technical2_v1.results.jsonl.gz) |
| confirmation_r3_full128_v1 | [128](raw_jsonl/confirmation_r3_full128_v1.native.jsonl.gz) | [128](raw_jsonl/confirmation_r3_full128_v1.results.jsonl.gz) |
| confirmation_r3_technical2_v1 | [2](raw_jsonl/confirmation_r3_technical2_v1.native.jsonl.gz) | [2](raw_jsonl/confirmation_r3_technical2_v1.results.jsonl.gz) |
| confirmation_r4_full128_v1 | [128](raw_jsonl/confirmation_r4_full128_v1.native.jsonl.gz) | [128](raw_jsonl/confirmation_r4_full128_v1.results.jsonl.gz) |
| confirmation_r4_technical2_v1 | [2](raw_jsonl/confirmation_r4_technical2_v1.native.jsonl.gz) | [2](raw_jsonl/confirmation_r4_technical2_v1.results.jsonl.gz) |
| depth3_mediation_technical2_v1 | [2](raw_jsonl/depth3_mediation_technical2_v1.native.jsonl.gz) | [2](raw_jsonl/depth3_mediation_technical2_v1.results.jsonl.gz) |
| depth3_mediation_validation299_v1 | [299](raw_jsonl/depth3_mediation_validation299_v1.native.jsonl.gz) | [299](raw_jsonl/depth3_mediation_validation299_v1.results.jsonl.gz) |
| depth4_mediation_technical2_v1 | [2](raw_jsonl/depth4_mediation_technical2_v1.native.jsonl.gz) | [2](raw_jsonl/depth4_mediation_technical2_v1.results.jsonl.gz) |
| depth4_mediation_validation299_v1 | [299](raw_jsonl/depth4_mediation_validation299_v1.native.jsonl.gz) | [299](raw_jsonl/depth4_mediation_validation299_v1.results.jsonl.gz) |
| mediation_pilot64_v1 | [64](raw_jsonl/mediation_pilot64_v1.native.jsonl.gz) | [64](raw_jsonl/mediation_pilot64_v1.results.jsonl.gz) |
| mediation_technical2_v1 | [2](raw_jsonl/mediation_technical2_v1.native.jsonl.gz) | [2](raw_jsonl/mediation_technical2_v1.results.jsonl.gz) |
| mediation_validation299_v1 | [299](raw_jsonl/mediation_validation299_v1.native.jsonl.gz) | [299](raw_jsonl/mediation_validation299_v1.results.jsonl.gz) |
| pilot64_full_v1 | [64](raw_jsonl/pilot64_full_v1.native.jsonl.gz) | [64](raw_jsonl/pilot64_full_v1.results.jsonl.gz) |
| query_pilot64_v1 | [64](raw_jsonl/query_pilot64_v1.native.jsonl.gz) | [64](raw_jsonl/query_pilot64_v1.results.jsonl.gz) |
| query_technical2_v1 | [2](raw_jsonl/query_technical2_v1.native.jsonl.gz) | [2](raw_jsonl/query_technical2_v1.results.jsonl.gz) |
| query_validation299_v1 | [299](raw_jsonl/query_validation299_v1.native.jsonl.gz) | [299](raw_jsonl/query_validation299_v1.results.jsonl.gz) |
| technical_full_v1 | [2](raw_jsonl/technical_full_v1.native.jsonl.gz) | [2](raw_jsonl/technical_full_v1.results.jsonl.gz) |
| validation299_full_v1 | [299](raw_jsonl/validation299_full_v1.native.jsonl.gz) | [299](raw_jsonl/validation299_full_v1.results.jsonl.gz) |

Use `python results/code/recover_small_results.py --release . --output recovered` to restore exact JSON paths and validate every original SHA256. Then `python results/code/recompute_scalar_results.py --recovered recovered --labels inputs/all_labels.jsonl.gz --output scalar_results.csv` recomputes all four gap families, exact raw-logit signs and decision changes without a model or tensors.

Exact state/SDPA tensor audits require the separately listed tensor files. The full original runtime tree is retained at `/root/autodl-tmp/ouro-e4-e5/repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112`. Credentials are not published.
