# Source and numerical audit

Checkpoint/source verified before loading and after initialization; actually imported dynamic model/config files match the expected hashes. Loader is copied from the starting commit's prior package. Source/model artifacts are unchanged.

```json
{
  "checkpoint_files_sha256": {
    "model.safetensors": "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af",
    "config.json": "ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3",
    "modeling_ouro.py": "c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413",
    "configuration_ouro.py": "950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb"
  },
  "actual_model_source_sha256": "c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413",
  "actual_body_source_sha256": "c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413",
  "actual_config_source_sha256": "950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb",
  "torch": "2.5.1+cu124",
  "cuda": "12.4",
  "transformers": "4.55.0",
  "gpu": "NVIDIA GeForce RTX 4090",
  "dtype": "BF16",
  "attn_implementation": "sdpa",
  "cache": false,
  "total_ut_steps": 2,
  "gpu_uuid": "GPU-d4e36609-350a-a3be-5024-3eefc3ad01fc",
  "python": "3.10.8 (main, Nov 24 2022, 14:13:03) [GCC 11.2.0]",
  "full_vocabulary_head_shape": "[1,T,49152] before R slicing",
  "source_runtime_difference": "torch2.5.1+cu124 on this device; historical13580 torch2.7.1+cu128; endpoints never pooled"
}
```

Device is a separate RTX4090 from the13580 server. Runtime Torch2.5.1+cu124 differs from previous Torch2.7.1+cu128; no prior endpoint is pooled as a native reference. Current Train1030 K1/K2 counts558/857 and Validation148/243 differ slightly from historical560/859 and149/244. These are distinct numerical cohorts, not unexplained identity claims.

Technical gates: all three ARC full49152 native repeats, official forcedK1/K2 full-shaped interface endpoints, and per-layer native non-R replay agree bitwise. B repeats the short-sequence replay/official gate. Actual SDPA call observation verifies causal=True, mask=None, no sliding exclusion. All640 B future-only mapping changes preserve source B bytes, and all640 same-variable exchanges preserve the full final vocabulary vector. C requires original B native/source endpoints to match bitwise before its route interventions; zero attention delta is exact on four fixed technical units. Replay executes complete original full-shaped layers while overwriting non-R inputs; it is an identity check, not a speedup or mechanism result.

Independent retained-tensor CPU audit: 21723 A/B endpoints and3616 C endpoints. Full logits hashes, candidate logits, predictions and tie lists are exact. GPU FP32 log-probabilities agree with FP64 reference with maximum error7.51486125e-07. The initial CPU FP32 reference had a2.38e-5 reduction error; LOGSOFTMAX_AUDIT_AMENDMENT.md records why it was replaced with FP64. The exact-head gate was never relaxed. Main margins use differences of raw BF16 candidate logits and are independent of log normalization.

A record field originally called stage_R_sha256 always captures K2 stage vectors, even beside a K1 score record. Public scalar records rename it captured_K2_stage_R_sha256; K1 endpoint logits and true B are separately preserved. This is a metadata clarification, not changing a forward. Source versions and transformations are retained.

Native B serialization initially reused world/unit keys. The original tensor files were preserved; missing native boundary states were recollected from frozen inputs, requiring original K1/K2 fullendpoint SHA equality in every one of640 restorations. B_NATIVE_SERIALIZATION_REPAIR.json documents this repair, and the CPU audit cross-checks the recovered raw tensors. No scientific samples, outputs or criteria changed. A complete additional archive replication using the corrected serializer subsequently reproduces every one of6400 native/intervention endpoint records, candidate scores, predictions, ties and captured K2 R-stage hashes exactly; all full native B and patch tensors are retained in data/B_replicated. B_FULL_ARCHIVE_REPLICATION_PASS.json is the authoritative receipt. This is audit work beyond the primary science-cost envelope.

Large raw B/history/full-vocabulary tensors stay on this server under /root/ouro-computation-sol-20261116/data and technical; hash/size manifests accompany the scalar release. Model weights, private connection material, and historical raw packages are excluded from GitHub. Code, frozen inputs, per-question candidate scores/predictions/ties, numerical receipts and scalar CPU reproduction are published.
