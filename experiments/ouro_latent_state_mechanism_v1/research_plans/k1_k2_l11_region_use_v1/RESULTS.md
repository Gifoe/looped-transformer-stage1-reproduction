# L11 receiver-region use: completed bounded study

Frozen fresh-confirmation outcome: **FAIL_FROZEN_CONFIRMATION_GATE**. Development selected mask 3 (Q + O, no direct R write). Overall mechanism-identification goal remains ACTIVE.

## Populations and operator

512 previously exposed development worlds / 16 basegroups and 2048 new confirmation worlds / 64 basegroups. Each group contains all32 first-hop, endpoint, selector, label, record-order worlds. Fresh seed2026111743; hash-disjoint from all2560old R worlds and2560same-binding C rewrites. Within-family new combinations, not an OOD or pretraining-unseen claim.

One K2/L11 event. The semantic intervention adds the ideal FP64 paired-message delta to the actual BF16 original SDPA base, casting once. It is not native BF16 K/V recomputation. Whole G records and all attention heads. Q is question text before first option; O includes all options and answer-format text except the final R token. O does not isolate only cabinet/label tokens.

Random direction uses the same full-H shape/seed across masks, with actual BF16 semantic row doses and zero target on closed rows. Closed immediate rows are bitwise unchanged. Parent V1 permutation-null failure remains preserved in the earlier release; self/construction identities are not empirical evidence.

## All development candidates

|mask|regions|F0 E0|F0 E1|F1 E0|F1 E1|min retention|eligible|
|---|---|---:|---:|---:|---:|---:|---|
|1|Q|0.065674|0.060303|0.048584|0.062500|0.286744|False|
|2|O|0.069336|0.091553|0.108887|0.063232|0.461676|False|
|3|Q+O|0.142822|0.141113|0.140137|0.118164|0.827089|True|
|4|R|0.014404|0.016357|0.032959|0.015137|0.101724|False|
|5|Q+R|0.078125|0.069580|0.064209|0.070312|0.378963|False|
|6|O+R|0.077881|0.103516|0.123291|0.085938|0.550000|False|
|7|Q+O+R|0.141602|0.157715|0.169434|0.136963|1.000000|True|

Effects are Y_semantic minus Y_random, where Y is opposite-endpoint raw logit minus mean of the two decoy logits. Retention concerns this specified intervention effect, not percentage of reasoning or accuracy. Development choice is not independent confirmation.

The selected Q+O candidate has no direct L11 R write. This prevents direct-at-that-event readout injection; it does not establish a unique later attention path or composition algorithm. Joint selection does not prove each region necessary. Descriptive development QO nonadditivity is small and mostly negative (see DEVELOPMENT_NONADDITIVITY.json); additive contributions can explain the joint threshold crossing.

## All 18 fresh contrasts

100000 paired basegroup bootstrap replicates, seed2026111745, Bonferroni18 simultaneous two-sided95% intervals; every32-world group retained.

|contrast|mean|lower|upper|
|---|---:|---:|---:|
|E_selected_F0_E0|0.117065|0.085686|0.151550|
|E_selected_F0_E1|0.138550|0.107300|0.171997|
|E_selected_F1_E0|0.147705|0.115601|0.182922|
|E_selected_F1_E1|0.116333|0.086670|0.149780|
|retention_F0_E0|0.009888|-0.002309|0.021814|
|retention_F0_E1|0.016431|0.004003|0.028640|
|retention_F1_E0|0.018115|0.005773|0.030457|
|retention_F1_E1|0.006909|-0.006213|0.019971|
|E_full_F0_E0|0.133972|0.100396|0.170776|
|E_full_F0_E1|0.152649|0.119324|0.188232|
|E_full_F1_E0|0.161987|0.126282|0.201660|
|E_full_F1_E1|0.136780|0.104546|0.174018|
|Y_selected_sem_vs_native|0.129868|0.104319|0.156454|
|G_selected_sem_vs_native|0.223785|0.194763|0.255436|
|native_K2_accuracy|0.750977|0.701172|0.796875|
|native_K2_minus_K1_accuracy|0.330078|0.270996|0.387695|
|E_selected_label0|0.129700|0.094815|0.166145|
|E_selected_label1|0.130127|0.097351|0.164188|

Frozen tests:
- E_selected_F0_E0: False
- E_selected_F0_E1: True
- E_selected_F1_E0: True
- E_selected_F1_E1: False
- E_full_F0_E0: True
- E_full_F0_E1: True
- E_full_F1_E0: True
- E_full_F1_E1: True
- Y_selected_sem_vs_native: True
- G_selected_sem_vs_native: True
- E_selected_label0: False
- E_selected_label1: False
- retention_F0_E0: False
- retention_F0_E1: True
- retention_F1_E0: True
- retention_F1_E1: False
- native_K2_accuracy_observed_atleast_0.70: True
- native_K2_gain_lower_positive: True
- all_eight_selector_positive: True
- selected_dose_QC: True
- full_dose_QC: True

A failed overall gate is retained even if some component contrasts are positive. No layer, mask, population or cutoff may be changed on the confirmation set.

## Interpretation limits

Q actual states/KV and immediate writes are bit-identical across future label changes in the audited semantic trajectories. This supports a pre-label computation boundary; Q can already encode the terminal cabinet. It does not prove an uncombined relation representation. The region intervention does not resolve where the first-hop relation is stored, whether G locally composes both relations, or whether a pure box/object/endpoint address exists. No attention-head/coefficient search, training, weight update, native-correct selection or ARC Test was performed.

Astra designed and scientifically reviewed the actual source. Sol implemented, ran GPU execution and did vector/numeric self-audits and independent arithmetic code replay. Astra did not independently replicate the model or numerical oracle.

## Reproduction and artifacts

Run python code/replay_region.py development, then python code/replay_region.py confirmation (NumPy1.26.4 for confirmation). Public compressed scalar archives include all7 development masks and every fresh condition. Shards are reassembled and hash-checked in memory. Full49152-dimensional BF16 heads and technical actualQKV/SDPA witnesses are retained on the authorized server and indexed in RAW_TENSOR_MANIFEST.json. A scalar replay does not replace a neural replication.

Original inference code preserves actual server paths. GPU replication requires the frozen parent releases (commits b743862f258db3a51a98d554200ef7327f277ae7 and12621805b8ffe9d274f01ff8fa8d7fd5f413d0e8), the gated original checkpoint, and restoration of their recorded folder layout. Hash manifests document all inherited inputs/source/anchors. The stdlib development replay and NumPy confirmation replay are portable within this release.

Technical36 forwards /72 heads /24 active-event witnesses passed mask0-native and mask7-parent bitwise bridges and independent NumPy full-KV oracle. The technical label check covered24world-conditions=12paired comparisons; full science label-pair checks are counted in the two full-vector audit receipts. An initial manifest-schema preflight failed before model loading; its logs and repair note are preserved.

Current study is complete; the full scientific goal is unresolved. Routing/retention alone cannot be upgraded to identification of the relation-composition computation.
