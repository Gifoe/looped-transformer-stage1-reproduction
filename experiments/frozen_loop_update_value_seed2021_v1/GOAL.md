# Discovering update value in frozen recurrent models

Identify an observable, answer-free signal that distinguishes beneficial from harmful recurrent readout updates. Prefer training-free inference; permit a small supervised head with the entire pretrained backbone frozen. A useful signal must improve actual held-out accuracy over Adaptive LoopCD and matched alternatives, including prompt ensembling. A failed experiment is a valid research outcome.

Historical ARC1172 and Qwen600 results are exposed evidence, never fresh confirmation. This directory is independent of their sealed manifests. Parent branch: `arc1172-qwen600-quality`, commit `06e40a3404f5f2f01418a52bad6942968a8c5fdf`; experiment branch `update-value-ouro14-20261029`.

Budget: at most 8 additional GPU-hours; cumulative cap 50 hours. Verified starting cumulative charge: 36265.481022056774 seconds. No unrelated generation job is resumed.
