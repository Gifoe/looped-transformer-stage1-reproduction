# CATC loop-control pilot

Completed original CATC study, Outcome C: no demonstrated improvement over native K4 and strong matched output baselines. Read FINAL_SCIENTIFIC_REVIEW.md for the scientific decision and GENERALIZATION_RESULTS.md for every evaluated method.

The study contains Train1094 x 31 fixed/control configurations, four bounded scalar fitting runs, full ARC Validation299, independent SciQ997, and separately matched Standard FullAnswer Validation299. Original Ouro weights remain frozen. ARC Test was not used.

Code is in code/, fitting and frozen evaluation artifacts in phaseC/. Follow REPRODUCE.md for two CPU statistical replays. Raw full-vocabulary tensors and Q/K/V witnesses remain on the server with hash manifests. The publication contains complete per-question scalar results and compressed, losslessly sharded archives.

Schema: directly_changed_K2_R_sites uses zero-based physical layer indices and records attempted writes, which can be unchanged after quantization. Norm fields are [requested_preclip_L2, cap_L2, total_scale, actual_BF16_writeback_L2, prepatch_SDPA_L2]. Attention probabilities normalize over the full causal sequence before region slicing. Native K2 endpoints are captured during full four-pass execution; no early-exit timing benefit is claimed.

SciQ: Welbl, Liu and Gardner (2017), Crowdsourcing Multiple Choice Science Questions; AllenAI dataset, CC-BY-NC 3.0. Only the Validation split was read. Supporting paragraphs were omitted; original ARC demonstrations were retained. Dataset archive and evaluated-input hashes are recorded.
