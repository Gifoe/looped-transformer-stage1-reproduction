# V2 candidate builder audit

`scripts/build_candidates.py` implements the fixed 14 recipes in CANDIDATE_DESIGN.json. It does not load a model onto a GPU. It requires COMPLETE_384, two exact recording parity cases, the matching discovery DATA_LOCK hash, all 384 unique training score rows, and four finite FP32 statistic arrays per question. Character-normalized scores, predictions and gold margins are independently recomputed from the recorded candidate likelihoods before grouping.

The transition tie string is `V2-trend|2021|3to4|question_id` (and analogously 4to5 and 5to6). Bottom and top groups have 96 questions each, with deterministic hash tie breaking. All four correctness-transition categories are recorded both globally and inside each quartile. Directions use training labels through gold-margin grouping; they must not be described as label-blind or as proven harmful computations.

Means, directional contrasts, site energies, column ranking and SVD/QR use CPU FP64 reductions of stored FP32 statistics. Frozen U tensors are FP32 and must pass a Gram-matrix check. Rank4 includes the rank1 vector plus three nonzero residual principal directions. Numerical rank deficiency aborts; it does not silently reduce rank or add random directions. Repeated singular values can permit different equivalent bases on another numerical library, so the saved factors and their hashes, followed by exact BF16 materialized tensors, are the replay authority.

The site nondegeneracy threshold is direction norm > 1e-12 and positive transition energy. MLP sites are ranked over all 24 physical layers; attention is independently ranked over all 24. The column all-zero fallback is the specified activation-weight energy, and is explicitly recorded. The output-site statistics and writer ranking are heuristic forward correlates. Subsequent edited-model outcomes, not this ranking, determine empirical utility.

`candidate_io.build_candidate(candidate_spec, model)` returns WeightEdit objects. It verifies the frozen factor file, individual basis tensor and actual BF16 base parameter hashes. The runner remains responsible for no-grad/eval mode, applying via `applied_edits` with cache invalidation on both entry and exit, saving the selected exact edited tensors with `save_edit_bundle`, and selection/validation/test access ordering.

CPU-only synthetic verification passed: quartile disjointness and category totals; normalized direction; rank4 orthogonality and rank1 inclusion; sparse BF16 changed coordinates; energy fallback; rank-deficiency rejection; helper materialization; exact parameter restore and two invalidation callbacks; whole-column attenuation. Both modules compile. No real training statistics had completed locally at this audit, so full 384-file execution remains a required server check. No held-out outcomes were read and no GPU job was launched.

Only new V2 files were written. The builder never imports runtime/load_model or accesses V1 experiment results.
