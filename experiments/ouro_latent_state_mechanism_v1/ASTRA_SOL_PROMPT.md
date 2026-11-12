# Astra → Sol: Autonomous Mechanistic Discovery of Ouro Recurrent Latent States

## Scientific Goal

Explain from evidence, rather than from preselected narratives, what changes inside **the frozen original Ouro-1.4B** between recurrent iterations, especially K1→K2, that is associated with the large improvement of **ARC-Challenge Joint-Option multiple-choice accuracy**. Discover a falsifiable and ideally causal mechanism; a reproducible negative conclusion is preferable to an ungrounded positive claim. Accuracy optimization is not the goal.

## Division of authority

**Astra (scientific director):** Has broad authority to inspect source, read related mechanistic literature, define observables, propose/revise/retire hypotheses, direct Sol, select controls, change methods and code after documented evidence, and plan additional experiments. Astra should decide what to measure next, not simply execute a predetermined list. Maintain a hypothesis and decision log. Do not require fresh user approval for ordinary reversible study iterations.

**Sol (implementation and execution lead):** Executes Astra's commands on the new server, downloads the pinned model and data, modifies/implements code, maintains exact runtime/identity audits, runs data collection and interventions, reports all outcomes including negative outcomes, and publishes reproducible artifacts. Sol must not silently alter the scoring protocol or choose a test-set winner.

**Resource scope:** Inspect the actual **6.1 TB** data/storage resource before proceeding. If this refers to a mounted dataset, inventory its paths, formats, sizes, licenses and relevance; if it refers to available storage, allocate it as a storage budget for streamed and compressed latent-state captures. Do not assume ARC-Challenge itself is 6.1 TB; ARC examples are small, but full hidden-state traces can be much larger. Respect available GPU, disk quotas, experiment logs and security boundaries. No weight updates unless explicitly separately authorized for a probe; model weights remain frozen throughout the primary study.

## Prior evidence — use as motivation, NOT a fixed mechanism

Historical full ARC test, 25-shot Joint-Option, Ouro-1.4B:
- Original Joint K1: 634/1172; K4: 1021/1172.
- Cyclic Joint K1: 657/1172; K4: 1036/1172.
- Only the earlier **24-question development audit** observed K1=10, K2=20, K3=19, K4=20. It did NOT establish a full-dataset K1→K2 gain.
- Historic Experiment A and B examined ranking shifts, option-order consistency and limited candidate perturbations; neither identified a latent-state causal mechanism.

Read the historical reports and related papers before proposing novelty claims. Do **not** reproduce Experiment B's 24-question pilot as the new scientific benchmark. Technical smoke tests may be tiny but must not be reported as accuracy evidence.

## Phase 0: Start with real source and checkpoint

1. Inspect server: `nvidia-smi`, disk mounts and free space, CUDA, Python, network. Clone `https://github.com/Gifoe/looped-transformer-stage1-reproduction` branch `codex/ouro-candidate-intervention-20261105`. Copy this kit under `experiments/ouro_latent_state_mechanism_v1/`.
2. Read `README.md` and execute the actual scripts: `bash setup.sh`, `python download_checkpoint.py`, `python prepare_arc.py --out data`, `python verify_runtime.py --input data/validation.jsonl`, `python storage_plan.py --inputs data/train.jsonl data/validation.jsonl data/test.jsonl --output-mount . --capture current --save-head`.
3. Download **ByteDance/Ouro-1.4B**, pinned revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, verify official model.safetensors SHA256 `8872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af` and tokenizer/model source. Keep BF16, original SDPA, uncached/no explicit attention-mask path.
4. Establish a gold-free model forward/scoring contract and separate label-only analysis. Add identity/no-op tests of all layer hooks and confirm all 24 physical layers execute four recurrent times. Note that normalized loop-end states and pre/post physical-layer residual states are different measurement objects.

## Phase 1: Whole-dataset behavior and latent-space inventory — no preselected H1/H2/H3/H4

Perform complete native K1–K4 **joint-letter** scoring on the **entire ARC-Challenge train, validation and test splits**, with historical 25-shot examples and option mapping preserved. Every model forward is label-free. Inference records must include the full cohort, every native depth, raw letter LL, original-option content rank, tie handling, and input/model hash.

- Execute `collect.py` on `data/train.jsonl`, `data/validation.jsonl`, and `data/test.jsonl`, each in its own run directory, with `--capture current --save-head`; run `analyze.py` afterwards. Do not rely on any sample of only 24 or 64 as the scientific population.
- In train analysis, exclude or flag the exact 25 demonstration examples and audit duplicate/near-duplicate questions. The test split was used extensively in historical work: it may be described, but **cannot be advertised as a fresh blinded validation set** for newly chosen hypotheses.
- Confirm full-test K1/K4 endpoint parity against the historical reports. If parity fails, investigate tokenizer, model hash, sample order, attention code, shot construction and ties before analyzing latent-space results.
- Verify full-cohort K2/K3 performance. Report the actual distribution of pairwise K1→K2, K2→K3 and K3→K4 rescues, damages, stable-but-wrong transitions, reversals and per-question gain saturation. The assumed “K1→K2 gives most improvement” must be treated as a *question to test*.
- Acquire K1–K4 states at all 24 physical layers for every question, with complete current-question + option token spans and answer readout positions. Optionally extend to entire contexts/rotations if observations require it and disk/time forecasts permit. Write state tensors by shard; never load entire cohort tensors in GPU/RAM at once.

The initial report should be an observation map: Which transition, positions, layers, representations, distributional changes and cases behave differently? Include any counterexample clusters, not just large deltas.

## Phase 2: Let findings determine hypotheses

**Do not begin from a forced list of mechanisms.** Astra reads the full-cohort outputs and first identifies repeatable irregularities, contrasts and prediction transitions. Only then write concrete falsifiable hypotheses. Optional inspiration—not mandatory hypotheses—includes information movement among question/options, representations becoming readable by the LM head, candidate-label correspondence, score-scale effects, recurrence-specific interactions, optimization-like refinement, or a mechanism not previously considered.

For each new hypothesis, Astra states:
- The exact observation that motivated it, with cohort, position, layer/loop and numeric evidence.
- At least one competing explanation that fits the same observation.
- A minimal intervention whose possible results distinguish the explanations.
- A negative control and what will count as **falsification** or an inconclusive result.
- What would make the result meaningful beyond already-published Ouro and mechanistic-intervention papers.

Do not rank methods by how appealing a mechanism story sounds. An unexpected failure, local nonlinearity, or disagreement between representation and decision can be the next research question.

## Phase 3: Independent Astra → Sol iterative execution

For every iteration, Astra writes `orders/ROUND_N.md` with the observation, discriminating question, exact dataset/split, target states/positions, planned intervention, source/recipient identity, baseline, controls, metric, compute and disk forecast, expected outcomes and stopping criteria. Sol executes it, independently checks numeric integrity, and writes `results/ROUND_N.md`, raw or compact records, plots and failure cases. Astra then updates `DISCOVERY_LOG.md` and selects the next round.

Astra has broad authority to install tools, inspect Ouro internals, change observation granularity, introduce justified new probe designs, run layer/token/component-level activation/path patching, and select suitable **development-data** subanalyses *after* whole-cohort behavior is mapped; reduced causal-intervention cohorts must be stratified and verified on held-out cases. Existing `collect.py --patch-*` is a generic same-forward K1↔K2 physical-layer output hold primitive, **not** a complete causal-mechanism finding; use only after a defensible target has been selected.

If a hypothesis fails, DO NOT change thresholds, flip signs, cherry-pick layers, reuse the same gold data to select a winner or keep rewriting the explanation. Retire or restrict the hypothesis and examine the *failure pattern* instead. If **all current explanations fail**, Astra must explicitly return to the observation stage and may:

1. Test whether the supposed behavior was an evaluation/readout artifact by using identity checks and matched-format controls.
2. Change the observable scale—from output margins to latent subspaces, token groups, physical-layer transitions, attention/MLP paths or intervention response—without asserting the new observable is correct a priori.
3. Broaden the competing explanation space, including hypotheses not named in this prompt; propose tests that separate at least two interpretations, not merely a correlation proxy.
4. Use meaningful negative or synthetic controls, different question families, option permutations or an additional independent dataset when necessary; maintain the same frozen Ouro checkpoint.
5. Report a concrete negative discovery or identifiability boundary and halt a particular line when further experiments are not informative.

Astra should always ask **“What observation would force us to change the scientific question?”**, and is free to replace the original tentative mechanism hypotheses accordingly. Avoid endless rounds of inconclusive correlation metrics.

## Phase 4: Causal validation and literature audit

When a candidate mechanism survives exploratory tests, validate a *necessary* intervention (remove/hold or disrupt the identified latent computation) and a *sufficiency or rescue* intervention (restore it without trivial final-answer copying). Include no-op, same-shape, unrelated-source, magnitude/position controls and sham patches when appropriate. Distinguish copying the correct terminal state from uncovering a process that causally forms it. Test fidelity on previously unseen question IDs and, if justified, a second benchmark.

For any mechanism claim, compare with actual published Ouro/state-patching/looped-transformer work. An attention heatmap, high probing accuracy, state-norm variation or large correct-answer margin is not itself a mechanism. Negative controls and predictive out-of-sample cases should precede strong causal wording.

## Persistence and deliverables

Maintain the following in the new experiment directory, committing reproducible source and small artifacts to a new Git branch (do not commit checkpoint weights, terabytes of tensors, secrets or private files):

- `REPRODUCTION.md` — setup, pinned checkpoint, dataset, prompt, exact numerical parity.
- `FULL_COHORT_TRAJECTORIES.md` — K1..4 transitions, latent states, full sample coverage, and meaningful effect sizes.
- `DISCOVERY_LOG.md` — observation → hypothesis → falsification → results → update, including all rejected hypotheses.
- `orders/ROUND_N.md`, `results/ROUND_N.md` — Astra/Sol actual execution contracts and outputs.
- `CAUSAL_EVIDENCE.md` — interventions, shams, cross-split results and limitations.
- `FINAL_RESEARCH_REPORT.md` — the best surviving explanation or explicit negative/identifiability finding, direct prior-work comparison, figures, and an actionable next step.

No arbitrary number of experiments is required. Iterate until evidence yields a robust, novel, falsifiable mechanistic insight **or** the remaining uncertainty is clearly articulated and the next experiment no longer justifies its cost.

**EXECUTE NOW:** As Sol, start with Phase 0: inspect server, clone repo, install a compatible environment, download and SHA-verify the correct Ouro-1.4B checkpoint, load the exact fixed25 Joint-Option template, prepare all official ARC-Challenge splits, forecast whole-cohort latent capture size, and begin whole-cohort K1–K4 collection. Astra takes over scientific decisions after reviewing the first complete behavior/latent inventory. Do not start with Experiment B's 24-question rerun or a prewritten favored hypothesis.
