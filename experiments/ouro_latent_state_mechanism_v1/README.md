# Ouro-1.4B latent-state mechanism discovery — full-data starting kit

Purpose: **discover** why native recurrent K1→K2 changes the Joint-Option choice behavior; do not preselect content/binding/readout explanations as truths. This is a runnable **initial data acquisition + diagnosis instrument**, not a discovered mechanism or a self-driving agent API.

## Placement

Copy this directory to `experiments/ouro_latent_state_mechanism_v1/` inside `Gifoe/looped-transformer-stage1-reproduction`. The sibling `ouro_weight_space_repair_v2/data/slcd_fixed_demonstrations.json` is required for historical 25-shot prompt parity. Preserve all R1–R33 files.

## New GPU server

```bash
git clone --branch codex/ouro-candidate-intervention-20261105 https://github.com/Gifoe/looped-transformer-stage1-reproduction.git
cd looped-transformer-stage1-reproduction/experiments
# place this directory here, then:
cd ouro_latent_state_mechanism_v1
nvidia-smi; df -h; python3 --version
bash setup.sh
source .venv/bin/activate
python download_checkpoint.py
python prepare_arc.py --out data
python verify_runtime.py --input data/validation.jsonl
python storage_plan.py --inputs data/train.jsonl data/validation.jsonl data/test.jsonl --capture current --save-head
```

**Note on historical compatibility:** this kit pins the original checkpoint revision and score construction. Hardware and environment differences can still change numerical outputs; verify against the historical K1/K4 full ARC counts before interpreting trajectories. Original server code uses BF16, uncached attention and no extra explicit attention mask. `setup.sh` may require Astra to install the correct CUDA Torch build for this specific host.

## Full-dataset acquisition (not Experiment B)

If the new server's 6.1 TB resource is a writable disk mount, set `OURO_OUTPUT_ROOT` to that mount rather than using the system disk. For the complete sequential workflow (all rows, full K1–K4 native states, post-hoc analysis), run:

```bash
OURO_OUTPUT_ROOT=/path/to/large/mount/ouro_state_outputs bash launch_full.sh
```

Otherwise run the corresponding commands separately below. If 6.1 TB refers to an existing **dataset** rather than free disk, index it read-only with `python inventory_volume.py --root /path/to/data` before selecting any additional external tasks.

```bash
# Full train + validation for exploration; full test is historical/exposed, use for descriptive audit.
python collect.py --input data/train.jsonl --out outputs/train_native --capture current --save-head
python collect.py --input data/validation.jsonl --out outputs/validation_native --capture current --save-head
python collect.py --input data/test.jsonl --out outputs/test_native --capture current --save-head

python analyze.py --predictions outputs/train_native/predictions.jsonl \
    --labels data/train.labels.jsonl --input-manifest data/train.jsonl \
    --exclude-fewshot --out analyses/train_native
python analyze.py --predictions outputs/validation_native/predictions.jsonl \
    --labels data/validation.labels.jsonl --out analyses/validation_native
python analyze.py --predictions outputs/test_native/predictions.jsonl \
    --labels data/test.labels.jsonl --out analyses/test_native --expect-full-arc-test
```

This processes all rows of the official splits, with separate labels inaccessible to the inference code. The train analysis excludes direct copies of the 25 few-shot examples; the raw outputs still retain every row. No arbitrary 24/64-question scientific subsample is used.

To collect all cyclic rotations for the *entire* split (extra inference cost):

```bash
python collect.py --input data/train.jsonl --out outputs/train_cyclic --rotations cyclic --capture current --save-head
python analyze.py --predictions outputs/train_cyclic/predictions.jsonl \
  --labels data/train.labels.jsonl --out analyses/train_cyclic --input-manifest data/train.jsonl --exclude-fewshot
```

`--capture current` records all recurrent/physical-layer residual states for the current question and options, including answer readout positions, plus token-level raw BF16 bits and K1–K4 final-head logits. `--capture all` collects the entire 25-shot context for *every* physical layer and loop, consuming much more disk; Astra should first inventory the 6.1 TB data/storage resource and forecast footprint. `--capture answer` is much smaller. Collection is resumable and never reads gold. All outputs are under a frozen per-run config and cannot silently mix settings.

## Controlled causal intervention (only after Astra identifies a discriminating target)

This kit exposes a literal same-forward, same-layer, same-position recurrent-state hold/restore primitive. For example, to replace the output of physical layer 12 at K2 with its K1 value across current-question positions:

```bash
python collect.py --input data/train.jsonl --out outputs/train_patch_layer12 \
  --capture current --patch-layer 12 --patch-source-loop 1 --patch-target-loop 2 --patch-fraction 0
```

`--patch-fraction 1` is the identity control. Both require separate run directories. **Astra must predefine where/why to intervene and what controls will distinguish explanations**; copying whole states alone is not a proof of mechanism. The module `latent_hooks.py` can be extended to position-specific or component-specific interventions after observing full-data transitions, with identity and first-three-loop parity checks retained.

## Correctness requirements

- **No generation:** all candidate labels are teacher-forced full-vocabulary `log_softmax/gather/sum`, as in historical R10/R11. The value returned is raw log-likelihood, not answer-text length normalized.
- **No target leakage:** `collect.py` reads only gold-free `.jsonl`; labels live in a separate file and are opened only by `analyze.py`.
- **Model:** immutable `ByteDance/Ouro-1.4B` revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`; `model.safetensors` SHA256 `8872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.
- **Depth:** K1..K4 from one native 4-loop forward; each physical layer's invocation counter must equal four. K2 and K3 are actual native states, not LL interpolations.
- **Position:** layer snapshots are *post-layer*, while normalized loop-final states are captured separately; don't conflate the two.
- **Numeric validation:** compare K1/K4 full historical ARC counts (634/1172 and 1021/1172); review discrepancies before mechanism inference. Cyclic reference is 657 and 1036 with all rotations.
- **Data:** keep full official ARC train/validation/test separate. Existing ARC test has been used historically; use train/validation for new mechanism selection and freeze explanations before a new independent benchmark.

## Agent handoff

Read `ASTRA_SOL_PROMPT_ZH.md` (Chinese) or `ASTRA_SOL_PROMPT.md` (English) as the agent operating brief. Astra selects experiments, Sol runs code and returns audited results. The scripts provide real command-line capabilities, not a fictitious Astra/Sol tool or future scheduled service. No step requires editing the checkpoint.
