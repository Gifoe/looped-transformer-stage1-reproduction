# Handoff: full cohort not run

Project: /home/luansiyu/loop-latent; repository: repo, research branch research/ouro-latent-mechanism-20261107. The checkpoint at model/model.safetensors has locally and remotely verified SHA256 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af; the supplied task text omitted its leading 5. The requested source package was supplied later, extracted unchanged and documented in SOURCE_PACKAGE_AUDIT.md; the automated runner uses the separately audited project implementation, not the package scripts. ARC split files are in data, and conda-env is project-local.

GPU at last check: 48,446 MiB used, 115 MiB free; another user's PID 82277 is serving VLLM. Do not stop or compete with it. No GPU gate, resource probe, full-cohort scores, activation cache, Observation Map, or causal findings exist. CPU tests cover synthetic hook/identity/scorer cases, manifest and shard integrity, behavior statistics, and representation-statistic edge cases; they do not establish GPU parity.

Frozen full input manifests include all 10,357 requests; validation and target data are both hash-bound. Gate and runner share the production request path; gate compares unmodified historical joint request and encoding, full positions, baseline, independent K, identity, full payload BF16 roundtrip and suffix causality, with exact-zero tolerance. Scripts are prototypes and have not completed GPU validation. Small validation subset results are not a full evidence census. Gate JSON/evidence are ordinary user-writable files: hashes detect accidental drift, not adversarial tampering. Coverage must complete before descriptive analysis.

From /home/luansiyu/loop-latent, after verifying nvidia-smi and that other jobs are absent or genuinely unaffected, set:

```bash
export HF_HOME=$PWD/cache/hf HF_HUB_OFFLINE=1
R=repo/experiments/ouro_latent_state_mechanism_v1
conda-env/bin/python "$R/resource_probe.py"
```

Inspect results/GPU_RESOURCE_PROBE.json, actual peak, complete-question RAM/disk, and residual disk space. The probe samples RSS at steps; it is not a continuous RAM upper bound. No full run should start if resources or parity are inadequate. Then run and inspect per-candidate GPU parity evidence for each target split:

```bash
for split in train validation test; do
  conda-env/bin/python "$R/gpu_parity_gate.py" --split "$split" --count 3 || break
done
```

Each gate is based on actual validation questions selected deterministically across 3/4/5 option structure and binds both validation and target data. An exact mismatch stops and writes failure evidence; do not widen tolerance after observing failures. Independently audit evidence, decoded token spans and full production path before any full run. Gate and probe reject GPU when free memory is under 40 GiB or 90% of device memory.

If and only if all checks and resource audits pass, one target split at a time:

```bash
split=validation
conda-env/bin/python "$R/cohort_runner.py" --split "$split" --gate "$R/results/GPU_GATE_${split}.json" --out "shards/$split"
conda-env/bin/python "$R/audit_coverage.py" --split "$split" --shards "shards/$split" --out "coverage_${split}.json"
conda-env/bin/python "$R/behavior_summary.py" --split "$split" --shards "shards/$split" --coverage "coverage_${split}.json" --out "behavior_${split}.json"
conda-env/bin/python "$R/representation_observation.py" --split "$split" --shards "shards/$split" --coverage "coverage_${split}.json" --out "representations_${split}.jsonl"
conda-env/bin/python "$R/summarize_representations.py" --split "$split" --input "representations_${split}.jsonl" --coverage "coverage_${split}.json" --out "representation_summary_${split}.json"
```

Repeat with split=train and split=test only after successful inspection and storage review. The batch is always one; positional arguments and attention mask are omitted for the native unpadded path, and actual token IDs/positions are recorded for replay. Option spans and labels are separate; tokens crossing character boundaries may appear in multiple roles. Representation summaries are descriptive, with equal question weights, not causal evidence. Test data has prior exposure; 25 demonstration IDs are in training and train summaries explicitly separate demonstration overlap.

Coverage defaults to same-GPU exact score replay: stored BF16 full-vocabulary logits are loaded to CUDA and processed by the original FP32 log_softmax/gather/sum sequence; scores must match exactly and the recorded GPU, PyTorch, CUDA, SDPA, BF16 and cache settings must match collection. CPU replay is exposed only as a synthetic test option and cannot admit behavior or representation analysis. GPU replay has not been executed or validated.

A bounded real-request CPU feasibility preflight exited without full inference: 128-token, one-loop BF16 SDPA took 67.33s on two threads; a coarse estimate for two four-loop passes on the 1,916-token request exceeded the 150s budget. This estimate is a stopping heuristic, not an observed runtime for the full request.

## User-authorized background execution

`ouro_watch.py` uses an OS flock for one project watcher and isolated run IDs under `/home/luansiyu/loop-latent/orchestration_runs/`; that lock cannot reserve GPU from other users. It waits until GPU free memory is at least 40 GiB and 90%, utilization below 10%, and no compute process appears, confirmed twice ten seconds apart. It also checks disk and RAM. Before and after every stage it rechecks frozen source/input/model hashes, current pip freeze and clean tracked/untracked repository state; any deviation or external GPU compute conflict stops the run. Parent termination cleans the run's own process group before releasing its lock. Resource/probe, gate, shards, coverage and reports all use the isolated run directory and must not overwrite existing outputs. The terminal success state is `OBSERVATION_READY`, meaning descriptive observations are ready for Astra, not a causal or mechanism result. See the active run's `status.json`, `provenance.json` and stage logs.
