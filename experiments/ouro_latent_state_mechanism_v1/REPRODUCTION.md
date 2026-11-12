# Phase 0 status

- 2026-11-06 server local time: GPU RTX A6000 49140 MiB, 45994 MiB occupied by another user VLLM EngineCore; not interrupted.
- Root filesystem 916G, 492G available; lsblk found only 931.5G physical device, no 6.1TB volume.
- Server DNS cannot resolve github.com; direct huggingface.co access timed out.
- User attachment directory contains only Pasted text.txt, no code package; branch tree has no experiments/ouro_latent_state_mechanism_v1.
- Repository branch commit 70dd18f18ca9a2f93c95b80c39350427dbcd7688 verified from local host; Git bundle transfer to source.bundle in progress.
- No checkpoint verified, no dataset prepared, no inference performed. Numerical protocol remains unvalidated.

- Local download and SHA256 verification against repository LFS metadata for ARC revision 210d026faf9955653af8916fad021475a3f00453, then uploaded data/train.parquet (e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb), validation.parquet (395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05), test.parquet (62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9). Remote readback hash pending.
- Local parquet read: train 1119, validation 299, test 1172; each split IDs unique, zero cross-split ID overlap. Case-fold plus whitespace normalized question set overlaps: train/validation 0, train/test 6, validation/test 2. Historical fixed 25 demo IDs all appear in train. Preserve and track demo overlap; no silent exclusions.
- Historical r10_core joint template: Question, lettered choices, Answer:; fixed demos separated by double newline and answers/continuations prefixed by one space. runtime encodes via lm_eval 0.4.9 TemplateLM._encode_pair, tokenizer add_special_tokens=False; r10_worker np.argmax selects first max. Real numerical parity remains unverified.
- Existing historical requirements include torch CUDA13, not automatically compatible with driver 560.35.03 (CUDA 12.6); server base conda lacks torch.

- Hugging Face HEAD metadata at pinned checkpoint revision returned 2,869,336,434 bytes and linked SHA256 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. The task text gives a 63-character value missing the leading 5. Metadata is not a local file integrity check; actual checkpoint has not been downloaded or verified.
- Remote sha256sum readback of all three parquets matched source metadata.

- data/manifest.json records split IDs, source hashes, demonstration IDs, and cross-split normalized-question overlap IDs.
- Created independent Miniconda conda-env using classic offline solver; torch 2.5.1+cu124 reused from temporary venv and import checks CUDA and BF16 available. Historical torch CUDA13 differs; runtime parity unverified. Installing lm_eval 0.4.9 and transformers 5.12.0 in conda-env; see phase0_python_install.log.
- Eight model configuration, code and tokenizer files fetched from mirror and verified by git blob IDs against official fixed revision metadata.
- Checkpoint full file downloaded and hashed on local host: SHA256 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. SSH transfer stopped; remote partial stable at 18448384 bytes and first 1 MiB matched local hash 9be8fc9c6bee0bb089d1cb52c640b194cf94cbfff4d368073980d1d82bd06a65. Server curl PID 113250 continues same partial via mirror range; phase0_model_download.log and model.sha256 record progress and final verification.

## Updates after initial status

- Git bundle clone completed on research/ouro-latent-mechanism-20261107 at 70dd18f18ca9a2f93c95b80c39350427dbcd7688.
- Checkpoint is model/model.safetensors relative to /home/luansiyu/loop-latent; source metadata and local and server full-file hashes match 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. Task text contains only 63 hexadecimal digits, missing first 5; do not silently rewrite the task.
- Miniconda prefix conda-env uses Python 3.11, torch 2.5.1+cu124, transformers 4.55.0, lm_eval 0.4.9, datasets 3.6.0; pip check passed. 5.12.0 failed OuroConfig and lm_eval imports. CPU BF16 model load has no missing or unexpected tensors. GPU still occupied; no GPU parity claim.
- Prompt span and candidate encoding audit: train 1119, validation 299, test 1172; no span failures and every target single token. CPU_SPAN_AUDIT.json holds lengths. Demonstrations appear in train; benchmark is historically exposed.
- Native capture smoke: CPU only 2-token input, exact read-only hook K4 logits and independent K1-K3 head equality, norm-to-next-layer selected position continuity; no identity writeback, historical scorer comparison or GPU validation completed.
- Full cohort inference and state collection are NOT RUN. No mechanism claim. Missing code package, downstream collection/restore runner and full GPU numeric gates remain outstanding.

## 2026-11-11 full-cohort observation update (supersedes earlier NOT RUN status)

Executed source commit 33ff0322934bebf423c9f3d087f9c71b3ecaa982. Project-local Miniconda conda-env: Python 3.11, torch 2.5.1+cu124, transformers 4.55.0, lm_eval 0.4.9; NVIDIA RTX A6000. Pinned model ByteDance/Ouro-1.4B revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1, checkpoint SHA256 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af. Task text omitted its leading 5; metadata and actual bytes agree on the 64-character hash.

Supplied package archive SHA256 7c3aa9cffd10c6caac13df60e3dba5cb12b21e089707652e0205baa91fe84485 is unchanged. The executed audited runner is a separately developed implementation, not the package collector. ARC-Challenge revision 210d026faf9955653af8916fad021475a3f00453; train/validation/test 1119/299/1172. Fixed demonstrations SHA256 8cf4b70e7c07c67a589560d5fcbc1fa8feb85a0e7af3b348fa21f147271419da. BF16, SDPA, eval, no cache, labels=None, true K=1–4, native full-vocabulary head and FP32 log-softmax; first-index argmax.

All three validation-sample GPU gates and maximum question resource probe passed; each split coverage has zero missing/failed questions and zero-tolerance CUDA logits replay. Source/environment/model/data hash guards passed before resumption. Original run stopped during test on a resource-conflict condition whose specific trigger was not logged; 759 existing test shard pairs were SHA-audited and 413 resumed under global lock. No old shard was overwritten. Paths and SHA256 are in results/ROUND_1_OBSERVATION_INDEX.json. Historical test exact reproduction remains unresolved: K1 626 vs 634, K4 1020 vs 1021.
