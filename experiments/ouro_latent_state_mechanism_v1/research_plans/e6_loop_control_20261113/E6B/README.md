# E6-B: natural latent update utility

Joint frozen Test: native K4 1022/1172; primary latent ZL rollback and Validation-calibrated latent rollback 1018/1172. Both tie their matched output-only controls in correct count. Natural features did not improve the final accuracy/compute frontier. Full-answer no-refit transfer is reported separately. Sol performed all E6 analysis/execution/review; Astra was later authorized separately for next-step direction only.

## CPU replay

With Python3.10+ and NumPy, from this directory run:

```sh
python E6B_REPLAY_RELEASE.py .
```

This checks all11720 frozen Joint Test controller decisions (prediction, selected depth, executed loops) and all10 frozen supervised diagnostic macro-F1 values. It reads released low-dimensional features and Train-fit coefficient NPZ, performs zero fits, and needs neither CUDA nor model weights. Labels are isolated evaluation data. The coefficient NPZ modifies no Ouro model parameter.

## Evidence and source layout

`E6B_FROZEN_SPEC.json` retains Train/Validation source hashes, policy thresholds, separate ZL/Cal permissions, selected diagnostic alphas and coefficient hash. `E6B_TEST_RECEIPT.json` binds the single frozen Test evaluation. `E6B_FEATURES.jsonl` contains all2565 non-demonstration natural observations with candidate scores and legal prefix features. `E6B_CONTROLLER_RESULTS.jsonl` retains Test decisions; development decisions are separate. FAS files have the `E6B_FAS_` prefix and use an independent prompt/character-normalized scoring protocol.

The repository sibling `shared_capture/` contains the complete original Joint/FAS score-and-feature capture exports and their SHA manifests, assembled by the E6A coordinator. Compressed high-dimensional R-only NPZ states remain on the experiment server; no full-token states are released. The frozen CPU evaluators load per-question JSON plus `IDENTITY.json` from the original capture directory layout; released decision replay above needs only this E6B directory.

`E6B_NATURAL_FEATURE_EXTRACTION_v3.py` and `E6_SHARED_FULL_ANSWER_CAPTURE_v2.py` are the actual frozen GPU producers. The required unsuffixed extraction filename is a byte-identical alias. `E6B_ANALYSIS_v2.py` and `E6B_TRANSFER_v2.py` implement development/freeze and evaluation phases. `E6B_ANALYSIS.py` supplies frozen metrics/probe helpers; its old monolithic main is not the executed formal workflow. `E6B_ONLINE_EARLY_EXIT.py` and the hook probe contain independent measured Validation/Train resource checks. Runtime JSON records their exact scope and raw timings.

These are project-internal ARC evaluations after historical ARC/Test exposure. No independent external benchmark or E6-C intervention was launched. Scientific review is explicitly current-model self-review.

## Lossless JSONL compression

Large JSONL artifacts are published as `.jsonl.gz`. Both replay scripts automatically read gzip, so decompression is optional. `LOSSLESS_JSONL_MANIFEST.json` records each exact uncompressed filename, byte count and SHA256, plus compressed SHA256. To restore the required unsuffixed artifact names with Python:

```python
import gzip, pathlib, hashlib, json
root = pathlib.Path('.')
for row in json.loads((root / 'LOSSLESS_JSONL_MANIFEST.json').read_text())['files']:
    data = gzip.decompress((root / row['path']).read_bytes())
    assert hashlib.sha256(data).hexdigest() == row['uncompressed_sha256']
    (root / row['uncompressed_path']).write_bytes(data)
```

After decompression `E6B_FEATURES.jsonl` and `E6B_CONTROLLER_RESULTS.jsonl` have their required names and exact original content. `E6B_NATURAL_FEATURE_EXTRACTION.py` is already present as the byte-identical alias of actual frozen v3 producer. No deployment helper, credentials, wheel archive, model weights or high-dimensional states are bundled.
