"""Whitelist-only publication: preserve scalars/scripts, exclude weights/private transport/raw activations."""
import pathlib,json,gzip,shutil,hashlib,datetime
R=pathlib.Path('/root/ouro-computation-sol-20261116');P=R/'publication';P.mkdir(exist_ok=True)
for name in ['code','data','inputs','freeze','analysis','audits','figures','source_versions','model_source']: (P/name).mkdir(exist_ok=True)
for f in (R/'reports').glob('*.md'):shutil.copy2(f,P/f.name)
for f in (R/'code').glob('*.py'):shutil.copy2(f,P/'code'/f.name)
for name in ['ab1_loader_draft.py','ab1_metrics.py','ab1_boundary_hooks.py']:shutil.copy2(R/'prior/code'/name,P/'code'/name)
for f in (R/'source_versions').rglob('*.py'):
 dest=P/'source_versions'/f.relative_to(R/'source_versions');dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(f,dest)
for name in ['modeling_ouro.py','configuration_ouro.py','config.json']:shutil.copy2(pathlib.Path('/root/autodl-tmp/ouro-e4-e5/model')/name,P/'model_source'/name)
for name in ['A_FREEZE.json','A_EXPANSION_FREEZE.json','B_FREEZE.json','C_FREEZE.json','ARC_STRUCTURE_FREEZE.json','ARC_STRUCTURAL_CANDIDATES.json','CODE_FREEZE_A.json','CODE_FREEZE_B.json','CONCURRENCY_AMENDMENT.json']:shutil.copy2(R/name,P/'freeze'/name)
for f in (R/'analysis').glob('*'):
 if f.suffix=='.gz':shutil.copy2(f,P/'data'/f.name)
 elif f.is_file():shutil.copy2(f,P/'analysis'/f.name)
def gzcopy(src,dst):
 with pathlib.Path(src).open('rb') as f,gzip.GzipFile(filename=str(dst),mode='wb',mtime=0) as g:shutil.copyfileobj(f,g)
for name in ['A_INPUTS.jsonl','B_INPUTS.jsonl','B_UNITS.jsonl']:gzcopy(R/name,P/'inputs'/(name+'.gz'))
# Only the1393 allowed ARC training/validation labels, never Test labels.
allowed={(q['split'],q['id']) for q in map(json.loads,(R/'A_INPUTS.jsonl').read_text().splitlines())};labels=[q for q in map(json.loads,(R/'prior/analysis_inputs/analysis_labels.jsonl').read_text().splitlines()) if (q['split'],q['id']) in allowed]
with gzip.GzipFile(filename=str(P/'inputs/ANALYSIS_LABELS.jsonl.gz'),mode='wb',mtime=0) as f:f.write(''.join(json.dumps(q)+'\n' for q in labels).encode())
for name in ['TECHNICAL_PASS.json','B_TECHNICAL_PASS.json','B_SDPA_ACTUAL_CALL_AUDIT.json','C_TECHNICAL_PASS.json','SOURCE_RUNTIME.json','INDEPENDENT_CPU_AUDIT.json','C_INDEPENDENT_CPU_AUDIT.json','B_NATIVE_SERIALIZATION_REPAIR.json','LOGSOFTMAX_AUDIT_AMENDMENT.md','RAW_TENSOR_MANIFEST.json','C_AND_TECHNICAL_RAW_MANIFEST.json','B_FULL_ARCHIVE_REPLICATION_PASS.json','GPU_COSTS.json','RESOURCE_SAMPLES.jsonl','A_development_COMPLETE.json','A_train_confirmation_COMPLETE.json','A_validation_COMPLETE.json','B_development_COMPLETE.json','B_heldout_COMPLETE.json','C_COMPLETE.json']:
 shutil.copy2(R/name,P/'audits'/name)
for f in (R/'figures').glob('*'):
 if 'preview' not in f.name:shutil.copy2(f,P/'figures'/f.name)
(P/'README.md').write_text('''# Ouro K1->K2 computational mechanism

Sol-only bounded study,2026-11-16. [Final mechanism report](FINAL_MECHANISM_REPORT.md) separates established/rejected/unresolved claims. Full ARC Train1094/Validation299; deterministic cross-world160pairs/640units; limited routing512units. Strong compact-interchange gate fails, while68/512 strictly new predicted crossed outputs and matched causal-route effects demonstrate partial semantic influence. This does not solve the entire K1->K2 algorithm.

- [Context reuse](K1_CONTEXT_REUSE_RESULTS.md)
- [Dataset specification](CROSS_WORLD_DATASET_SPEC.md)
- [Intermediate exchange](INTERMEDIATE_VARIABLE_PATCHING.md)
- [Computational abstraction and routing](COMPUTATIONAL_ABSTRACTION.md)
- [Controls and counterexamples](CAUSAL_CONTROLS_AND_COUNTEREXAMPLES.md)
- [ARC structural transfer limits](ARC_MECHANISM_TRANSFER.md)
- [Source and numerical audit](SOURCE_AND_NUMERICAL_AUDIT.md)
- [Freeze and amendments](EXPERIMENT_FREEZE.md)

![Complete experiment results](figures/COMPUTATIONAL_MECHANISM.png)

## Files

`code/`: executable GPU collectors, CPU analyses/reproducer, auditing, figures and packaging. `freeze/`: exact cohorts, semantic graphs/sites, seeds, gates and concurrency amendment. `inputs/`: losslessly gzipped gold-free ARC input IDs, synthetic graphs/prompts/inputs and separate allowed ARC analysis labels. `data/`: every question-level candidate logit/log-probability, prediction, tie, control, dose and selected trajectory hash. `analysis/`: complete group/bootstrap summaries and positive/negative examples. `audits/`: exact numerical receipts, repair/reference amendments, resource costs and raw tensor manifest. `source_versions/`: original collector versions for serialization provenance. `model_source/`: pinned model/config source; pretrained weights/tokenizer files are external.

The stage metadata in public scalar rows is explicitly named `captured_K2_stage_R_sha256`, including beside K1 score rows. K1 endpoint logits/B are separate actual tensors. No generated prose substitutes for per-question evidence.

## CPU reproduction

Use Python3.10 and NumPy1.26.4. No torch/model/GPU is needed for scalar recomputation:

```bash
python code/reproduce_release.py --release .
```

It losslessly expands the scalar archives in a temporary directory, independently re-executes all paired/group/bootstrap calculations, and requires exact equality with A/B/C_SUMMARY.json. The retained-tensor audits additionally check every full49152 endpoint; those large tensors stay on the experiment server and are not in this release.

## GPU execution

Restore the exact external model checkpoint/tokenizer identified by SOURCE_RUNTIME.json. Original execution used the paths in code/core.py, the40321 RTX4090 UUID, Torch2.5.1cu124/transformers4.55.0/NumPy1.26.4, BF16/SDPA,cachefalse. Do not run collectors against existing frozen output paths: use a fresh root and adapt only paths/device binding, preserving frozen input IDs, source hashes, mode/scoring and numerical gates. Rebuild native anchors on a different device/runtime. Run `run_A.py technical` before its development/train_confirmation/validation phases; run `run_B.py technical` before development/heldout; then C technical/heldout under its exposed-cohort extension freeze. Each process runs the entire real2-loop body. `core.py` imports three included original AB1 helper modules; the publication's code directory is self-contained after path/device adaptation. Original `source_versions` retain the serialization implementation; repaired current B uses separate native:: and patch:: keys.

The existing scientific parameters/gates were not refitted. Historical E6/H2 and prior B2 evidence remain at the starting commit e7da5c78676bf18ef5bd74a967788ad9875e9ea9. No ARC Test was opened for this study. Raw tensors and checkpoint remain at /root/ouro-computation-sol-20261116 and the original model directory on the server.
''')
manifest=[]
for f in sorted(P.rglob('*')):
 if f.is_file() and f.name!='PUBLICATION_MANIFEST.json':
  data=f.read_bytes();manifest.append({'path':str(f.relative_to(P)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()})
(P/'PUBLICATION_MANIFEST.json').write_text(json.dumps({'files':manifest,'total_bytes':sum(f['bytes'] for f in manifest),'source_commit':'e7da5c78676bf18ef5bd74a967788ad9875e9ea9'},indent=2));print('PACKAGED',len(manifest),'files',sum(f['bytes'] for f in manifest),'bytes')
