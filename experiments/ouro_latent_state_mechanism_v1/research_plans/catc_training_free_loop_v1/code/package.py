"""Whitelist the completed original CATC pilot; raw tensors stay on server."""
import pathlib,shutil,json,gzip,hashlib
R=pathlib.Path('/root/ouro-catc-sol-20261116');P=R/'publication';P.mkdir(exist_ok=True)
assert json.loads((R/'STATUS.json').read_text())['pilot_complete']
for name in ['code','audit_source','phaseC']:(P/name).mkdir(exist_ok=True)
for f in (R/'code').rglob('*.py'):
 dest=P/f.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
for f in R.glob('*.md'):
 if f.name!='ACTIVE_RUN.md':shutil.copy2(f,P/f.name)
for f in R.glob('*.json'):
 if f.name not in ['PROGRESS.json','GITHUB_PUBLICATION_RECEIPT.json']:shutil.copy2(f,P/f.name)
for f in [R/'technical/CATC_before_actual_bound.py']:
 if f.exists():shutil.copy2(f,P/'audit_source'/f.name)
for f in [R/'DEVICE_IDENTITY.txt']:
 if f.exists():shutil.copy2(f,P/f.name)
def gz(src,dst):
 with src.open('rb') as a,gzip.GzipFile(filename=str(dst),mode='wb',mtime=0) as b:shutil.copyfileobj(a,b)
for f in [R/'INPUTS.jsonl',R/'TRAINING_FREE_ALL_ROWS.jsonl']:gz(f,P/(f.name+'.gz'))
for f in (R/'phaseC').glob('*.json'):
 if f.name not in ['FINAL_VALIDATION_ROWS.json','validation_native.json']:shutil.copy2(f,P/'phaseC'/f.name)
for f in (R/'phaseC').glob('*INPUTS.jsonl'):gz(f,P/'phaseC'/(f.name+'.gz'))
for f in (R/'phaseC').glob('*LABELS.jsonl'):gz(f,P/'phaseC'/(f.name+'.gz'))
shutil.copy2(R/'phaseC/ADAPTATION_QUESTION_ROWS.jsonl.gz',P/'phaseC/ADAPTATION_QUESTION_ROWS.jsonl.gz')
with gzip.GzipFile(filename=str(P/'phaseC/FULL_ANSWER_QUESTION_ROWS.jsonl.gz'),mode='wb',mtime=0) as g:
 for f in sorted((R/'phaseC/fas_results').glob('*.json')):g.write((json.dumps(json.loads(f.read_text()),separators=(',',':'))+'\n').encode())
for name in ['catc14','unstructured14','value7','routing7']:
 d=P/'phaseC'/name;d.mkdir(exist_ok=True)
 for f in (R/'phaseC'/name).glob('*.json'):
  if f.name in ['COMPLETE.json','history.json']:shutil.copy2(f,d/f.name)
  else:gz(f,d/(f.name+'.gz'))
shutil.copy2(R/'code/CATC_IMPLEMENTATION.py',P/'CATC_IMPLEMENTATION.py')
(P/'REPRODUCE.md').write_text('''# Reproduce the published scalar results

Use Python 3.10 and NumPy 1.26.4. No Torch, model download or GPU is needed for the two statistical replays:

```
python code/reproduce_release.py --release .
python code/reproduce_adaptation.py --release .
```

The scripts verify and assemble any lossless archive shards automatically. The first reconstructs all 1094 training questions and requires exact equality of the fixed-setting summaries, gates, correction norms and capture runtime. The second reconstructs the 299 Joint-Option validation questions and 997 SciQ questions, reproduces their statistics exactly, and verifies all 299 FullAnswer decisions, token sums, confidence intervals and selected-coefficient histories.

CPU replay verifies reported arithmetic and statistical decisions; it does not repeat GPU forwards. Full 49152-dimensional tensors and Q/K/V witnesses remain on the server, identified in RAW_TENSOR_MANIFEST.json. GPU collectors use the frozen external Ouro checkpoint and source hashes, BF16 SDPA and four recurrent passes. For an independent GPU rerun, use a fresh output root, adapt paths/device binding, and run the numerical gates before collecting effects. Preserve the original formal output files.

The executed evaluation source hashes are in the corresponding freeze receipts. A later LoopCD 0.5 helper addition derives a standing baseline from saved native vectors without changing the original completed forward executions or frozen statistical gates.
''')
print('CATC_PACKAGE_READY')
