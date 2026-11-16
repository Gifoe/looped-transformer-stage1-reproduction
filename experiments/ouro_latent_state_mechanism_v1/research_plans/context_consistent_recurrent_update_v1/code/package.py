import pathlib,json,gzip,shutil,hashlib
R=pathlib.Path('/root/ouro-context-update-sol-20261117');P=R/'publication';P.mkdir(exist_ok=True);assert json.loads((R/'INDEPENDENT_CPU_AUDIT.json').read_text())['status']=='PASS';(P/'code').mkdir(exist_ok=True)
for f in (R/'code').glob('*.py'):shutil.copy2(f,P/'code'/f.name)
for f in R.glob('*.md'):
 if f.name!='ACTIVE_RUN.md':shutil.copy2(f,P/f.name)
for f in R.glob('*.json'):
 if f.stat().st_size<2*1024*1024 and f.name!='LABELS.json':shutil.copy2(f,P/f.name)
 else:
  with f.open('rb') as src,gzip.GzipFile(filename=str(P/(f.name+'.gz')),mode='wb',mtime=0) as dst:shutil.copyfileobj(src,dst)
for f in [R/'INPUTS.jsonl',R/'ALL_QUESTION_RESULTS.jsonl']:
 with f.open('rb') as src,gzip.GzipFile(filename=str(P/(f.name+'.gz')),mode='wb',mtime=0) as dst:shutil.copyfileobj(src,dst)
shutil.copy2(R/'code/INTERVENTION_IMPLEMENTATION.py',P/'INTERVENTION_IMPLEMENTATION.py')
(P/'README.md').write_text('''# Context-consistent recurrent update

All nine development conditions on256questions and five frozen confirmation conditions on512questions. Original25shot ARC Joint-Option, no Test/training/search. Read ASTRA_FINAL_DECISION.md and CONTEXT_UPDATE_RESULTS.md. All work by Sol; requested ASTRA filename retained.

CPU replay: Python3.10, NumPy1.26.4, `python code/reproduce_release.py --release .`. This assembles lossless shards and exactly recomputes strict effects, paired intervals, trajectories, attention associations and doses. It audits every input/candidate/tie/norm and exact-input-group separation. It does not rerun GPU forwards.

Raw BF16 attention observation and full-vocabulary technical tensors stay on server, identified by RAW_TENSOR_MANIFEST.json. GPU reruns require pinned external checkpoint and fresh root; adapt paths, run technical.py, then run_study.py and finish scripts. Original source/execution hashes and freezes remain preserved.

Strict utility requires unique native and modified top candidates and keeps the full cohort denominator. Random norms match actual structured Q doses per token; unmatched cases remain. Norm row scale for Random is the *structured reference* clipping scale, while the randomized scalar amplitude is in dose_matching.amplitude. P/R/Q total doses differ, limiting region-semantics attribution. Attention observations are descriptive associations, not mediation proof.
''');print('PACKAGE_READY')
