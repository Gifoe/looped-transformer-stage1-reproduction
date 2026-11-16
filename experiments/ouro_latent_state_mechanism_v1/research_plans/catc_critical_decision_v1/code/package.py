import pathlib,shutil,json,gzip,hashlib,re
R=pathlib.Path('/root/ouro-catc-critical-sol-20261116');P=R/'publication';P.mkdir(exist_ok=True)
assert json.loads((R/'INDEPENDENT_CPU_AUDIT.json').read_text())['input_sequences_verified']==1393
for folder in ['code','prior']:
 for f in (R/folder).rglob('*'):
  if f.is_file() and f.suffix in ['.py','.md','.json'] and '__pycache__' not in f.parts:
   dest=P/f.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
for pattern in ['*.md','*.json','*INPUTS.jsonl.gz','DEVICE_IDENTITY.txt']:
 for f in R.glob(pattern):
  if f.name not in ['ACTIVE_RUN.md','GITHUB_PUBLICATION_RECEIPT.json']:shutil.copy2(f,P/f.name)
for n in ['QUESTION_LEVEL_RESULTS.jsonl','DEVELOPMENT_ROWS.jsonl']:
 with (R/n).open('rb') as src,gzip.GzipFile(filename=str(P/(n+'.gz')),mode='wb',mtime=0) as dst:shutil.copyfileobj(src,dst)
for f in (P/'code').glob('*.py'):
 if f.name=='package.py':continue # scanner's own pattern literals
 assert not re.search('password|BEGIN.*PRIVATE KEY|github_pat_|ghp_|remote40321_private',f.read_text(),re.I),f
print('CRITICAL_PACKAGE_READY')
