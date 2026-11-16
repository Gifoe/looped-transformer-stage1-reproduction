import pathlib,json,gzip,shutil,tempfile,subprocess,sys,os,argparse
p=argparse.ArgumentParser();p.add_argument('--release',default='.');a=p.parse_args();pub=pathlib.Path(a.release).resolve()
from assemble_release import assemble
assemble(pub)
with tempfile.TemporaryDirectory(prefix='ccru-replay-') as tmp:
 r=pathlib.Path(tmp)
 for n in ['FREEZE.json','SELECTED_METHOD.json']:shutil.copy2(pub/n,r/n)
 for name in ['INPUTS.jsonl','ALL_QUESTION_RESULTS.jsonl']:
  with gzip.open(pub/(name+'.gz'),'rb') as src,(r/name).open('wb') as dst:shutil.copyfileobj(src,dst)
 env={**os.environ,'CCRU_ROOT':str(r)};subprocess.run([sys.executable,str(pub/'code/analyze.py')],env=env,check=True);subprocess.run([sys.executable,str(pub/'code/audit.py')],env=env,check=True)
 for n in ['SUMMARY.json','STRICT_NON_TIE_RESCUES.json','PROPAGATION.json','ATTENTION_ASSOCIATIONS.json','NORMS_AND_COSTS.json']:
  expected=json.loads((pub/n).read_text()) if (pub/n).exists() else json.loads(gzip.decompress((pub/(n+'.gz')).read_bytes()));assert json.loads((r/n).read_text())==expected,n
 print('CONTEXT_PUBLIC_CPU_EXACT_REPLAY_PASS')
