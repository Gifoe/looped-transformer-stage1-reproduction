import pathlib,argparse,json,gzip,tempfile,shutil,subprocess,sys,os
p=argparse.ArgumentParser();p.add_argument('--release',default='.');a=p.parse_args();pub=pathlib.Path(a.release).resolve()
from assemble_release import assemble
assemble(pub)
with tempfile.TemporaryDirectory(prefix='edge-scalar-replay-') as tmp:
 r=pathlib.Path(tmp)
 for n in ['FREEZE.json','EXECUTION_FREEZE.json','LOCALIZED_BOUNDARY_REFERENCE.json']:shutil.copy2(pub/n,r/n)
 for n in ['INPUTS.jsonl','ALL_QUESTION_RESULTS.jsonl']:
  with gzip.open(pub/(n+'.gz'),'rb') as src,(r/n).open('wb') as dst:shutil.copyfileobj(src,dst)
 env={**os.environ,'EDGE_ROOT':str(r)}
 for phase in ['development','fresh_confirmation']:
  subprocess.run([sys.executable,str(pub/'code/analyze_A.py'),phase],env=env,check=True);n=phase+'_SUMMARY.json';assert json.loads((r/n).read_text())==json.loads((pub/n).read_text()),n
 subprocess.run([sys.executable,str(pub/'code/audit_scalars.py')],env=env,check=True);subprocess.run([sys.executable,str(pub/'code/compare_boundary.py')],env=env,check=True);assert json.loads((r/'BOUNDARY_REFERENCE_COMPARISON.json').read_text())==json.loads((pub/'BOUNDARY_REFERENCE_COMPARISON.json').read_text());assert json.loads((r/'SELECTED_CANDIDATE.json').read_text())['candidate']==json.loads((pub/'SELECTED_CANDIDATE.json').read_text())['candidate'];print('EDGE_PUBLIC_CPU_EXACT_REPLAY_PASS')
