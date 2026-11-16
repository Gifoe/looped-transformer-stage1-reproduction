"""CPU-only exact summary replay from losslessly compressed scalar rows."""
import pathlib,argparse,tempfile,json,gzip,shutil,subprocess,os,sys
p=argparse.ArgumentParser();p.add_argument('--release',default='.');a=p.parse_args();pub=pathlib.Path(a.release).resolve()
if (pub/'SHARDED_ARTIFACTS.json').exists():
 from assemble_release import assemble
 assemble(pub)
with tempfile.TemporaryDirectory(prefix='catc-replay-') as tmp:
 r=pathlib.Path(tmp);(r/'results').mkdir();(r/'code').mkdir()
 for n in ['analyze.py','E6A_METHODS.py']:shutil.copy2(pub/'code'/n,r/'code'/n)
 shutil.copy2(pub/'FREEZE.json',r/'FREEZE.json')
 with gzip.open(pub/'TRAINING_FREE_ALL_ROWS.jsonl.gz','rt') as f:
  labels=[]
  for line in f:
   q=json.loads(line);labels.append({'id':q['id'],'split':'train','gold':q.pop('gold')});(r/'results'/f'{q["id"]}.json').write_text(json.dumps(q))
 (r/'ANALYSIS_LABELS.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in labels))
 env=os.environ.copy();env['CATC_ROOT']=str(r);subprocess.run([sys.executable,str(r/'code/analyze.py')],env=env,check=True)
 for n in ['SUMMARY.json','GATES.json','CORRECTION_NORMS.json','RUNTIME_AND_MEMORY.json']:
  actual=json.loads((r/n).read_text());expected=json.loads((pub/n).read_text());assert actual==expected,n;print(n,'EXACT_RECOMPUTATION_PASS')
print('CATC_PUBLIC_REPLAY_PASS')
