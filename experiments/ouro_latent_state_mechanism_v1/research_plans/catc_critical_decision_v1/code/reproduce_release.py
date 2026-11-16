"""CPU-only exact replay, including retrospective duplicate repair; no fitting."""
import pathlib,json,gzip,shutil,tempfile,os,sys,subprocess,argparse,numpy as np
p=argparse.ArgumentParser();p.add_argument('--release',default='.');a=p.parse_args();pub=pathlib.Path(a.release).resolve()
from assemble_release import assemble
assemble(pub)
with tempfile.TemporaryDirectory(prefix='catc-critical-replay-') as tmp:
 r=pathlib.Path(tmp)
 for n in ['FREEZE.json','SELECTED_METHOD.json','COHORT_DUPLICATE_REPAIR.json','TRAIN_INPUTS.jsonl.gz','VALIDATION_INPUTS.jsonl.gz']:shutil.copy2(pub/n,r/n)
 with gzip.open(pub/'QUESTION_LEVEL_RESULTS.jsonl.gz','rb') as src,(r/'QUESTION_LEVEL_RESULTS.jsonl').open('wb') as dst:shutil.copyfileobj(src,dst)
 env=os.environ.copy();env['CATC_CRITICAL_ROOT']=str(r)
 for script in ['analyze_critical.py','audit_scalar_rows.py']:subprocess.run([sys.executable,str(pub/'code'/script)],env=env,check=True)
 for n in ['CRITICAL_SUMMARY.json','DOSE_MATCHING_AUDIT.json']:assert json.loads((r/n).read_text())==json.loads((pub/n).read_text()),n
 rows=[json.loads(l) for l in (r/'QUESTION_LEVEL_RESULTS.jsonl').read_text().splitlines()];dev=[q for q in rows if q['cohort']=='development'];f=json.loads((r/'FREEZE.json').read_text());means={};oracle=0
 for name in ['Native','V+','V-','P+','P-']:
  losses=[]
  for q in dev:
   x=np.asarray(q['fixed_direction_methods'][name]['scores']['4']['candidate_logits'],dtype=np.float64);m=x.max();losses.append(float(m+np.log(np.exp(x-m).sum())-x[q['gold']]))
  means[name]=float(np.mean(losses))
 selected=min(['V+','V-','P+','P-'],key=lambda n:(means[n],['V+','V-','P+','P-'].index(n)));assert selected==json.loads((r/'SELECTED_METHOD.json').read_text())['name'];assert all(means[n]==json.loads((pub/'DIRECTIONAL_SUMMARY.json').read_text())['full_development']['4']['methods'][n]['mean_candidate_CE'] for n in means)
 assert sum(any(v['scores']['4']['prediction']==q['gold'] for v in q['fixed_direction_methods'].values()) for q in dev)==229
 print('CRITICAL_EXACT_REPLAY_AND_SELECTION_PASS')
