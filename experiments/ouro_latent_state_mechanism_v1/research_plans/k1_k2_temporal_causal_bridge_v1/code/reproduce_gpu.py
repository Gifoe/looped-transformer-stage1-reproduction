"""Run an explicitly selected frozen model stage on a fresh workspace.
Only path strings/deadline are adapted; executed sources remain byte-preserved in release.
"""
import pathlib,argparse,json,datetime,subprocess,sys,gzip,shutil
from assemble_release import assemble
p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--work-dir',required=True);p.add_argument('--stage',required=True,choices=['technical-A','A-development','A-confirmation','B-development','B-confirmation','technical-C','C-development','C-confirmation']);a=p.parse_args();P=pathlib.Path(__file__).resolve().parent.parent;R=pathlib.Path(a.work_dir).resolve();model=pathlib.Path(a.model_dir).resolve();assert R!=P
assemble(P);R.mkdir(parents=True,exist_ok=True);(R/'code').mkdir(exist_ok=True);(R/'raw').mkdir(exist_ok=True);(R/'results').mkdir(exist_ok=True)
for name in ['INPUTS.jsonl','UNITS.jsonl']:
 if not(R/name).exists():
  with gzip.open(P/(name+'.gz'),'rb') as s,(R/name).open('wb') as d:shutil.copyfileobj(s,d)
for name in ['A_BACKGROUND_SELECTION.json','B_SELECTION.json','B_CONFIRMATION_GATE.json','C_TECHNICAL_PASS.json','CAUSAL_CUT_NUMERICAL_AUDIT.json','confirmation_A_SUMMARY.json']:
 # Frozen gates document actual study. Technical stage overwrites its own receipt;
 # B/C execution also checks fresh predecessor raw outputs via exact endpoint hashes.
 if not(R/name).exists():shutil.copyfile(P/name,R/name)
if not(R/'FREEZE.json').exists():
 f=json.loads((P/'FREEZE.json').read_text());f['reproduction_original_deadline_utc']=f['deadline_utc'];f['deadline_utc']=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=180)).isoformat();f['reproduction_path_only_adapter']=True;(R/'FREEZE.json').write_text(json.dumps(f,indent=2))
for f in (P/'code').glob('*.py'):shutil.copyfile(f,R/'code'/f.name)
source=P/'code'/'source_versions'/'bridge_A_B_EXECUTED.py' if a.stage=='technical-A' or a.stage[0] in ['A','B'] else P/'code'/'bridge.py'
s=source.read_text().replace('/root/autodl-tmp/ouro-temporal-bridge-sol-20261117/raw',str(R/'raw')).replace('/root/ouro-temporal-bridge-sol-20261117',str(R)).replace('/root/autodl-tmp/ouro-e4-e5/model',str(model));(R/'code'/'bridge.py').write_text(s)
if a.stage.startswith('technical-'):cmd=[sys.executable,str(R/'code'/('technical.py' if a.stage.endswith('A') else 'technical_C.py'))]
else:
 stage,phase=a.stage.split('-');cmd=[sys.executable,str(R/'code'/('run_'+stage+'.py')),phase]
subprocess.run(cmd,check=True,cwd=R)
if not a.stage.startswith('technical-'):subprocess.run([sys.executable,str(R/'code'/('analyze_'+stage+'.py')),phase],check=True,cwd=R,env={**__import__('os').environ,'BRIDGE_ROOT':str(R)})
