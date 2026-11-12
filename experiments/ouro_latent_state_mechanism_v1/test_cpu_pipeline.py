"""Tiny CPU integration fixture: not a dataset run or GPU admission."""
import hashlib,json,sys,tempfile,os,shutil
from pathlib import Path
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
import cohort_runner as run
import audit_coverage as coverage
import behavior_summary as behavior
import representation_observation as rep
import summarize_representations as summary

def main():
 torch.set_num_threads(2);row={'id':'fixture','question':'test','answerKey':'A','choices':{'label':['A','B'],'text':['one','two']}}
 fake_table=type('FixtureTable',(),{'to_pylist':lambda self:[row]})()
 with tempfile.TemporaryDirectory(prefix='cpu_pipeline_',dir=run.PROJECT) as t:
  root=Path(t);shards=root/'shards';shards.mkdir();(root/'results').mkdir();dummy=root/'dummy';dummy.write_text('fixture data');dataset_sha=run.sha(dummy);hashes={'fixture':'v1'}
  for module in (coverage,behavior,rep,summary):module.pq.read_table=lambda path:fake_table;module.split_data_path=lambda split:dummy
  for module in (coverage,behavior,rep):module.bound_hashes=lambda split:hashes
  coverage.validate_manifest=lambda *a:None
  shutil.copyfile(Path(coverage.__file__),root/'audit_coverage.py');shutil.copyfile(Path(coverage.__file__).with_name('ledger_guard.py'),root/'ledger_guard.py');coverage.__file__=str(root/'audit_coverage.py')
  model=AutoModelForCausalLM.from_pretrained(run.PROJECT/'model',trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval();model.config.use_cache=False;model.model.config.use_cache=False
  prompt='synthetic';records=[];tensors={};manifest=[]
  with torch.inference_mode():
   for j,target in enumerate((340,422)):
    enc={'input_ids':[330,389],'context_ids':[330,389],'target_ids':[target],'predecessor_positions':[1],'spans':{'question':[0],'answer_marker':[1]},'offset_mapping':[[0,1],[1,2]],'truncated':False};execution=run.execute_request(model,enc);scores=execution['scores'];positions=execution['positions'];digest=hashlib.sha256(json.dumps(enc['input_ids']).encode()).hexdigest()
    for name,value in execution['tensors'].items():tensors[f'candidate{j}_{name}']=value
    records.append({'candidate':j,'input_ids_sha256':digest,'encoding':enc,'captured_positions':positions,'scores':scores,'event_sequence':execution['events']});manifest.append({'id':'fixture','candidate':j,'input_sha256':digest,'selected_token_count':len(positions)})
  name=hashlib.sha256(b'fixture').hexdigest();payload=shards/(name+'.safetensors');save_file(tensors,str(payload))
  from safetensors.torch import load_file
  reloaded=load_file(str(payload))
  for j,record in enumerate(records):
   targets=torch.tensor(record['encoding']['target_ids'],dtype=torch.long)
   for k in range(1,5):
    logits=reloaded[f'candidate{j}_loop{k}_lm_head_logits'];score=float(logits.float().log_softmax(-1).gather(-1,targets[:,None]).sum())
    if score!=record['scores'][k-1]:raise RuntimeError('stored full-vocabulary logits do not reconstruct exact CPU scores')
  gold=0;ranks=[sorted(range(2),key=lambda j:-records[j]['scores'][k]) for k in range(4)];pred=[order[0] for order in ranks];margins=[records[0]['scores'][k]-records[1]['scores'][k] for k in range(4)]
  meta={'id':'fixture','split':'validation','gold_index':gold,'rankings':ranks,'predictions':pred,'gold_margins':margins,'sha256':hashes,'tensor_sha256':run.sha(payload),'records':records};run.write_json(shards/(name+'.json'),meta)
  (root/'results/INPUT_MANIFEST_validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest));ledger=root/'ledger.json'
  os.environ['OURO_SYNTHETIC_TEST']='1'
  sys.argv=['audit_coverage','--split','validation','--shards',str(shards),'--out',str(ledger),'--score-device','cpu'];coverage.main();c=json.loads(ledger.read_text());assert c['counts']['complete_questions']==1 and c['counts']['complete_requests']==2
  behavior.require_production_ledger=lambda _:None;rep.require_production_ledger=lambda _:None # synthetic fixture only
  meta_path=shards/(name+'.json');meta['records'][0]['scores'][0]+=1.0;run.write_json(meta_path,meta)
  bad_ledger=root/'bad_ledger.json';sys.argv[sys.argv.index('--out')+1]=str(bad_ledger)
  try:coverage.main()
  except SystemExit:pass
  else:raise AssertionError('altered score was accepted')
  bad=json.loads(bad_ledger.read_text());assert bad['counts']['failed_questions']==1 and 'score replay mismatch' in bad['questions'][0]['errors'][0]
  meta['records'][0]['scores'][0]-=1.0;run.write_json(meta_path,meta);ledger=root/'ledger_final.json';sys.argv[sys.argv.index('--out')+1]=str(ledger);coverage.main()
  sys.argv=['behavior_summary','--split','validation','--shards',str(shards),'--coverage',str(ledger),'--out',str(root/'behavior.json')];behavior.main();assert json.loads((root/'behavior.json').read_text())['all']['n']==1
  obs=root/'observation.jsonl';sys.argv=['representation_observation','--split','validation','--shards',str(shards),'--coverage',str(ledger),'--out',str(obs)];rep.main()
  sys.argv=['summarize_representations','--split','validation','--input',str(obs),'--coverage',str(ledger),'--out',str(root/'summary.json')];summary.main();assert json.loads((root/'summary.json').read_text())['question_count']==1
 print('PASS synthetic CPU production request -> shard -> coverage -> behavior -> representation summary')
if __name__=='__main__':main()
