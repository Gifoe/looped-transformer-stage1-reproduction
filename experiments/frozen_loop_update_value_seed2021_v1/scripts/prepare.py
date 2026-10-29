"""CPU-only acquisition and disjoint, deterministic official-task manifests."""
import runtime_bootstrap
import hashlib, json, random, sys, shutil, os
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yaml
from datasets import Dataset, DatasetDict
from lm_eval.api.task import ConfigurableTask
import lm_eval

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'looped_stage1_trajectory_audit_seed2021_v1'
S=requests.Session();S.trust_env=False
def dump(p,x):
 p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf8')
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return x.item()
 return x
def norm(s):return ' '.join(s.casefold().split())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def acquire(repo,folder,split,meta):
 files=sorted(f['rfilename'] for f in meta['siblings'] if (f['rfilename'].startswith(folder+'/') and '/'+split+'-' in f['rfilename'] and f['rfilename'].endswith('.parquet')) or (repo=='baber/piqa' and f['rfilename']=='piqa_'+split+'.parquet'))
 assert files,(repo,split)
 out=[];records=[]
 for f in files:
  p=BASE/'data/snapshots'/repo.replace('/','--')/f
  if not p.exists():
   p=ROOT/'data/snapshots'/repo.replace('/','--')/f;p.parent.mkdir(parents=True,exist_ok=True)
   if not p.exists():
    r=S.get(f'https://hf-mirror.com/datasets/{repo}/resolve/{meta["sha"]}/{f}',timeout=90);r.raise_for_status();p.write_bytes(r.content)
  out+= [clean(x) for x in pd.read_parquet(p).to_dict('records')]
  records.append(dict(repo=repo,revision=meta['sha'],file=f,sha256=sha(p)))
 return out,records
def main():
 old=json.loads((BASE/'data/manifest_audit.json').read_text())['sources']
 # Read only question-bearing historical data manifests. No trajectories are needed.
 excluded=set();history=[]
 for p in BASE.rglob('*.jsonl'):
  if 'data' not in p.parts or 'snapshots' in p.parts:continue
  n=0
  with p.open(encoding='utf8') as f:
   for line in f:
    if not line.strip():continue
    r=json.loads(line)
    if isinstance(r.get('question'),str):excluded.add(norm(r['question']));n+=1
  if n:history.append(dict(path=str(p.relative_to(BASE)),n=n,sha256=sha(p)))
 specs=[('openbookqa','allenai/openbookqa','main',['train']),('sciq','allenai/sciq','data',['test']),('piqa','baber/piqa','',['validation'])]
 allrows={};snapshots=[];cross=set(excluded);duplicates=[]
 for task,repo,folder,splits in specs:
  if repo in old:meta=old[repo]
  else:
   r=S.get('https://hf-mirror.com/api/datasets/'+repo,timeout=45);r.raise_for_status();meta=r.json()
  dump('sources/'+task+'_metadata.json',meta)
  yaml_path=Path(lm_eval.__file__).parent/'tasks'/task/(task+'.yaml')
  if task=='openbookqa':yaml_path=yaml_path.with_name('openbookqa.yaml')
  config=yaml.safe_load(yaml_path.read_text());shutil.copyfile(yaml_path,ROOT/'sources'/('harness_'+task+'.yaml'))
  rows=[]
  for split in splits:
   docs,rec=acquire(repo,folder,split,meta);snapshots+=rec
   config.update(custom_dataset=lambda **kw:DatasetDict({split:Dataset.from_list(docs)}),test_split=split,validation_split=None,training_split=None,dataset_kwargs={},num_fewshot=0)
   engine=ConfigurableTask(config=config);engine.set_fewshot_seed(2021)
   for i,d in enumerate(docs):
    q=d['question_stem'] if task=='openbookqa' else d['question'] if task=='sciq' else d['goal']
    uid=f'{task}/{split}/{d.get("id",i)}'
    if norm(q) in cross:duplicates.append(uid);continue
    cross.add(norm(q))
    prompt=engine.fewshot_context(d,0);choices=engine.doc_to_choice(d);gold=int(engine.doc_to_target(d))
    req=engine.construct_requests(d,prompt,metadata=(task,i,1));pairs=[list(r.args) for r in req]
    assert all(x[0]==prompt for x in pairs)
    # Alter only task-preserving format; no answer choices or labels in context.
    support=d.get('support','').lstrip()
    alt=(support+'\n' if task=='sciq' else '')+'Q: '+q+'\nA:'
    variants={'official':pairs,'format':[[alt,p[1]] for p in pairs]}
    if task=='sciq':variants['no_support']=[['Question: '+q+'\nAnswer:',p[1]] for p in pairs]
    rows.append(dict(id=uid,task=task,official_split=split,source_index=i,question=q,choices=choices,gold=gold,variants=variants,choice_char_lengths=[len(c) for c in choices],primary_metric='acc_norm',dataset_revision=meta['sha'],fewshot_count=0))
  allrows[task]=rows
 pool=allrows['openbookqa'];random.Random(2021).shuffle(pool)
 assert len(pool)>=4400,len(pool)
 groups={'discovery':pool[:400],'head_train':pool[400:3600],'validation':pool[3600:4400],'final_sciq':allrows['sciq'],'final_piqa':allrows['piqa']}
 manifest={}
 for name,rows in groups.items():
  p=ROOT/'data'/(name+'.jsonl');payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
  if p.exists():assert p.read_text(encoding='utf8')==payload,'Manifest is immutable'
  p.write_text(payload,encoding='utf8',newline='\n');manifest[name]=dict(n=len(rows),sha256=sha(p),ids=[r['id'] for r in rows])
 dump('data/manifest_audit.json',dict(passed=True,manifests=manifest,snapshots=snapshots,exposure_scan=history,excluded_question_count=len(excluded),excluded_duplicates=duplicates,split_seed=2021,final_policy='All eligible official final questions; exact normalized duplicates excluded before predictions; report exclusion count',pretraining_contamination='Not assessable from experiment exposure audit'))
 print(json.dumps({k:v['n'] for k,v in manifest.items()}),flush=True)
if __name__=='__main__':main()
