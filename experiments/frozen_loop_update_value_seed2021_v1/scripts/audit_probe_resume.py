"""Question-only probe availability audit and checksum-bound failure recovery."""
import runtime_bootstrap
import json,hashlib,shutil
from pathlib import Path
from score import ROOT,BASE,read,encode,probes,sha,dump
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(BASE/'models/Ouro-1.4B',local_files_only=True)
missing={};counts={}
for split in ['discovery','head_train','validation','final_sciq','final_piqa']:
 absent=[];total=0
 for row in read('data/'+split+'.jsonl'):
  for view,pairs in row['variants'].items():
   prefix,_=encode(tok,*pairs[0]);total+=1
   if not probes(tok,pairs[0][0],row['question'],prefix):absent.append(dict(id=row['id'],view=view))
 missing[split]=absent;counts[split]=dict(views=total,missing=len(absent))
dump('results/PROBE_AVAILABILITY_AUDIT.json',dict(passed=True,counts=counts,missing=missing,label_access=False))
p=ROOT/'results/discovery.jsonl';rows=read('results/discovery.jsonl');assert len(rows)==307
sig=rows[0]['signature'];assert all(r['signature']==sig for r in rows)
dump('configs/RESUME_discovery.json',dict(completed=307,partial_sha256=sha(p),previous_signature=sig,reason='Causal question probe unavailable for single-token official prompt; no changes to valid-case scoring',old_source_commit='967d2eb'))
dest=ROOT/'results/discovery_attempt1.jsonl';assert not dest.exists();shutil.copyfile(p,dest)
for name in ['discovery_exit.json','discovery.err','discovery.out','discovery_launcher.log','discovery_launcher_exit.json']:
 old=ROOT/'logs'/name
 if old.exists():
  new=ROOT/'logs'/('attempt1_'+name);assert not new.exists();old.rename(new)
print(json.dumps(dict(passed=True,counts=counts,reused_questions=307)),flush=True)
