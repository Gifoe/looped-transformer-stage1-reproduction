"""Audit every ARC prompt without inference; HF_HOME should point inside project."""
import sys,json,collections
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
P=Path(__file__).resolve();repo=P.parents[2];project=repo.parent
sys.path.insert(0,str(repo/'experiments/ouro_weight_space_repair_v2/scripts'))
from prompt_spans import make_question,token_spans
from r10_core import question_prompt,joint_prefix
t=AutoTokenizer.from_pretrained(project/'model',local_files_only=True,trust_remote_code=True)
prefix=joint_prefix(json.loads((repo/'experiments/ouro_weight_space_repair_v2/data/slcd_fixed_demonstrations.json').read_text(encoding='utf-8')))
report={'scope':'CPU prompt span and target tokenization only; no inference','splits':{}}
for split in ('train','validation','test'):
 rows=pq.read_table(project/'data'/f'{split}.parquet').to_pylist();errors=[];counts=collections.Counter();longest=0
 for row in rows:
  try:
   prompt,spans=make_question(prefix,row['question'],row['choices']['text'],question_prompt)
   for i in range(len(row['choices']['text'])):
    result=token_spans(t,prompt,' '+chr(65+i),spans);counts[len(result['target_ids'])]+=1;longest=max(longest,len(result['input_ids']))
  except Exception as e: errors.append({'id':row['id'],'error':repr(e)})
 report['splits'][split]={'n':len(rows),'errors':errors,'target_length_counts':dict(counts),'max_input_tokens':longest}
(P.parent/'results/CPU_SPAN_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
if any(x['errors'] for x in report['splits'].values()):raise SystemExit('span audit failed')
