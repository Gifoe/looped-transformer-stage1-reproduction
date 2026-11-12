"""Create complete split/request input coverage manifest without model inference."""
import sys,json,hashlib,os
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from prompt_spans import make_question,token_spans
from cohort_runner import PROJECT,REPO,split_data_path,sha,write_json
sys.path.insert(0,str(REPO/'experiments/ouro_weight_space_repair_v2/scripts'))
from r10_core import joint_prefix,question_prompt,joint_requests
from runtime import encode_pair
t=AutoTokenizer.from_pretrained(PROJECT/'model',local_files_only=True,trust_remote_code=True)
demo=json.loads((REPO/'experiments/ouro_weight_space_repair_v2/data/slcd_fixed_demonstrations.json').read_text());prefix=joint_prefix(demo)
out=Path(__file__).parent/'results';summary={'scope':'full input audit only; no inference','splits':{}}
for split in ('train','validation','test'):
 rows=pq.read_table(split_data_path(split)).to_pylist();seen=set();counts={};errors=[];path=out/f'INPUT_MANIFEST_{split}.jsonl';temp=path.with_suffix('.jsonl.partial')
 with temp.open('w',encoding='utf8') as f:
  for row in rows:
   try:
    if row['id'] in seen:raise ValueError('duplicate ID')
    seen.add(row['id']);choices=row['choices']['text'];prompt,spans=make_question(prefix,row['question'],choices,question_prompt);historical=joint_requests({'question':row['question'],'choices':choices},prefix)
    for j in range(len(choices)):
     enc=token_spans(t,prompt,' '+chr(65+j),spans);hp,ht=encode_pair(t,*historical[j]);assert historical[j]==(prompt,' '+chr(65+j));assert hp==enc['context_ids'] and ht==enc['target_ids']
     assert all(0<=k<len(enc['context_ids']) for points in enc['spans'].values() for k in points)
     assert all(0<=k<len(enc['input_ids']) for k in enc['predecessor_positions'])
     if len(enc['target_ids'])!=1:raise ValueError('non-single-token continuation not admitted by GPU gate')
     key=str(len(choices));counts[key]=counts.get(key,0)+1
     rec={'id':row['id'],'split':split,'candidate':j,'option_count':len(choices),'raw_labels':row['choices']['label'],'answer_key':row['answerKey'],'target_token_count':len(enc['target_ids']),'input_length':len(enc['input_ids']),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'input_sha256':hashlib.sha256(json.dumps(enc['input_ids']).encode()).hexdigest(),'selected_token_count':len(set(sum(enc['spans'].values(),[])+enc['predecessor_positions'])),'span_token_counts':{k:len(v) for k,v in enc['spans'].items()},'demo_overlap':row['id'] in demo['ids']}
     f.write(json.dumps(rec,ensure_ascii=False)+'\n')
   except Exception as e:errors.append({'id':row['id'],'error':repr(e)})
 os.replace(temp,path);summary['splits'][split]={'expected_questions':len(rows),'unique_questions':len(seen),'candidate_counts_by_option_count':counts,'errors':errors,'manifest_sha256':sha(path),'data_sha256':sha(split_data_path(split))}
write_json(out/'INPUT_MANIFEST_SUMMARY.json',summary)
if any(x['errors'] for x in summary['splits'].values()):raise SystemExit('input audit failed')
print('PASS all split input manifests')
