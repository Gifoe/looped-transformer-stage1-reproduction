"""R24 gold-free full576 support extraction, tokenizer/text only; no model/states."""
import os,time,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from r24_common import ROOT,LOCK,read,sha,digest,save,gate
from r24_support import accepted_positions,specific_sets,candidate_support

def structured_prefix(demos):
 text='';fields=[]
 for demo in demos:
  text+='Question: '+demo['question']+'\nAnswer:';start=len(text);text+=' '+demo['answer'];end=len(text);fields.append(dict(demo_id=demo['id'],start=start,end=end));text+='\n\n'
 return text,fields

def main():
 spec=gate();manifest=read('configs/r24_INPUT_MANIFEST.json');assert manifest['n']==len(manifest['rows'])==576
 from transformers import AutoTokenizer
 from runtime import encode_pair
 tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Ouro-1.4B'),local_files_only=True,use_fast=True);assert tok.is_fast
 prefix,fields=structured_prefix(manifest['demonstrations']);canonical=None;count=0;started=time.monotonic()
 with (ROOT/'results/r24_support.jsonl').open('x',encoding='utf8') as out:
  for row in manifest['rows']:
   assert time.monotonic()<float(os.environ['R24_CPU_DEADLINE_MONOTONIC'])
   current='Question: '+row['current_question']+'\nAnswer:';assert prefix+current==row['prompt'];encoded=tok(row['prompt'],add_special_tokens=False,return_offsets_mapping=True);offsets=encoded['offset_mapping'];entries=[];boundaries=[]
   for field in fields:
    legal,cross=accepted_positions(offsets,field['start'],field['end'])
    entries.extend(dict(value_id=encoded['input_ids'][i],demo_id=field['demo_id'],source_token_position=i,key_prediction_position=i-1,offset=list(offsets[i])) for i in legal)
    boundaries.extend(dict(demo_id=field['demo_id'],source_token_position=i,offset=list(offsets[i]),value_id=encoded['input_ids'][i]) for i in cross)
   if canonical is None:
    all_values,specific,df,excluded=specific_sets(entries,lambda v:tok.decode([v],skip_special_tokens=False,clean_up_tokenization_spaces=False));canonical=dict(entries=entries,boundaries=boundaries,answer_char_fields=fields,S_all=sorted(all_values),S_specific=sorted(specific),demo_df=df,excluded_tokens=excluded,specific_demo_ids=sorted({e['demo_id'] for e in entries if e['value_id'] in specific}),fewshot_prefix_sha256=__import__('hashlib').sha256(prefix.encode()).hexdigest());save('results/r24_VALUE_INVENTORY.json',canonical)
   else:assert entries==canonical['entries'] and boundaries==canonical['boundaries']
   targets=[];prefix_ids=None
   for j,(context,continuation) in enumerate(row['request_pairs']):
    assert context==row['prompt'];p,t=encode_pair(tok,context,continuation);assert p==encoded['input_ids'] and digest(t)==row['target_id_sha256'][j] and len(t)==row['choice_token_counts'][j]
    assert continuation==' '+row['choice_texts'][j] and len(row['choice_texts'][j])==row['choice_char_lengths'][j] and len(t)>0
    if prefix_ids is None:prefix_ids=p
    else:assert p==prefix_ids
    targets.append(t)
   assert len(targets)==len(row['choice_char_lengths']);supports=candidate_support(targets,row['R6_winner'],all_values,specific)
   record=dict(id=row['id'],split=row['split'],R6_winner=row['R6_winner'],candidates=supports,audit=dict(prompt_structured_bytes_exact=True,offsets_match_actual_encode_pair_prefix_IDs=True,target_ids_and_candidate_order_exact=True,prefix_ids_sha256=digest(prefix_ids),candidate_target_id_sha256=[digest(t) for t in targets],answer_boundary_token_count=len(boundaries),value_occurrences=len(entries),value_inventory_sha256=sha('results/r24_VALUE_INVENTORY.json')))
   out.write(json.dumps(record,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno());count+=1
 assert count==576
 save('results/r24_SUPPORT_COMPLETE.json',dict(status='complete_gold_free_support',n=576,protocol_sha256=sha(LOCK),support_sha256=sha('results/r24_support.jsonl'),value_inventory_sha256=sha('results/r24_VALUE_INVENTORY.json'),all_prompt_offset_candidate_audits_passed=True,CPU_wall_seconds=time.monotonic()-started,GPU_worker_seconds=0,model_or_state_or_keys_loaded=False))
 print('R24_SUPPORT_COMPLETE',count)
if __name__=='__main__':main()
