"""Independent all-unit prompt semantics audit, without model outcomes or query gold."""
import json,re,hashlib,collections
from pathlib import Path
from tokenizers import Tokenizer
R=Path(__file__).resolve().parent;t=Tokenizer.from_file('/root/autodl-tmp/ouro-e4-e5/model/tokenizer.json')
def parse(s):
 question,ans=s.rsplit('\nAnswer:',1);parts=re.split(r'\n([A-E])\. ',question);return parts[0],parts[1::2],parts[2::2],ans
sources={(q['split'],q['id']):q for q in map(json.loads,(R/'REENCODING_STRUCTURED_TRAIN_VAL.jsonl').read_text().splitlines())};demos=json.loads((R/'REENCODING_DEMONSTRATIONS.json').read_text());cells=collections.defaultdict(set);labels=collections.Counter();n=0;identity=0
with (R/'E7_VARIANTS.jsonl').open() as f:
 for line in f:
  v=json.loads(line);q=sources[v['split'],v['id']];c=len(q['choices']);p,s=v['p'],v['s'];rename={ch:'ABCD'[(i+s)%4] for i,ch in enumerate('ABCD')};rename['E']='E';assert not any(k in v for k in ['gold','query_gold','correct'])
  blocks=re.split(r'(?:^|\n\n)Question: ',v['prompt']);assert blocks[0]=='' and len(blocks)==27
  for original,block in zip(demos,blocks[1:26]):
   stem,ls,options,answer=parse(block);assert stem==original['stem'] and options==original['choices'],'P must not reorder demonstrations';assert ls==[rename[x] for x in original['labels']];assert answer==' '+rename[original['answer_text'].strip()]
  stem,ls,options,ans=parse(blocks[-1]);mapping=[(j+p)%c for j in range(c)];assert stem==q['stem'] and ans=='' and options==[q['choices'][i] for i in mapping];assert ls==[rename[x] for x in q['labels']]==v['labels'];assert mapping==v['display_to_original'];assert v['targets']==[t.encode(' '+x,add_special_tokens=False).ids[0] for x in ls]
  assert t.decode(v['ids'],skip_special_tokens=False)==v['prompt'];assert hashlib.sha256(json.dumps(v['ids']).encode()).hexdigest()==v['input_ids_sha256'];assert v['R']==len(v['ids'])-1
  if p==s==0:assert v['input_ids_sha256']==q['source_input_ids_sha256'];identity+=1
  key=v['split'],v['id'];assert (p,s) not in cells[key];cells[key].add((p,s));labels[v['split']+':m'+str(c)+':'+''.join(ls)]+=1;n+=1
assert n==22280 and len(cells)==1393 and identity==1393
for key,cc in cells.items():m=len(sources[key]['choices']);assert cc=={(p,s) for p in range(m) for s in range(4)}
result={'status':'PASS','variant_units':n,'questions':len(cells),'identity_roundtrip':identity,'all_demo_contents_and_order_unchanged_underP':True,'all_demo_labels_and_answers_consistent_underS':True,'all_query_stems_and_candidate_contents_preserved':True,'all_semantic_display_maps_bijective':True,'complete_counterbalance':True,'all_token_text_decodes_exact':True,'no_current_query_gold':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'label_set_counts':dict(labels),'scope_limit':'Structure and lexical/manual prior audit preserve task semantics as checked; not a formal proof of arbitrary natural-language equivalence and not an accuracy result.'}
(R/'E7_ALL_VARIANT_SEMANTIC_AUDIT.json').write_text(json.dumps(result,indent=2));print('E7_ALL_VARIANT_SEMANTIC_AUDIT_PASS',n,len(cells),flush=True)
