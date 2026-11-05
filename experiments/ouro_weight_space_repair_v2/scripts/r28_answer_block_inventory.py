"""Fixed25 complete answer-field blocks, exact length strata and one cycle; CPU only."""
from pathlib import Path
import json,hashlib,sys,importlib.metadata
ROOT=Path(__file__).resolve().parents[1]
TOKENIZER_SHA='fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 from tokenizers import Tokenizer
 path=Path(sys.argv[1]);assert sha(path)==TOKENIZER_SHA;tok=Tokenizer.from_file(str(path))
 memory=json.loads((ROOT/'configs/r25_MEMORY_MANIFEST.json').read_text());old=json.loads((ROOT/'results/r24_VALUE_INVENTORY.json').read_text());text=memory['key_text'];assert text.endswith('Question:');prefix=text[:-len('Question:')];assert hashlib.sha256(prefix.encode()).hexdigest()==old['fewshot_prefix_sha256']
 encoded=tok.encode(text,add_special_tokens=False);ids=encoded.ids;offsets=encoded.offsets;blocks=[];errors=[]
 assert len(old['answer_char_fields'])==25
 for index,field in enumerate(old['answer_char_fields']):
  start,end=field['start'],field['end'];positions=[i for i,(a,b) in enumerate(offsets) if b>a and a<end and b>start];contained=all(start<=offsets[i][0] and offsets[i][1]<=end for i in positions);contiguous=bool(positions) and positions==list(range(positions[0],positions[-1]+1))
  if not contained or not contiguous:errors.append(dict(demo_index=index,reason='Full field intersects boundary token or non-contiguous/empty token block; no clipping/padding'))
  blocks.append(dict(demo_index=index,demo_id=field['demo_id'],field_start=start,field_end=end,field_text=text[start:end],field_definition='Frozen R24 complete answer field including the scaffold leading space, excluding following newline',positions=positions,first_position=positions[0] if positions else None,last_position=positions[-1] if positions else None,length=len(positions),token_ids=[ids[i] for i in positions],token_offsets=[list(offsets[i]) for i in positions],all_tokens_fully_contained=contained,contiguous=contiguous))
 groups=[];mapping=[]
 for length in sorted({b['length'] for b in blocks}):
  group=[b for b in blocks if b['length']==length];indices=[b['demo_index'] for b in group];groups.append(dict(token_length=length,demo_indices=indices,group_size=len(group),singleton_identity=len(group)==1))
  for i,recipient in enumerate(group):
   donor=group[(i+1)%len(group)];mapping.append(dict(recipient_demo_index=recipient['demo_index'],donor_demo_index=donor['demo_index'],token_length=length,identity_by_demo=recipient['demo_index']==donor['demo_index'],identical_token_content=recipient['token_ids']==donor['token_ids'],different_token_positions=sum(x!=y for x,y in zip(recipient['token_ids'],donor['token_ids']))))
 flat=[(p,t) for b in blocks for p,t in zip(b['positions'],b['token_ids'])];historical=[(e['source_token_position'],e['value_id']) for e in memory['entries']];assert flat==historical
 total=sum(b['length'] for b in blocks);moved=[m for m in mapping if not m['identity_by_demo']];different=[m for m in mapping if not m['identical_token_content']]
 summary=dict(demos=25,total_answer_field_tokens=total,length_groups=len(groups),non_singleton_demos=len(moved),non_singleton_demo_fraction=len(moved)/25,non_singleton_tokens=sum(m['token_length'] for m in moved),non_singleton_token_fraction=sum(m['token_length'] for m in moved)/total,content_changed_demos=len(different),content_changed_demo_fraction=len(different)/25,actual_different_token_positions=sum(m['different_token_positions'] for m in mapping),singleton_demos=25-len(moved),all25_exact_boundaries=not errors,historical143occurrence_positions_ids_exact=True)
 out=dict(status='complete_CPU_inventory_only',blocks=blocks,length_groups=groups,cycle_mapping=sorted(mapping,key=lambda m:m['recipient_demo_index']),cycle_definition='Within each exact-token-length group in original demo order, recipient i takes donor (i+1)%group_size; singleton identity; no search/resampling/clipping/padding',summary=summary,errors=errors,tokenizer_sha256=TOKENIZER_SHA,tokenizer_library_version=importlib.metadata.version('tokenizers'),fixed_prefix_sha256=old['fewshot_prefix_sha256'],source_sha256=sha(__file__),input_sha256={p:sha(ROOT/p) for p in ['configs/r25_MEMORY_MANIFEST.json','results/r24_VALUE_INVENTORY.json']},scope='Fixed25 text/token metadata only; no current ARC gold/outcomes, model, activations, estimator, function evidence or GPU authorization')
 p=ROOT/'results/r28_ANSWER_BLOCK_INVENTORY.json'
 with p.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(summary));print('inventory_sha256',sha(p))
if __name__=='__main__':main()
