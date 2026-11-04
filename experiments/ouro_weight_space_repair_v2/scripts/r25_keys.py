"""R25 key construction and selected/full native4 exact audit, no hooks."""
import time
from r25_common import ROOT,sha,digest,read

def key_tokens(tok,memory):
 tokens=tok.encode(memory['key_text'],add_special_tokens=False);entries=memory['entries'];assert len(entries)==143 and all(entries[i]['source_token_position']<entries[i+1]['source_token_position'] for i in range(len(entries)-1))
 for e in entries:assert e['source_token_position']>0 and e['key_prediction_position']==e['source_token_position']-1 and tokens[e['source_token_position']]==e['value_id']
 return tokens

def audit_inputs_before_model(tok,memory,data):
 from runtime import encode_pair
 tokens=key_tokens(tok,memory);assert memory['key_text'].endswith('Question:');prefix=memory['key_text'][:-len('Question:')];encoded=tok(memory['key_text'],add_special_tokens=False,return_offsets_mapping=True);assert encoded['input_ids']==tokens
 for e in memory['entries']:assert list(encoded['offset_mapping'][e['source_token_position']])==e['offset']
 through=max(e['source_token_position'] for e in memory['entries'])+2;records=[]
 for chosen,clean,expected,state in data:
  for j,request in enumerate(clean['request_pairs']):
   context,continuation=request;assert context[:len(prefix)]==prefix and context[len(prefix):].startswith('Question: ')
   p,t=encode_pair(tok,*request);assert p[:through]==tokens[:through] and digest(t)==state['target_id_sha256'][j] and len(t)==state['choice_token_counts'][j]
   records.append(dict(id=chosen['id'],candidate=j,prefix_through_memory_boundary_exact=True,target_ids_sha256=digest(t),prefix_ids_sha256=digest(p)))
 assert len(data)==24
 return tokens,dict(status='passed_CPU_all24_pre_model_input_audit',n=24,candidates=len(records),records=records,key_input_ids_sha256=digest(tokens),all143value_ids_offsets_predecessors_exact=True,model_or_key_or_query_loaded=False,scope='CPU tokenizer/input metadata only, before load_model and before key extraction; repeated candidate checks remain')

def build(model,scorer,tokens,memory,timing):
 import torch
 from r23_readouts import heads
 from r25_math import torch_sqrt
 positions=[e['key_prediction_position'] for e in memory['entries']];assert max(positions)<len(tokens)
 def run(selected):
  scorer.kmax=4;scorer._set_depth(4);assert model.config._attn_implementation=='sdpa'
  return scorer._run(tokens,positions=selected,use_cache=False)[1]
 torch.cuda.synchronize();start=time.monotonic();selected=run(positions);logits=heads(model,selected,[4])[4];keys=torch_sqrt(logits);torch.cuda.synchronize();timing['key_build']=time.monotonic()-start
 start=time.monotonic();full=run(range(max(positions)+1));ix=torch.as_tensor(positions,device=full[3].device);assert torch.equal(selected[3],full[3].index_select(1,ix));fullhead=heads(model,full,[4])[4];assert torch.equal(logits,fullhead[torch.as_tensor(positions,device=fullhead.device)]);torch.cuda.synchronize();timing['key_audit']=time.monotonic()-start
 audit=dict(selected_vs_full_native4_BF16_states_exact=True,selected_vs_full_native4_head_fullV_exact=True,positions=positions,key_input_ids_sha256=digest(tokens),key_input_length=len(tokens),key_count=len(positions),head_shape=list(logits.shape),sqrt_key_bytes=keys.numel()*keys.element_size(),original_uncached_run_source_sha256=sha('scripts/ouro_cached_scoring.py'),evidence_scope='Source-bound original _run entry arguments/backend, not an actual HF call interception receipt',call_contract='Original _run(tokens,positions, use_cache=False), depth4; immutable source executes model.model(input_ids=x,use_cache=False); no hooks/explicitmask/positionids/pastkv changes',attention_backend=model.config._attn_implementation,keys_input_no_current_question_or_answer=True)
 return keys,logits,audit
