"""Label-free deterministic input construction and finite interpretation."""
V=49152
def variants(tokens,cuts):
 assert len(cuts)==2 and 0<cuts[0]<cuts[1]<len(tokens)-1
 assert all(isinstance(x,int) and 0<=x<V for x in tokens)
 def changed(c):return tokens[:c+1]+[(x+1)%V for x in tokens[c+1:]]
 return [('A',list(tokens),list(cuts)),('B1',changed(cuts[0]),list(cuts)),('B2',changed(cuts[1]),list(cuts)),('A_repeat',list(tokens),list(cuts)),('C1',tokens[:cuts[0]+1],[cuts[0]]),('C2',tokens[:cuts[1]+1],[cuts[1]])]
def build_inputs(tok,cohort,rows,refs,memory):
 from runtime import encode_pair
 chosen=None;selection_audit=[]
 for item in cohort:
  for j,request in enumerate(rows[item['id']]['request_pairs']):
   pairs=[encode_pair(tok,*request)]+[encode_pair(tok,refs[n],request[1]) for n in ['N0','N1','N2']]
   selection_audit.append(dict(id=item['id'],candidate=j,target_lengths=[len(t) for p,t in pairs],eligible=all(len(t)>=3 for p,t in pairs)))
   if all(len(t)>=3 for p,t in pairs):chosen=(item,j,pairs);break
  if chosen:break
 assert chosen is not None
 item,j,pairs=chosen;contexts=[]
 for name,(p,t) in zip(['Q','N0','N1','N2'],pairs):
  tokens=p+t[:-1];cuts=[len(p)-1,len(p)]
  contexts.append(dict(name=name,depth=5 if name=='Q' else 4,tokens=tokens,cuts=cuts,prefix_length=len(p),target_length=len(t)))
 encoded=tok(memory['key_text'],add_special_tokens=False,return_offsets_mapping=True);tokens=encoded['input_ids'];entries=memory['entries'];cuts=[entries[0]['key_prediction_position'],entries[-1]['key_prediction_position']]
 assert all(list(encoded['offset_mapping'][e['source_token_position']])==e['offset'] for e in entries)
 assert len(entries)==143 and all(tokens[e['source_token_position']]==e['value_id'] and e['key_prediction_position']==e['source_token_position']-1 for e in entries)
 through=max(e['source_token_position'] for e in entries)+2
 assert pairs[0][0][:through]==tokens[:through]
 contexts.append(dict(name='key',depth=4,tokens=tokens,cuts=cuts))
 for c in contexts:variants(c['tokens'],c['cuts'])
 return dict(id=item['id'],split=item['split'],candidate=j,contexts=contexts,selection_audit=selection_audit,selection='First cohort/candidate with all Q/N012 targets >=3; no labels')
def validate_records(records):
 assert [r['context'] for r in records]==['Q','N0','N1','N2','key']
 total=0
 for r in records:
  d=5 if r['context']=='Q' else 4;assert r['complete'] and r['depth']==d
  cuts=r['cuts'];assert len(cuts)==2 and 0<cuts[0]<cuts[1]
  expected={(v,k,p) for v,positions in [('A_repeat',cuts),('B1',cuts[:1]),('B2',cuts[1:]),('C1',cuts[:1]),('C2',cuts[1:])] for k in range(1,d+1) for p in positions}
  actual=[(x['variant'],x['depth'],x['position']) for x in r['comparisons']]
  assert len(actual)==len(set(actual))==6*d and set(actual)==expected
  inputs=r['inputs'];assert len(inputs)==6
  plan=variants(inputs[0]['input_ids'],cuts)
  from r7_core import digest
  for x,(name,t,p) in zip(inputs,plan):
   assert x['variant']==name and x['input_ids']==t and x['positions']==p and x['shape']==[1,len(t)] and x['input_sha256']==digest(t)
  for x in r['comparisons']:
   for domain in ['state','head']:
    q=x[domain];assert type(q['exact']) is bool and len(q['shape'])==3 and q['shape'][:2]==[1,1]
    assert q['shape'][2]==(V if domain=='head' else 2048)
    assert q['dtype']==('torch.float32' if domain=='head' else 'torch.bfloat16')
    assert all(len(q[k])==64 for k in ['sha256_A','sha256_B'])
    import math
    assert all(math.isfinite(q[k]) and q[k]>=0 for k in ['max_abs','RMS','unequal_elements'])
    assert q['exact']==(q['unequal_elements']==0)==(q['sha256_A']==q['sha256_B'])
    if q['exact']:assert q['max_abs']==q['RMS']==0
  total+=len(actual)
 assert total==126
 return total

def summarize(records):
 repeat=all(x['state']['exact'] and x['head']['exact'] for r in records for x in r['comparisons'] if x['variant']=='A_repeat')
 suffix=all(x['state']['exact'] and x['head']['exact'] for r in records for x in r['comparisons'] if x['variant'] in ['B1','B2'])
 length=all(x['state']['exact'] and x['head']['exact'] for r in records for x in r['comparisons'] if x['variant'] in ['C1','C2'])
 category='inconclusive_repeat' if not repeat else ('content_invariance_not_established' if not suffix else ('length_bitwise_sensitive' if not length else 'finite_config_prefix_evidence'))
 return dict(A_repeat_all_exact=repeat,A_B_all_exact=suffix,A_C_all_exact=length,category=category,comparisons=sum(len(r['comparisons']) for r in records),scope='Finite selected-context evidence only; no semantic leakage, universal causality or accuracy claim')

def interpret(records):
 validate_records(records)
 out=summarize(records);out['per_context']={r['context']:summarize([r]) for r in records};return out
