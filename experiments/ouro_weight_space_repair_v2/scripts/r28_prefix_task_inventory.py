"""Fixed24 known-prefix graph; selectively reads request/id fields only; CPU no model."""
from pathlib import Path
import json,hashlib,ast,types,sys,time
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def selected(text,wanted):
 # Skip unrequested JSON values syntactically without materializing gold/prediction/score fields.
 dec=json.JSONDecoder();i=1;out={}
 def white(i):
  while i<len(text) and text[i].isspace():i+=1
  return i
 def skip(i):
  start=i;stack=[];quoted=False
  while i<len(text):
   c=text[i]
   if quoted:
    if c=='\\':i+=2;continue
    if c=='"':quoted=False
   else:
    if c=='"':quoted=True
    elif c in '[{':stack.append(c)
    elif c in ']}':
     if not stack:return i
     stack.pop()
     if not stack:return i+1
    elif c==',' and not stack:return i
   i+=1
  return i
 while True:
  i=white(i)
  if text[i]=='}':return out
  key,i=dec.raw_decode(text,i);i=white(i);assert text[i]==':';i=white(i+1)
  if key in wanted:out[key],i=dec.raw_decode(text,i)
  else:i=skip(i)
  i=white(i)
  if text[i]=='}':return out
  assert text[i]==',';i+=1
class Adapter:
 def __init__(self,t):self.t=t
 def encode(self,s,**kw):return self.t.encode(s,add_special_tokens=False).ids
 def decode(self,ids,**kw):return self.t.decode(ids,skip_special_tokens=False)
def pair(tok,context,continuation):
 spaces=len(context)-len(context.rstrip())
 if spaces:continuation=context[-spaces:]+continuation;context=context[:-spaces]
 p=tok.encode(context);return p,tok.encode(context+continuation)[len(p):]
def stats(tasks):
 lengths=[t['length'] for t in tasks];return dict(unique_forwards=len(tasks),input_tokens=sum(lengths),min_length=min(lengths) if lengths else None,max_length=max(lengths) if lengths else None,depth_weighted_input_tokens=sum(t['length']*t['depth'] for t in tasks))
def main():
 started=time.monotonic()
 from tokenizers import Tokenizer
 tokenizer=Path(sys.argv[1]);assert sha(tokenizer)=='fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d';rawtok=Tokenizer.from_file(str(tokenizer));tok=Adapter(rawtok)
 private=ROOT/'transport/r28_LMEVAL_MODEL_SOURCE_PRIVATE.py';assert sha(private)=='8f7104bc45cb388c8df049bc5938ea0d639b867fa8c4b9c36c0d255f5b7f4e3e';tree=ast.parse(private.read_text(encoding='utf8'));fn=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='TemplateLM' for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_encode_pair');fn.returns=None
 for a in fn.args.args:a.annotation=None
 ns={'transformers':types.SimpleNamespace(AutoModelForSeq2SeqLM=object())};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'private_source_encode_pair','exec'),ns);original=lambda c,t:ns['_encode_pair'](types.SimpleNamespace(tok_encode=lambda s,**kw:tok.encode(s)),c,t)
 cohort=selected((ROOT/'configs/r20_COHORT_ADMISSION_LOCK.json').read_text(encoding='utf8'),{'cohort'})['cohort'];assert len(cohort)==24
 wanted={r['id'] for r in cohort};rows={}
 for split in ['discovery','selection']:
  for line in (ROOT/('data/slcd_'+split+'.jsonl')).read_text(encoding='utf8').splitlines():
   identity=selected(line,{'id'})['id']
   if identity in wanted:rows[identity]=selected(line,{'id','request_pairs'})
 assert set(rows)==wanted
 refs=json.loads((ROOT/'configs/r7_reference_manifest.json').read_text(encoding='utf8'))['references'];memory=json.loads((ROOT/'configs/r25_MEMORY_MANIFEST.json').read_text(encoding='utf8'));prefix=memory['key_text'][:-len('Question:')];blocks=json.loads((ROOT/'results/r28_ANSWER_BLOCK_INVENTORY.json').read_text(encoding='utf8'));source_positions=[e['source_token_position'] for e in memory['entries']];source_values=[e['value_id'] for e in memory['entries']]
 from r7_references import prefix_alignment
 prefixes={};tasks={};usages=[];failures=[];candidate_records=[];context_positions=Counter();within_question={};groups={};q_group={}
 mapdemo={b['demo_index']:b for b in blocks['blocks']};source_map=[]
 for m in blocks['cycle_mapping']:
  a=mapdemo[m['recipient_demo_index']]['positions'];b=mapdemo[m['donor_demo_index']]['positions'];assert len(a)==len(b);source_map.extend(zip(a,b))
 source_map=sorted(source_map);mapsha=digest(source_map)
 def prefix_pool(context,ids,text):
  key=digest(ids)
  if key not in prefixes:
   offsets=rawtok.encode(text,add_special_tokens=False).offsets;assert rawtok.encode(text,add_special_tokens=False).ids==ids
   if context!='N0':
    assert text.startswith(prefix) and [ids[p] for p in source_positions]==source_values
    for block in blocks['blocks']:
     assert [list(offsets[p]) for p in block['positions']]==block['token_offsets'] and max(block['positions'])<len(ids)
   prefixes[key]=dict(ids=ids,length=len(ids),source25_answer_fields_exact=context!='N0',source_token_count=143 if context!='N0' else 0)
  return key
 def add(context,mode,depth,p,t,j,text):
  ids=p+t[:j];known=tok.encode(text.rstrip()+tok.decode(t[:j]));exact=known==ids
  if not exact:
   first=next((i for i,(a,b) in enumerate(zip(known,ids)) if a!=b),min(len(known),len(ids)));failures.append(dict(context=context,mode=mode,position=j,known_consumed_text=tok.decode(t[:j]),expected_prefix_ids=ids,independently_encoded_ids=known,first_difference=first))
  pref=prefix_pool(context,p,text.rstrip());identity=dict(input_ids=ids,context=context,mode=mode,depth=depth,checkpoint_revision='574fa66cb8bf5abdc979642d01cf2b79b16bfab1',backend='sdpa',dtype='BF16',primitive='true_prefix_single_position_head',source_map_sha256=mapsha if mode=='answer_block_CF4' else None);key=digest(identity)
  if context!='N0':assert all(pos<len(ids) and ids[pos]==v for pos,v in zip(source_positions,source_values))
  record=dict(key=key,context=context,mode=mode,depth=depth,prefix_ref=pref,consumed_ids=t[:j],input_ids_sha256=digest(ids),length=len(ids),read_position=len(ids)-1,head_shape=[1,1,2048],independent_consumed_encoding_exact=exact,source_map_sha256=identity['source_map_sha256'])
  if key in tasks:assert tasks[key]==record
  else:tasks[key]=record
  return key
 for qi,item in enumerate(cohort):
  qid=item['id'];within_question[qid]=set()
  for candidate,request in enumerate(rows[qid]['request_pairs']):
   qp=pair(tok,*request);assert qp==original(*request);pairs=[pair(tok,refs[n],request[1]) for n in ['N0','N1','N2']];assert all(p==original(refs[n],request[1]) for n,p in zip(['N0','N1','N2'],pairs));mapping,reason=prefix_alignment(tok,request,qp,pairs)
   if mapping is None:failures.append(dict(id=qid,candidate=candidate,reason=reason));continue
   candidate_records.append(dict(id=qid,candidate=candidate,Q_target_positions=len(qp[1]),reference_target_positions={n:len(p[1]) for n,p in zip(['N0','N1','N2'],pairs)},reference_position_maps={n:m for n,m in zip(['N0','N1','N2'],mapping)}))
   for t in range(len(qp[1])):
    qkey=add('Q','native',5,*qp,t,request[0]);keys={'Q':qkey};context_positions['Q']+=1
    for n,rp,m in zip(['N0','N1','N2'],pairs,mapping):
     keys[n+'_native']=add(n,'native',4,*rp,m[t],refs[n]);context_positions[n+'_native']+=1
     if n!='N0':keys[n+'_CF']=add(n,'answer_block_CF4',4,*rp,m[t],refs[n]);context_positions[n+'_CF']+=1
    groupkey=digest({k:v for k,v in keys.items() if k!='Q'})
    groups.setdefault(groupkey,dict(reference_tasks={k:v for k,v in keys.items() if k!='Q'},Q_consumers=[]))
    if qkey not in groups[groupkey]['Q_consumers']:groups[groupkey]['Q_consumers'].append(qkey)
    if qkey in q_group:assert q_group[qkey]==groupkey
    q_group[qkey]=groupkey
    usages.append(dict(id=qid,candidate=candidate,Q_position=t,tasks=keys,stream_group=groupkey));within_question[qid].update(keys.values())
 summary={}
 for name,context,mode in [('Q','Q','native'),('N0','N0','native'),('N1_native','N1','native'),('N1_CF','N1','answer_block_CF4'),('N2_native','N2','native'),('N2_CF','N2','answer_block_CF4')]:summary[name]=stats([x for x in tasks.values() if x['context']==context and x['mode']==mode])
 UQ,U0,U1,U2=[summary[n]['unique_forwards'] for n in ['Q','N0','N1_native','N2_native']];T=.405*UQ+.0739125*U0+.650*(U1+U2);planning=1.25*T+100
 out=dict(status='passed_CPU_prefix_admission' if not failures else 'CPU_prefix_admission_failed_no_model',cohort=[{'id':r['id'],'split':r['split']} for r in cohort],candidate_records=candidate_records,prefix_pool=prefixes,tasks=list(tasks.values()),usages=usages,stream_groups=list(groups.values()),failure_counterexamples=failures,counts=dict(by_context=summary,total=stats(list(tasks.values())),original_Q_positions=len(usages),context_position_uses=dict(context_positions),sum_unique_tasks_per_question=sum(len(v) for v in within_question.values()),global_unique_tasks=len(tasks),stream_groups=len(groups),Q_shared_across_candidate_uses=len(usages)-UQ),planning_heuristic=dict(T_seconds=T,one_point25T_plus100=planning,threshold1400_pass=planning<=1400,scope='Predeclared design heuristic, not measured runtime or hard bound'),metadata=dict(tokenizer_sha256=sha(tokenizer),lm_eval_source_sha256=sha(private),runtime_source_sha256=sha(ROOT/'scripts/runtime.py'),alignment_source_sha256=sha(ROOT/'scripts/r7_references.py'),source_map_sha256=mapsha,source_sha256=sha(__file__),CPU_wall_seconds=time.monotonic()-started,model_loaded=False,GPU_worker_seconds=0),scope='Fixed24 all candidates; selectively parsed IDs/request_pairs only, no gold/prediction/score; no unknown future IDs in task input/cache key; exact input memo only, not KV/full teacher-forcing cache')
 p=ROOT/'configs/r28_TRUE_PREFIX_TASK_INVENTORY.json'
 with p.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(status=out['status'],counts=out['counts'],failures=len(failures),planning=out['planning_heuristic'])));print('manifest_sha256',sha(p))
if __name__=='__main__':main()
