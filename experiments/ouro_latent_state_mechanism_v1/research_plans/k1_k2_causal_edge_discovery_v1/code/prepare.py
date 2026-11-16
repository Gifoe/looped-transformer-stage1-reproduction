"""Reuse the original symbolic generator, add frozen fresh wording/order and source controls."""
import pathlib,json,ast,random,hashlib,datetime
from transformers import AutoTokenizer
R=pathlib.Path('/root/ouro-causal-edge-sol-20261117');O=pathlib.Path('/root/ouro-computation-sol-20261116');tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/ouro-e4-e5/model',local_files_only=True,trust_remote_code=True);src=(O/'code/prepare_B.py').read_text();tree=ast.parse(src);prefix=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prefix' for t in n.targets));labels=list('ABCD');targets=[tok.encode(' '+c,add_special_tokens=False)[0] for c in labels];assert targets==[330,389,340,422]
oldunits=[q for q in map(json.loads,(O/'B_UNITS.jsonl').read_text().splitlines()) if q['split']=='development'];oldids={u[k] for u in oldunits for k in ['receiver','donor','text_edit','same_variable']};oldrows=[q for q in map(json.loads,(O/'B_INPUTS.jsonl').read_text().splitlines()) if q['id'] in oldids];pairs=[];rows=[]
function=src[src.index('def build('):src.index('for i in range(160):')]
function=function.replace("  else:\n   clause=f'The box", "  elif template==1:\n   clause=f'The box")
needle="  text=prefix+clause+mapping+question"
replacement="""  else:
   clause=f'Object {obj} occupies Box {variable}.\\n';mapping=f'Box {ba} rests within Cabinet {maps[0]}.\\nBox {bb} rests within Cabinet {maps[1]}.\\n';question=f'Question: In which cabinet does object {obj} ultimately reside?\\n'
  if reverse_mapping:mapping=''.join(reversed(mapping.splitlines(keepends=True)))
  text=prefix+clause+mapping+question"""
assert needle in function;function=function.replace(needle,replacement);exec(compile(function,'reused_generator','exec'));rng=random.Random(2026111701);oldcombos={(q['intermediate'],tuple(q['mappings'].items())) for q in oldrows};used=set()
for i in range(64):
 template=0 if i<16 else 1 if i<32 else 2;obj=rng.choice(list('IJKL'));boxes=rng.sample(list('IJKLMNOP'),2);values=rng.sample([5,6,7,8,9],4);order=rng.sample(values,4);key=(obj,tuple(boxes),tuple(values));assert key not in used;used.add(key);reverse_mapping=bool(rng.getrandbits(1))
 for rev in [False,True]:build(f'NEW{i:03d}','fresh_confirmation',template,obj,boxes,values,order,rev)
for q in rows:q['mapping_clause_order_varied']=True
rows=oldrows+rows;pairs=oldunits+pairs
for q in rows:
 enc=tok(q['text'],add_special_tokens=False,return_offsets_mapping=True);assert enc['input_ids']==q['ids'];start=q['text'].rfind('\n\n')+2;stop=q['text'].index('Question:',start);eligible=[i for i,(a,b) in enumerate(enc['offset_mapping']) if a>=start and b<=stop and any(c.isalnum() for c in q['text'][a:b])];q['M']=eligible[-1];assert q['source_span'][1]<=q['mapping_start']<=q['M']<q['R'];q['candidate_texts']=['Cabinet '+str(x) for x in q['options']]
lookup={q['id']:q for q in rows};aux=[]
for u in pairs:
 q=lookup[u['receiver']];variable=next(x for x in 'ABCDEFGHJKLMNOPQRSTUVWXYZ' if x not in q['mappings']);text=q['text'];start=text.rfind('\n\n')+2;pos=text.index('Box '+q['intermediate'],start)+4;new=text[:pos]+variable+text[pos+len(q['intermediate']):];enc=tok(new,add_special_tokens=False,return_offsets_mapping=True);z={**q,'id':q['id']+'_unrelated_source','text':new,'ids':enc['input_ids'],'input_ids_sha256':hashlib.sha256(json.dumps(enc['input_ids']).encode()).hexdigest(),'intermediate':variable,'correct_index':None,'auxiliary_not_scored':True};assert len(z['ids'])==len(q['ids']) and z['ids'][:q['source_span'][0]]==q['ids'][:q['source_span'][0]];aux.append(z);u['unrelated_source']=z['id']
for q in aux:
 if q['id'] not in lookup:rows.append(q);lookup[q['id']]=q
for u in pairs:
 qr,qd,qe,qs=[lookup[u[k]] for k in ['receiver','donor','text_edit','same_variable']];assert qr['M']==qd['M']==qe['M']==qs['M'];assert len({u['predicted_cross_index'],u['original_index'],u['donor_index']})==3;assert qe['correct_index']==u['predicted_cross_index'];assert qr['ids'][:qr['source_span'][0]]==qd['ids'][:qd['source_span'][0]]
(R/'INPUTS.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in rows));(R/'UNITS.jsonl').write_text(''.join(json.dumps(u)+'\n' for u in pairs));(R/'GENERATOR_AUDIT.json').write_text(json.dumps({'status':'PASS','development_basepairs':32,'development_units':128,'fresh_basepairs':64,'fresh_units':256,'fresh_seed':2026111701,'new_template':'Object X occupies Box Y; Box Y rests within Cabinet Z; in which cabinet ultimately resides','fresh_objects':'IJKL','fresh_boxes':'IJKLMNOP','fresh_cabinets':[5,6,7,8,9],'mapping_order_varied':True,'symbolic_cross_labels_verified':True,'source_positions_lengths_andM_aligned':True,'auxiliary_control':'third-boxfirsthopbeforemapping,not_scored asfulltask','sha256':{n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['INPUTS.jsonl','UNITS.jsonl']},'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2));print('GENERATOR_PASS',len(rows),len(pairs))
