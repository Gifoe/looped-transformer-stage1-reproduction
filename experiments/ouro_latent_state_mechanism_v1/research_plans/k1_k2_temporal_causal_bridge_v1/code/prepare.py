import pathlib,json,ast,random,hashlib,datetime
from transformers import AutoTokenizer
R=pathlib.Path('/root/ouro-temporal-bridge-sol-20261117');O=pathlib.Path('/root/ouro-computation-sol-20261116');tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/ouro-e4-e5/model',local_files_only=True,trust_remote_code=True);src=(O/'code/prepare_B.py').read_text();tree=ast.parse(src);prefix=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prefix' for t in n.targets));labels=list('ABCD');targets=[tok.encode(' '+c,add_special_tokens=False)[0] for c in labels];assert targets==[330,389,340,422];pairs=[];rows=[]
function=src[src.index('def build('):src.index('for i in range(160):')].replace("  else:\n   clause=f'The box","  elif template==1:\n   clause=f'The box")
replacement="""  else:
   clause=f'Object {obj} is stored in Box {variable}.\\n';mapping=f'Box {ba} is contained in Cabinet {maps[0]}.\\nBox {bb} is contained in Cabinet {maps[1]}.\\n';question=f'Question: Which cabinet ultimately encloses object {obj}?\\n'
  if reverse_mapping:mapping=''.join(reversed(mapping.splitlines(keepends=True)))
  text=prefix+clause+mapping+question"""
function=function.replace('  text=prefix+clause+mapping+question',replacement);exec(compile(function,'reused_legal_world_generator','exec'));rng=random.Random(2026111711);groups=[];used=set()
for i in range(96):
 split='development' if i<32 else 'confirmation';j=i if i<32 else i-32;n=32 if i<32 else 64;template=0 if j<n//4 else 1 if j<n//2 else 3;obj=rng.choice(list('ABCD') if i<32 else list('EFGH'));boxes=rng.sample(list('QRSTUVWX'),2);values=rng.sample(list(range(1,10)),4);order=rng.sample(values,4);reverse_mapping=bool(rng.getrandbits(1));key=(obj,tuple(boxes),tuple(values),template,reverse_mapping);assert key not in used;used.add(key);pid=f'TB{i:03d}';groups.append({'id':pid,'split':split,'template':template,'object':obj,'boxes':boxes,'values':values,'mapping_reversed':reverse_mapping})
 for rev in [False,True]:build(pid,split,template,obj,boxes,values,order,rev)
lookup={q['id']:q for q in rows}
for q in rows:
 enc=tok(q['text'],add_special_tokens=False,return_offsets_mapping=True);assert enc['input_ids']==q['ids'];start=q['text'].rfind('\n\n')+2;firstnl=q['text'].index('\n',start)+1;query=q['text'].index('Question:',firstnl);b=min(i for i,(a,e) in enumerate(enc['offset_mapping']) if e>firstnl);query_start=min(i for i,(a,e) in enumerate(enc['offset_mapping']) if e>query);q['b']=b;q['query_start']=query_start;q['regions']={'map':[b,query_start],'query':[query_start,q['R']],'read':[q['R'],q['R']+1]};assert 0<b<query_start<q['R'];q['prefix_len']=b
units=[]
for group in groups:
 pid=group['id']
 for rev in [0,1]:
  for mapping,(x,y) in enumerate([('original','original_edit'),('donor_edit','donor')]):
   for direction,(ref,altered) in enumerate([(x,y),(y,x)]):
    q0=lookup[f'{pid}_L{rev}_{ref}'];q1=lookup[f'{pid}_L{rev}_{altered}'];assert q0['b']==q1['b'] and q0['query_start']==q1['query_start'] and len(q0['ids'])==len(q1['ids']);b=q0['b'];assert q0['ids'][b:]==q1['ids'][b:];assert q0['mappings']==q1['mappings'] and q0['options']==q1['options'] and q0['correct_index']!=q1['correct_index'];assert q0['mappings'][q0['intermediate']]==q0['correct_cabinet'] and q1['mappings'][q1['intermediate']]==q1['correct_cabinet'];units.append({'id':f'{pid}_L{rev}_M{mapping}_D{direction}','pair_id':pid,'split':group['split'],'template':group['template'],'mapping':mapping,'direction':direction,'reverse_labels':bool(rev),'reference_W0':q0['id'],'input_W1':q1['id'],'answer0':q0['correct_index'],'answer1':q1['correct_index'],'cabinet0':q0['correct_cabinet'],'cabinet1':q1['correct_cabinet'],'b':b,'regions':q1['regions']})
for split in ['development','confirmation']:
 h={q['input_ids_sha256'] for q in rows if q['split']==split}
 if split=='development':old=h
 else:assert not old&h
(R/'INPUTS.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in rows));(R/'UNITS.jsonl').write_text(''.join(json.dumps(u)+'\n' for u in units));(R/'GROUPS.json').write_text(json.dumps(groups,indent=2));audit={'status':'PASS','groups':96,'development_units':256,'confirmation_units':512,'legal_worlds':len(rows),'factorial_relation2_mapping2_optionorders2_directions2':True,'D_token_ids_exact_between_paired_worlds':True,'complete_regions_partition_D':True,'new_template':'stored / contained in / ultimately encloses','seed':2026111711,'sha256':{n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['INPUTS.jsonl','UNITS.jsonl','GROUPS.json']},'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'GENERATOR_AUDIT.json').write_text(json.dumps(audit,indent=2));print('TEMPORAL_GENERATOR_PASS',len(rows),len(units))
