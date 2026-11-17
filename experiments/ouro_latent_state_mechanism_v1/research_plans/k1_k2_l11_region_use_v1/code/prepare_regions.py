import pathlib,json,ast,hashlib,datetime
from transformers import AutoTokenizer
R=pathlib.Path('/root/ouro-region-use-sol-20261117');P=pathlib.Path('/root/ouro-semantic-read-sol-20261117');D=pathlib.Path('/root/ouro-binding-interface-sol-20261117')
src=(P/'code/prepare.py').read_text().replace("i=j if split=='development' else j+16","i=j+80")
t=ast.parse(src);ix=next(i for i,n in enumerate(t.body) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [x.id for x in n.target.elts]==['split','count','seed','pool'])
t.body=t.body[:ix+1];t.body[-1].iter=ast.parse("[('region_confirmation',64,2026111743,list('EFGHIJKL'))]",mode='eval').body
ns={'__name__':'region_generator'};exec(compile(ast.fix_missing_locations(t),'inherited_generator','exec'),ns)
fresh=ns['rows'];groups=ns['groups'];tok=ns['tok'];old=list(map(json.loads,(P/'INPUTS.jsonl').read_text().splitlines()));dev=list(map(json.loads,(D/'DEV_INPUTS.jsonl').read_text().splitlines()))
oldhash={q['input_ids_sha256'] for q in old};chash=set()
for q in old:
 prefix=ns['prefix'];assert q['text'].startswith(prefix);s=q['text'][len(prefix):];end=s.index('Question:');a=s[:end];b=s[end:]
 changes={0:[(' is in Box ',' is inside Box '),(' is in Cabinet ',' is inside Cabinet ')],1:[('The box containing object ','The box holding object '),(' is located inside Cabinet ',' is placed inside Cabinet ')],3:[(' is stored in Box ',' is kept in Box '),(' is contained in Cabinet ',' is located in Cabinet ')]}[q['template']]
 for a0,a1 in changes:a=a.replace(a0,a1)
 text=prefix+a+b;ids=tok.encode(text,add_special_tokens=False);assert len(ids)==len(q['ids']);chash.add(hashlib.sha256(json.dumps(ids).encode()).hexdigest())
assert len(fresh)==2048 and len(groups)==64 and len({q['input_ids_sha256'] for q in fresh})==2048 and not {q['input_ids_sha256'] for q in fresh}&(oldhash|chash)
rows=dev+fresh
for q in rows:
 en=tok(q['text'],add_special_tokens=False,return_offsets_mapping=True);assert en['input_ids']==q['ids'];off=en['offset_mapping'];start=q['text'].rfind('\nA. Cabinet ')+1;assert start>0;positions=[i for i,(a,b) in enumerate(off) if a<start and b>off[q['H_start']][0]];assert positions==list(range(q['H_start'],max(positions)+1));qb=max(positions)+1;assert all(off[i][1]<=start for i in positions) and off[qb][0]>=start;q['Q_span']=[q['H_start'],qb];q['O_span']=[qb,q['R']];assert q['H_start']<qb<q['R'];assert list(range(*q['Q_span']))+list(range(*q['O_span']))+[q['R']]==list(range(q['H_start'],q['R']+1))
for split in ['development','region_confirmation']:
 z=[q for q in rows if q['split']==split]
 for role in ['correct_index','opposite_index']:assert len({sum(q[role]==j for q in z) for j in range(4)})==1
 for pid in {q['pair_id'] for q in z}:
  g=[q for q in z if q['pair_id']==pid];assert len(g)==32
  for q in g:
   p=next(p for p in g if all(p[k]==q[k] for k in ['firsthop_factor','endpoint_factor','selector','record_order']) and p['label_factor']!=q['label_factor']);assert q['ids'][:q['Q_span'][1]]==p['ids'][:p['Q_span'][1]] and q['Q_span']==p['Q_span'] and q['O_span']==p['O_span']
(R/'INPUTS.jsonl').write_text(''.join(json.dumps(q,separators=(',',':'))+'\n' for q in rows));(R/'FRESH_GROUPS.json').write_text(json.dumps(groups,indent=2))
out={'status':'PASS','development_worlds':512,'fresh_confirmation_worlds':2048,'fresh_basegroups':64,'seed':2026111743,'old_R_hashes':len(oldhash),'old_samebinding_C_hashes':len(chash),'fresh_hash_disjoint':True,'Q_O_R_exact_disjoint_exhaustive':True,'label_prefix_token_identity_allworlds':True,'new_model_outcomes':0,'no_filtering':True,'within_family_only':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'INPUT_CPU_AUDIT.json').write_text(json.dumps(out,indent=2));print('REGION_INPUT_CPU_PASS',out,flush=True)
