import json,hashlib,pathlib,os,datetime
import pyarrow.parquet as pq
from transformers import AutoTokenizer
P=pathlib.Path('/home/luansiyu/loop-latent');R=P/'repo';E=R/'experiments/ouro_latent_state_mechanism_v1';D=R/'experiments/ouro_weight_space_repair_v2/data/slcd_fixed_demonstrations.json';O=pathlib.Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();h=lambda s:hashlib.sha256(s.encode('utf8')).hexdigest()
def qp(question,choices):return 'Question: '+question+'\n'+'\n'.join(chr(65+i)+'. '+a for i,a in enumerate(choices))+'\nAnswer:'
demo=json.loads(D.read_text()); assert len(demo['documents'])==25 and [d['id'] for d in demo['documents']]==demo['ids'];prefix='\n\n'.join(qp(d['question'],d['choices']['text'])+' '+chr(65+d['choices']['label'].index(d['answerKey'])) for d in demo['documents'])+'\n\n';tok=AutoTokenizer.from_pretrained(P/'model',local_files_only=True,trust_remote_code=True);stats={};sources={str(p):sha(p) for p in [D,E/'prompt_spans.py',R/'experiments/ouro_weight_space_repair_v2/scripts/r10_core.py',P/'model/tokenizer.json',P/'model/tokenizer_config.json']}
with (O/'boundary_ledger.jsonl').open('x',encoding='utf8') as f:
 for split in ('train','validation','test'):
  path=P/'data'/(split+'.parquet'); mf=E/'results'/('INPUT_MANIFEST_'+split+'.jsonl');sources[str(path)]=sha(path);sources[str(mf)]=sha(mf);records={}
  for line in mf.open(encoding='utf8'):
   v=json.loads(line)
   if v['candidate']==0:
    if v['id'] in records:raise ValueError('duplicate manifest ID')
    records[v['id']]=v
  rows=pq.read_table(path).to_pylist();assert len(records)==len(rows) and set(records)=={r['id'] for r in rows};counts={'n':0,'cross_boundary_questions':0,'unknown_zero_offset_questions':0,'R_in_prefix':0,'R_crossing':0,'R_current':0}
  for row in rows:
   prompt=prefix+qp(row['question'],row['choices']['text']);enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];offsets=enc['offset_mapping'];r=records[row['id']];assert h(prompt)==r['prompt_sha256'] and hashlib.sha256(json.dumps(ids).encode()).hexdigest()==r['input_sha256'] and len(ids)==r['input_length'];boundary=len(prefix);groups={'prefix':[],'current':[],'crossing':[],'zero_offset':[]}
   for i,(a,b) in enumerate(offsets):
    if b<=a:groups['zero_offset'].append(i)
    elif b<=boundary:groups['prefix'].append(i)
    elif a>=boundary:groups['current'].append(i)
    else:groups['crossing'].append(i)
   Rpos=len(ids)-1; assert sum(len(x) for x in groups.values())==len(ids);Rgroup=next(k for k,v in groups.items() if Rpos in v);counts['n']+=1;counts['cross_boundary_questions']+=bool(groups['crossing']);counts['unknown_zero_offset_questions']+=bool(groups['zero_offset']);counts['R_in_prefix']+=Rgroup=='prefix';counts['R_crossing']+=Rgroup=='crossing';counts['R_current']+=Rgroup=='current'
   obj={'split':split,'id':row['id'],'prompt_sha256':r['prompt_sha256'],'input_sha256':r['input_sha256'],'prefix_chars':boundary,'input_length':len(ids),'readout_position':Rpos,'readout_group':Rgroup,'groups':groups,'crossing_offsets':[{'index':i,'offset':offsets[i]} for i in groups['crossing']]};f.write(json.dumps(obj,separators=(',',':'))+'\n')
  stats[split]=counts
summary={'status':'INPUT_BOUNDARIES_ONLY_NOT_INTERVENTION','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prefix_sha256':h(prefix),'prefix_chars':len(prefix),'partitions':'offset ending at/before prefix boundary; starting at/after boundary; strict crossings and zero offsets separately; readout is retained separately as label','sources':sources,'stats':stats,'ledger_sha256':sha(O/'boundary_ledger.jsonl')}
with (O/'summary.json').open('x') as f:json.dump(summary,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({'stats':stats,'ledger_sha256':summary['ledger_sha256'],'summary_sha256':sha(O/'summary.json'),'prefix_chars':len(prefix)},sort_keys=True))
