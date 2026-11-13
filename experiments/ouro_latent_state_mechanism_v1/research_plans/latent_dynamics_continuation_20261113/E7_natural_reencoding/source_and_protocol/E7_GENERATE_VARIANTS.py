"""Frozen full counterbalance; no query gold and no model-outcome input."""
import json,hashlib,collections,datetime
from pathlib import Path
from tokenizers import Tokenizer
from REENCODING_RENDER import prompt
R=Path(__file__).resolve().parent;tokpath=Path('/root/autodl-tmp/ouro-e4-e5/model/tokenizer.json');tok=Tokenizer.from_file(str(tokpath))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def streamsha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 rows=[json.loads(l) for l in (R/'REENCODING_STRUCTURED_TRAIN_VAL.jsonl').read_text().splitlines()];folds=json.loads((R/'FOLD_MANIFEST.json').read_text());trainids={x['id'] for x in folds};rows=[q for q in rows if q['split']=='validation' or q['id'] in trainids];assert collections.Counter(q['split'] for q in rows)=={'train':1094,'validation':299};demos=json.loads((R/'REENCODING_DEMONSTRATIONS.json').read_text());assert len(demos)==25 and all(len(q['choices'])==4 for q in demos)
 rank=lambda q:hashlib.sha256(('E7-reencoding-technical-20261113:'+q['source_input_ids_sha256']).encode()).hexdigest();eligible=sorted([q for q in rows if q['split']=='train' and len(q['choices'])==4],key=lambda q:(rank(q),q['id']));tech=[];groups=set()
 for q in eligible:
  if q['source_input_ids_sha256'] in groups:continue
  groups.add(q['source_input_ids_sha256']);tech.append(q['id'])
  if len(tech)==2:break
 tech += [q['id'] for q in rows if q['split']=='train' and len(q['choices']) in [3,5]];assert len(tech)==4
 expected=collections.Counter();manifest=[];lengths=collections.defaultdict(list);identity=0
 pref=R/'E7_INITIAL_PROTOCOL_FREEZE.json';assert not pref.exists()
 spec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'director_sha256':sha(R/'ASTRA_E7_NATURAL_REENCODING_ORDER.md'),'generator_sha256':sha(__file__),'renderer_sha256':sha(R/'REENCODING_RENDER.py'),'tokenizer_sha256':sha(tokpath),'structured_source_sha256':sha(R/'REENCODING_STRUCTURED_TRAIN_VAL.jsonl'),'folds_sha256':sha(R/'FOLD_MANIFEST.json'),'technical_ids':tech,'expected_units':{'train':17504,'validation':4776,'total':22280,'technical':64},'P':'query-only leftrotatep; demonstrations order fixed','S':'global ABCD rotation s, Eidentity; all demos/query structural labels and demoanswers only','no_query_gold':True,'analysis':'Astra formalorder fixed three K1to2 gaincontrasts;10000questionblockbootstrap95 and98.333333CI;Train/Val separate','pair':'identityK2 and p=floor(m/2),s0 K2;FP64 centered FP32LP semantic average equalweight;semanticfirst tie','strong_output_frontier':'Original cal_early_0.6 entropy0.5 fixed using existingTrain before variants; E6 original primary weighted4 and Cal LoopCDadaptive identities unchanged','stock_gate':'Identity only separate full-body fixedCDF thresholds.25,.5,.75,1.0; default1.0 parity technical; no claimed online stopping','input_execution_order':'Train then Validation, originalquestionIDs sorted; perquestion variants sorted by SHA256 of split:ID:p:s for labelblind execution order','Goal':'ACTIVE, no overall time deadline; no newTest or internalstate interventions'}
 pref.write_text(json.dumps(spec,indent=2));data=R/'E7_VARIANTS.jsonl';techfile=R/'E7_TECHNICAL_VARIANTS.jsonl';assert not data.exists() and not techfile.exists();demoaudit=[]
 for s in range(4):
  ren={c:'ABCD'[(i+s)%4] for i,c in enumerate('ABCD')};ren['E']='E';_,qq,ds=prompt(demos,{**rows[0],'answer_text':''},0,ren);demoaudit.append({'s':s,'rename':ren,'blocks':ds})
 (R/'E7_DEMONSTRATION_MAPPING.json').write_text(json.dumps(demoaudit,ensure_ascii=False),encoding='utf8')
 with data.open('w',encoding='utf8') as out,techfile.open('w',encoding='utf8') as tout:
  for qi,q in enumerate(sorted(rows,key=lambda x:(x['split']!='train',x['id']))):
   c=len(q['choices']);configs=[];texts=[]
   for p in range(c):
    for s in range(4):
     ren={l:'ABCD'[(i+s)%4] for i,l in enumerate('ABCD')};ren['E']='E';text,v,ds=prompt(demos,{**q,'answer_text':''},p,ren);configs.append((p,s,text,v));texts.append(text);texts.extend(text+' '+label for label in v['labels'])
   enc=tok.encode_batch(texts,add_special_tokens=False);variants=[];offset=0
   for p,s,text,v in configs:
    ids=enc[offset].ids;targets=[]
    for j,label in enumerate(v['labels']):
     full=enc[offset+1+j].ids;assert full[:len(ids)]==ids and len(full)==len(ids)+1;single=tok.encode(' '+label,add_special_tokens=False).ids;assert full[len(ids):]==single;targets.append(single[0])
    offset+=c+1;hashids=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    if p==s==0:assert hashids==q['source_input_ids_sha256'];identity+=1
    assert all(v['choices'][j]==q['choices'][orig] for j,orig in enumerate(v['display_to_original']))
    vrow={'id':q['id'],'split':q['split'],'variant_id':q['id']+f'__p{p}_s{s}','p':p,'s':s,'ids':ids,'R':len(ids)-1,'input_ids_sha256':hashids,'targets':targets,'display_to_original':v['display_to_original'],'labels':v['labels'],'original_input_ids_sha256':q['source_input_ids_sha256'],'generator_sha256':spec['generator_sha256'],'renderer_sha256':spec['renderer_sha256'],'prompt':text}
    variants.append(vrow);lengths[q['split']+f':m{c}:p{p}:s{s}'].append(len(ids))
   variants.sort(key=lambda v:hashlib.sha256((v['split']+':'+v['variant_id']).encode()).hexdigest())
   for v in variants:
    line=json.dumps(v,ensure_ascii=False)+'\n';out.write(line)
    if q['split']=='train' and q['id'] in tech:tout.write(line)
    expected[q['split']]+=1;manifest.append({k:v[k] for k in ['id','split','variant_id','p','s','R','input_ids_sha256','targets','display_to_original','labels']})
   if (qi+1)%100==0:print('E7_VARIANTS_PREPARED',qi+1,len(rows),flush=True)
 assert expected=={'train':17504,'validation':4776} and identity==1393
 (R/'E7_VARIANT_MANIFEST.json').write_text(json.dumps({'n':len(manifest),'units':manifest},indent=2));(R/'E7_TOKEN_LENGTH_AUDIT.json').write_text(json.dumps({k:{'n':len(v),'min':min(v),'max':max(v),'mean':sum(v)/len(v)} for k,v in lengths.items()},indent=2));spec.update({'status':'FROZEN_BEFORE_NEW_VARIANT_OUTCOMES','identity_roundtrip_questions':identity,'counts':dict(expected),'variants_sha256':streamsha(data),'technical_variants_sha256':streamsha(techfile),'variant_manifest_sha256':streamsha(R/'E7_VARIANT_MANIFEST.json'),'initial_freeze_sha256':sha(pref),'token_length_audit_sha256':sha(R/'E7_TOKEN_LENGTH_AUDIT.json'),'collector_source_sha256':sha(R/'E7_REENCODING_COLLECT.py') if (R/'E7_REENCODING_COLLECT.py').exists() else 'collector must be separately source-frozen before technical launch'})
 (R/'E7_PROTOCOL_FREEZE.json').write_text(json.dumps(spec,indent=2));print('E7_ALL_VARIANTS_FROZEN',dict(expected),streamsha(data),flush=True)
if __name__=='__main__':main()
