import pathlib,json,re,hashlib,datetime
from tokenizers import Tokenizer
R=pathlib.Path('/root/ouro-catc-sol-20261116');P=R/'phaseC';tok=Tokenizer.from_file('/root/autodl-tmp/ouro-e4-e5/model/tokenizer.json');qs=list(map(json.loads,(P/'VALIDATION_INPUTS.jsonl').read_text().splitlines()));rows=[];pre=None
spec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'population':'fullValidation299, historicallyexposed, noFASparameterselection','protocol':'exactoriginalE6FASquestion-only fulltext25demonstrations and alloptioncontinuations; primarysumfullvocabFP32tokenlogprob/answercharsexcludingleadingspace; secondarysum/tokenmean','execution':'sequentialteacherforcedprefixes,noKVcache; at eachnexttokenpredictionR=lastprefixposition. CATC modifiesRinK2only; Qincludescurrentquestion+prioranswerprefix, excludingR. Sameprefixshapes fornative/CATC/LoopCD. Freshsame-devicebaselines, nohistoricalFASscorespooled. This executionformneednotbe bitwiseidenticaltohistoricalbatchedfullsequenceTF.','methods':['nativeK2','nativeK4','LoopCD_logits_0.25','LoopCD_logits_1','fixedV+.1','selectedcatc14'],'source_protocol_commit':'7857ee2e4a7413d52fb1fc8f2d641e10056a406c','expected_demonstration_prefix_sha256':'e181a54c18e9e1aaf4b336d7031c8647bb3dd0cf8f28d8cb0df735074485d7c9'}
(P/'FULL_ANSWER_FREEZE.json').write_text(json.dumps(spec,indent=2))
def parse(block):
 q,ans=block.rsplit('\nAnswer:',1);parts=re.split(r'\n([A-E])\. ',q);assert parts[1::2]==list('ABCDE'[:len(parts[2::2])]);return parts[0],parts[2::2],ans.strip()
for q in qs:
 text=tok.decode(q['ids'],skip_special_tokens=False);assert tok.encode(text,add_special_tokens=False).ids==q['ids'];blocks=re.split(r'(?:^|\n\n)Question: ',text);assert blocks[0]=='' and len(blocks)==27;demos=[]
 for b in blocks[1:26]:
  stem,opts,answer=parse(b);demos.append('Question: '+stem+'\nAnswer: '+opts[ord(answer)-65])
 prefix='\n\n'.join(demos)+'\n\n'
 if pre is None:pre=prefix
 assert prefix==pre;stem,opts,answer=parse(blocks[-1]);assert answer=='';prompt=pre+'Question: '+stem+'\nAnswer:';ids=tok.encode(prompt,add_special_tokens=False).ids;enc=tok.encode(prompt,add_special_tokens=False);boundary=next(i for i,(start,end) in enumerate(enc.offsets) if end>len(pre));pid=ids[:boundary];assert boundary>0;targets=[]
 for option in opts:
  full=tok.encode(prompt+' '+option,add_special_tokens=False).ids;assert full[:len(ids)]==ids;targets.append(full[len(ids):])
 rows.append({'id':q['id'],'split':'validation','ids':ids,'R':len(ids)-1,'prefix_len':len(pid),'answer_targets':targets,'answer_char_lengths':[len(o) for o in opts],'input_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'question':stem,'choices':opts})
assert hashlib.sha256(pre.encode()).hexdigest()==spec['expected_demonstration_prefix_sha256'];(P/'FULL_ANSWER_INPUTS.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in rows));(P/'FULL_ANSWER_PREPARATION_AUDIT.json').write_text(json.dumps({'status':'PASS','n':len(rows),'prefix_sha256':hashlib.sha256(pre.encode()).hexdigest(),'prefix_tokens':rows[0]['prefix_len'],'target_tokens':sum(len(t) for q in rows for t in q['answer_targets']),'input_sha256':hashlib.sha256((P/'FULL_ANSWER_INPUTS.jsonl').read_bytes()).hexdigest()},indent=2));print('FAS299_FROZEN',rows[0]['prefix_len'],sum(len(t) for q in rows for t in q['answer_targets']))
