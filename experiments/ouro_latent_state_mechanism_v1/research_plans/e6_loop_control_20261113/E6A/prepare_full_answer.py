"""Recover exact question/options from fixed joint prompts; freeze separate FAS."""
import json,re,hashlib,collections
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent
tok=Tokenizer.from_file(r'D:/chenyu-iclr/experiments/ouro_weight_space_repair_v1/models/Ouro-1.4B/tokenizer.json')
source=Path(r'D:/ouro-e4-e5-20261112/inputs/runner_inputs.jsonl')
def parse(block):
 q,ans=block.rsplit('\nAnswer:',1);parts=re.split(r'\n([A-E])\. ',q);stem=parts[0];labels=parts[1::2];choices=parts[2::2]
 assert labels==list('ABCDE'[:len(choices)]) and 3<=len(choices)<=5
 return stem,choices,ans.strip()
rows=[];prefix=None;texts=collections.defaultdict(list)
for q in map(json.loads,source.read_text(encoding='utf8').splitlines()):
 text=tok.decode(q['ids'],skip_special_tokens=False);assert tok.encode(text,add_special_tokens=False).ids==q['ids'];blocks=re.split(r'(?:^|\n\n)Question: ',text);assert blocks[0]=='' and len(blocks)==27
 demos=[]
 for b in blocks[1:26]:
  stem,opts,answer=parse(b);assert answer in 'ABCDE'[:len(opts)];demos.append('Question: '+stem+'\nAnswer: '+opts[ord(answer)-65])
 pre='\n\n'.join(demos)+'\n\n'
 if prefix is None:prefix=pre
 assert pre==prefix
 stem,opts,answer=parse(blocks[-1]);assert answer=='' and len(opts)==len(q['targets']);prompt=pre+'Question: '+stem+'\nAnswer:';context=tok.encode(prompt,add_special_tokens=False).ids;targets=[]
 for option in opts:
  ids=tok.encode(prompt+' '+option,add_special_tokens=False).ids;assert ids[:len(context)]==context;target=ids[len(context):];assert target;targets.append(target)
 key=re.sub(r'\W+',' ',stem+' '+' '.join(opts)).lower().strip();texts[key].append([q['split'],q['id']])
 rows.append({'id':q['id'],'split':q['split'],'ids':context,'R':len(context)-1,'answer_targets':targets,'answer_char_lengths':[len(x) for x in opts],'choices':opts,'question':stem,'input_ids_sha256':hashlib.sha256(json.dumps(context).encode()).hexdigest(),'original_joint_input_ids_sha256':q['input_ids_sha256']})
out=ROOT/'FULL_ANSWER_INPUTS.jsonl';out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
spec={'status':'FROZEN_BEFORE_ANY_FULL_ANSWER_RESULTS','protocol':'ARC question-only complete-answer conditional likelihood, same original25 demonstrations transformed to full text, no candidate options in context','primary_aggregation':'sum full-vocabulary FP32 token logprob / answer character count excluding leading continuation space','secondary_aggregations':['sum token logprob','mean token logprob'],'token_boundary':'context encode must exactly prefix encode(context + leading-space full answer); no silent token slicing mismatches','dtype':'BF16','attention':'SDPA','cache':False,'loops':[1,2,3,4],'argmax':'first index','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'preparation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inputs_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'n':len(rows),'prefix_sha256':hashlib.sha256(prefix.encode()).hexdigest(),'normalized_text_duplicate_groups':[v for v in texts.values() if len(v)>1],'methods':'E6A original frozen candidates and parameters unchanged; no Full-Answer Test tuning; Cal remains labelled Joint-development calibrated transfer'}
(ROOT/'FULL_ANSWER_PROTOCOL_FREEZE.json').write_text(json.dumps(spec,indent=2),encoding='utf8');print({'rows':len(rows),'inputs_sha256':spec['inputs_sha256'],'normalized_duplicates':len(spec['normalized_text_duplicate_groups'])})
