"""Pre-inference constructor audit. Gold is used only to protect the true option."""
from pathlib import Path
import json,hashlib,sys
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from r10_core import joint_prefix,joint_requests
H=lambda b:hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows():
 protocol=read(ROOT/'configs/r10_PROTOCOL.json');data=[]
 for split in ['discovery','selection']:
  p=ROOT/('data/slcd_'+split+'.jsonl');assert H(p.read_bytes())==protocol['dataset_sha256'][split]
  data.extend(json.loads(x) for x in p.read_text(encoding='utf8').splitlines())
 return sorted(data,key=lambda x:(H(('2021|'+x['id']).encode('utf8')),x['id']))
def wrong_index(r):
 return min((i for i in range(len(r['choices'])) if i!=r['gold']),key=lambda i:(H(('2021|'+r['id']+'|swap|'+str(i)).encode('utf8')),i))
def render(prefix,question,choices):
 p=prefix+'Question: ';fields=dict(question_start=len(p));p+=question;fields['question_end']=len(p);p+='\n';fields['options']=[]
 for i,c in enumerate(choices):
  p+=chr(65+i)+'. ';a=len(p);p+=c;fields['options'].append(dict(index=i,start=a,end=len(p)));p+='\n' if i<len(choices)-1 else ''
 p+='\nAnswer:';assert p==joint_requests(dict(question=question,choices=choices),prefix)[0][0];return p,fields
def field_counts(tok,p,fields,j):
 e=tok.encode(p,add_special_tokens=False);f=fields['options'][j];touch=[k for k,(a,b) in enumerate(e.offsets) if a<b and a<f['end'] and b>f['start']]
 return dict(total_tokens=len(e.ids),field_tokens=len(touch),positions=touch)
def tokenizer_path():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--tokenizer-json');args=ap.parse_args()
 if args.tokenizer_json:return Path(args.tokenizer_json)
 for p in [ROOT/'models/Ouro-1.4B/tokenizer.json',ROOT/'artifacts/r24_TOKENIZER_AUDIT_PRIVATE.json']:
  if p.exists():return p
 raise FileNotFoundError('Pass --tokenizer-json for pinned ByteDance/Ouro-1.4B tokenizer.json revision574fa66cb8bf5abdc979642d01cf2b79b16bfab1; SHA fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d')
def main():
 data=rows();selected=read(OUT/'COHORT_PREREGISTRATION.json');assert [x['id'] for x in data[:24]]==selected['pilot24_ids'];assert H(json.dumps(selected['pilot24_ids'],separators=(',',':')).encode())=='f37bb5c339125292fab8be3b4a2dedf24130900377a7eed08e0887a456497fec'
 tp=tokenizer_path();assert H(tp.read_bytes())=='fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d';tok=Tokenizer.from_file(str(tp));prefix=joint_prefix(read(ROOT/'data/slcd_fixed_demonstrations.json'))
 pools=[['18'],['increase in the population size'],['The Moon moves toward Earth.'],['less than -1\u00b0C','below minus 28\u00b0C','less than minus 28\u00b0C']];result=[]
 for rank,r in enumerate(data[:24]):
  j=wrong_index(r);p,f=render(prefix,r['question'],r['choices']);orig=field_counts(tok,p,f,j);candidates=[]
  if rank<4:
   for text in pools[rank]:
    choices=list(r['choices']);choices[j]=text;q,g=render(prefix,r['question'],choices);new=field_counts(tok,q,g,j)
    candidates.append(dict(text=text,token_length_matched=new['total_tokens']==orig['total_tokens'] and new['field_tokens']==orig['field_tokens'],original_total_tokens=orig['total_tokens'],modified_total_tokens=new['total_tokens'],original_field_tokens=orig['field_tokens'],modified_field_tokens=new['field_tokens'],duplicate_existing_option=text in r['choices']))
  result.append(dict(rank=rank,id=r['id'],question=r['question'],choices=r['choices'],gold=r['gold'],fixed_wrong_index=j,original_prompt_tokens=orig['total_tokens'],original_wrong_field_tokens=orig['field_tokens'],secondary_bounded_eligible=rank<4,authored_pre_inference_secondary_candidates=candidates))
 payload=dict(status='CPU_pre_inference_constructor_semantic_review_no_model',secondary_scope='First4 SHA-ranked pilot questions only; fixed candidate priority; exact whole-prompt and option-field token lengths required plus independent semantic approval. Other20 are scope-excluded, not semantic rejections.',primary_placeholder='Not a valid answer',rows=result)
 # ASCII JSON is deliberate: preserve Unicode field identity even with external default-codepage readers.
 (OUT/'SEMANTIC_CONSTRUCTOR_REVIEW.json').write_text(json.dumps(payload,indent=2,ensure_ascii=True)+'\n',encoding='utf8')
 original={r['id']:r for r in data}
 for r in result:assert r['question']==original[r['id']]['question'] and r['choices']==original[r['id']]['choices']
 print(json.dumps(dict(status=payload['status'],secondary_candidates=[dict(id=r['id'],candidates=r['authored_pre_inference_secondary_candidates']) for r in result[:4]],constructor_sha256=H((OUT/'SEMANTIC_CONSTRUCTOR_REVIEW.json').read_bytes()))))
if __name__=='__main__':main()
