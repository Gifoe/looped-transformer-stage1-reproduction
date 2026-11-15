"""Independent every-endpoint candidate/logit/normalizer audit, CPU only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from safetensors.torch import load_file
from C1_CAUSAL_ANALYSIS import ENDPOINTS,sha,write,read,lines
from C1_AUDIT_TENSOR_IO import load_checked

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 p.add_argument('--split',choices=['train','validation'],required=True);a=p.parse_args()
 assert not torch.cuda.is_initialized()
 start=time.time();rows=list(lines(a.root/'inputs'/(a.split+'.jsonl')))
 assert len(rows)==(1094 if a.split=='train' else 299)
 cases=[];maxerr=0.;undefined=[];endpoint_count=0
 for row in rows:
  stem=hashlib.sha256(row['id'].encode()).hexdigest()
  p=a.root/'references'/a.split/(stem+'.json');meta=read(p)
  tp=p.with_suffix('.safetensors');reference,archive=load_checked(tp,meta)
  assert hashlib.sha256(json.dumps(row['ids']).encode()).hexdigest()==row['input_ids_sha256']
  entries=[(name,meta['conditions'][name],reference[name+'_logits'],sha(p),meta['tensor_sha256']) for name in ['K1','N','P','U']]
  for name in ENDPOINTS:
   if name in ['N','P','U']:continue
   p=a.root/'treatments'/a.split/stem/(name+'.json');c=read(p)
   if 'candidate_logits' not in c:
    undefined.append({'id':row['id'],'condition':name,'metadata_sha256':sha(p),'status':c.get('status')});continue
   tp=p.with_suffix('.safetensors');content,archive=load_checked(tp,c);value=content['logits']
   entries.append((name,c,value,sha(p),c['tensor_sha256']))
  case={'id':row['id'],'input_ids_sha256':row['input_ids_sha256'],'endpoints':{}}
  for name,c,t,mhash,thash in entries:
   assert t.device.type=='cpu' and t.dtype==torch.bfloat16 and tuple(t.shape) in [(49152,),(1,49152)]
   rawsha=hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
   assert rawsha==c['Rlogits_sha256'] and rawsha==c['full_logits_sha256'],(row['id'],name,'rawhash')
   v=t.reshape(49152).float().numpy().astype(np.float64);assert np.isfinite(v).all()
   candidate=v[row['targets']].tolist();assert candidate==c['candidate_logits']
   top=max(candidate);ties=[i for i,x in enumerate(candidate) if x==top]
   assert c['prediction']==ties[0] and c['top_ties']==ties
   maximum=float(np.max(v));lse=maximum+math.log(float(np.exp(v-maximum).sum(dtype=np.float64)))
   exact=np.asarray(candidate)-lse;produced=np.asarray(c['scores'],dtype=np.float64)
   error=float(np.max(np.abs(exact-produced)))
   # A priori conservative FP32 reduction/rounding envelope, not a fitted tolerance.
   envelope=8*np.finfo(np.float32).eps*(1+float(np.max(np.abs(v)))+abs(lse))
   assert error<=envelope,(row['id'],name,'FP32 normalizer error',error,envelope)
   assert abs(float(c['logsumexp_fp32'])-lse)<=envelope,(row['id'],name,'logsumexp')
   maxerr=max(maxerr,error);endpoint_count+=1
   case['endpoints'][name]={'metadata_sha256':mhash,'tensor_sha256':thash,'Rlogits_sha256':rawsha,
       'candidate_logits_exact':True,'first_ties_exact':True,'FP64_logsumexp':lse,
       'max_FP32_score_error':error,'a_priori_FP32_envelope':envelope}
  cases.append(case)
 assert not torch.cuda.is_initialized()
 out={'status':'PASS','split':a.split,'n_questions':len(rows),'n_defined_endpoints':endpoint_count,
      'n_undefined_endpoints':len(undefined),'undefined':undefined,'cases':cases,
      'source_sha256':sha(__file__),'analysis_source_sha256':sha(Path(__file__).with_name('C1_CAUSAL_ANALYSIS.py')),
      'tensor_reader_sha256':sha(Path(__file__).with_name('C1_AUDIT_TENSOR_IO.py')),
      'CPU_seconds':time.time()-start,'max_FP32_score_error_vs_independent_FP64':maxerr,
      'scope':'All saved full-vocabulary BF16 endpoints checked against metadata, offered candidates, first ties and independent FP64 normalizer. No model forward, GPU initialization or proof of causal hook semantics.'}
 write(a.root/'analysis'/(a.split.upper()+'_ALL_ENDPOINT_CPU_AUDIT.json'),out)
 print('C1_ALL_ENDPOINT_CPU_AUDIT_PASS',a.split,len(rows),endpoint_count,maxerr,flush=True)

if __name__=='__main__':main()
