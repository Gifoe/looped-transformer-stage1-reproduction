import tempfile,torch
from pathlib import Path
from test_controlled import Model
from controlled import capture_natural,boundary,score_forward
from artifacts import score
from source_guard import FrozenSources,state_digest,tensor_sha,sha_file
from single_question import run_one
m=Model().eval().bfloat16();ids=torch.tensor([[1,2,3,4,5]]);regions={'P':[0],'Q':[1,2,3],'R':[4]};targets=[1,2]
with tempfile.TemporaryDirectory() as td:
 f=Path(td)/'source';f.write_text('fixed');expected={'model':state_digest(m),'ids':tensor_sha(ids),'files':{str(f.resolve()):sha_file(f)}};guard=FrozenSources(m,ids,[f],expected)
 natural=capture_natural(m,ids,regions['R'],guard);k1,k2=natural[0][0],natural[1][0];prior={}
 with torch.inference_mode():
  for corner in ('211','221','212','222'):
   with boundary(m,2,k1,k2,regions,corner):prior[corner]=score_forward(m,ids,regions['R'])
 entry={'id':'toy','split':'train','ids':ids.tolist()[0],'targets':targets,'regions':regions,'natural_scores':[score(x[1],targets,'cpu') for x in natural],'prior_logits':prior,'prior_scores':{k:score(v,targets,'cpu') for k,v in prior.items()},'source_hashes':{'test':'x'}}
 r=run_one(m,ids,entry,guard,Path(td)/'result',10,4,'cpu');assert r['status']=='ONE_QUESTION_STAGING_ONLY'
 print('SINGLE_QUESTION_CPU_PASS; NOT GPU OR COVERAGE')
