"""One-question staging integration, not a production service or approved probe."""
import json,torch,hashlib
from pathlib import Path
from controlled import run_question
from source_guard import tensor_sha
from artifacts import score,save,replay,sha,canonical

def run_one(model,ids,entry,guard,out,expected_vocab,expected_hidden,device):
 if model.training or ids.ndim!=2 or ids.shape[0]!=1 or ids.tolist()[0]!=entry['ids']:raise ValueError('model/input mode')
 if str(ids.device)!=str(device):raise ValueError('device')
 if next(model.parameters()).dtype!=torch.bfloat16:raise ValueError('dtype')
 prior={k:v.to(device) for k,v in entry['prior_logits'].items()}
 for k,z in prior.items():
  if score(z,entry['targets'],device)!=entry['prior_scores'][k]:raise ValueError('prior score bridge')
 natural,results=run_question(model,ids,entry['regions'],guard,prior)
 for i in (0,1):
  if score(natural[i][1],entry['targets'],device)!=entry['natural_scores'][i]:raise ValueError('natural scores drift')
 guard.finish()
 identity={'id':entry['id'],'split':entry['split'],'input_ids_tensor_sha256':tensor_sha(ids),'source_hashes':entry['source_hashes']}
 save(out,identity,results,entry['targets'],entry['regions']['P'],entry['regions']['R'],expected_vocab,expected_hidden,prior,guard)
 evidence=replay(out,identity,entry['targets'],entry['regions']['P'],entry['regions']['R'],expected_vocab,expected_hidden,prior,device,guard)
 guard.finish()
 # This is a staging receipt, not coverage or a GPU acceptance gate.
 result={'status':'ONE_QUESTION_STAGING_ONLY','identity':identity,'replay':evidence}
 p=Path(out)/'staging_receipt.json'
 with p.open('x') as f:json.dump(result,f,sort_keys=True);f.flush()
 return result
