"""Server CPU-only gold-free complete576 cached-state feature extraction."""
import os,time,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from r22_common import ROOT,read,sha,digest,save,gate

def main():
 spec=gate();manifest=read('configs/r22_CHALLENGE_MANIFEST.json');assert digest(manifest['rows'])==manifest['rows_sha256']
 import torch,numpy as np
 from r22_math import torch_feature,numpy_feature
 torch.set_num_threads(2)
 from transformers import AutoTokenizer
 from runtime import encode_pair
 tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Ouro-1.4B'),local_files_only=True)
 count=0;checks=[];totalbytes=0;started=time.monotonic()
 with (ROOT/'results/r22_features.jsonl').open('x',encoding='utf8') as f:
  for row in manifest['rows']:
   assert time.monotonic()<float(os.environ['R22_STOP_NEW_FILE_MONOTONIC'])
   entry=row['state'];assert sha(entry['file'])==entry['sha256'];totalbytes+=entry['bytes'];payload=torch.load(ROOT/entry['file'],map_location='cpu',weights_only=True);assert payload['id']==row['id'];candidates=payload['candidate_states'];assert len(candidates)==entry['candidate_count'];features=[]
   pairs=[encode_pair(tok,*p) for p in row['request_pairs']]
   if not all(p[0]==pairs[0][0] for p in pairs):
    save('results/r22_INPUT_INCONCLUSIVE.json',dict(id=row['id'],reason='Candidate boundary tokenization gives different prefix IDs; strict prerequisite not satisfied',status='inconclusive_no_feature_interpretation'));raise RuntimeError('Prefix IDs differ')
   assert all(digest(t)==entry['target_id_sha256'][j] for j,(_,t) in enumerate(pairs))
   first_exact=[]
   for k in range(6):
    reference=candidates[0]['normalized'][k][:,:1];diff=max(float((c['normalized'][k][:,:1].double()-reference.double()).abs().max()) for c in candidates);first_exact.append(diff)
   if any(v!=0 for v in first_exact):
    save('results/r22_INPUT_INCONCLUSIVE.json',dict(id=row['id'],first_position_maxdiff_by_depth=first_exact,status='inconclusive_no_feature_interpretation',reason='Strict first-position invariance prerequisite unmet; sequence-shape BF16 rounding is possible, not evidence of corruption or causal leakage'));raise RuntimeError('First-position exact prerequisite failed')
   for j,c in enumerate(candidates):
    assert c['candidate']==j and digest(c['target_ids'])==entry['target_id_sha256'][j] and len(c['target_ids'])==entry['choice_token_counts'][j]
    states=c['normalized'];assert len(states)==6 and all(tuple(h.shape)==(1,len(c['target_ids']),2048) for h in states)
    value=torch_feature(states)
    if count==0 and j==0:
     independent=numpy_feature([h.float().numpy() for h in states]);error=max(abs(value[k]-independent[k]) for k in value);assert error<=1e-12;checks.append(dict(first_NumPy_Torch_max_error=error))
    value.update(candidate=j,C_margin=row['native6_scores'][j],C_log_tokens=float(np.log(value['tokens'])),C_neg_log_tokens=float(-np.log(value['tokens'])));features.append(value)
   for a in row['challengers']:
    b=row['R6_winner']
    if features[a]['tokens']==features[b]['tokens']==1:assert features[a]['V']==features[b]['V']
   record=dict(prefix_ids_sha256=digest(pairs[0][0]),all_candidates_prefix_ids_exact=True,first_position_maxdiff_by_depth=first_exact,id=row['id'],split=row['split'],candidates=features,state_sha256=entry['sha256']);f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno());count+=1;del payload,candidates
 assert count==576
 save('results/r22_FEATURES_COMPLETE.json',dict(status='complete_gold_free_features',n=count,features_sha256=sha('results/r22_features.jsonl'),challenge_sha256=sha('configs/r22_CHALLENGE_MANIFEST.json'),protocol_sha256=sha('configs/r22_TRAJECTORY_LOCK.json'),all_state_SHA_ID_candidate_target_position_shape_audits_passed=True,first_real_checks=checks,bytes_read=totalbytes,CPU_wall_seconds=time.monotonic()-started,GPU_worker_seconds=0,model_loaded=False,threads=2))
 print('R22_FEATURES_COMPLETE',count)
if __name__=='__main__':main()
