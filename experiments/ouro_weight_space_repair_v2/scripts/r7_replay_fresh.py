"""Original-model replay without undistributed historical hidden-state files.

Requires completed frozen artifacts and the bounded supervisor's deadline.
Does not select/modify a method; strict equality against archived scores.
"""
import argparse,time,os
import numpy as np
from r7_core import ROOT,read,save,sha,gated_rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['discovery','selection','confirmation','validation','test'],required=True)
    p.add_argument('--job',required=True);p.add_argument('--limit',type=int);a=p.parse_args()
    assert a.job.replace('_','').isalnum() and 'GPU_DEADLINE_UNIX' in os.environ,'Run through r7_supervise.py with an authorized budget'
    lock=read('configs/r7_FINAL_METHOD_LOCK.json');assert lock['test_access_authorized']
    for name,value in lock['inference_source_sha256'].items():assert sha('scripts/'+name)==value
    from r7_evaluate import rows
    expected={r['id']:r for r in rows(a.split)};data=gated_rows(a.split)
    if a.limit:assert a.split!='test';data=data[:a.limit]
    from runtime import load_model,check_budget
    from ouro_cached_scoring import OuroCachedScorer
    from r7_inference import infer_question
    tok,model=load_model();scorer=OuroCachedScorer(model,5)
    for name,value in lock['model_identity']['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+name)==value
    assert sha('models/Ouro-1.4B/model.safetensors')==lock['model_identity']['safetensors_sha256']
    prompts=read('configs/r7_reference_manifest.json')['references'];prompts=[prompts[k] for k in ['N0','N1','N2']]
    start=time.time();records=[]
    for row in data:
        check_budget(45);old=expected[row['id']];names=list(old['method_scores'])
        values=infer_question(model,scorer,tok,{'request_pairs':row['request_pairs'],'choice_char_lengths':row['choice_char_lengths']},names,prompts)
        error=0.
        for field in ['method_scores','baseline_scores']:
            for name,vector in old[field].items():
                err=float(np.max(np.abs(np.asarray(vector)-np.asarray(values[field][name]))));error=max(error,err)
                assert err==0.,('Replay mismatch',row['id'],name,err)
        records.append(dict(id=row['id'],max_candidate_score_abs_error=error))
    save(f'results/r7_replay_{a.job}.json',dict(status='passed',split=a.split,n=len(data),historical_states_used=False,
        actual_main_depth=5,actual_reference_depth=4,final_method_lock_sha256=sha('configs/r7_FINAL_METHOD_LOCK.json'),
        replay_source_sha256=sha('scripts/r7_replay_fresh.py'),elapsed_seconds=time.time()-start,records=records))
    print('R7_FRESH_REPLAY_PASSED',len(records),flush=True)


if __name__=='__main__':main()
