"""Replay the frozen edited checkpoint only; never rerun candidate selection.

Run with supervise.py so GPU time is charged. Preserves all original results.
"""
import argparse
import json
import time
from runtime import ROOT,load_model,encode_pair,check_budget,save
from ouro_cached_scoring import OuroCachedScorer
from weight_edits import load_edit_bundle,applied_edits,file_sha256
from run_edit_pipeline import methods


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['test','discovery_screen'],default='test')
    ap.add_argument('--n',type=int);a=ap.parse_args()
    final=json.loads((ROOT/'configs/FINAL_LOCK.json').read_text());bundle=ROOT/'edited_weights/selected'
    assert file_sha256(bundle/'manifest.json')==final['bundle_manifest_sha256']
    data=[json.loads(l) for l in (ROOT/'data'/(a.split+'.jsonl')).read_text().splitlines()]
    if a.n is not None:
        assert a.split!='test','Official test replay must retain all1172 questions'
        assert 0<a.n<=len(data);data=data[:a.n]
    if a.split=='test':assert len(data)==1172
    output=ROOT/'results'/('independent_replay_'+a.split+'_'+str(time.time_ns()))
    output.mkdir();tok,model=load_model();scorer=OuroCachedScorer(model,6)
    edits,manifest=load_edit_bundle(bundle,model)
    reference_path=ROOT/'results'/('final_test' if a.split=='test' else 'screen')/final['selected_candidate_id']/'scores.jsonl'
    reference={r['id']:r for r in map(json.loads,reference_path.read_text().splitlines())} if reference_path.exists() else {}
    maximum=0.;correct={name:0 for name in ['native_k3','native_k4','native_k5','native_k6','loopcd_k4_fixed','loopcd_k4_adaptive']}
    with applied_edits(model,edits,invalidate=scorer.invalidate):
        for i,row in enumerate(data):
            check_budget(120);pairs=[encode_pair(tok,*p) for p in row['request_pairs']]
            scored=methods(scorer.score_uncached(pairs),row)
            for name in correct:correct[name]+=scored[name]['correct']
            if row['id'] in reference:
                old=reference[row['id']];assert old['prompt_sha256']==row['prompt_sha256']
                maximum=max(maximum,max(abs(x-y) for name,v in scored.items() for x,y in zip(v['sum_scores'],old['methods'][name]['sum_scores'])))
            r=dict(id=row['id'],gold=row['gold'],prompt_sha256=row['prompt_sha256'],methods=scored,
                bundle_manifest_sha256=final['bundle_manifest_sha256'])
            with (output/'scores.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r)+'\n')
            if (i+1)%16==0:print('replay',i+1,len(data),flush=True)
    save(str(output.relative_to(ROOT))+'/COMPLETE.json',dict(status='complete',n=len(data),correct=correct,
        selected_candidate_id=final['selected_candidate_id'],bundle_manifest_sha256=final['bundle_manifest_sha256'],
        max_sum_score_error_against_published_reference=maximum,
        exact_replay_observed=maximum==0,actual_changed_parameters=manifest['actual_changed_parameters']))
    print('REPLAY_COMPLETE',str(output),correct,'max_error',maximum,flush=True)


if __name__=='__main__':main()
