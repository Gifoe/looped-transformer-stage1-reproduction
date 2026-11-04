"""Frozen E1 release gate: no labels, predictions, argmax or tolerances.
Compare ALL v0 row/candidate keys recursively; no shared numerical exclusions.
Only execution metadata outside raw rows (protocol, clocks, memory, namespace,
publication and all-parameter receipts) are excluded from overlap comparison.
"""
import json,math
from r20_e1_common import ROOT,read,sha,save,gate

def main():
    spec=gate();done=read('results/r20_e1_pilot_COMPLETE.json');sup=read('logs/r20_e1_SUPERVISOR.json')
    assert done['status']==sup['status']=='complete' and done['n']==24 and not sup.get('timed_out',False) and not sup['overrun']
    assert done['all_parameter_hashes_before']==done['all_parameter_hashes_after'] and not done['weights_modified']
    assert done['protocol_sha256']==sha('configs/r20_e1_VALUE_ABLATION_PILOT_LOCK.json') and done['score_file_sha256']==sha('results/r20_e1_pilot.jsonl')
    assert sha('results/r20_pilot.jsonl')==spec['v0_partial_sha256']
    old=[json.loads(x) for x in (ROOT/'results/r20_pilot.jsonl').read_text().splitlines()];new=[json.loads(x) for x in (ROOT/'results/r20_e1_pilot.jsonl').read_text().splitlines()]
    assert len(old)==19 and len(new)==24
    assert [r['id'] for r in new]==[c['id'] for c in spec['cohort']]
    counts={};maxima={};errors=[]
    def compare(a,b,path):
        if type(a) is not type(b):errors.append(path+':type');return
        if isinstance(a,dict):
            if set(a)!=set(b):errors.append(path+':keys');return
            for k in a:compare(a[k],b[k],path+'.'+k)
        elif isinstance(a,list):
            if len(a)!=len(b):errors.append(path+':length');return
            for x,y in zip(a,b):compare(x,y,path+'[]')
        else:
            counts[path]=counts.get(path,0)+1
            if isinstance(a,(int,float)) and not isinstance(a,bool):maxima[path]=max(maxima.get(path,0),abs(a-b))
            if isinstance(a,float) and (not math.isfinite(a) or not math.isfinite(b)):errors.append(path+':nonfinite')
            elif a!=b:errors.append(path+':nonzero_or_unequal')
    for a,b in zip(old,new[:19]):compare(a,b,'row')
    out=dict(status='passed_strict_zero' if not errors else 'failed_stop_no_effects',overlap_questions=19,complete_questions=24,comparison_scope='All raw row/candidate keys recursively, exhaustive key/type/list-length equality, numeric strict0; no raw-field exclusions.',excluded_scope='Only metadata outside raw rows: clocks/memory/namespace/protocol/publication/final parameter receipts',per_field={k:dict(count=v,max_abs_diff=maxima.get(k,0)) for k,v in counts.items()},failures=errors,protocol_sha256=done['protocol_sha256'],e1_scores_sha256=done['score_file_sha256'],v0_scores_sha256=sha('results/r20_pilot.jsonl'),no_gold_or_predictions_accessed=True)
    save('results/r20_e1_OVERLAP_PARITY.json',out);print(out['status'])
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
