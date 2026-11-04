"""E2 frozen double-overlap gate; no gold/argmax, ALL raw fields, no exclusions."""
import json,math

def compare_records(old,new):
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
    compare(old,new,'records')
    return dict(status='passed_strict_zero' if not errors else 'failed_stop_no_effects',per_field={k:dict(count=v,max_abs_diff=maxima.get(k,0)) for k,v in counts.items()},failures=errors)

def main():
    from r20_e2_common import ROOT,read,sha,save,gate
    spec=gate();done=read('results/r20_e2_pilot_COMPLETE.json');sup=read('logs/r20_e2_SUPERVISOR.json')
    assert done['status']==sup['status']=='complete' and done['n']==24 and not sup.get('timed_out',False) and not sup['overrun']
    assert done['all_parameter_hashes_before']==done['all_parameter_hashes_after'] and not done['weights_modified']
    assert done['protocol_sha256']==sha('configs/r20_e2_VALUE_ABLATION_PILOT_LOCK.json') and done['score_file_sha256']==sha('results/r20_e2_pilot.jsonl')
    hashes={'V0':sha('results/r20_pilot.jsonl'),'E1':sha('results/r20_e1_pilot.jsonl'),'E2':sha('results/r20_e2_pilot.jsonl')}
    assert hashes['V0']==spec['v0_partial_sha256'] and hashes['E1']==spec['e1_partial_sha256']
    new=[json.loads(x) for x in (ROOT/'results/r20_e2_pilot.jsonl').read_text().splitlines()];assert len(new)==24 and [r['id'] for r in new]==[c['id'] for c in spec['cohort']]
    allpass=True
    for tag,path,n in [('V0','results/r20_pilot.jsonl',19),('E1','results/r20_e1_pilot.jsonl',23)]:
        old=[json.loads(x) for x in (ROOT/path).read_text().splitlines()];assert len(old)==n
        out=compare_records(old,new[:n]);out.update(overlap_questions=n,complete_questions=24,all_score_sha256=hashes,protocol_sha256=done['protocol_sha256'],comparison_scope='ALL raw keys/types/list lengths/numeric strict0/strings/booleans/input hashes/positions; no raw exclusions',excluded_scope='Only metadata outside raw rows: clocks/memory/namespace/protocol/publication/finalparams receipts',no_gold_or_predictions_accessed=True)
        save('results/r20_e2_OVERLAP_'+tag+'.json',out);allpass=allpass and out['status']=='passed_strict_zero'
    print('BOTH_OVERLAPS_PASSED' if allpass else 'OVERLAP_FAILED_NO_EFFECTS')
    if not allpass:raise SystemExit(1)
if __name__=='__main__':main()
