"""Independent counts, frozen dependencies and controller replay checks."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'frozen_loop_update_value_seed2021_v1'
PILOT=ROOT.parent/'closed_loop_action_value_20261029'
ALPHAS=['0.0','-1.0','-0.5','0.5']
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    lock=load(ROOT/'configs/LOCK.json');amend=load(ROOT/'configs/PACKING_AMENDMENT.json')
    assert sha(ROOT/'configs/LOCK.json')==amend['original_lock_sha256']
    for rel,h in lock['files'].items():assert sha(ROOT/rel)==h,rel
    for rel,h in lock['dependencies'].items():assert sha(ROOT.parent/rel)==h,rel
    for rel,h in amend['new_files'].items():assert sha(ROOT/rel)==h,rel
    with (ROOT/'results/development_packed.jsonl').open(encoding='utf8') as f:rows=[json.loads(l) for l in f]
    with (OLD/'data/discovery.jsonl').open(encoding='utf8') as f:data=[json.loads(l) for l in f]
    with (OLD/'results/discovery.jsonl').open(encoding='utf8') as f:baseline={r['id']:r['views']['official']['sum_scores']['adaptive'] for r in map(json.loads,f)}
    with (PILOT/'results/pilot.jsonl').open(encoding='utf8') as f:pilot={r['id']:r for r in map(json.loads,f)}
    assert len(rows)==400 and [r['id'] for r in rows]==lock['ids']
    complete=load(ROOT/'results/COMPLETE_PACKED.json');assert complete['n']==400 and complete['sha256']==sha(ROOT/'results/development_packed.jsonl')
    assert complete['weights_unchanged'] and complete['all_nonzero_actions_propagated']
    correct=[];base_correct=[];selected_counts={a:0 for a in ALPHAS};pilot_pred_equal=True
    for r,doc in zip(rows,data):
        assert r['id']==doc['id'] and r['gold']==doc['gold'] and r['choice_char_lengths']==doc['choice_char_lengths']
        d=r['decision'];loss=d.get('losses')
        if loss:
            proposal=min(ALPHAS[1:],key=lambda a:loss[a]['A'])
            va=loss['0.0']['A']-loss[proposal]['A'];vb=loss['0.0']['B']-loss[proposal]['B']
            assert str(d['proposal'])==proposal and abs(d['value']-min(va,vb))<1e-12
            assert str(d['alpha'])==(proposal if min(va,vb)>.001 else '0.0')
        else:assert d['alpha']==0.
        confidence=max(ALPHAS,key=lambda a:r['controls'][a]['margin']);assert str(d['confidence_alpha'])==confidence
        for group in ['actions','terminal']:
            for a,v in r[group].items():
                assert a in ALPHAS and np.isfinite(v['sum_scores']).all()
                assert v['prediction']==int(np.argmax(np.array(v['sum_scores'])/r['choice_char_lengths']))
        assert r['terminal']['0.0']['sum_scores']==r['actions']['0.0']['sum_scores']
        error=max(abs(a-b) for a,b in zip(r['actions']['0.0']['sum_scores'],baseline[r['id']]))
        assert error==r['baseline_score_error'] and error<1e-5
        assert r['actions']['0.0']['prediction']==int(np.argmax(np.array(baseline[r['id']])/r['choice_char_lengths']))
        for drift in r['downstream_state_rms'].values():assert drift[0]==0 and min(drift[1:])>0
        a=str(d['alpha']);selected_counts[a]+=1
        correct.append(r['actions'][a]['prediction']==r['gold']);base_correct.append(r['actions']['0.0']['prediction']==r['gold'])
        if r['id'] in pilot:
            prior=pilot[r['id']];assert prior['decision']['alpha']==d['alpha']
            pilot_pred_equal &= all(prior['actions'][a]['prediction']==r['actions'][a]['prediction'] for a in ALPHAS)
    assert pilot_pred_equal
    y=np.array(correct);b=np.array(base_correct);plus=int((y&~b).sum());minus=int((~y&b).sum())
    report=load(ROOT/'results/ANALYSIS.json')['full400'];summary=report['summary_vs_adaptive']['probe']
    assert summary['correct']==int(y.sum()) and summary['corrections']==plus and summary['degradations']==minus
    p=binomtest(plus,plus+minus).pvalue if plus+minus else 1.
    assert abs(p-summary['mcnemar_p'])<1e-12 and report['selected_action_counts']==selected_counts
    verified={}
    for directory in [OLD,PILOT,ROOT.parent/'update_value_followup_audit_20261029']:
        files=load(directory/'ARTIFACT_MANIFEST.json')['files']
        for rel,x in files.items():assert sha(directory/rel)==x['sha256'],str(directory/rel)
        verified[directory.name]=len(files)
    receipt=dict(n=400,source_and_data_locks_passed=True,all_controller_decisions_replayed=True,baseline_predictions_unchanged=True,pilot_decisions_and_predictions_preserved=True,paired=dict(correct=int(y.sum()),corrections=plus,degradations=minus,mcnemar_p=float(p)),historical_payloads_verified=verified,final_tasks_loaded=False)
    (ROOT/'results/VERIFICATION.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print(json.dumps(receipt))
if __name__=='__main__':main()
