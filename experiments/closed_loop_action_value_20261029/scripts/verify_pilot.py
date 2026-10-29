"""CPU artifact verification. This cannot establish benchmark efficacy."""
import json,hashlib,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'frozen_loop_update_value_seed2021_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    lock=load(ROOT/'configs/PILOT_LOCK.json')
    for rel,digest in lock['files'].items():assert sha(ROOT/rel)==digest
    for rel,digest in lock['original_files'].items():assert sha(OLD/rel)==digest
    tree=ast.parse((ROOT/'scripts/pilot.py').read_text())
    for fn in tree.body:
        if isinstance(fn,ast.FunctionDef) and fn.name in ['choose','evaluate']:
            assert not any(isinstance(n,ast.Constant) and n.value=='gold' for n in ast.walk(fn))
    with (ROOT/'results/pilot.jsonl').open(encoding='utf8') as f:rows=[json.loads(l) for l in f]
    assert [r['id'] for r in rows]==lock['ids']
    complete=load(ROOT/'results/PILOT_COMPLETE.json')
    assert complete['pilot_sha256']==sha(ROOT/'results/pilot.jsonl')
    assert complete['parity']['zero_feedback_exact_all_depths'] and complete['weights_unchanged']
    for r in rows:
        d=r['decision'];loss=d.get('losses')
        if loss:
            proposal=min([-1.,-.5,.5],key=lambda a:loss[str(a)]['A'])
            va=loss['0.0']['A']-loss[str(proposal)]['A']
            vb=loss['0.0']['B']-loss[str(proposal)]['B']
            assert proposal==d['proposal'] and abs(min(va,vb)-d['value'])<1e-12
            assert d['alpha']==(proposal if min(va,vb)>.001 else 0.)
        assert r['baseline_score_error']<1e-5
        assert r['selected_prediction']==r['actions'][str(d['alpha'])]['prediction']
        for drift in r['downstream_state_rms'].values():assert drift[0]==0 and min(drift[1:])>0
    assert sum(r['selected_prediction']==r['gold'] for r in rows)==complete['selected_correct']
    assert load(ROOT/'results/SUPERVISOR_EXIT.json')['exit_code']==0
    original=load(OLD/'ARTIFACT_MANIFEST.json')['files']
    for rel,entry in original.items():assert sha(OLD/rel)==entry['sha256']
    report=dict(n=len(rows),locks_verified=True,controller_decisions_replayed=True,baseline_score_error_max=max(r['baseline_score_error'] for r in rows),original_payload_hashes_verified=len(original),scope='Implementation and artifact verification only; no efficacy conclusion')
    (ROOT/'results/LOCAL_VERIFY.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report))
if __name__=='__main__':main()
