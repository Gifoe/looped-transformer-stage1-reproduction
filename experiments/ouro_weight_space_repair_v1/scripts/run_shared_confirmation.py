"""Conditional discovery-only shared-invocation check; never trains or edits weights.

Requires completed layer screen + CPU analyzer. Runs only analyzer-approved leads
(<=2 modules), alpha .5 post-branch-norm damping. Uses native uncached Kmax6.
Old representatives' loop5 results are reused verbatim from the completed screen;
their K3 result is explicitly sourced from native because a loop5-only hook cannot
affect K3. Added unresolved harms receive new matched loop5 forwards.
Run under supervise.py. This file does not access validation/test data.
"""
import argparse
import hashlib
import json
import time

from runtime import ROOT, load_model, encode_pair, outcome, save, check_budget
from ouro_cached_scoring import OuroCachedScorer
from causal_hooks import OuroCausalHooks, BranchIntervention


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--screen-study', default='layer_screen_v1')
    ap.add_argument('--study', default='shared_confirmation_v1')
    a=ap.parse_args()
    assert all(s.replace('_','').isalnum() for s in [a.screen_study,a.study])
    screen=ROOT/'results'/a.screen_study
    out=ROOT/'results'/a.study
    out.mkdir(exist_ok=True)
    finished=json.loads((screen/'COMPLETE.json').read_text())
    screen_frozen=json.loads((screen/'FROZEN.json').read_text())
    analysis=json.loads((screen/'MECHANISM_ANALYSIS.json').read_text())
    assert finished['status']==analysis['status']=='complete'
    assert finished['implementation_sha256']==screen_frozen['implementation_sha256']
    shortlist=analysis['top_at_most_two_modules']
    assert len(shortlist)<=2
    allowed=('localization_only_requires_shared_invocation_check',
             'eligible_for_bounded_label_blind_direction_diagnostic')
    shortlist=[item for item in shortlist if item['decision'] in allowed]
    if not shortlist:
        save(str((out/'SKIPPED.json').relative_to(ROOT)),dict(status='skipped',
             reason='Completed-screen analyzer did not identify a lead warranting shared-invocation confirmation',
             screen_analysis_sha256=sha(screen/'MECHANISM_ANALYSIS.json'), gpu_model_loaded=False))
        print('No screen-qualified modules; no model loaded or experiment run.',flush=True)
        return
    assert len({(s['layer'],s['branch']) for s in shortlist})==len(shortlist)
    for item in shortlist:
        assert 0<=item['layer']<24 and item['branch'] in ('attention','mlp')
    rep_path=ROOT/'configs/representative_cases.json'
    assert sha(rep_path)==screen_frozen['representative_sha256']
    representatives=json.loads(rep_path.read_text())
    source_study=representatives['source_study']
    assert source_study.replace('_','').isalnum()
    source=ROOT/'results'/source_study
    assert (source/f"COMPLETE_{representatives['source_n']}.json").exists()
    discovery_analysis=json.loads((source/'ANALYSIS.json').read_text())
    assert discovery_analysis['status']=='complete' and discovery_analysis['n']==representatives['source_n']
    native_path=source/'scores.jsonl'
    discovery_path=ROOT/'data/discovery.jsonl'
    assert sha(native_path)==screen_frozen['native_scores_sha256']
    assert sha(discovery_path)==screen_frozen['discovery_sha256']
    native={r['id']:r for r in read_lines(native_path)}
    discovery={r['id']:r for r in read_lines(discovery_path)}
    old_ids=screen_frozen['ids']
    assert 1<=len(old_ids)<=8
    unresolved=[]
    for uid,row in native.items():
        assert row['scope']=='official_train_discovery_only' and row['cached'] is False
        methods=row['methods']
        if methods['native_k4']['correct'] and not methods['native_k5']['correct']:
            if not any(methods[f'normalized_chord_{step}']['correct'] for step in ['0.125','0.25','0.5']):
                unresolved.append(uid)
    added=sorted(set(unresolved)-set(old_ids))
    ids=list(old_ids)+added
    def group(uid):
        m=native[uid]['methods'];c4,c5=m['native_k4']['correct'],m['native_k5']['correct']
        return 'harm' if c4 and not c5 else 'rescue' if c5 and not c4 else 'stable_correct' if c5 else 'stable_wrong'
    for uid in ids:
        assert '/train/' in uid and uid in discovery
        assert discovery[uid]['prompt_sha256']==native[uid]['prompt_sha256']
    screen_rows={(r['id'],r['condition']):r for r in read_lines(screen/'scores.jsonl')}
    modules=[dict(layer=s['layer'],branch=s['branch'],alpha=.5,screen_rank=s['rank'],
                  screening_decision=s['decision']) for s in shortlist]
    code_names=['runtime.py','ouro_cached_scoring.py','causal_hooks.py','run_shared_confirmation.py']
    codehash=hashlib.sha256(''.join(sha(ROOT/'scripts'/n) for n in code_names).encode()).hexdigest()
    frozen=dict(implementation_sha256=codehash,screen_study=a.screen_study,source_study=source_study,
        screen_analysis_sha256=sha(screen/'MECHANISM_ANALYSIS.json'),
        screen_scores_sha256=sha(screen/'scores.jsonl'),representative_sha256=sha(rep_path),
        native_scores_sha256=sha(native_path),discovery_analysis_sha256=sha(source/'ANALYSIS.json'),
        discovery_sha256=sha(discovery_path),ids=ids,representative_ids=old_ids,
        added_unresolved_harm_ids=added,all_unresolved_harm_ids=sorted(unresolved),modules=modules,
        scopes=['loop5','all_loops'],scoring='native_uncached_Kmax6',
        note='All-loop scalar branch damping is not a raw projection-weight edit.')
    frozen_path=out/'FROZEN.json'
    if frozen_path.exists():assert json.loads(frozen_path.read_text())==frozen,'Resume inputs changed'
    else:save(str(frozen_path.relative_to(ROOT)),frozen)
    path=out/'scores.jsonl';done={}
    if path.exists():
        for r in read_lines(path):
            key=(r['id'],r['layer'],r['branch'],r['scope'])
            assert key not in done and r['implementation_sha256']==codehash
            done[key]=r
    def commit(item):
        with path.open('a',encoding='utf8') as stream:stream.write(json.dumps(item)+'\n')
        done[(item['id'],item['layer'],item['branch'],item['scope'])]=item
    def make_record(uid,module,scope,methods,seconds,evidence):
        base=native[uid]['methods']
        return dict(id=uid,group=group(uid),case_source='frozen_representative' if uid in old_ids else 'added_grid_unresolved_harm',
            chosen_source='configs/representative_cases.json' if uid in old_ids else 'frozen_discovery_normalized_chord_grid',
            data_scope='official_train_discovery_only',layer=module['layer'],branch=module['branch'],scope=scope,
            alpha=.5,methods=methods,elapsed_seconds=seconds,evidence=evidence,
            delta_margin_vs_native={f'k{k}':methods[f'native_k{k}']['margin']-base[f'native_k{k}']['margin'] for k in (3,4,5,6)},
            implementation_sha256=codehash)
    # Reuse already-completed exact matching loop5 intervention; no duplicate GPU cost.
    for module in modules:
        condition=f"damp_loop5_{module['branch']}_layer{module['layer']:02d}"
        for uid in old_ids:
            key=(uid,module['layer'],module['branch'],'loop5')
            if key in done:continue
            record=screen_rows[(uid,condition)]
            assert record['scope']=='official_train_discovery_only' and record['cached'] is False
            assert record['native_k4_max_abs_error']==0
            methods=dict(record['methods'],native_k3=native[uid]['methods']['native_k3'])
            commit(make_record(uid,module,'loop5',methods,0,
                dict(kind='reuse_completed_screen',condition=condition,
                     k3='native score, structurally unchanged by loop5-only intervention',k4_k5_k6='verbatim screen candidate scores')))
    check_budget(90)
    started=time.time();tok,model=load_model();scorer=OuroCachedScorer(model,6)
    import torch
    pairs={uid:[encode_pair(tok,*p) for p in discovery[uid]['request_pairs']] for uid in ids}
    def evaluate(uid,module,loops,record=False):
        values=[]
        spec=BranchIntervention(module['layer'],module['branch'],mode='record' if record else 'damp',alpha=0 if record else .5,loops=loops)
        try:
            with OuroCausalHooks(model,[spec]) as hooks:
                scorer.invalidate()
                for prefix,target in pairs[uid]:
                    check_budget(30)
                    hooks.set_capture_positions(list(range(len(prefix)-1,len(prefix)+len(target)-1)) if record else [])
                    values.append(scorer.score_uncached([(prefix,target)],loopcd=False)[0])
        finally:scorer.invalidate()
        return {f'native_k{k}':outcome([s['sum_loglikelihood'][k] for s in values],
            discovery[uid]['choice_char_lengths'],discovery[uid]['gold']) for k in (3,4,5,6)}
    # Recheck no-op on every process startup, even on resume.
    check_budget(90);noop=evaluate(ids[0],modules[0],None,record=True)
    errors={f'k{k}':max(abs(a-b) for a,b in zip(noop[f'native_k{k}']['sum_scores'],native[ids[0]]['methods'][f'native_k{k}']['sum_scores'])) for k in (3,4,5,6)}
    save(str((out/'NOOP_PARITY.json').relative_to(ROOT)),dict(id=ids[0],errors=errors,passed=all(v==0 for v in errors.values())))
    assert all(v==0 for v in errors.values()),'Exact recording-noop failed'
    work=[(uid,m,'loop5',(5,)) for m in modules for uid in added]+[(uid,m,'all_loops',None) for m in modules for uid in ids]
    expected=len(ids)*len(modules)*2
    for uid,module,scope,loops in work:
        key=(uid,module['layer'],module['branch'],scope)
        if key in done:continue
        check_budget(90);began=time.time();methods=evaluate(uid,module,loops)
        if scope=='loop5':
            for k in (3,4):assert methods[f'native_k{k}']['sum_scores']==native[uid]['methods'][f'native_k{k}']['sum_scores']
        torch.cuda.synchronize()
        commit(make_record(uid,module,scope,methods,time.time()-began,dict(kind='new_native_uncached_forward')))
        progress=dict(status='running',completed=len(done),expected=expected,last_id=uid,last_module=module,last_scope=scope,elapsed_seconds=time.time()-started)
        save(str((out/'PROGRESS.json').relative_to(ROOT)),progress);print(json.dumps(progress),flush=True)
    assert len(done)==expected
    contrasts=[]
    for module in modules:
        for uid in ids:
            local=done[(uid,module['layer'],module['branch'],'loop5')]['methods']
            shared=done[(uid,module['layer'],module['branch'],'all_loops')]['methods']
            for k in (3,4,5,6):
                l,g=local[f'native_k{k}'],shared[f'native_k{k}']
                contrasts.append(dict(id=uid,group=group(uid),case_source='representative' if uid in old_ids else 'added_unresolved',
                    layer=module['layer'],branch=module['branch'],k=k,loop5_correct=l['correct'],all_loops_correct=g['correct'],
                    all_loops_minus_loop5_margin=g['margin']-l['margin'],
                    all_loops_minus_loop5_normalized_scores=[a-b for a,b in zip(g['normalized_scores'],l['normalized_scores'])]))
    save(str((out/'GLOBAL_VS_LOOP5.json').relative_to(ROOT)),dict(contrasts=contrasts))
    save(str((out/'COMPLETE.json').relative_to(ROOT)),dict(status='complete',completed=len(done),expected=expected,
        modules=modules,representative_n=len(old_ids),added_unresolved_n=len(added),elapsed_seconds=time.time()-started,
        implementation_sha256=codehash,
        limitation='Shared scalar-damping failure does not falsify projection directions; no H2 or weight-edit claim.'))


if __name__=='__main__':main()
