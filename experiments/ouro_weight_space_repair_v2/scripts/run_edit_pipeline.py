"""Fixed search -> held-out selection -> one exact edit -> locked regression.

The supervisor charges this entire process, including loading and bookkeeping.
No validation/test rows are opened before their respective frozen lock files.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from runtime import ROOT, REVISION, load_model, encode_pair, outcome, save, check_budget
from ouro_cached_scoring import OuroCachedScorer
from candidate_io import build_candidate
from weight_edits import (applied_edits, save_edit_bundle, load_edit_bundle,
                          file_sha256, tensor_sha256)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line]


def lock(path, value):
    p=ROOT/path
    if p.exists():
        assert read(p)==value, ('Immutable lock mismatch',path)
    else:
        save(path,value)
    return file_sha256(p)


def aggregate(records):
    names=list(records[0]['methods'])
    return {name:dict(n=len(records),correct=sum(r['methods'][name]['correct'] for r in records),
        accuracy=sum(r['methods'][name]['correct'] for r in records)/len(records),
        mean_gold_margin=sum(r['methods'][name]['margin'] for r in records)/len(records)) for name in names}


def order(item):
    v=item['metrics']['native_k4']
    return (-v['correct'],-v['mean_gold_margin'],item['actual_changed_parameters'],item['candidate_id'])


def promote(results,per_family,number):
    selected=[]
    for family in ['A','B']:
        group=sorted([r for r in results if r['family']==family],key=order)
        selected.extend(r['candidate_id'] for r in group[:per_family])
    for r in sorted(results,key=order):
        if len(selected)>=number:break
        if r['candidate_id'] not in selected:selected.append(r['candidate_id'])
    assert len(selected)==number and len(set(selected))==number
    return selected


def stage_data(name,n):
    """Copy existing exact V1 manifests ONLY after their upstream locks exist."""
    assert name in ['validation','test']
    prerequisite='VALIDATION_LOCK.json' if name=='validation' else 'FINAL_LOCK.json'
    assert (ROOT/'configs'/prerequisite).is_file()
    source=ROOT.with_name('ouro_weight_space_repair_v1')/'data'/(name+'.jsonl')
    destination=ROOT/'data'/(name+'.jsonl')
    if not destination.exists():shutil.copy2(source,destination)
    assert file_sha256(source)==file_sha256(destination)
    data=rows(destination)
    assert len(data)==n and len({r['id'] for r in data})==n
    training=rows(ROOT/'data/discovery_expanded.jsonl')+rows(ROOT/'data/selection.jsonl')
    assert not ({r['id'] for r in data}&{r['id'] for r in training})
    save('logs/'+name.upper()+'_STAGING.json',dict(source_experiment='V1_immutable_fixed25',
        sha256=file_sha256(destination),n=n,prerequisite_lock=prerequisite,
        prerequisite_sha256=file_sha256(ROOT/'configs'/prerequisite)))
    return data


def methods(values,row):
    result={f'native_k{k}':outcome([v['sum_loglikelihood'][k] for v in values],
                                  row['choice_char_lengths'],row['gold']) for k in range(1,7)}
    for name in ['fixed','adaptive']:
        result['loopcd_k4_'+name]=outcome([v['loopcd'][name] for v in values],
                                        row['choice_char_lengths'],row['gold'])
    return result


def own_logits_check(model,value,target):
    """Independently recompute both CD formulas from this edited forward's states."""
    import torch
    early=model.lm_head(value['normalized_states'][0].to('cuda'))[0].float()
    final=model.lm_head(value['normalized_states'][3].to('cuda'))[0].float()
    top=final.softmax(-1).topk(2,dim=-1).values
    omega=1-(top[:,0]-top[:,1])
    targets=torch.tensor(target,device='cuda')[:,None]
    errors={}
    for name,z in [('fixed',final+.5*(final-early)),('adaptive',final+omega[:,None]*(final-early))]:
        actual=float(z.log_softmax(-1).gather(-1,targets).sum())
        errors[name]=abs(actual-value['loopcd'][name])
    assert max(errors.values())==0, errors
    return dict(errors=errors,source='same edited forward normalized K1/K4 states',
                coefficients=dict(fixed=.5,adaptive='1 - final_top_probability_gap, cap1'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stop-after',choices=['screen','selection','validation','final'],default='final')
    a=ap.parse_args();start=time.time()
    complete=read(ROOT/'results/training_statistics/COMPLETE_384.json')
    assert complete['status']=='complete' and complete['n']==384
    parity=read(ROOT/'results/training_statistics/PARITY.json');assert parity['exact_recording_and_v1_original']
    frozen=read(ROOT/'configs/CANDIDATES_FROZEN.json')
    specs={c['id']:c for c in frozen['candidates']}
    assert len(specs)==14 and frozen['model_revision']==REVISION and not frozen['baseline_eligible_for_final_selection']
    code_files=['run_edit_pipeline.py','runtime.py','ouro_cached_scoring.py','weight_edits.py','candidate_io.py']
    implementation={p:file_sha256(ROOT/'scripts'/p) for p in code_files}
    signature=hashlib.sha256(json.dumps(implementation,sort_keys=True).encode()).hexdigest()
    data_lock=read(ROOT/'configs/DATA_LOCK.json')
    training_hashes={name:file_sha256(ROOT/'data'/name) for name in
        ['discovery_expanded.jsonl','discovery_screen.jsonl','selection.jsonl','fixed_demonstrations.json']}
    assert training_hashes['selection.jsonl']==data_lock['selection_sha256']
    assert training_hashes['fixed_demonstrations.json']==data_lock['demonstrations_sha256']
    for name in ['discovery_expanded','discovery_screen']:
        assert training_hashes[name+'.jsonl']==data_lock['cohorts'][name]['sha256']
    lock('configs/SEARCH_LOCK.json',dict(model_revision=REVISION,implementation=implementation,
        implementation_sha256=signature,candidates_frozen_sha256=file_sha256(ROOT/'configs/CANDIDATES_FROZEN.json'),
        data_lock_sha256=file_sha256(ROOT/'configs/DATA_LOCK.json'),training_manifest_sha256=training_hashes,
        selection_policy='screen top2 per family; selection best1 per family plus best remaining; validation exactly1 by nativeK4 count/margin/changed/id',
        baseline_eligible=False,final_plain_depths=[3,4,5,6],primary_combo=dict(k=4,omega=.5),secondary_combo=dict(k=4,cap=1)))
    tok,model=load_model();scorer=OuroCachedScorer(model,6)
    total=sum(p.numel() for p in model.parameters());assert total==1434652673
    # Materialize once. Every subsequent stage loads the same BF16 replacement bits.
    manifests={}
    for cid,spec in specs.items():
        check_budget(120);directory=ROOT/'edited_weights/candidates'/cid
        if (directory/'manifest.json').exists():
            edits,manifest=load_edit_bundle(directory,model)
            assert manifest['metadata']['candidate_spec']==spec
            assert manifest['metadata']['candidates_frozen_sha256']==file_sha256(ROOT/'configs/CANDIDATES_FROZEN.json')
        else:
            edits=build_candidate(spec,model)
            manifest=save_edit_bundle(directory,edits,total,metadata=dict(candidate_spec=spec,
                candidates_frozen_sha256=file_sha256(ROOT/'configs/CANDIDATES_FROZEN.json')))
        manifests[cid]=manifest
        assert manifest['actual_changed_parameters']>0
        del edits
    bundle_hashes={cid:file_sha256(ROOT/'edited_weights/candidates'/cid/'manifest.json') for cid in specs}
    lock('configs/MATERIALIZED_WEIGHTS_LOCK.json',dict(candidate_bundle_manifest_sha256=bundle_hashes,
        implementation_sha256=signature,real_bf16_parameters=True))
    discovery=rows(ROOT/'data/discovery_expanded.jsonl')
    screen=rows(ROOT/'data/discovery_screen.jsonl');selection=rows(ROOT/'data/selection.jsonl')
    assert len(screen)==64 and len(selection)==192
    assert {r['id'] for r in screen} <= {r['id'] for r in discovery}
    assert not ({r['id'] for r in selection}&{r['id'] for r in discovery})
    # Changed-path gates use training data only. A prediction flip is not required.
    gate_path=ROOT/'results/EDITED_PATH_GATE.json'
    if not gate_path.exists():
        row=screen[0];pairs=[encode_pair(tok,*p) for p in row['request_pairs']]
        baseline=scorer.score_uncached(pairs)
        gates=[]
        for cid in ['A01','B01']:
            edits,m=load_edit_bundle(ROOT/'edited_weights/candidates'/cid,model)
            with applied_edits(model,edits,invalidate=scorer.invalidate):
                values=scorer.score_uncached(pairs,return_states=True)
                own=own_logits_check(model,values[0],pairs[0][1])
                depth=scorer.check_depth_invariance(pairs[:1],depths=(3,4,5),atol=0)
                assert depth['passed']
                changed_scores=max(abs(x['sum_loglikelihood'][4]-y['sum_loglikelihood'][4]) for x,y in zip(values,baseline))
                assert changed_scores>0
            restored=scorer.score_uncached(pairs)
            assert all(x['sum_loglikelihood']==y['sum_loglikelihood'] and x['loopcd']==y['loopcd'] for x,y in zip(restored,baseline))
            gates.append(dict(candidate_id=cid,actual_changed_parameters=m['actual_changed_parameters'],
                changed_k4_sum_score_max=changed_scores,own_edited_logits=own,
                separate_depth_parity=depth,exact_original_restoration=True,bundle_sha256=bundle_hashes[cid]))
        save('results/EDITED_PATH_GATE.json',dict(status='passed',training_question_id=row['id'],
            real_nn_Parameter_copy=True,no_activation_hooks=True,implementation_sha256=signature,cases=gates))
    else:
        gate=read(gate_path)
        assert gate['implementation_sha256']==signature and gate['status']=='passed'
        assert {g['candidate_id'] for g in gate['cases']}=={'A01','B01'}
        for g in gate['cases']:
            assert g['bundle_sha256']==bundle_hashes[g['candidate_id']] and g['actual_changed_parameters']>0
            assert g['changed_k4_sum_score_max']>0 and g['exact_original_restoration']
            assert g['separate_depth_parity']['passed'] and max(g['own_edited_logits']['errors'].values())==0

    def evaluate(stage,cid,data,bundle=None):
        directory=ROOT/'results'/stage/cid;directory.mkdir(parents=True,exist_ok=True)
        bundle=bundle or ROOT/'edited_weights/candidates'/cid
        edits,manifest=load_edit_bundle(bundle,model)
        path=directory/'scores.jsonl';existing=rows(path) if path.exists() else []
        expected={r['id']:r for r in data};done={}
        bundle_sha=file_sha256(bundle/'manifest.json')
        data_sha=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        for r in existing:
            assert r['id'] in expected and r['id'] not in done
            assert r['scope']==stage and r['candidate_id']==cid and r['stage_data_sha256']==data_sha
            assert r['implementation_sha256']==signature and r['bundle_manifest_sha256']==bundle_sha
            assert r['prompt_sha256']==expected[r['id']]['prompt_sha256'] and r['gold']==expected[r['id']]['gold']
            assert r['choice_char_lengths']==expected[r['id']]['choice_char_lengths']
            done[r['id']]=r
        with applied_edits(model,edits,invalidate=scorer.invalidate):
            for row in data:
                if row['id'] in done:continue
                check_budget(120);beg=time.time();pairs=[encode_pair(tok,*p) for p in row['request_pairs']]
                values=scorer.score_uncached(pairs)
                r=dict(id=row['id'],gold=row['gold'],candidate_id=cid,scope=stage,
                    prompt_sha256=row['prompt_sha256'],choice_char_lengths=row['choice_char_lengths'],
                    choice_token_counts=[len(t) for _,t in pairs],methods=methods(values,row),
                    bundle_manifest_sha256=bundle_sha,implementation_sha256=signature,
                    stage_data_sha256=data_sha,
                    elapsed_seconds=time.time()-beg,loopcd_source='this edited checkpoint own K1/K4 logits')
                with path.open('a',encoding='utf8') as f:f.write(json.dumps(r)+'\n')
                done[r['id']]=r
                if len(done)%8==0 or len(done)==len(data):
                    progress=dict(stage=stage,candidate_id=cid,completed=len(done),target=len(data),
                        worker_elapsed_seconds=time.time()-start)
                    save('results/PIPELINE_PROGRESS.json',progress);print(json.dumps(progress),flush=True)
            # Audit exact tensor bits at exit; same parameter object is shared at all loops.
            params=dict(model.named_parameters())
            assert all(tensor_sha256(params[e.name])==tensor_sha256(e.edited) for e in edits)
        ordered=[done[r['id']] for r in data]
        summary=dict(candidate_id=cid,family=specs[cid]['family'],stage=stage,
            n=len(data),metrics=aggregate(ordered),actual_changed_parameters=manifest['actual_changed_parameters'],
            delta_frobenius=manifest['actual_delta_frobenius'],bundle_manifest_sha256=bundle_sha,
            scores_sha256=file_sha256(path),implementation_sha256=signature)
        summary['stage_data_sha256']=data_sha
        save(str(directory.relative_to(ROOT))+'/COMPLETE.json',summary)
        return summary

    screen_results=[evaluate('screen',cid,screen) for cid in specs]
    chosen=promote(screen_results,per_family=2,number=4)
    lock('configs/SELECTION_LOCK.json',dict(candidates=chosen,rule='best2 per family from 64 training discovery questions',
        ranking=sorted(screen_results,key=order),candidate_bundles=bundle_hashes))
    if a.stop_after=='screen':return
    selection_results=[evaluate('selection',cid,selection) for cid in chosen]
    finalists=promote(selection_results,per_family=1,number=3)
    lock('configs/VALIDATION_LOCK.json',dict(candidates=finalists,rule='best1 per family plus best remaining from 192 training selection',
        ranking=sorted(selection_results,key=order),candidate_bundles={c:bundle_hashes[c] for c in finalists},
        no_further_edits_or_strength_search=True))
    if a.stop_after=='selection':return
    validation=stage_data('validation',299)
    val_results=[evaluate('validation',cid,validation) for cid in finalists]
    winner=sorted(val_results,key=order)[0]['candidate_id']
    source=ROOT/'edited_weights/candidates'/winner;selected=ROOT/'edited_weights/selected'
    if not selected.exists():shutil.copytree(source,selected)
    assert file_sha256(source/'manifest.json')==file_sha256(selected/'manifest.json')
    selected_edits,selected_manifest=load_edit_bundle(selected,model)
    assert selected_manifest['actual_changed_parameters']>0
    del selected_edits
    lock('selected_edit_manifest.json',dict(**selected_manifest,selected_candidate_id=winner,
        selected_bundle='edited_weights/selected',selected_bundle_manifest_sha256=file_sha256(selected/'manifest.json'),
        validation_ranking=sorted(val_results,key=order),original_baseline_eligible=False,
        selection_criterion='native K4 correct count then mean gold margin then actual BF16 changed count then ID',
        selected_even_if_inferior_to_original=True))
    lock('configs/FINAL_LOCK.json',dict(selected_candidate_id=winner,
        selected_manifest_sha256=file_sha256(ROOT/'selected_edit_manifest.json'),
        bundle_manifest_sha256=file_sha256(selected/'manifest.json'),model_revision=REVISION,
        plain_depths=[3,4,5,6],primary_combination=dict(name='fixed_loopcd',k=4,reference_k=1,omega=.5),
        secondary_combination=dict(name='adaptive_loopcd',k=4,reference_k=1,cap=1.),
        loopcd_source='same selected edited checkpoint own logits; no original caches',
        test_role='previously used official ARC-Challenge locked regression benchmark',
        no_test_driven_edit_or_coefficient_changes=True))
    if a.stop_after=='validation':return
    test=stage_data('test',1172)
    final=evaluate('final_test',winner,test,bundle=selected)
    # Only now read old official test baseline rows; never use them to change edits.
    for split in ['selection','validation','test']:
        src=ROOT.with_name('ouro_weight_space_repair_v1')/'results'/('frozen_'+split)/'scores.jsonl'
        dst=ROOT/'results/original_reference'/split/'scores.jsonl';dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(src,dst)
        assert file_sha256(src)==file_sha256(dst)
        reference=rows(dst);current=selection if split=='selection' else validation if split=='validation' else test
        reference_by_id={r['id']:r for r in reference};assert len(reference_by_id)==len(current)
        assert all(reference_by_id[r['id']]['prompt_sha256']==r['prompt_sha256'] and reference_by_id[r['id']]['gold']==r['gold'] for r in current)
    save('results/PIPELINE_COMPLETE.json',dict(status='complete',selected_candidate_id=winner,
        final=final,elapsed_seconds=time.time()-start,implementation_sha256=signature,
        test_was_opened_after_final_lock=True,original_test_reference_opened_after_selected_evaluation=True))
    print('V2_EDIT_PIPELINE_COMPLETE',winner,flush=True)


if __name__=='__main__':main()
